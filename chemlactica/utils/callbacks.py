import os
import time
import hashlib
import glob
import gc
import json

from .dataset_utils import process_dataset
from datasets import load_dataset
from .model_utils import load_model
from tqdm.auto import tqdm
from sklearn.metrics import root_mean_squared_error


from aim.hugging_face import AimCallback
import torch
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    ProgressCallback,
)
from transformers.training_args import TrainingArguments
import accelerate
from accelerate.logging import get_logger

logger = get_logger(__name__)


def calc_hash_for_binary_file(path):
    with open(path, "rb") as _file:
        file_content = _file.read()
        hex_hash = hashlib.md5(file_content).hexdigest()
        return hex_hash


class CustomProgressCallback(ProgressCallback):
    def __init__(self, early_stopping_steps, total_theoretical_peak_flops):
        self.training_bar = None
        self.prediction_bar = None
        self.early_stopping_steps = early_stopping_steps
        self._start_flos = 0
        self._total_theoretical_peak_flops = total_theoretical_peak_flops
        self._start_time = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(
                total=self.early_stopping_steps, dynamic_ncols=True
            )
        self.current_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed_time = time.time() - self._start_time
        new_flos = state.total_flos
        elapsed_flos = (new_flos - self._start_flos) / elapsed_time
        mfu = (elapsed_flos / self._total_theoretical_peak_flops) * 100
        self._start_time = time.time()
        self._start_flos = new_flos
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            logs.update({"mfu": mfu})
            self.training_bar.write(str(logs))
            logger.info(str(logs))


class CustomAimCallback(AimCallback):
    def __init__(
        self, checkpoints_dict_name, model, blocksize, run_hash, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._checkpoints_dict_name = checkpoints_dict_name
        self.model = model
        self._run_hash = run_hash
        self.setup()
        self.embedding_norm_1 = 0
        self.embedding_norm_2 = 0
        self.activations_norm = 0
        self._run[self._checkpoints_dict_name] = {}
        self.blocksize = blocksize
        self.start_time = None
        self._run["repo_path"] = str(os.path.abspath(os.getcwd()))

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)

        for arg_name, arg_value in vars(args).items():
            self._run["TrainingArguments/" + arg_name] = str(arg_value)
        for state_name, state_value in vars(state).items():
            self._run["TrainerState/" + state_name] = str(state_value)
        # Log the model configuration
        if self.model:
            for config_name, config_value in vars(self.model.config).items():
                self._run["ModelConfig/" + config_name] = str(config_value)
        self.model = None

    def on_save(self, args, state, control=None, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        checkpoints_dict = self._run[self._checkpoints_dict_name]
        checkpoints_dict[state.global_step] = {}
        for file_name in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file_name)
            if os.path.isfile(file_path):
                checkpoints_dict[state.global_step][
                    file_name
                ] = calc_hash_for_binary_file(file_path)
        self._run[self._checkpoints_dict_name] = checkpoints_dict

    # def on_step_begin(self, args, state, control, **kwargs):
    #     self.start_time = time.time()

    # def on_step_end(self, args, state, control, **kwargs):
    # Get batch size (first dimension of inputs)
    # self.embedding_norm_1 = torch.linalg.norm(
    #     self.model.get_input_embeddings().weight, ord=1
    # )
    # self.embedding_norm_2 = torch.linalg.norm(
    #     self.model.get_input_embeddings().weight, ord=2
    # )
    # self.embedding_norm_1, self.embedding_norm_2 = (
    #     0,
    #     0,
    # )  # embedding norm should be modified to work with fsdp wrapped model

    # self.experiment.track(self.embedding_norm_1, name="embedding l1 norm")
    # self.experiment.track(self.embedding_norm_2, name="embedding l2 norm")
    # pass


class WPSCounterCallback(TrainerCallback):
    def __init__(self, block_size, aim_run=None):
        self._aim_run = aim_run
        self._block_size = block_size
        self._start_time = None
        self.cum_words_seen = 0

    def on_step_begin(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero and self._aim_run is not None:
            if self._start_time is not None:
                batch_size = args.per_device_train_batch_size
                # Calculate tokens in batch
                num_words = (
                    batch_size
                    * self._block_size
                    * args.world_size
                    * args.gradient_accumulation_steps
                )
                self.cum_words_seen += num_words
                # Calculate time taken for this step
                elapsed_time = time.time() - self._start_time
                # Calculate words per second
                words_per_second = num_words / elapsed_time
                self._aim_run.track(words_per_second, name="words per second")
                self._aim_run.track(self.cum_words_seen, name="cum_words_seen")
            self._start_time = time.time()


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class EpochCallback(TrainerCallback):
    def __init__(self, num_epochs=1):
        self._num_epochs = num_epochs
        self._current_epoch = 0

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._current_epoch += 1
        if self._current_epoch == self._num_epochs:
            control.should_training_stop = True


class ReproducabilityCallback(TrainerCallback):
    def __init__(self, train_config, model_config, use_flash_attn=False):
        self.train_config = train_config
        self.model_config = model_config
        self.use_flash_attn = use_flash_attn

    def on_save(self, args, state, control, model, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

        training_data_files = glob.glob(".small_data/valid" + "/*.jsonl")

        dataset = load_dataset(
            "text",
            data_files={"data": training_data_files},
            streaming=True,
        )

        processed_dataset = process_dataset(
            dataset=dataset,
            train_config=self.train_config,
            model_config=self.model_config,
            process_batch_sizes=(100, 100),
        )

        batches = []
        for i, inp in enumerate(processed_dataset["data"]):
            del inp["token_type_ids"]
            inp = {k: inp[k].unsqueeze(0).to(model.device) for k in inp.keys()}
            batches.append(inp)
            if i == 20:
                break

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        model.eval()
        model_logits = []
        # model_gen_toks = {}
        with torch.no_grad():
            for i, batch in enumerate(batches):
                model_logits.append(model(**batch))

        if torch.distributed.get_rank() == 0:
            print(
                f"Loading from checkpoint: {checkpoint_dir} (process {torch.distributed.get_rank()})"  # noqa
            )
            saved_model = load_model(
                checkpoint_dir, use_flash_attn=self.use_flash_attn, dtype=torch.bfloat16
            )
            saved_model.to(model.device)

            saved_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(batches):
                    out = model_logits[i]
                    saved_md_out = saved_model(**batch)
                    # print(i, batch)

                    logits_diff = torch.abs(out.logits - saved_md_out.logits)
                    if logits_diff.max() != 0:
                        print(
                            f"MISMATCH: logits difference {i} min {logits_diff.min()}, max {logits_diff.max()}, mean {logits_diff.mean()}, median {logits_diff.median()}"  # noqa
                        )
                    loss_diff = torch.abs(out.loss - saved_md_out.loss)
                    if loss_diff != 0:
                        print(f"MISMATCH: loss difference {loss_diff}")
                    different_tokens_count = torch.sum(
                        out.logits.softmax(-1).argmax(-1)
                        != saved_md_out.logits.softmax(-1).argmax(-1)
                    )
                    if different_tokens_count != 0:
                        print(
                            "MISMATCH: different token count",
                            different_tokens_count.item(),
                        )

                # for cont in contexts:
                #     max_length = 400
                #     inputs = get_tokenizer()(cont, return_tensors="pt").to(saved_model.device)
                #     generated_toks = model_gen_toks[cont]
                #     saved_md_generated_toks = saved_model.generate(
                #         inputs["input_ids"],
                #         max_length=max_length,
                #         do_sample=False
                #     )
                #     generated_toks = generated_toks.squeeze()
                #     saved_md_generated_toks = saved_md_generated_toks.squeeze()
                #     maximum = max(len(generated_toks), len(saved_md_generated_toks))
                #     print(len(saved_md_generated_toks), len(generated_toks), maximum)
                #     generated_toks = F.pad(generated_toks,
                # pad=(0, maximum - len(generated_toks)),
                # mode='constant', value=0)
                #     saved_md_generated_toks = F.pad(saved_md_generated_toks,
                # pad=(0, maximum - len(saved_md_generated_toks)),
                # mode='constant', value=0)
                #     print(generated_toks.shape, saved_md_generated_toks.shape)

        torch.distributed.barrier()


# return the usual dataloader, no batches skipped
accelerate.skip_first_batches = lambda dataloader, num_batches=0: dataloader


class JsonlDatasetResumeCallback(TrainerCallback):
    def __init__(self, shared_jsonl_files):
        self.shared_jsonl_files = shared_jsonl_files

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.resume_from_checkpoint:  # resume training
            print("Resuming from saved jsonl states.")
            with open(
                os.path.join(args.resume_from_checkpoint, "jsonl_states.json"), "r"
            ) as file:
                jsonl_states = json.load(file)

            # assert not self.shared_jsonl_files
            for name, state in jsonl_states.items():
                print(f"loadeding state {name}: {state}")
                self.shared_jsonl_files[name] = state

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        assert self.shared_jsonl_files
        jsonl_states = {key: value for key, value in self.shared_jsonl_files.items()}
        print(jsonl_states)

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        print("Saving jsonl states")
        for name, state in jsonl_states.items():
            print(name, state)
        with open(os.path.join(checkpoint_dir, "jsonl_states.json"), "w") as file:
            json.dump(jsonl_states, file, indent=4)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_steps):
        self.early_stopping_steps = early_stopping_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.early_stopping_steps:
            control.should_training_stop = True


class SFTNumericalEval(TrainerCallback):
    def __init__(self, dataset, aim_callback, separator_token) -> None:
        super().__init__()
        self.dataset = dataset
        self.aim = aim_callback
        self.separator_token = separator_token

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        tokenizer,
        **kwargs,
    ):
        super().on_evaluate(args, state, control, **kwargs)
        model.eval()
        ground_truths, gens, diffs = [], [], []
        eos_token_id = tokenizer.encode("[/PROPERTY]")[0]
        for sample in self.dataset["validation"]:
            ground_truth = round(sample["activity"], 2)
            prompt = (
                f"{self.separator_token}[START_SMILES]{sample['smiles']}"
                "[END_SMILES][PROPERTY]activity"
            )
            prompt = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                prompt.input_ids,
                do_sample=False,
                eos_token_id=eos_token_id,
                max_new_tokens=100,
            )
            out = tokenizer.batch_decode(out)[0]
            try:
                gen = out[
                    out.find("activity ")
                    + len("activity ") : out.find("[/PROPERTY]")  # noqa
                ]
                gen = float(gen)
                diff = abs(ground_truth - gen)
                ground_truths.append(ground_truth)
                gens.append(gen)
                diffs.append(diff)
            except ValueError:
                print(f"could not generate for {sample['smiles']}")
                pass
        rmse = root_mean_squared_error(ground_truths, gens)
        self.aim._run.track({"numerical eval rmse": rmse}, step=state.global_step)
        print(f"{rmse=}")


class GradientAccumulationScheduler(TrainerCallback):
    def __init__(
        self,
        aim_callback,
        dynamic_ga,
        max_ga=256,
        ga_delta_steps=100,
        ga_delta_percentage=0.1,
        patience=1000,
    ) -> None:
        super().__init__()
        print("init dynamic ", dynamic_ga)
        self.aim = aim_callback
        self.dynamic_grad_ac = dynamic_ga
        self.max_ga = max_ga
        self.ga_delta_steps = ga_delta_steps
        self.ga_delta_percentage = ga_delta_percentage
        self.wait = 0
        self.patience = patience
        # 20 is also an arbitrary number, look at the comment bellow.
        assert self.ga_delta_steps * 20 < self.patience

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(
            f"is local process zero: {state.is_local_process_zero}, "
            f"step: {state.global_step}, grad acc: {args.gradient_accumulation_steps}"
        )
        if self.wait == self.patience:
            if self.dynamic_grad_ac:
                # 20 and 19 are arbitrary numbers.
                # taking the average of loss for a window of [-2000:-1900] for delta steps=100
                last_far_loss = [
                    s["loss"]
                    for s in state.log_history[
                        -20 * self.ga_delta_steps : -19 * self.ga_delta_steps  # noqa
                    ]  # noqa
                ]  # noqa
                last_near_loss = [
                    s["loss"] for s in state.log_history[-self.ga_delta_steps :]  # noqa
                ]  # noqa
                mean_far = sum(last_far_loss) / self.ga_delta_steps
                mean_near = sum(last_near_loss) / self.ga_delta_steps
                if mean_far - mean_near < mean_far * self.ga_delta_percentage:
                    args.gradient_accumulation_steps *= 2
                print(f"far 100 mean: {mean_far}, near 100 mean: {mean_near}")
            else:
                args.gradient_accumulation_steps *= 2
            args.gradient_accumulation_steps = min(
                args.gradient_accumulation_steps, self.max_ga
            )
            self.wait = 0
            if state.is_local_process_zero:
                print(
                    "gradient accumulation updated to "
                    f"{args.gradient_accumulation_steps} at step {state.global_step}"
                )
        else:
            self.wait += 1
        if state.is_world_process_zero and self.aim is not None:
            self.aim._run.track(
                {"gradient accumulation steps": args.gradient_accumulation_steps},
                step=state.global_step,
            )
        super().on_step_begin(args, state, control, **kwargs)
