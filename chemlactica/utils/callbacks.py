import os
import time
import hashlib
import glob
import gc
import json

from chemlactica.config.create_train_config import model_train_configs
from .dataset_utils import process_dataset
from datasets import load_dataset
from .model_utils import load_model
from tqdm.auto import tqdm

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
                # Calculate time taken for this step
                elapsed_time = time.time() - self._start_time
                # Calculate words per second
                words_per_second = num_words / elapsed_time
                self._aim_run.track(words_per_second, name="words per second")

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
    def __init__(self, model_config, use_flash_attn=False):
        self.model_config = model_config
        self.train_config = model_train_configs[model_config]
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
            train_config={"block_size": 2048},
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
