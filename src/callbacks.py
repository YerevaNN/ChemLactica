import os
import time
import hashlib
import glob
import gc

from dataset_utils import process_dataset
from datasets import load_dataset
from custom_transformer import load_model
from utils import chemlactica_special_tokens

from aim.hugging_face import AimCallback
import torch
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    ProgressCallback,
)
from transformers.training_args import TrainingArguments
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def calc_hash_for_binary_file(path):
    with open(path, "rb") as _file:
        file_content = _file.read()
        hex_hash = hashlib.md5(file_content).hexdigest()
        return hex_hash


class CustomProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))
            logger.info(str(logs))


class CustomAimCallback(AimCallback):
    def __init__(self, checkpoints_dict_name, model, blocksize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoints_dict_name = checkpoints_dict_name
        self.model = model
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
                num_words = batch_size * self._block_size * args.world_size
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
    def __init__(self, accelerator, model_config, train_config):
        self.accelerator = accelerator
        self.model_config = model_config
        self.train_config = train_config
        
    def on_save(self, args, state, control, model, **kwargs):
        print(f"Process {torch.distributed.get_rank()}")
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
            if i == 20: break

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        print("Loading from checkpoint:", checkpoint_dir)

        accelerator = Accelerator()
        saved_model = load_model(f"facebook/galactica-{self.model_config}", flash_att=True, dtype=torch.bfloat16)
        saved_model.resize_token_embeddings(
            self.train_config["vocab_size"] + len(chemlactica_special_tokens)
        )
        saved_model = accelerator.prepare(saved_model)
        accelerator.load_state(checkpoint_dir)
        saved_model.to(accelerator.device)

        # saved_model = load_model(checkpoint_dir, flash_att=True, dtype=torch.bfloat16)
        # saved_model.to(model.device)

        model.eval()
        saved_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(batches):
                out = model(**batch)
                saved_md_out = saved_model(**batch)
                # print(i, batch)

                logits_diff = torch.abs(out.logits - saved_md_out.logits)
                print(f"Logits difference {i} min {logits_diff.min():.6f}, max {logits_diff.max():.6f}, mean {logits_diff.mean():.6f}, median {logits_diff.median():.6f}")
                loss_diff = torch.abs(out.loss - saved_md_out.loss)
                print(f"loss difference {loss_diff}")
                different_tokens_count = torch.sum(out.logits.softmax(-1).argmax(-1) != saved_md_out.logits.softmax(-1).argmax(-1))
                print("different token count", different_tokens_count.item())

        contexts = [
            "[CLOGP 100][START_SMILES]",
            "[SAS 1][START_SMILES]",
            "[WEIGHT 41.123][START_SMILES]",
            "random input",
        ]

        # with torch.no_grad():
        #     for cont in contexts:
        #         max_length = 400
        #         inputs = get_tokenizer()(cont, return_tensors="pt").to(model.device)
        #         generated_toks = saved_model.generate(
        #             inputs["input_ids"],
        #             max_length=max_length,
        #             do_sample=False
        #         )
        #         saved_md_generated_toks = saved_model.generate(
        #             inputs["input_ids"],
        #             max_length=max_length,
        #             do_sample=False
        #         )
        #         generated_toks = generated_toks.squeeze()
        #         saved_md_generated_toks = saved_md_generated_toks.squeeze()
        #         maximum = max(len(generated_toks), len(saved_md_generated_toks))
        #         print(len(saved_md_generated_toks), len(generated_toks), maximum)
        #         generated_toks = F.pad(generated_toks, pad=(0, maximum - len(generated_toks)), mode='constant', value=0)
        #         saved_md_generated_toks = F.pad(saved_md_generated_toks, pad=(0, maximum - len(saved_md_generated_toks)), mode='constant', value=0)
        #         print(generated_toks.shape, saved_md_generated_toks.shape)
        #         diff_gen_tokens = torch.sum(generated_toks.squeeze() != saved_md_generated_toks.squeeze())
        #         print(f"Checking diff generated tokens (max_length={max_length}) '{cont}': count {diff_gen_tokens}")
