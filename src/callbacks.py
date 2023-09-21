import os
import time
import hashlib
import glob
import json
import pickle

from dataset_utils import process_dataset

from aim.hugging_face import AimCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers import OPTForCausalLM
import torch
from transformers.training_args import TrainingArguments
from datasets import load_dataset
import accelerate


def calc_hash_for_binary_file(path):
    with open(path, "rb") as _file:
        file_content = _file.read()
        hex_hash = hashlib.md5(file_content).hexdigest()
        return hex_hash


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
    def on_save(self, args, state, control, model, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        saved_model = OPTForCausalLM.from_pretrained(checkpoint_dir).to(model.device)

        training_data_files = glob.glob(".small_data/train" + "/*.jsonl")

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

        is_repr = True
        for inp in processed_dataset["data"]:
            del inp["token_type_ids"]
            inp = {k: inp[k].unsqueeze(0).to(model.device) for k in inp.keys()}
            out = model(**inp)
            saved_out = saved_model(**inp)

            ok = torch.allclose(out.logits, saved_out.logits, atol=1e-4)
            if not ok:
                is_repr = False
            break

        if is_repr:
            print(f"Model at step {state.global_step} is reproducable.")
        else:
            print(f"Model at step {state.global_step} is not reproducable.")


class JsonlDatasetResumeCallback(TrainerCallback):

    def __init__(
        self,
        communication_dir="/tmp/jsonl_states",
        file_name_to_store_states="jsonl_states"
    ):
        if not os.path.exists(communication_dir):
            os.mkdir(communication_dir)
        self.jsonl_datasets_states = {}
        self.communication_dir = communication_dir
        self.file_name_to_store_states = file_name_to_store_states
        self.pickle_states_path = os.path.join(self.communication_dir, f"{self.file_name_to_store_states}.pickle")
        if os.path.exists(self.pickle_states_path):
            os.remove(self.pickle_states_path)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.resume_from_checkpoint: # resume training
            jsonl_states_path = os.path.join(args.resume_from_checkpoint, f"{self.file_name_to_store_states}.json")
            with open(jsonl_states_path, "r") as file:
                self.jsonl_datasets_states = json.load(file)

            with open(self.pickle_states_path, "wb") as file:
                pickle.dump(self.jsonl_datasets_states, file)
            
            accelerate.skip_first_batches = lambda: None # disable this function to not skip any steps

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        assert os.path.exists(self.pickle_states_path)
        os.remove(self.pickle_states_path)
        os.rmdir(self.communication_dir)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.distributed.get_rank() == 0:
            assert os.path.exists(self.pickle_states_path)

            with open(self.pickle_states_path, "rb") as file:
                jsonl_states = pickle.load(file)

            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            jsonl_states_path = os.path.join(checkpoint_dir, f"{self.file_name_to_store_states}.json")
            with open(jsonl_states_path, "w") as file:
                json.dump(jsonl_states, file, indent=4)
            print(f"Jsonl datasets states saved to {jsonl_states_path}")
