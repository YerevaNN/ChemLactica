from aim.hugging_face import AimCallback
import os
import hashlib
import torch
import time
from transformers import TrainerCallback


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

    def on_save(self, args, state, control=None, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        checkpoints_dict = self._run[self._checkpoints_dict_name]
        checkpoints_dict[state.global_step] = {}
        for file_name in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file_name)
            checkpoints_dict[state.global_step][file_name] = calc_hash_for_binary_file(
                file_path
            )
        self._run[self._checkpoints_dict_name] = checkpoints_dict

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # Get batch size (first dimension of inputs)
        self.embedding_norm_1 = torch.linalg.norm(
            self.model.get_input_embeddings().weight, ord=1
        )
        self.embedding_norm_2 = torch.linalg.norm(
            self.model.get_input_embeddings().weight, ord=2
        )

        self.experiment.track(self.embedding_norm_1, name="embedding l1 norm")
        self.experiment.track(self.embedding_norm_2, name="embedding l2 norm")

        batch_size = (
            args.per_device_train_batch_size
        )  # kwargs['model_inputs']['input_ids'].shape[0]
        # Calculate tokens in batch
        num_words = batch_size * self.blocksize  # args.blocksize
        # Calculate time taken for this step
        elapsed_time = time.time() - self.start_time  # Calculate words per second
        words_per_second = num_words / elapsed_time
        self.experiment.track(words_per_second, name="words per second")


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
