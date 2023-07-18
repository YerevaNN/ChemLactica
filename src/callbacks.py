from aim.hugging_face import AimCallback
import os
import hashlib


def calc_hash_for_binary_file(path):
    with open(path, "rb") as _file:
        file_content = _file.read()
        hex_hash = hashlib.md5(file_content).hexdigest()
        return hex_hash


class CustomAimCallback(AimCallback):
    def __init__(self, checkpoints_dict_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoints_dict_name = checkpoints_dict_name
        self.setup()
        self._run[self._checkpoints_dict_name] = {}

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
