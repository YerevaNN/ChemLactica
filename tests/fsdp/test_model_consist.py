import subprocess
import unittest
import gc
import os

# import sys
import shutil

import torch

from test_utils import create_train_command, TD_PATH, TEST_DIR


class TestConsistencyOfModelOutput(unittest.TestCase):
    def setUp(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.exists(TEST_DIR):
            print(f"Removing {TEST_DIR}")
            shutil.rmtree(TEST_DIR)
        os.mkdir(TEST_DIR)
        os.mkdir(os.path.join(TEST_DIR, "checkpoints"))

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

        # clean up
        gc.collect()
        torch.cuda.empty_cache()

    def test_consist_of_model_output(self):
        command = create_train_command(
            module="accelerate.commands.launch",
            module_args={
                "config_file": "chemlactica/config/test_configs/fsdp_config.yaml"
            },
            script="chemlactica/train.py",
            script_args={
                "from_pretrained": "facebook/galactica-125m",
                "model_config": "125m",
                "training_data_dirs": f"{os.path.join(TD_PATH, 'comp_train')} {os.path.join(TD_PATH, 'assay_train')}",  # noqa
                "dir_data_types": "computed assay",
                "valid_data_dir": f"{os.path.join(TD_PATH, 'comp_valid')}",
                "train_batch_size": 4,
                "max_steps": 20,
                "eval_steps": 5,
                "save_steps": 5,
                "dataloader_num_workers": 1,
                "checkpoints_root_dir": os.path.join(TEST_DIR, "checkpoints"),
                "experiment_name": "fsdp_model_output_consist",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "check_reproducability": "",
                "flash_attn": "",
            },
        )
        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=False)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())


if __name__ == "__main__":
    unittest.main(verbosity=2)
