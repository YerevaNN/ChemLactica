import subprocess
import unittest
import gc
import os
import sys
import shutil

import torch

from test_utils import create_train_command


test_directory = "/tmp/chemlactica_fsdp_precommit_test"


class TestConsistencyOfModelOutput(unittest.TestCase):

    def setUp(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.exists(test_directory):
            print(f"Removing {test_directory}")
            shutil.rmtree(test_directory)
        os.mkdir(test_directory)
        os.mkdir(f"{test_directory}/checkpoints")

    def tearDown(self):
        shutil.rmtree(test_directory)

        # clean up
        gc.collect()
        torch.cuda.empty_cache()

    def test_consist_of_model_output(self):
        command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": "src/config/test_configs/fsdp_config.yaml"},
            script="src/train.py",
            script_args={
                "from_pretrained": "facebook/galactica-125m",
                "model_config": "125m",
                "training_data_dir": ".small_data/train",
                "valid_data_dir": ".small_data/valid",
                "train_batch_size": 4,
                "max_steps": 20,
                "eval_steps": 5,
                "save_steps": 5,
                "dataloader_num_workers": 1,
                "checkpoints_root_dir": f"{test_directory}/checkpoints",
                "experiment_name": "fsdp_model_output_consist",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "check_reproducability": "",
                "flash_attn": "",
            }
        )
        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=True)
        if out.returncode != 0:
            print(f"error: {out.stderr}")
            raise Exception()


if __name__ == "__main__":
    unittest.main(verbosity=2)
