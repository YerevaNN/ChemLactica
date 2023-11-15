import subprocess
import unittest
import gc
import os
import sys
import shutil

import torch

from test_utils import create_train_command

test_directory = "/tmp/chemlactica_fsdp_precommit_test"


class TestModelTraining(unittest.TestCase):

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

    def test_model_train(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

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
                "max_steps": 1000,
                "eval_steps": 2000,
                "save_steps": 2000,
                "dataloader_num_workers": 1,
                "checkpoints_root_dir": f"{test_directory}/checkpoints",
                "experiment_name": "fsdp_model_train",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            }
        )

        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=True)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())
        else:
            print(out.stdout.decode())

    def test_model_valid(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

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
                "max_steps": 100,
                "eval_steps": 10,
                "save_steps": 2000,
                "dataloader_num_workers": 1,
                "checkpoints_root_dir": f"{test_directory}/checkpoints",
                "experiment_name": "fsdp_model_valid",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            }
        )

        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=True)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())
        else:
            print(out.stdout.decode())

    def test_model_resume(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        first_command = create_train_command(
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
                "eval_steps": 10,
                "save_steps": 10,
                "dataloader_num_workers": 1,
                "checkpoints_root_dir": f"{test_directory}/checkpoints",
                "experiment_name": "fsdp_model_resume",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            }
        )

        print(f"Running command: {first_command}")
        out = subprocess.run(first_command, shell=True, capture_output=True)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())
        else:
            print(out.stdout.decode())

        second_command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": "src/config/test_configs/fsdp_config.yaml"},
            script="src/train.py",
            script_args={
                "from_pretrained": f"{test_directory}/checkpoints/facebook/galactica-125m/none/checkpoint-{20}",
                "model_config": "125m",
                "training_data_dir": ".small_data/train",
                "valid_data_dir": ".small_data/valid",
                "train_batch_size": 4,
                "max_steps": 40,
                "eval_steps": 10,
                "save_steps": 10,
                "dataloader_num_workers": 1,
                "checkpoints_root_dir": f"{test_directory}/checkpoints",
                "experiment_name": "fsdp_model_resume",
                "gradient_accumulation_steps": 1,
                "no_track": "", 
                "flash_attn": "",
            }
        )

        print(f"Running command: {second_command}")
        out = subprocess.run(second_command, shell=True, capture_output=True)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())
        else:
            print(out.stdout.decode())


if __name__ == "__main__":
    unittest.main(verbosity=2)