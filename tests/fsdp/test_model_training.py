import subprocess
import unittest
import gc
import os

# import sys
import shutil

import torch

from test_utils import create_train_command, TEST_DIR, TD_PATH


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.config_file = "chemlactica/config/test_configs/fsdp_config.yaml"
        self.script = "chemlactica/train.py"
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

    def test_model_train(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": self.config_file},
            script=self.script,
            script_args={
                "from_pretrained": "facebook/galactica-125m",
                "model_config": "125m",
                "training_data_dirs": f"{os.path.join(TD_PATH, 'assay_train')}",
                "dir_data_types": "assay",
                "valid_data_dir": f"{os.path.join(TD_PATH, 'comp_valid')}",
                "train_batch_size": 4,
                "shuffle_buffer_size": 4,
                "max_steps": 300,
                "eval_steps": 2000,
                "save_steps": 2000,
                "dataloader_num_workers": 8,
                "checkpoints_root_dir": os.path.join(TEST_DIR, "checkpoints"),
                "experiment_name": "fsdp_model_train",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            },
        )

        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=False)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())

    def test_model_train_interleaved(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": self.config_file},
            script=self.script,
            script_args={
                "from_pretrained": "facebook/galactica-125m",
                "model_config": "125m",
                "training_data_dirs": f"{os.path.join(TD_PATH, 'comp_train')} {os.path.join(TD_PATH, 'assay_train')}",  # noqa
                "dir_data_types": "computed assay",
                "valid_data_dir": f"{os.path.join(TD_PATH, 'comp_valid')}",
                "train_batch_size": 4,
                "shuffle_buffer_size": 4,
                "max_steps": 300,
                "eval_steps": 2000,
                "save_steps": 2000,
                "dataloader_num_workers": 8,
                "checkpoints_root_dir": os.path.join(TEST_DIR, "checkpoints"),
                "experiment_name": "fsdp_model_train",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            },
        )

        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=False)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())

    def test_model_valid(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": self.config_file},
            script=self.script,
            script_args={
                "from_pretrained": "facebook/galactica-125m",
                "model_config": "125m",
                "training_data_dirs": f"{os.path.join(TD_PATH, 'comp_train')} {os.path.join(TD_PATH, 'assay_train')}",  # noqa
                "dir_data_types": "computed assay",
                "valid_data_dir": f"{os.path.join(TD_PATH, 'comp_valid')}",
                "train_batch_size": 4,
                "max_steps": 100,
                "eval_steps": 10,
                "save_steps": 2000,
                "dataloader_num_workers": 8,
                "checkpoints_root_dir": os.path.join(TEST_DIR, "checkpoints"),
                "experiment_name": "fsdp_model_valid",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            },
        )

        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=False)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())

    def test_model_resume(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        first_command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": self.config_file},
            script=self.script,
            script_args={
                "from_pretrained": "facebook/galactica-125m",
                "model_config": "125m",
                "training_data_dirs": f"{os.path.join(TD_PATH, 'comp_train')} {os.path.join(TD_PATH, 'assay_train')}",  # noqa
                "dir_data_types": "computed assay",
                "valid_data_dir": f"{os.path.join(TD_PATH, 'comp_valid')}",
                "train_batch_size": 4,
                "max_steps": 20,
                "eval_steps": 10,
                "save_steps": 10,
                "dataloader_num_workers": 8,
                "checkpoints_root_dir": os.path.join(TEST_DIR, "checkpoints"),
                "experiment_name": "fsdp_model_resume",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            },
        )

        print(f"Running command: {first_command}")
        out = subprocess.run(first_command, shell=True, capture_output=False)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())

        second_command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": self.config_file},
            script=self.script,
            script_args={
                "from_pretrained": os.path.join(
                    TEST_DIR, "checkpoints/facebook/galactica-125m/none/checkpoint-20"
                ),
                "model_config": "125m",
                "training_data_dirs": f"{os.path.join(TD_PATH, 'comp_train')} {os.path.join(TD_PATH, 'assay_train')}",  # noqa
                "dir_data_types": "computed assay",
                "valid_data_dir": f"{os.path.join(TD_PATH, 'comp_valid')}",
                "train_batch_size": 4,
                "max_steps": 40,
                "eval_steps": 10,
                "save_steps": 10,
                "dataloader_num_workers": 8,
                "checkpoints_root_dir": os.path.join(TEST_DIR, "checkpoints"),
                "experiment_name": "fsdp_model_resume",
                "gradient_accumulation_steps": 1,
                "no_track": "",
                "flash_attn": "",
            },
        )

        print(f"Running command: {second_command}")
        out = subprocess.run(second_command, shell=True, capture_output=False)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())


if __name__ == "__main__":
    unittest.main(verbosity=2)
