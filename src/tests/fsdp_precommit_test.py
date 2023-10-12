import subprocess
import argparse
import unittest
import gc
import os
import sys

import torch
import torch.distributed as dist
from train import train


test_directory = "/tmp/chemlactica_fsdp_precommit_test"


def create_train_command(module, module_args, script, script_args):
    train_command = "python3 -m "
    train_command += (
        f"{module} {''.join([f'--{arg} {val} ' for arg, val in module_args.items()])}"
    )
    train_command += (
        f"{script} {''.join([f'--{arg} {val} ' for arg, val in script_args.items()])}"
    )
    return train_command


# class TestNetwork(unittest.TestCase):
#     def setUp(self):
#         script_path = os.path.dirname(os.path.abspath(__file__))
#         # Building absolute paths
#         self.train_data_dir = os.path.join(
#             script_path, "..", "..", ".small_data", "train"
#         )
#         self.valid_data_dir = os.path.join(
#             script_path, "..", "..", ".small_data", "valid"
#         )

#     def test_train_eval_save(self):
#         gc.collect()
#         torch.cuda.empty_cache()

#         train_command = create_train_command(
#             module="accelerate.commands.launch",
#             module_args={"config_file": "src/config/config.yaml"},
#             script="src/train.py",
#             script_args={
#                 "from_pretrained": "facebook/galactica-125m",
#                 "model_config": "125m",
#                 "training_data_dir": self.train_data_dir,
#                 "valid_data_dir": self.valid_data_dir,
#                 "train_batch_size": "16",
#                 "max_steps": "20",
#                 "eval_steps": "5",
#                 "save_steps": "5",
#                 "dataloader_num_workers": "0",
#                 "experiment_name": "gal125m_test_train_eval_save",
#                 "checkpoints_root_dir": "../checkpoints/facebook/galactica-125m",
#                 "no-track": "",
#             },
#         )
#         print(train_command)
#         executed_prog = subprocess.run(
#             train_command,
#             shell=True,
#         )
#         if executed_prog.returncode != 0:
#             raise Exception(f"\n\tExit code: {executed_prog.returncode}")


class TestReproducabilityOfModelOutput(unittest.TestCase):

    def setUp(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        subprocess.run(f"mkdir {test_directory}", shell=True)
        subprocess.run(f"mkdir {test_directory}/checkpoints", shell=True)

    def tearDown(self):
        subprocess.run(f"rm -rf {test_directory}", shell=True)

        # clean up
        gc.collect()
        torch.cuda.empty_cache()

    def test_repr_of_model_output(self):
        args = {
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
            "experiment_name": "fsdp_pretest_reproducability",
            "track": False,
            "check_reproducability": True,
            "use_flash_attn": True,
            "gradient_accumulation_steps": 1,
        }
        train(**args)
        dist.barrier() # process sync in the end


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--gpu_devices",
    #     type=str,
    #     metavar="GD",
    #     dest="gpu_devices",
    #     required=True,
    #     help="comma seperated gpu device indices",
    # )
    # parser.add_argument(
    #     "--test_directory",
    #     type=str,
    #     metavar="TD",
    #     dest="test_directory",
    #     required=False,
    #     help="dir where to create intermediate test files (this dir will be deleted at the end)",
    #     default="/tmp/chemlactica_precommit_test"
    # )

    # args = parser.parse_args()
    # gpu_devices = args.gpu_devices
    # test_directory = args.test_directory

    # script_path = os.path.dirname(os.path.abspath(__file__))
    # src_code_path = os.path.join(script_path, "..")

    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # # os.environ["PYTHONPATH"] = src_code_path
    # print(f"TESTING WITH DEVICES '{gpu_devices}'")

    unittest.main(verbosity=2)
