import subprocess
import unittest
import gc
import os
import sys

# sys.path.append(os.path.join(script_path, ".."))
import torch


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


class TestReproducability(unittest.TestCase):
    def test_reproducability_of_logits(self):
        gc.collect()
        torch.cuda.empty_cache()

        # script_path = os.path.dirname(os.path.abspath(__file__))
        # print("script path", os.getcwd())
        subprocess.run("mkdir test_dir", shell=True)
        subprocess.run("mkdir test_dir/checkpoints", shell=True)

        train_command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": "src/tests/test_config.yaml"},
            script="src/train.py",
            script_args={
                "from_pretrained": "facebook/galactica-125m",
                "model_config": "125m",
                "training_data_dir": ".small_data/train",
                "valid_data_dir": ".small_data/valid",
                "train_batch_size": "4",
                "max_steps": "20",
                "eval_steps": "5",
                "save_steps": "5",
                "dataloader_num_workers": "1",
                "experiment_name": "gal125m_test_reproducability_of_logits",
                "checkpoints_root_dir": "test_dir/checkpoints/facebook/galactica-125m",
                "no-track": "",
                "check-reproducability": ""
            },
        )
        print(train_command)
        executed_prog = subprocess.run(
            train_command,
            shell=True,
        )
        subprocess.run("rm -rf test_dir", shell=True)
        if executed_prog.returncode != 0:
            raise Exception(f"\n\tExit code: {executed_prog.returncode}")




if __name__ == "__main__":
    unittest.main(verbosity=2)
