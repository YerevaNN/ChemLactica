import subprocess
import unittest
import gc
import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, ".."))
import glob

import transformers
from transformers import OPTForCausalLM
from config.create_train_config import model_train_configs
from datasets import load_dataset
from dataset_utils import process_dataset
import torch
from accelerate import Accelerator

test_directory = "/tmp/chemlactica_test_dir"


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

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

        subprocess.run(f"mkdir {test_directory}", shell=True)
        subprocess.run(f"mkdir {test_directory}/checkpoints", shell=True)

    def tearDown(self):
        subprocess.run(f"rm -rf {test_directory}", shell=True)

    def test_reproducability_of_logits(self):

        # script_path = os.path.dirname(os.path.abspath(__file__))
        # print("script path", os.getcwd())
        model_config = "125m"
        train_config = model_train_configs[model_config]
        output_dir = f"{test_directory}/checkpoints"
        train_batch_size = 4
        max_steps = 5
        eval_steps = 5
        save_steps = 5
        dataloader_num_workers = 1
        training_data_dir = ".small_data/train"
        valid_data_dir = ".small_data/valid"
        gradient_accumulation_steps = 1

        train_command = create_train_command(
            module="accelerate.commands.launch",
            module_args={"config_file": "src/tests/test_config.yaml"},
            script="src/train.py",
            script_args={
                "from_pretrained": f"facebook/galactica-{model_config}",
                "model_config": model_config,
                "training_data_dir": training_data_dir,
                "valid_data_dir": valid_data_dir,
                "train_batch_size": train_batch_size,
                "max_steps": max_steps,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "dataloader_num_workers": dataloader_num_workers,
                "experiment_name": "gal125m_test_reproducability",
                "checkpoints_root_dir": output_dir,
                "no-track": "",
                "check-reproducability": ""
            },
        )
        print(train_command)
        executed_prog = subprocess.run(
            train_command,
            shell=True,
        )
        if executed_prog.returncode != 0:
            raise Exception(f"\n\tExit code: {executed_prog.returncode}")

        # saved_model = OPTForCausalLM.from_pretrained(f"{test_directory}/checkpoints/facebook/galactica-{model_config}/none/checkpoint-5")
        # accelerator = Accelerator()
        # saved_model = OPTForCausalLM.from_pretrained(f"facebook/galactica-{model_config}")
        # saved_model = accelerator.prepare(accelerator)
        # accelerator.load_state(f"{test_directory}/checkpoints/facebook/galactica-{model_config}/none/checkpoint-5")

        # training_args = transformers.TrainingArguments(
        #     output_dir=output_dir,
        #     per_device_train_batch_size=train_batch_size,
        #     per_device_eval_batch_size=train_batch_size,
        #     # log_level = "info",
        #     log_on_each_node=True,
        #     # learning_rate=train_config["max_learning_rate"],
        #     # lr_scheduler_type="linear",
        #     # weight_decay=train_config["weight_decay"],
        #     # adam_beta1=train_config["adam_beta1"],
        #     # adam_beta2=train_config["adam_beta2"],
        #     # warmup_steps=train_config["warmup_steps"],
        #     max_grad_norm=train_config["global_gradient_norm"],
        #     evaluation_strategy="steps",
        #     max_steps=max_steps,
        #     eval_steps=eval_steps,
        #     save_steps=save_steps,
        #     dataloader_drop_last=True,
        #     dataloader_pin_memory=True,
        #     dataloader_num_workers=dataloader_num_workers,
        #     logging_steps=1,
        #     gradient_checkpointing=False,
        #     gradient_accumulation_steps=gradient_accumulation_steps,
        #     save_total_limit=4,
        # )

        # valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
        # eval_dataset = load_dataset(
        #     "text", data_files={"validation": valid_data_files}, streaming=False
        # )

        # processed_eval_dataset = process_dataset(
        #     dataset=eval_dataset,
        #     train_config=train_config,
        #     process_batch_sizes=(50, 50),
        #     is_eval=True,
        # )

        # trainer = transformers.Trainer(
        #     model=saved_model,
        #     args=training_args,
        #     train_dataset=processed_eval_dataset["validation"],
        #     eval_dataset=processed_eval_dataset["validation"]
        # )
        # trainer.evaluate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
