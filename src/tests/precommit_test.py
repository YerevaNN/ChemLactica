import subprocess
import unittest
import os

# from config.create_train_config import model_train_configs
# import torch.distributed as dist
import torch

# import transformers
# from transformers import TrainingArguments, AutoModelForCausalLM
# from datasets import load_dataset
# from eval_metrics import compute_metrics, preprocess_logits_for_metrics
# import argparse
# import glob
# import gc
# from utils import load_model
# from callbacks import CustomAimCallback, WPSCounterCallback, ProfCallback
# from optimum.bettertransformer import BetterTransformer
# from custom_trainer import CustomTrainer
# from dataset_utils import process_dataset
# from contextlib import nullcontext
import random
import numpy

torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)


class TestNetworkTraining(unittest.TestCase):
    def test_small_opt_model(self):
        script_path = os.path.dirname(os.path.abspath(__file__))

        # Building absolute paths
        train_script_path = os.path.join(script_path, "..", "train.py")
        training_data_dir_path = os.path.join(
            script_path, "..", "..", ".small_data", "train"
        )
        valid_data_dir_path = os.path.join(
            script_path, "..", "..", ".small_data", "valid"
        )
        os.makedirs("checkpoints")
        train_command = f"python3 {train_script_path} \
                --from_pretrained small_opt \
                --model_config small_opt \
                --tokenizer_checkpoint facebook/galactica-125m \
                --training_data_dir {training_data_dir_path} \
                --valid_data_dir {valid_data_dir_path} \
                --max_steps 4 \
                --eval_steps 2 \
                --save_steps 2 \
                --tokenizer 125m \
                --checkpoints_root_dir ../checkpoints/ \
                --track_dir ../aim/"
        # --profile \
        # --profile_dir ./profiling"

        print(train_command)
        subprocess.run("mkdir profiling", shell=True)
        executed_prog = subprocess.run(
            train_command,
            shell=True,
        )
        if executed_prog.returncode != 0:
            raise Exception(f"\n\tExit code: {executed_prog.returncode}")


# class TestFSDPNetworkReproducibility(unittest.TestCase):
#     def test_network_reproducibility(self):
#         gc.collect()
#         torch.cuda.empty_cache()

#         from_pretrained = "facebook/galactica-125m"
#         model_config = "125m"
#         train_config = model_train_configs[model_config]

#         model = load_model(from_pretrained)
#         # Converts the model to use PyTorch’s native attention implementation
#         model = BetterTransformer.transform(model)

#         # Not sure if this will not cause issues like initializing two distributed groups
#         # comment out to run without accelerate
#         dist.init_process_group()

#         trainer_callback_dict = {}
#         experiment_hash = "none"
#         communication_list = [experiment_hash]

#         dist.broadcast_object_list(communication_list, src=0)
#         experiment_hash = communication_list[0]
#         print(f"Process {dist.get_rank()} aim hash: {experiment_hash}")

#         training_args = TrainingArguments(
#             output_dir=".",
#             per_device_train_batch_size=train_config["batch_size"],
#             per_device_eval_batch_size=train_config["batch_size"],
#             learning_rate=train_config["max_learning_rate"],
#             lr_scheduler_type="linear",
#             weight_decay=train_config["weight_decay"],
#             adam_beta1=train_config["adam_beta1"],
#             adam_beta2=train_config["adam_beta2"],
#             warmup_steps=train_config["warmup_steps"],
#             max_grad_norm=train_config["global_gradient_norm"],
#             evaluation_strategy="steps",
#             eval_steps=eval_steps,
#             max_steps=max_steps,
#             save_steps=save_steps,
#             # gradient_accumulation_steps=4,
#             dataloader_drop_last=True,
#             dataloader_pin_memory=True,
#             # torch_compile=True,
#             # # torch_compile requires to set use_orig_params=true
#             # which has some conflict with saving checkpoints
#             dataloader_num_workers=num_workers,
#             logging_steps=eval_steps // 2,
#             gradient_checkpointing=False,
#         )

#         dataset = load_dataset(
#             "text",
#             data_files={"train": training_data_files, "validation": valid_data_files},
#             streaming=True,
#         )

#         processed_dataset = process_dataset(
#             dataset=dataset, train_config=train_config, process_batch_sizes=(100, 100)
#         )

#         trainer = CustomTrainer(
#             model=model,
#             args=training_args,
#             compute_metrics=compute_metrics,
#             train_dataset=processed_dataset["train"],
#             eval_dataset=processed_dataset["validation"],
#             callbacks=list(trainer_callback_dict.values()),
#             preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#         )


if __name__ == "__main__":
    unittest.main(verbosity=2)
