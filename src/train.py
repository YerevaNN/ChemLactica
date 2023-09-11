from config.create_train_config import model_train_configs

# import torch.distributed as dist
from accelerate.utils import broadcast_object_list
import torch
from transformers import TrainingArguments
from datasets import load_dataset
from eval_metrics import compute_metrics, preprocess_logits_for_metrics
import argparse
import accelerate
from accelerate import Accelerator
import glob
from callbacks import (
    CustomAimCallback,
    WPSCounterCallback,
    ProfCallback,
    EpochCallback,
    ReproducabilityCallback,
)
import os
from utils import load_model, CustomTokenizer
from optimum.bettertransformer import BetterTransformer
from custom_trainer import CustomTrainer
from dataset_utils import process_dataset
from contextlib import nullcontext
import random
import numpy

torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)


def train(
    from_pretrained,
    model_config,
    training_data_dir,
    valid_data_dir,
    max_steps,
    eval_steps,
    save_steps,
    train_batch_size,
    valid_batch_size,
    experiment_name,
    checkpoints_root_dir,
    dataloader_num_workers,
    track,
    track_dir,
    profile,
    profile_dir,
    gradient_accumulation_steps,
):
    accelerator = Accelerator()
    accelerate.tracking.AimTracker(run_name=experiment_name, logging_dir=track_dir)
    print("test process", accelerator.process_index)

    if not valid_batch_size:
        valid_batch_size = train_batch_size

    # absolute_path = os.path.dirname(os.path.abspath(__file__))

    train_config = model_train_configs[model_config]

    model = load_model(from_pretrained, train_config)
    if os.path.isdir(from_pretrained):
        resume_from_checkpoint = from_pretrained
    else:
        resume_from_checkpoint = False
    # Converts the model to use PyTorch’s native attention implementation

    # model = BetterTransformer.transform(model)

    CustomTokenizer.set_model_size(model_config)

    # Not sure if this will not cause issues like initializing two distributed groups
    # comment out to run without accelerate

    # dist.init_process_group()

    trainer_callback_dict = {}
    experiment_hash = "none"
    communication_list = [experiment_hash]
    if track:
        if accelerator.is_main_process:
            aim_callback = CustomAimCallback(
                checkpoints_dict_name="checkpoints_hashes",
                repo=track_dir,
                experiment=experiment_name,
                model=model,
                blocksize=train_config["block_size"],
            )

            trainer_callback_dict["aim_callback"] = aim_callback

            experiment_hash = aim_callback._run_hash
            communication_list = [experiment_hash]

    accelerator.wait_for_everyone()
    broadcast_object_list(communication_list)
    experiment_hash = communication_list[0]

    print(f"Process {accelerator.process_index} aim hash: {experiment_hash}")

    if profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(profile_dir, experiment_hash)
            ),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        )
        trainer_callback_dict["profiller_callback"] = ProfCallback(prof)

    wps_counter_callback = WPSCounterCallback(
        train_config["block_size"],
        trainer_callback_dict.get("aim_callback")._run
        if trainer_callback_dict.get("aim_callback") is not None
        else None,
    )
    trainer_callback_dict["wps_counter_callback"] = wps_counter_callback

    trainer_callback_dict["epoch_callback"] = EpochCallback(num_epochs=1)
    trainer_callback_dict["reproducability_callback"] = ReproducabilityCallback()

    checkpoints_dir = os.path.join(
        checkpoints_root_dir, "facebook", f"galactica-{model_config}", experiment_hash
    )
    accelerator.print("resuming from checkpoint:", resume_from_checkpoint)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=valid_batch_size,
        learning_rate=train_config["max_learning_rate"],
        lr_scheduler_type="linear",
        weight_decay=train_config["weight_decay"],
        adam_beta1=train_config["adam_beta1"],
        adam_beta2=train_config["adam_beta2"],
        warmup_steps=train_config["warmup_steps"],
        max_grad_norm=train_config["global_gradient_norm"],
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # torch_compile=True,
        # torch_compile requires to set use_orig_params=true
        # which has some conflict with saving checkpoints
        dataloader_num_workers=dataloader_num_workers,
        logging_steps=1,
        gradient_checkpointing=False,
        save_total_limit=4,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    training_data_files = glob.glob(training_data_dir + "/*.jsonl")
    valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
    dataset = load_dataset(
        "text",
        data_files={"train": training_data_files, "validation": valid_data_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset, train_config=train_config, process_batch_sizes=(50, 50)
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        callbacks=list(trainer_callback_dict.values()),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    prof_context_manager = (
        trainer_callback_dict.get("profiller_callback").prof
        if trainer_callback_dict.get("profiller_callback") is not None
        else nullcontext()
    )

    with prof_context_manager as prof:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="none")

    parser.add_argument(
        "--from_pretrained",
        type=str,
        metavar="FP",
        dest="from_pretrained",
        required=True,
        help="the path to the model dir",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        metavar="MC",
        dest="model_config",
        required=True,
        help="the model configuration to use",
    )
    parser.add_argument(
        "--training_data_dir",
        type=str,
        metavar="DT",
        dest="training_data_dir",
        required=True,
        help="path to directory containing training data",
    )
    parser.add_argument(
        "--valid_data_dir",
        type=str,
        metavar="VD",
        dest="valid_data_dir",
        required=True,
        help="path to directory containing validation data",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        metavar="MS",
        dest="max_steps",
        required=True,
        help="the number of steps to train (overrides the n_epochs)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        metavar="ES",
        dest="eval_steps",
        required=True,
        help="the number of training steps after which to evaluate the model",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        metavar="SS",
        dest="save_steps",
        required=True,
        help="the number of steps to save a model checkpoint",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        metavar="TBS",
        required=True,
        help="train batch size (per GPU when using dist training)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        metavar="TBS",
        required=False,
        help="valid batch size (per GPU when using dist validation)",
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        metavar="EN",
        dest="experiment_name",
        required=False,
        help="the name of the experiment",
        default="none",
    )
    parser.add_argument(
        "--checkpoints_root_dir",
        type=str,
        metavar="CRD",
        dest="checkpoints_root_dir",
        required=True,
        help="directory where to save checkpoints",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        metavar="NW",
        dest="dataloader_num_workers",
        required=False,
        help="number of processes to use for dataloading",
        default=0,
    )
    parser.add_argument(
        "--track",
        action="store_true",
        dest="track",
        help="whether or not track the training using aim",
    )
    parser.add_argument(
        "--no-track",
        action="store_false",
        dest="track",
        help="the directory to save the aim tracking information",
    )
    parser.set_defaults(track=True)
    parser.add_argument(
        "--track_dir",
        type=str,
        metavar="TD",
        dest="track_dir",
        required=False,
        help="aim track directory",
        default=None,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        dest="profile",
        help="whether or not profile the training",
    )
    parser.add_argument(
        "--no-profile",
        action="store_false",
        dest="profile",
        help="whether or not profile the training",
    )
    parser.set_defaults(profile=False)
    parser.add_argument(
        "--profile_dir",
        type=str,
        metavar="PD",
        dest="profile_dir",
        required=False,
        help="profiling directory",
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        metavar="GA",
        dest="gradient_accumulation_steps",
        required=False,
        help="the number of steps to over which to accumulate gradients",
        default=1
    )

    args = parser.parse_args()
    train(**args.__dict__)
