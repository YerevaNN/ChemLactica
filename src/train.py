from config.create_train_config import model_train_configs

# import torch.distributed as dist
from accelerate.utils import broadcast_object_list
import torch
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.data.data_collator import default_data_collator
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
from torch.utils.data import DataLoader
import tqdm


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


# def train(
#     from_pretrained,
#     model_config,
#     training_data_dir,
#     valid_data_dir,
#     max_steps,
#     eval_steps,
#     save_steps,
#     train_batch_size,
#     valid_batch_size,
#     experiment_name,
#     checkpoints_root_dir,
#     dataloader_num_workers,
#     track,
#     track_dir,
#     profile,
#     profile_dir,
#     gradient_accumulation_steps,
# ):
#     accelerator = Accelerator()
#     accelerate.tracking.AimTracker(run_name=experiment_name, logging_dir=track_dir)
#     print("test process", accelerator.process_index)

    # if not valid_batch_size:
    #     valid_batch_size = train_batch_size

    # # absolute_path = os.path.dirname(os.path.abspath(__file__))

    # train_config = model_train_configs[model_config]

    # model = load_model(from_pretrained, train_config)
    # if os.path.isdir(from_pretrained):
    #     resume_from_checkpoint = from_pretrained
    # else:
    #     resume_from_checkpoint = False
    # Converts the model to use PyTorch’s native attention implementation

#     model = BetterTransformer.transform(model)

#     CustomTokenizer.set_model_size(model_config)

#     # Not sure if this will not cause issues like initializing two distributed groups
#     # comment out to run without accelerate

#     # dist.init_process_group()

#     trainer_callback_dict = {}
#     experiment_hash = "none"
#     communication_list = [experiment_hash]
#     if track:
#         if accelerator.is_main_process:
#             aim_callback = CustomAimCallback(
#                 checkpoints_dict_name="checkpoints_hashes",
#                 repo=track_dir,
#                 experiment=experiment_name,
#                 model=model,
#                 blocksize=train_config["block_size"],
#             )

#             trainer_callback_dict["aim_callback"] = aim_callback

#             experiment_hash = aim_callback._run_hash
#             communication_list = [experiment_hash]

#     accelerator.wait_for_everyone()
#     broadcast_object_list(communication_list)
#     experiment_hash = communication_list[0]

#     print(f"Process {accelerator.process_index} aim hash: {experiment_hash}")

#     if profile:
#         prof = torch.profiler.profile(
#             activities=[
#                 torch.profiler.ProfilerActivity.CPU,
#                 torch.profiler.ProfilerActivity.CUDA,
#             ],
#             schedule=torch.profiler.schedule(
#                 skip_first=3, wait=1, warmup=1, active=2, repeat=2
#             ),
#             on_trace_ready=torch.profiler.tensorboard_trace_handler(
#                 os.path.join(profile_dir, experiment_hash)
#             ),
#             profile_memory=True,
#             with_stack=True,
#             record_shapes=True,
#         )
#         trainer_callback_dict["profiller_callback"] = ProfCallback(prof)

#     wps_counter_callback = WPSCounterCallback(
#         train_config["block_size"],
#         trainer_callback_dict.get("aim_callback")._run
#         if trainer_callback_dict.get("aim_callback") is not None
#         else None,
#     )
#     trainer_callback_dict["wps_counter_callback"] = wps_counter_callback

#     trainer_callback_dict["epoch_callback"] = EpochCallback(num_epochs=1)
#     trainer_callback_dict["reproducability_callback"] = ReproducabilityCallback()

#     checkpoints_dir = os.path.join(
#         checkpoints_root_dir, "facebook", f"galactica-{model_config}", experiment_hash
#     )
#     accelerator.print("resuming from checkpoint:", resume_from_checkpoint)

#     training_args = TrainingArguments(
#         output_dir=checkpoints_dir,
#         per_device_train_batch_size=train_batch_size,
#         per_device_eval_batch_size=valid_batch_size,
#         learning_rate=train_config["max_learning_rate"],
#         lr_scheduler_type="linear",
#         weight_decay=train_config["weight_decay"],
#         adam_beta1=train_config["adam_beta1"],
#         adam_beta2=train_config["adam_beta2"],
#         warmup_steps=train_config["warmup_steps"],
#         max_grad_norm=train_config["global_gradient_norm"],
#         evaluation_strategy="steps",
#         eval_steps=eval_steps,
#         max_steps=max_steps,
#         save_steps=save_steps,
#         dataloader_drop_last=True,
#         dataloader_pin_memory=True,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         # torch_compile=True,
#         # torch_compile requires to set use_orig_params=true
#         # which has some conflict with saving checkpoints
#         dataloader_num_workers=dataloader_num_workers,
#         logging_steps=eval_steps // 2,
#         gradient_checkpointing=False,
#         save_total_limit=4,
#         resume_from_checkpoint=resume_from_checkpoint,
#     )

#     training_data_files = glob.glob(training_data_dir + "/*.jsonl")
#     valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
#     dataset = load_dataset(
#         "text",
#         data_files={"train": training_data_files, "validation": valid_data_files},
#         streaming=True,
#     )

#     processed_dataset = process_dataset(
#         dataset=dataset, train_config=train_config, process_batch_sizes=(50, 50)
#     )

#     trainer = CustomTrainer(
#         model=model,
#         args=training_args,
#         compute_metrics=compute_metrics,
#         train_dataset=processed_dataset["train"],
#         eval_dataset=processed_dataset["validation"],
#         callbacks=list(trainer_callback_dict.values()),
#         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#     )

#     prof_context_manager = (
#         trainer_callback_dict.get("profiller_callback").prof
#         if trainer_callback_dict.get("profiller_callback") is not None
#         else nullcontext()
#     )

#     with prof_context_manager as prof:
#         trainer.train(resume_from_checkpoint=resume_from_checkpoint)

#     return trainer

def save_model(_accelerate, _overall_step):
    output_dir = f"step_{_overall_step}"
    output_dir = os.path.join("checkpoints")
    _accelerate.save_state(output_dir)


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
    # Initialize accelerator
    accelerator = Accelerator()
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    if not valid_batch_size:
        valid_batch_size = train_batch_size

    # absolute_path = os.path.dirname(os.path.abspath(__file__))
    train_config = model_train_configs[model_config]

    model = load_model(from_pretrained, train_config)
    if os.path.isdir(from_pretrained):
        resume_from_checkpoint = from_pretrained
    else:
        resume_from_checkpoint = False

    model = BetterTransformer.transform(model)
    CustomTokenizer.set_model_size(model_config)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    model = accelerator.prepare(model)

    # Instantiate optimizer
    optimizer = transformers.AdamW(
        params=model.parameters(),
        lr=train_config["max_learning_rate"],
        weight_decay=train_config["weight_decay"],
        betas=(train_config["adam_beta1"], train_config["adam_beta2"])
    )

    num_training_steps = max_steps // gradient_accumulation_steps
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
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
    data_collator = default_data_collator
    remove_columns_collator = RemoveColumnsCollator(
        data_collator=data_collator,
        signature_columns=["input_ids", "attention_mask", "labels"],
        model_name=model.__class__.__name__,
    )
    train_dataloader = DataLoader(
        processed_dataset["train"],
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        collate_fn=remove_columns_collator,
    )
    valid_dataloader = DataLoader(
        processed_dataset["validation"],
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        collate_fn=remove_columns_collator,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # New Code #
    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0
    resume_step = 0

    # We need to load the checkpoint back in before training here with `load_state`
    # The total number of epochs is adjusted based on where the state is being loaded from,
    # as we assume continuation of the same training script
    if resume_from_checkpoint:
        if resume_from_checkpoint is not None or resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            path = os.path.basename(resume_from_checkpoint)
            
        training_difference = os.path.splitext(path)[0]

        resume_step = int(training_difference.replace("step_", ""))

    model.train()
    # New Code #
    if resume_from_checkpoint and resume_step is not None:
        # We need to skip steps until we reach the resumed step
        active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        overall_step += resume_step
    else:
        # After the first iteration though, we need to go back to the original dataloader
        active_dataloader = train_dataloader
    with tqdm.tqdm(total=num_training_steps) as pbar:
        for step, batch in enumerate(active_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # New Code #
            overall_step += 1

            # New Code #
            # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
            # These are saved to folders named `step_{overall_step}`
            # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
            # If mixed precision was used, will also save a "scalar.bin" file
            if overall_step % save_steps == 0:
                save_model(accelerator, overall_step)
                
            pbar.update(1)

    model.eval()
    with tqdm.tqdm(total=40000) as pbar:
        for step, batch in enumerate(valid_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True` (the default).
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            # TODO: compute metrics and track them
        pbar.update(1)


if __name__ == "__main__":
    set_seed(42)
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
        metavar="GAS",
        dest="gradient_accumulation_steps",
        required=False,
        help="the number of steps to over which to accumulate gradients",
        default=1
    )

    args = parser.parse_args()
    train(**args.__dict__)
