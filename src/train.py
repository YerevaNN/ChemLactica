import os
from datasets import interleave_datasets
import signal
import traceback
import argparse
from datetime import timedelta
import random
import glob
import multiprocessing
from contextlib import nullcontext
import numpy
import transformers
from transformers import (
    # Trainer,
    TrainingArguments,
    ProgressCallback,
    get_polynomial_decay_schedule_with_warmup,
)
from accelerate import Accelerator, logging, InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list
import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from datasets.dataset_dict import IterableDatasetDict

from callbacks import (
    CustomAimCallback,
    WPSCounterCallback,
    ProfCallback,
    EpochCallback,
    CustomProgressCallback,
    ReproducabilityCallback,
    JsonlDatasetResumeCallback,
)
from config.create_train_config import model_train_configs
from eval_metrics import compute_metrics, preprocess_logits_for_metrics
from utils import signal_handler, get_tokenizer_special_tokens
from model_utils import load_model
from custom_trainer import CustomTrainer
from dataset_utils import process_dataset
from jsonl_dataset import samples_generator

torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)
logger = logging.get_logger("transformers")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def train(
    from_pretrained,
    model_config,
    training_data_dirs,
    dir_data_types,
    valid_data_dir,
    max_steps,
    eval_steps,
    save_steps,
    train_batch_size,
    shuffle_buffer_size,
    experiment_name,
    checkpoints_root_dir,
    dataloader_num_workers,
    use_flash_attn,
    gradient_accumulation_steps,
    gradient_checkpointing,
    track=False,
    track_dir=None,
    check_reproducability=False,
    valid_batch_size=None,
    profile=False,
    profile_dir=None,
):
    transformers.logging.set_verbosity_info()
    transformers.utils.logging.enable_explicit_format()

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))

    accelerator = Accelerator(
        kwargs_handlers=[kwargs], log_with="all", project_dir=track_dir
    )

    train_config = model_train_configs[model_config]
    if os.path.isdir(from_pretrained):
        resume_from_checkpoint = from_pretrained
    else:
        resume_from_checkpoint = False

    # auth_token = os.environ["HF_TOKEN"]
    model = load_model(
        from_pretrained,
        use_flash_attn=use_flash_attn,
        train_config=train_config,
        # auth_token=auth_token,
    )
    special_tokens = get_tokenizer_special_tokens()
    print(f"{len(special_tokens)} {special_tokens} additional special tokens.")

    if os.path.isdir(from_pretrained):
        resume_from_checkpoint = from_pretrained
    else:
        resume_from_checkpoint = False

    if not resume_from_checkpoint:
        model.resize_token_embeddings(train_config["vocab_size"] + len(special_tokens))

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

    broadcast_object_list(communication_list)
    experiment_hash = communication_list[0]
    print(f"Process {accelerator.process_index} aim hash: {experiment_hash}")

    if not valid_batch_size:
        valid_batch_size = train_batch_size

    logger.info(f"Process {accelerator.process_index} aim hash: {experiment_hash}")

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
    if check_reproducability:
        trainer_callback_dict["reproducability_callback"] = ReproducabilityCallback(
            accelerator, model_config, use_flash_attn
        )
    trainer_callback_dict["progress_callback"] = CustomProgressCallback()

    accelerator.wait_for_everyone()

    with multiprocessing.Manager() if accelerator.is_main_process else nullcontext() as manager:
        shared_jsonl_files = None
        if accelerator.is_main_process:
            shared_jsonl_files = manager.dict()
            trainer_callback_dict[
                "json_dataset_resume_callback"
            ] = JsonlDatasetResumeCallback(shared_jsonl_files)
            print(f"shared jsonl files {shared_jsonl_files}")
        checkpoints_dir = os.path.join(
            checkpoints_root_dir,
            "facebook",
            f"galactica-{model_config}",
            experiment_hash,
        )
        accelerator.print("resuming from checkpoint:", resume_from_checkpoint)

        # optimizer = AdamW(
        #     model.parameters(),
        #     lr=train_config["max_learning_rate"],
        #     betas=[train_config["adam_beta1"], train_config["adam_beta2"]],
        #     weight_decay=train_config["weight_decay"],
        # )

        # lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=train_config["warmup_steps"],
        #     num_training_steps=max_steps,
        #     lr_end=0.0 * train_config["max_learning_rate"],
        #     power=1.0,
        # )

        training_args = TrainingArguments(
            output_dir=checkpoints_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=valid_batch_size,
            # log_level = "info",
            log_on_each_node=True,
            bf16=True,
            fp16=False,
            logging_dir=track_dir,
            learning_rate=train_config["max_learning_rate"],
            weight_decay=train_config["weight_decay"],
            adam_beta1=train_config["adam_beta1"],
            adam_beta2=train_config["adam_beta2"],
            # warmup_steps=train_config["warmup_steps"],
            max_grad_norm=train_config["global_gradient_norm"],
            evaluation_strategy="steps",
            max_steps=max_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            dataloader_drop_last=True,
            dataloader_pin_memory=True,
            # torch_compile=True,
            # torch_compile requires to set use_orig_params=true
            # which has some conflict with saving checkpoints
            dataloader_num_workers=dataloader_num_workers,
            logging_steps=1,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_total_limit=4,
            resume_from_checkpoint=resume_from_checkpoint,
            lr_scheduler_type="linear",
            optim="adamw_torch"
            # load_best_model=True
        )

        # print("TRAINING directories", training_data_dirs, dir_data_types)
        assert len(training_data_dirs) == len(dir_data_types)
        train_dataset_dict = {}
        print("---Training dataset names---")
        for i, (training_data_dir, dir_data_type) in enumerate(zip(training_data_dirs, dir_data_types)):
            training_data_files = glob.glob(training_data_dir + "/*.jsonl")
            ds_name = f"{dir_data_type}_1"
            is_assay_split = "assay" in dir_data_type
            dataset = IterableDataset.from_generator(
                samples_generator,
                gen_kwargs = {
                    "files" : training_data_files,
                    "shared_jsonl_files" : shared_jsonl_files
                }
            )
            dataset = process_dataset(
                dataset=dataset,
                train_config=train_config,
                process_batch_sizes=(50, 50),
                is_eval=False,
                assay=is_assay_split
            )
            if is_assay_split:
                dataset.shuffle(buffer_size=shuffle_buffer_size)
            print(f"Dataset {i}: {ds_name}")
            train_dataset_dict[ds_name] = dataset

        valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
        eval_dataset = load_dataset(
            "text", data_files={"validation": valid_data_files}, streaming=False
        )

        # for split_name in train_dataset.keys():
        #     is_assay_split = "assay" in split_name
        #     train_dataset[split_name] = process_dataset(
        #         dataset=train_dataset[split_name],
        #         train_config=train_config,
        #         process_batch_sizes=(50, 50),
        #         is_eval=False,
        #         assay=is_assay_split
        #     )
        #     if is_assay_split:
        #         train_dataset.shuffle(buffer_size=shuffle_buffer_size)

        train_dataset = list(train_dataset_dict.values())
        if len(train_dataset) > 1:
            train_dataset = interleave_datasets(train_dataset)
        else:
            train_dataset = train_dataset[0]

        # processed_train_dataset = concatenate_datasets(split_datasets)
        # final_train_dataset = interleave_datasets(split_train_datasets)
        # final_train_dataset = final_train_dataset.shuffle(buffer_size=shuffle_buffer_size)
        # processed_train_dataset = process_dataset(
        #     dataset=train_dataset,
        #     train_config=train_config,
        #     process_batch_sizes=(50, 50),
        #     is_eval=False,
        #     assay=True,
        # )

        # for i, sample in enumerate(final_train_dataset):
        # This code section is useful to verify interleaving
        #     if i >= 5: # change 5 to the number of samples you want to inspect
        #         break
        #     print(f"Sample {i}:")
        #     print(max(sample["input_ids"]))
        #     # print(sample)

        processed_eval_dataset = process_dataset(
            dataset=eval_dataset,
            train_config=train_config,
            process_batch_sizes=(50, 50),
            is_eval=True,
            assay=False,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=processed_eval_dataset["validation"],
            # optimizers=[optimizer, lr_scheduler],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        trainer.remove_callback(ProgressCallback)
        for additional_callback in list(trainer_callback_dict.values()):
            trainer.add_callback(additional_callback)

        prof_context_manager = (
            trainer_callback_dict.get("profiller_callback").prof
            if trainer_callback_dict.get("profiller_callback") is not None
            else nullcontext()
        )

        for sample in train_dataset:
            print(sample)

        with prof_context_manager as prof:
            try:
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            except Exception as e:
                traceback_info = traceback.format_exc()
                logger.error(e, traceback_info)
            except KeyboardInterrupt:
                with accelerator.main_process_first():
                    logger.error("KeyboardInterrupt")
            if not (max_steps % eval_steps == 0):
                trainer.evaluate()


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
        "--training_data_dirs",
        metavar="DT",
        nargs='*',
        dest="training_data_dirs",
        required=True,
        help="path to directory containing training data",
    )
    parser.add_argument(
        "--dir_data_types",
        metavar="DD",
        nargs='*',
        dest="dir_data_types",
        required=True,
        help="corresponding type of data for directory (in same order)",
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
        "--shuffle_buffer_size",
        type=int,
        metavar="SBS",
        dest="shuffle_buffer_size",
        required=False,
        help="the buffer size of the buffered shuffle",
        default=4
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
        "--no_track",
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
        "--no_profile",
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
        "--flash_attn",
        action="store_true",
        dest="use_flash_attn",
        help="whether or not to use flash attn)",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_false",
        dest="use_flash_attn",
        help="whether or not to use flash attn",
    )
    parser.set_defaults(use_flash_attn=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        metavar="GA",
        dest="gradient_accumulation_steps",
        required=False,
        help="the number of steps to over which to accumulate gradients",
        default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        dest="gradient_checkpointing",
        default=False,
        help="whether or not to use gradient_checkpointing",
    )
    parser.add_argument(
        "--check-reproducability",
        action="store_true",
        dest="check_reproducability",
        help="whether or not check reproducability (should only be use for testing)",
    )
    parser.add_argument(
        "--no_check_reproducability",
        action="store_false",
        dest="check_reproducability",
        help="whether or not check reproducability (should only be use for testing)",
    )
    parser.set_defaults(profile=False)

    args = parser.parse_args()
    train(**args.__dict__)
