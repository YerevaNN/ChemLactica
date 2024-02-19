import os

# import sys
from datasets import interleave_datasets
import signal
import traceback
from chemlactica.utils.distributed_utils import get_experiment_hash
from datetime import timedelta
import random
from chemlactica.utils.parseargs import init_parser
import glob
import multiprocessing
from contextlib import nullcontext
import numpy
import transformers
from transformers import (
    # Trainer,
    # TrainingArguments,
    ProgressCallback,
    # get_polynomial_decay_schedule_with_warmup,
)
from accelerate import Accelerator, logging, InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list
import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from chemlactica.utils.flop_counter import get_theoretical_peak_flops

# from datasets.dataset_dict import IterableDatasetDict

from chemlactica.utils.callbacks import (
    CustomAimCallback,
    WPSCounterCallback,
    ProfCallback,
    EpochCallback,
    CustomProgressCallback,
    ReproducabilityCallback,
    JsonlDatasetResumeCallback,
    EarlyStoppingCallback,
)
from chemlactica.config.create_train_config import model_train_configs
from chemlactica.eval_metrics import compute_metrics, preprocess_logits_for_metrics
from chemlactica.utils.utils import (
    signal_handler,
    get_tokenizer_special_tokens,
    get_called_command,
    remove_extraneous_args,
)
from chemlactica.utils.model_utils import load_model
from chemlactica.custom_trainer import CustomTrainer, CustomArguments
from chemlactica.utils.dataset_utils import process_dataset, DIR_DATA_TYPES
from chemlactica.jsonl_dataset import samples_generator

torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)
logger = logging.get_logger("transformers")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def train(
    command,
    slurm_eval,
    from_pretrained,
    model_config,
    training_data_dirs,
    dir_data_types,
    valid_data_dir,
    max_steps,
    scheduler_max_steps,
    eval_steps,
    save_steps,
    train_batch_size,
    shuffle_buffer_size,
    experiment_name,
    checkpoints_root_dir,
    dataloader_num_workers,
    flash_attn,
    gradient_accumulation_steps,
    gradient_checkpointing,
    evaluate_only,
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
    checkpoint_path_components = from_pretrained.split(os.path.sep)
    if os.path.isdir(from_pretrained):
        organization = checkpoint_path_components[-4]
        model_name = checkpoint_path_components[-3]
        resume_from_checkpoint = from_pretrained
    else:
        resume_from_checkpoint = False
        organization = checkpoint_path_components[-2]
        model_name = checkpoint_path_components[-1]

    # auth_token = os.environ["HF_TOKEN"]
    model = load_model(
        from_pretrained,
        use_flash_attn=flash_attn,
        train_config=train_config,
        gradient_checkpointing=gradient_checkpointing,
        # auth_token=auth_token,
    )

    special_tokens = get_tokenizer_special_tokens(train_config["tokenizer_path"])
    print(f"{len(special_tokens)} {special_tokens} additional special tokens.")

    if not resume_from_checkpoint:
        # if we are continuing training, embeddings already resized
        model.resize_token_embeddings(
            train_config["vocab_size"] + len(special_tokens), pad_to_multiple_of=8
        )

    trainer_callback_dict = {}
    experiment_hash = get_experiment_hash(from_pretrained)
    experiment_hash_list = [experiment_hash]
    if track:
        if accelerator.is_main_process:
            aim_callback = CustomAimCallback(
                checkpoints_dict_name="checkpoints_hashes",
                repo=track_dir,
                experiment=experiment_name,
                model=model,
                blocksize=train_config["block_size"],
                run_hash=experiment_hash if experiment_hash != "none" else None,
            )
            trainer_callback_dict["aim_callback"] = aim_callback
            experiment_hash_list = [aim_callback._run_hash]

    broadcast_object_list(experiment_hash_list)
    print(f"Process {accelerator.process_index} aim hash: {experiment_hash_list[0]}")

    if not valid_batch_size:
        valid_batch_size = train_batch_size

    logger.info(f"Process {accelerator.process_index} aim hash: {experiment_hash}")

    # TODO: separate profiling
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

    trainer_callback_dict["early stop callback"] = EarlyStoppingCallback(
        early_stopping_steps=(max_steps)
    )

    trainer_callback_dict["epoch_callback"] = EpochCallback(num_epochs=1)
    if check_reproducability:
        trainer_callback_dict["reproducability_callback"] = ReproducabilityCallback(
            accelerator, model_config, flash_attn
        )

    total_theoretical_peak_flops = get_theoretical_peak_flops(accelerator)
    trainer_callback_dict["progress_callback"] = CustomProgressCallback(
        max_steps, total_theoretical_peak_flops
    )

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
            organization,
            f"{model_name}",
            experiment_hash,
        )
        accelerator.print("resuming from checkpoint:", resume_from_checkpoint)

        if not scheduler_max_steps:
            # If we don't explicitly specify when the scheduler should plan to finish:
            # it will finish at max_steps when training ends so we anneal to 0 LR.
            scheduler_max_steps = max_steps

        training_args = CustomArguments(
            command=command,
            slurm_eval=slurm_eval,
            experiment_name=experiment_name,
            do_train=not evaluate_only,
            output_dir=checkpoints_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=valid_batch_size,
            # log_level = "info",
            log_on_each_node=True,
            bf16=True,
            bf16_full_eval=True,
            fp16=False,
            logging_dir=track_dir,
            learning_rate=train_config["max_learning_rate"],
            weight_decay=train_config["weight_decay"],
            adam_beta1=train_config["adam_beta1"],
            adam_beta2=train_config["adam_beta2"],
            warmup_steps=train_config["warmup_steps"],
            max_grad_norm=train_config["global_gradient_norm"],
            evaluation_strategy="steps",
            max_steps=scheduler_max_steps,
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
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=gradient_accumulation_steps,
            # save_total_limit=4, in order for offline eval to work, we keep all of them for now
            resume_from_checkpoint=resume_from_checkpoint,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            # load_best_model=True
        )

        # print("TRAINING directories", training_data_dirs, dir_data_types)
        assert len(training_data_dirs) == len(dir_data_types)
        train_dataset_dict = {}
        print("---Training dataset names---")
        for i, (training_data_dir, dir_data_type) in enumerate(
            zip(training_data_dirs, dir_data_types)
        ):
            if dir_data_type.lower() not in DIR_DATA_TYPES:
                raise ValueError(
                    f"""Unknown data type {dir_data_type},
                    the following data types are supported: {DIR_DATA_TYPES}"""
                )
            training_data_files = glob.glob(training_data_dir + "/*.jsonl")
            ds_name = f"{dir_data_type}_{i}"
            is_assay_split = "assay" in dir_data_type
            dataset = IterableDataset.from_generator(
                samples_generator,
                gen_kwargs={
                    "files": training_data_files,
                    "shared_jsonl_files": shared_jsonl_files,
                },
            )
            dataset = process_dataset(
                dataset=dataset,
                train_config=train_config,
                process_batch_sizes=(50, 50),
                is_eval=False,
                assay=is_assay_split,
            )
            if is_assay_split:
                dataset.shuffle(buffer_size=shuffle_buffer_size)
            print(f"Dataset {i}: {ds_name}")
            train_dataset_dict[ds_name] = dataset

        valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")

        train_dataset = list(train_dataset_dict.values())
        if len(train_dataset) > 1:
            train_dataset = interleave_datasets(train_dataset)
        else:
            train_dataset = train_dataset[0]

        if evaluate_only or not slurm_eval:
            eval_dataset = load_dataset(
                "text", data_files={"validation": valid_data_files}, streaming=False
            )
            processed_eval_dataset = process_dataset(
                dataset=eval_dataset,
                train_config=train_config,
                process_batch_sizes=(50, 50),
                is_eval=True,
                assay=False,
            )
        else:
            processed_eval_dataset = None

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset if not evaluate_only else None,
            eval_dataset=processed_eval_dataset["validation"]
            if not evaluate_only or slurm_eval
            else None,
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

        with prof_context_manager as prof:
            try:
                if not evaluate_only:
                    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                else:
                    pass
            except Exception as e:
                traceback_info = traceback.format_exc()
                logger.error(e, traceback_info)
            except KeyboardInterrupt:
                with accelerator.main_process_first():
                    logger.error("KeyboardInterrupt")
            if evaluate_only:
                trainer.evaluate()


if __name__ == "__main__":
    parser = init_parser()

    args = parser.parse_args()

    command = get_called_command(args)

    remove_extraneous_args(args)

    train(command, **args.__dict__)
