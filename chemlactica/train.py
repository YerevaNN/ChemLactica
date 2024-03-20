import os
import random
import signal
import traceback
import multiprocessing
from datetime import timedelta
from contextlib import nullcontext

import torch
import numpy
import transformers
from transformers import (
    ProgressCallback,
)
from accelerate import Accelerator, logging, InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list

from chemlactica.custom_trainer import CustomArguments
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
from chemlactica.utils.utils import (
    signal_handler,
    get_tokenizer_special_tokens,
    get_called_command,
    remove_extraneous_args,
)
from chemlactica.utils.parseargs import init_parser
from chemlactica.utils.model_utils import load_model
from chemlactica.utils.utils import get_model_train_config
from chemlactica.utils.distributed_utils import get_experiment_hash
from chemlactica.utils.flop_counter import get_theoretical_peak_flops
from chemlactica.get_dataset import get_dataset
from chemlactica.get_trainer import get_trainer

torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)
logger = logging.get_logger("transformers")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def train(
    command,
    slurm_eval,
    train_type,
    from_pretrained,
    model_config,
    training_data_dirs,
    dir_data_types,
    valid_data_dir,
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
    max_steps,
    num_train_epochs=1,
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

    model_config, train_config = get_model_train_config(model_config)
    checkpoint_path_components = from_pretrained.split(os.path.sep)
    if os.path.isdir(from_pretrained):
        organization = checkpoint_path_components[-4]
        model_name = checkpoint_path_components[-3]
        resume_from_checkpoint = from_pretrained if train_type == "pretrain" else False
    else:
        resume_from_checkpoint = False
        organization = checkpoint_path_components[-2]
        model_name = checkpoint_path_components[-1]

    # auth_token = os.environ["HF_TOKEN"]
    model = load_model(
        from_pretrained,
        use_flash_attn=flash_attn,
        model_config=model_config,
        gradient_checkpointing=gradient_checkpointing,
        # auth_token=auth_token,
    )

    special_tokens = get_tokenizer_special_tokens(train_config.tokenizer_path)
    print(f"{len(special_tokens)} {special_tokens} additional special tokens.")

    if not resume_from_checkpoint:
        # if we are continuing training, embeddings already resized
        model.resize_token_embeddings(
            model_config.vocab_size + len(special_tokens), pad_to_multiple_of=8
        )

    trainer_callback_dict = {}
    experiment_hash = get_experiment_hash(from_pretrained, train_type)
    experiment_hash_list = [experiment_hash]
    if track:
        if accelerator.is_main_process:
            aim_callback = CustomAimCallback(
                checkpoints_dict_name="checkpoints_hashes",
                repo=track_dir,
                experiment=experiment_name,
                model=model,
                blocksize=model_config.block_size,
                run_hash=experiment_hash if experiment_hash != "none" else None,
            )
            trainer_callback_dict["aim_callback"] = aim_callback
            experiment_hash_list = [aim_callback._run_hash]

    broadcast_object_list(experiment_hash_list)
    experiment_hash = experiment_hash_list[0]
    logger.info(f"Process {accelerator.process_index} aim hash: {experiment_hash}")

    if not valid_batch_size:
        valid_batch_size = train_batch_size

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
        model_config.block_size,
        trainer_callback_dict.get("aim_callback")._run
        if trainer_callback_dict.get("aim_callback") is not None
        else None,
    )
    trainer_callback_dict["wps_counter_callback"] = wps_counter_callback

    if train_type == "pretrain":
        trainer_callback_dict["early stop callback"] = EarlyStoppingCallback(
            early_stopping_steps=(max_steps)
        )
        trainer_callback_dict["epoch_callback"] = EpochCallback(num_epochs=1)

    if check_reproducability and train_type == "pretrain":
        trainer_callback_dict["reproducability_callback"] = ReproducabilityCallback(
            model_config, flash_attn
        )

    total_theoretical_peak_flops = get_theoretical_peak_flops(accelerator)
    trainer_callback_dict["progress_callback"] = CustomProgressCallback(
        max_steps, total_theoretical_peak_flops
    )

    accelerator.wait_for_everyone()

    with multiprocessing.Manager() if accelerator.is_main_process else nullcontext() as manager:
        shared_jsonl_files = None
        if accelerator.is_main_process and train_type == "pretrain":
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
        logger.info(f"Checkpoint dir: {checkpoints_dir}")
        accelerator.print("resuming from checkpoint:", resume_from_checkpoint)

        if not scheduler_max_steps:
            # If we don't explicitly specify when the scheduler should plan to finish:
            # it will finish at max_steps when training ends so we anneal to 0 LR.
            scheduler_max_steps = max_steps

        training_args = CustomArguments(
            command=command,
            slurm_eval=slurm_eval,
            experiment_name=experiment_name,
            # train_config=train_config,
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
            learning_rate=train_config.max_learning_rate,
            weight_decay=train_config.weight_decay,
            adam_beta1=train_config.adam_beta1,
            adam_beta2=train_config.adam_beta2,
            warmup_steps=train_config.warmup_steps,
            max_grad_norm=train_config.global_gradient_norm,
            evaluation_strategy="steps",
            max_steps=scheduler_max_steps,
            num_train_epochs=num_train_epochs,
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

        dataset = get_dataset(
            train_type,
            training_data_dirs,
            valid_data_dir,
            dir_data_types,
            train_config,
            shared_jsonl_files,
            evaluate_only,
            slurm_eval,
            shuffle_buffer_size,
            accelerator,
        )
        trainer = get_trainer(
            train_type, model, dataset, training_args, evaluate_only, slurm_eval
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
                raise
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
