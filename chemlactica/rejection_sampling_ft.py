from typing import List, Dict
import os
import gc
import signal
import traceback
import argparse
import random
import glob
import tqdm

from chemlactica.config.create_finetine_config import model_fine_tune_configs
from chemlactica.eval_metrics import compute_metrics, preprocess_logits_for_metrics
from chemlactica.utils.utils import signal_handler, get_tokenizer_special_tokens, get_tokenizer
from chemlactica.utils.dataset_utils import process_dataset
from chemlactica.utils.model_utils import load_model
from custom_trainer import CustomIterativeSFTTrainer
from generation.rejection_sampling_utils import generate_dataset

import torch
import numpy as np
import pandas as pd
import transformers
from transformers import logging
from transformers import (
    TrainingArguments,
    ProgressCallback,
    get_polynomial_decay_schedule_with_warmup,
)
from datasets import load_dataset

from utils.callbacks import (
    CustomAimCallback,
    WPSCounterCallback,
    CustomProgressCallback,
    ReproducabilityCallback,
)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logger = logging.get_logger("transformers")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def fine_tine(
    from_pretrained,
    model_config,
    valid_data_dir,
    rounds,
    steps_per_round,
    eval_steps,
    save_steps,
    train_batch_size,
    max_learning_rate,
    experiment_name,
    checkpoints_root_dir,
    dataloader_num_workers,
    use_flash_attn,
    gradient_accumulation_steps,
    gradient_checkpointing,
    device,
    seed,
    track=False,
    track_dir=None,
    check_reproducability=False,
    valid_batch_size=None,
    profile=False,
    profile_dir=None,
):
    # transformers.logging.set_verbosity_info()
    transformers.utils.logging.enable_explicit_format()
    set_seed(seed)

    train_config = model_fine_tune_configs[model_config]
    train_config["max_learning_rate"] = max_learning_rate
    model = load_model(
        from_pretrained,
        use_flash_attn=use_flash_attn,
        train_config=train_config,
        # auth_token=auth_token,
    )
    special_tokens = get_tokenizer_special_tokens()
    print(f"{len(special_tokens)} {special_tokens} additional special tokens.")
    tokenizer = get_tokenizer()

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

    experiment_hash = communication_list[0]
    print(f"Process {os.getpid()} aim hash: {experiment_hash}")

    if not valid_batch_size:
        valid_batch_size = train_batch_size

    wps_counter_callback = WPSCounterCallback(
        train_config["block_size"],
        trainer_callback_dict.get("aim_callback")._run
        if trainer_callback_dict.get("aim_callback") is not None
        else None,
    )
    trainer_callback_dict["wps_counter_callback"] = wps_counter_callback
    
    if check_reproducability:
        trainer_callback_dict["reproducability_callback"] = ReproducabilityCallback(
            model_config, use_flash_attn
        )
    trainer_callback_dict["progress_callback"] = CustomProgressCallback()

    checkpoints_dir = os.path.join(
        checkpoints_root_dir,
        "facebook",
        f"galactica-{model_config}",
        experiment_hash,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["max_learning_rate"],
        betas=[train_config["adam_beta1"], train_config["adam_beta2"]],
        weight_decay=train_config["weight_decay"],
    )

    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config["warmup_steps"],
        num_training_steps=steps_per_round * rounds,
        lr_end=0.1 * train_config["max_learning_rate"],
        power=1.0,
    )

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
        warmup_steps=train_config["warmup_steps"],
        max_grad_norm=train_config["global_gradient_norm"],
        evaluation_strategy="steps",
        max_steps=steps_per_round * rounds,
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
        # load_best_model=True
    )

    print("Warmup steps", train_config["warmup_steps"])

    valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
    
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

    trainer = CustomIterativeSFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        eval_dataset=processed_eval_dataset["validation"],
        optimizers=[optimizer, lr_scheduler],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.remove_callback(ProgressCallback)
    for additional_callback in list(trainer_callback_dict.values()):
        trainer.add_callback(additional_callback)

    # trainer.evaluate()
    trainer.control = trainer.callback_handler.on_train_begin(training_args, trainer.state, trainer.control)
    trainer.control = trainer.callback_handler.on_epoch_begin(training_args, trainer.state, trainer.control)
    for i in tqdm.tqdm(range(1, rounds + 1)):
        print(f"---------Rej Sampling ROUND {i}---------")
        if trainer.state.global_step == 0:
            generator_checkpoint_path = from_pretrained
        else:
            generator_checkpoint_path = os.path.join(training_args.output_dir, f"checkpoint-{trainer.state.global_step}")
        print(f"Generator model: {generator_checkpoint_path}")
        # lead_molecule = "CC(=O)NCCNC(=O)c1cnn(-c2ccc(C)c(Cl)c2)c1C1CC1"
        lead_molecule = "c1ccc(-c2cc(N3C[C@H]4[C@@H]5CC[C@@H](O5)[C@H]4C3)c3ccccc3[nH+]2)cc1"

        # generate and save rejection sampled samples
        train_ds_name = generate_dataset(
            checkpoint_path=generator_checkpoint_path,
            run_hash=experiment_hash,
            round=i,
            num_samples=steps_per_round,
            max_similars_in_prompt=5,
            lead_molecule=lead_molecule,
            use_flash_attn=use_flash_attn,
            device=device,
            seed=seed
        )
        gc.collect()
        torch.cuda.empty_cache()
        # read the rejection sampled samples
        df_samples = pd.read_csv(train_ds_name)
        del df_samples["Unnamed: 0"]
        optimized_molecule_mask = np.bitwise_and(df_samples['qed'].values >= 0.9, df_samples['morgan_sim_to_lead'].values >= 0.4)
        found_optimized_molecule = len(np.where(optimized_molecule_mask == True)[0]) > 0
        print(f"Found optimized molecule: {found_optimized_molecule}")
        if trainer_callback_dict.get("aim_callback"):
            trainer_callback_dict["aim_callback"]._run.track(found_optimized_molecule, name='optimized in round')

        rej_sampled_samples = list(df_samples["samples"].values)
        random.shuffle(rej_sampled_samples)
        for i in tqdm.tqdm(range(len(rej_sampled_samples) // train_batch_size)):
            batch_of_texts = rej_sampled_samples[i * train_batch_size:min((i+1) * train_batch_size, len(rej_sampled_samples))]
            if batch_of_texts:
                trainer.control = trainer.callback_handler.on_step_begin(training_args, trainer.state, trainer.control)
                trainer.step(texts=batch_of_texts)
                trainer.control = trainer.callback_handler.on_step_end(training_args, trainer.state, trainer.control)
        trainer._save_checkpoint(model=None, trial=None)
        trainer.control = trainer.callback_handler.on_save(training_args, trainer.state, trainer.control)
        gc.collect()
        torch.cuda.empty_cache()
    
    trainer.control = trainer.callback_handler.on_epoch_end(training_args, trainer.state, trainer.control)
    trainer.control = trainer.callback_handler.on_train_end(training_args, trainer.state, trainer.control)


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
        "--valid_data_dir",
        type=str,
        metavar="VD",
        dest="valid_data_dir",
        required=True,
        help="path to directory containing validation data",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        dest="rounds",
        required=True,
        help="the number of rounds",
    )
    parser.add_argument(
        "--steps_per_round",
        type=int,
        dest="steps_per_round",
        required=True,
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
        "--max_learning_rate",
        type=float,
        required=True,
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
        "--check_reproducability",
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
    parser.set_defaults(check_reproducability=False)
    parser.add_argument(
        "--device",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=False,
        default=42
    )

    args = parser.parse_args()
    fine_tine(**args.__dict__)