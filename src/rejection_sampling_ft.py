from typing import List, Dict
import os
import signal
import argparse
from datetime import timedelta
import random
import glob
import tqdm
import numpy as np
import transformers
from transformers import (
    TrainingArguments,
    ProgressCallback,
    get_polynomial_decay_schedule_with_warmup,
)
import torch
from datasets import load_dataset
from generation import generate
from dataclasses import dataclass
from utils import get_tokenizer
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem import MACCSkeys

from callbacks import (
    CustomAimCallback,
    WPSCounterCallback,
    ProfCallback,
    CustomProgressCallback,
    ReproducabilityCallback,
)
from config.create_train_config import model_fine_tune_configs
from eval_metrics import compute_metrics, preprocess_logits_for_metrics
from utils import signal_handler, get_tokenizer_special_tokens
from model_utils import load_model
from custom_trainer import CustomIterativeSFTTrainer
from dataset_utils import process_dataset

from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_inchi(smiles: str):
    return Chem.MolToInchi(Chem.MolFromSmiles(smiles))


def same_molecule(smiles1: str, smiles2: str) -> bool:
    try:
        return get_inchi(smiles1) == get_inchi(smiles2)
    except:
        return True


def get_morgan_fingerprint(smiles: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)


def get_maccs_fingerprint(smiles: str):
    return MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))


def tanimoto_dist_func(smiles1: str, smiles2: str, fingerprint: str="morgan"):
    return DataStructs.TanimotoSimilarity(
        get_morgan_fingerprint(smiles1) if fingerprint == 'morgan' else get_maccs_fingerprint(smiles1),
        get_morgan_fingerprint(smiles2) if fingerprint == 'morgan' else get_maccs_fingerprint(smiles2),
    )


def compute_qed(smiles: str):
    return qed(Chem.MolFromSmiles(smiles))


def find(string: str, start: str, end: str="\n"):
    s = string.find(start)
    e = string.find(end)
    if e != -1:
        return string[s + len(start):e]


@dataclass
class OptimPromptEntry:
    prompt: str=""
    qed_of_prompt: float=None

    def __str__(self):
        return f"{__class__.__name__}: '{self.prompt}'"
    
    def __repr__(self):
        return str(self)


@dataclass
class MoleculeEntry:
    smiles: str=""
    qed: float=-1
    morgan_sim_to_lead: float=-1
    maccs_sim_to_lead: float=-1
    inchi: str=None

    def score(self):
        return self.morgan_sim_to_lead + self.qed

    def __eq__(self, other):
        return self.inchi == other.inchi

    def __lt__(self, other):
        return self.score() < other.score()
    
    def __str__(self):
        return f"smiles: {self.smiles}, qed: {self.qed:.4f}, morgan sim to lead: {self.morgan_sim_to_lead:.4f}, maccs sim to lead: {self.maccs_sim_to_lead:.4f}"
    
    def __repr__(self):
        return str(self)


def fine_tine(
    from_pretrained,
    model_config,
    valid_data_dir,
    max_steps,
    eval_steps,
    save_steps,
    train_batch_size,
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
    # transformers.logging.set_verbosity_info()
    # transformers.utils.logging.enable_explicit_format()

    device = "cuda:0"

    train_config = model_fine_tune_configs[model_config]
    if os.path.isdir(from_pretrained):
        resume_from_checkpoint = from_pretrained
    else:
        resume_from_checkpoint = False

    model = load_model(
        from_pretrained,
        use_flash_attn=use_flash_attn,
        train_config=train_config,
        # auth_token=auth_token,
    ).to(device)
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
        num_training_steps=max_steps,
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

    def move_to_device(examples):
        for key in list(examples.keys()):
            examples[key] = torch.tensor(examples[key])
        return examples
    
    processed_eval_dataset = processed_eval_dataset.map(
        move_to_device,
        batched=True
    )

    trainer = CustomIterativeSFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        eval_dataset=processed_eval_dataset["validation"],
        optimizers=[optimizer, lr_scheduler],
        max_length=train_config["block_size"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.remove_callback(ProgressCallback)
    for additional_callback in list(trainer_callback_dict.values()):
        trainer.add_callback(additional_callback)

    pubchem_molecules_file = open("/mnt/sxtn/chem/pubchem_inchi/CID-SMILES")
    def next_molecule():
        return pubchem_molecules_file.readline().split("\t")[-1].rstrip("\n")

    generator_model = load_model(from_pretrained, use_flash_attn=True, dtype=torch.bfloat16).to(device)
    sample_gen_args = {
        "max_new_tokens": 50,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "eos_token_id": 2
    }
    rej_sample_args = {
        "max_new_tokens": 50,
        "temperature": 1.3,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "num_return_sequences": 80,
        "eos_token_id": 2
    }
    def next_input_sample(lead: str):
        num_of_similar = random.randint(0, 5)
        input_text = "</s>"
        try:
            lead_molecule = MoleculeEntry(
                smiles=lead, qed=compute_qed(lead),
                morgan_sim_to_lead=random.uniform(0.9, 0.99),
                inchi=get_inchi(lead)
            )
        except Exception:
            return None
        similar_molecules_in_prompt = []
        if num_of_similar:
            sample_gen_args["num_return_sequences"] = num_of_similar * 2
            for outputs in generate(
                    prompts=[
                        f"</s>[SIMILAR]{lead_molecule.smiles} {random.uniform(0.9, 0.99):.2f}[/SIMILAR][QED]{random.uniform(0.9, 0.99):.2f}[/QED][START_SMILES]"
                        for i in range(num_of_similar)
                    ],
                    model=generator_model,
                    **sample_gen_args
                ).values():
                for out in outputs:
                    smiles = find(out, "[START_SMILES]", "[END_SMILES]")
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        gen_molecule = MoleculeEntry(
                            smiles=smiles,
                            qed=compute_qed(smiles),
                            morgan_sim_to_lead=tanimoto_dist_func(smiles, lead),
                            inchi=get_inchi(smiles)
                        )
                        if gen_molecule != lead_molecule:
                            similar_molecules_in_prompt.append(gen_molecule)
                    except Exception as e:
                        pass
        similar_molecules_in_prompt = list(np.unique(similar_molecules_in_prompt))
        similar_molecules_in_prompt = similar_molecules_in_prompt[-num_of_similar:]
        similar_molecules_in_prompt.append(lead_molecule)
        random.shuffle(similar_molecules_in_prompt)
        for mol in similar_molecules_in_prompt:
            input_text += f"[SIMILAR]{mol.smiles} {mol.morgan_sim_to_lead:.2f}[/SIMILAR]"
        input_text += f"[QED]{random.uniform(0.9, 0.99):.2f}[/QED][START_SMILES]"
        candidate_target_molecules = []
        for outputs in generate(
                input_text,
                model=generator_model,
                **rej_sample_args
            ).values():
            for out in outputs:
                smiles = find(out, "[START_SMILES]", "[END_SMILES]")
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    candidate_target_molecules.append(
                        MoleculeEntry(
                            smiles=smiles,
                            qed=compute_qed(smiles),
                            morgan_sim_to_lead=tanimoto_dist_func(smiles, lead),
                            inchi=get_inchi(smiles)
                        )
                    )
                except Exception as e:
                    pass
        for target_mol in np.unique(candidate_target_molecules)[::-1]:
            if target_mol not in similar_molecules_in_prompt:
                input_text += f"{target_mol.smiles}[END_SMILES]</s>"
                return input_text

    trainer.evaluate()
    trainer.control = trainer.callback_handler.on_train_begin(training_args, trainer.state, trainer.control)
    trainer.control = trainer.callback_handler.on_epoch_begin(training_args, trainer.state, trainer.control)
    for i in range(1, args.max_steps + 1):
        trainer.control = trainer.callback_handler.on_step_begin(training_args, trainer.state, trainer.control)
        train_batch = []
        progress_bar = tqdm.tqdm(total=train_batch_size)
        while len(train_batch) < train_batch_size:
            sample = next_input_sample(lead=next_molecule())
            if sample:
                train_batch.append(sample)
                progress_bar.update(1)
            if len(train_batch) == train_batch_size:
                break
        trainer.step(texts=train_batch)
        trainer.control = trainer.callback_handler.on_step_end(training_args, trainer.state, trainer.control)
        # print(f"Step {i}")
        if i % save_steps == 0:
            trainer._save_checkpoint(model=None, trial=None)
            trainer.control = trainer.callback_handler.on_save(training_args, trainer.state, trainer.control)
    trainer.control = trainer.callback_handler.on_epoch_end(training_args, trainer.state, trainer.control)
    trainer.control = trainer.callback_handler.on_train_end(training_args, trainer.state, trainer.control)

    # with prof_context_manager as prof:
    #     try:
    #         trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    #     except Exception as e:
    #         traceback_info = traceback.format_exc()
    #         logger.error(e, traceback_info)
    #     except KeyboardInterrupt:
    #         with accelerator.main_process_first():
    #             logger.error("KeyboardInterrupt")
    #     if not (max_steps % eval_steps == 0):
    #         trainer.evaluate()
    pubchem_molecules_file.close()


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

    args = parser.parse_args()
    fine_tine(**args.__dict__)