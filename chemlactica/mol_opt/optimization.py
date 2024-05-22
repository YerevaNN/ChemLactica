from typing import List
import torch
from datasets import Dataset
import multiprocessing
import gc
import math
import tqdm
import random
from functools import partial
import numpy as np
from transformers import OPTForCausalLM
from chemlactica.mol_opt.utils import OptimEntry, MoleculeEntry, Pool, generate_random_number, tanimoto_dist_func
from chemlactica.mol_opt.tunning import supervised_fine_tune


def create_similar_mol_entries(pool, mol_entry, num_similars):
    similar_entries = [e.last_entry for e in pool.random_subset(num_similars + 1)]
    count = 0
    valid_similar_entries = []
    for similar_entry in similar_entries:
        if count >= num_similars:
            break
        if similar_entry == mol_entry:
            continue
        valid_similar_entries.append(similar_entry)
        count += 1
    return valid_similar_entries


def create_optimization_entries(num_entries, pool, config):
    optim_entries = []
    for i in range(num_entries):
        mol_entries = [e.last_entry for e in pool.random_subset(config["num_mols"])]
        entries = []
        for mol_entry in mol_entries:
            similar_mol_entries = create_similar_mol_entries(pool, mol_entry, num_similars=config["num_similars"])
            mol_entry.similar_mol_entries = similar_mol_entries
            entries.append(mol_entry)
        optim_entries.append(OptimEntry(None, entries))
    return optim_entries


def create_molecule_entry(output_text):
    start_smiles_tag, end_smiles_tag = "[START_SMILES]", "[END_SMILES]"
    start_ind = output_text.rfind(start_smiles_tag)
    end_ind = output_text.rfind(end_smiles_tag)
    if start_ind == -1 or end_ind == -1:
        return None
    generated_smiles = output_text[start_ind+len(start_smiles_tag):end_ind]
    if len(generated_smiles) == 0:
        return None

    try:
        molecule = MoleculeEntry(
            smiles=generated_smiles,
        )
        return molecule
    except:
        return None


def optimize(
        model, tokenizer,
        oracle, config,
        additional_properties={}
    ):
    with open(config["log_dir"], "w") as file:
        print("config", config)
        # print("molecule generation arguments", config["generation_config"])
        pool = Pool(config["pool_size"], validation_perc=config["validation_perc"])

        max_score = 0
        tol_level = 0
        num_iter = 0
        prev_train_iter = 0
        while True:
            model.eval()
            iter_optim_entries: List[OptimEntry] = []
            while len(iter_optim_entries) < config["num_gens_per_iter"]:
                optim_entries = create_optimization_entries(
                    config["num_gens_per_iter"], pool,
                    config=config
                )
                for i in range(len(optim_entries)):
                    last_entry = MoleculeEntry(smiles="")
                    last_entry.similar_mol_entries = create_similar_mol_entries(
                        pool, last_entry, config["num_similars"]
                    )
                    for prop_name, prop_spec in additional_properties.items():
                        last_entry.add_props[prop_name] = prop_spec
                    optim_entries[i].last_entry = last_entry

                prompts = [
                    optim_entry.to_prompt(is_generation=True, include_oracle_score=prev_train_iter != 0, config=config)
                    for optim_entry in optim_entries
                ]
                output_texts = []
                for i in range(0, len(prompts), config["generation_batch_size"]):
                    prompt_batch = prompts[i: min(len(prompts), i + config["generation_batch_size"])]
                    data = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(model.device)
                    if type(model) == OPTForCausalLM:
                        del data["token_type_ids"]
                    for key, value in data.items():
                        data[key] = value[:, -2048 + config["generation_config"]["max_new_tokens"]:]
                    output = model.generate(
                        **data,
                        **config["generation_config"]
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    output_texts.extend(tokenizer.batch_decode(output))

                current_mol_entries = []
                current_optim_entries = []
                with multiprocessing.Pool(processes=config["num_processes"]) as pol:
                    for i, entry in enumerate(pol.map(create_molecule_entry, output_texts)):
                        if entry and not optim_entries[i].contains_entry(entry):
                            current_mol_entries.append(entry)
                            current_optim_entries.append(optim_entries[i])

                if getattr(oracle, "takes_entry", False):
                    oracle_scores = oracle(current_mol_entries)
                else:
                    oracle_scores = oracle([e.smiles for e in current_mol_entries])
                for i, oracle_score in enumerate(oracle_scores):
                    entry = current_mol_entries[i]
                    entry.score = oracle_score
                    entry.similar_mol_entries = current_optim_entries[i].last_entry.similar_mol_entries
                    for prop_name, prop_spec in additional_properties.items():
                        entry.add_props[prop_name] = prop_spec
                        entry.add_props[prop_name]["value"] = entry.add_props[prop_name]["calculate_value"](entry)
                    current_optim_entries[i].last_entry = entry
                    iter_optim_entries.append(current_optim_entries[i])
                    file.write(f"generated smiles: {entry.smiles}, score: {entry.score:.4f}\n")
                    if entry.score > max_score:
                        max_score = entry.score
                        tol_level = 0
                    if oracle.finish or len(iter_optim_entries) >= config["num_gens_per_iter"]:
                        break

                if oracle.finish:
                    break

            if oracle.finish:
                break
            initial_num_iter = num_iter
            num_iter = len(oracle.mol_buffer) // config["num_gens_per_iter"]
            if num_iter > initial_num_iter:
                tol_level += 1
            print(f"num_iter: {num_iter}, tol_level: {tol_level}, prev_train_iter: {prev_train_iter}")

            # diversity_score = 1 / (1 + math.log(1 + repeated_max_score) / math.log(10))
            pool.add(iter_optim_entries)
            file.write("Pool\n")
            for i, optim_entry in enumerate(pool.optim_entries):
                file.write(f"\t{i} smiles: {optim_entry.last_entry.smiles}, score: {optim_entry.last_entry.score:.4f}\n")

            if "rej-sample-v2" in config["strategy"]:
                # round_entries.extend(current_entries)
                # round_entries = list(np.unique(round_entries))[::-1]
                # top_k = int(len(all_entries) * config["rej_sample_config"]["rej_perc"])
                # if top_k >= config["rej_sample_config"]["num_samples_per_round"]:
                if config["rej_sample_config"]["should_train"](num_iter, tol_level, prev_train_iter):
                    train_entries, validation_entries = pool.get_train_valid_entries()
                    print(f"Num of training examples: {len(train_entries)}, num of validation examples: {len(validation_entries)}.")
                    file.write("Training entries\n")
                    for i, optim_entry in enumerate(train_entries):
                        file.write(f"\t{i} smiles: {optim_entry.last_entry.smiles}, score: {optim_entry.last_entry.score:.4f}\n")
                    file.write("Validation entries\n")
                    for i, optim_entry in enumerate(validation_entries):
                        file.write(f"\t{i} smiles: {optim_entry.last_entry.smiles}, score: {optim_entry.last_entry.score:.4f}\n")
                    
                    train_dataset = Dataset.from_dict({
                        "sample": [
                            optim_entry.to_prompt(is_generation=False, include_oracle_score=True, config=config)
                            for optim_entry in train_entries
                        ]
                    })
                    validation_dataset = Dataset.from_dict({
                        "sample": [
                            optim_entry.to_prompt(is_generation=False, include_oracle_score=True, config=config)
                            for optim_entry in validation_entries
                        ]
                    })
                    train_dataset.shuffle(seed=42)
                    validation_dataset.shuffle(seed=42)
                    config["rej_sample_config"]["formatting_func"] = lambda x: x["sample"]
                    supervised_fine_tune(
                        model, tokenizer,
                        train_dataset, validation_dataset,
                        config["rej_sample_config"]
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    prev_train_iter = num_iter
