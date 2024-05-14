import torch
from datasets import Dataset
import multiprocessing
import gc
import math
import tqdm
import random
import numpy as np
from transformers import OPTForCausalLM
from chemlactica.mol_opt.utils import MoleculeEntry, MoleculePool, generate_random_number
from chemlactica.mol_opt.tunning import supervised_fine_tune


def create_optimization_prompts(num_prompts, molecule_pool, max_mols_in_prompt: int, strategy: str, eos_token: str, post_processor=None):
    prompts = []
    for i in range(num_prompts):
        similars_in_prompt = molecule_pool.random_subset(max_mols_in_prompt)
        prompt = eos_token
        oracle_scores_of_mols_in_prompt = []
        for mol in similars_in_prompt:
            if "default" in strategy:
                prompt += f"[SIMILAR]{mol.smiles} {generate_random_number(0.8, 0.9):.2f}[/SIMILAR]"
            elif "rej-sample" in strategy:
                prompt += f"[ORACLE_SCORE]{mol.score:.2f}[/ORACLE_SCORE][START_SMILES]{mol.smiles}[END_SMILES]"
            # prompt += f"[START_SMILES]{mol.smiles}[END_SMILES]"
            oracle_scores_of_mols_in_prompt.append(mol.score)
        if post_processor:
            prompt = post_processor(prompt)
        q_0_9 = np.quantile(oracle_scores_of_mols_in_prompt, 0.9) if oracle_scores_of_mols_in_prompt else 0
        required_oracle_score = generate_random_number(q_0_9, 1.0) # TODO: change the hard coded 1.0
        if "default" in strategy:
            prompt += f"[START_SMILES]"
        elif "rej-sample" in strategy:
            prompt += f"[ORACLE_SCORE]{required_oracle_score:.2f}[/ORACLE_SCORE][START_SMILES]"
        
        prompts.append(prompt)
    return prompts


def create_molecule_entry(output_text):
    start_smiles_tag, end_smiles_tag = "[START_SMILES]", "[END_SMILES]"
    start_ind = output_text.rfind(start_smiles_tag)
    end_ind = output_text.rfind(end_smiles_tag)
    if start_ind == -1 or end_ind == -1:
        return None
    generated_smiles = output_text[start_ind+len(start_smiles_tag):end_ind]

    for output in output_text.split(start_smiles_tag)[:-1]:
        smiles_in_prompt = output.split(end_smiles_tag)[0]
        if generated_smiles == smiles_in_prompt:
            return None
    try:
        return MoleculeEntry(
            smiles=generated_smiles,
        )
    except:
        return None


# def query_molecule_properties(model, tokenizer, smiles, property_tag, prop_pred_kwargs):
#     property_start_tag, property_end_tag = f"[{property_tag}]", f"[/{property_tag}]"
#     prompts = [f"</s>[START_SMILES]{smiles}[END_SMILES][{property_tag}]"]
#     data = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
#     del data["token_type_ids"]
#     outputs = model.generate(
#         **data,
#         **prop_pred_kwargs
#     )
#     predicted_property_values = []
#     output_texts = tokenizer.batch_decode(outputs)
#     for output_text in output_texts:
#         start_ind = output_text.find(property_start_tag)
#         end_ind = output_text.find(property_end_tag)
#         if start_ind != -1 and end_ind != -1:
#             predicted_property_values.append(output_text[start_ind+len(property_start_tag):end_ind])
#         else:
#             predicted_property_values.append(None)
#     return predicted_property_values


def optimize(
        model, tokenizer,
        oracle, config
    ):
    file = open(config["log_dir"], "w")
    print("config", config)
    # print("molecule generation arguments", config["generation_config"])
    molecule_pool = MoleculePool(config["molecule_pool_size"])

    if "rej-sample" in config["strategy"]:
        all_entries = {}

    max_score = 0
    tol_level = 0
    num_iter = 0
    while True:
        model.eval()
        current_entries = []
        while len(current_entries) < config["num_gens_per_iter"]:
            prompts = create_optimization_prompts(
                config["num_gens_per_iter"],
                molecule_pool,
                max_mols_in_prompt=config["max_mols_in_prompt"],
                eos_token=config["eos_token"],
                strategy=config["strategy"],
                post_processor=config.get("prompts_post_processor")
            )
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

            with multiprocessing.Pool(processes=config["num_processes"]) as pol:
                for i, entry in enumerate(pol.map(create_molecule_entry, output_texts)):
                    if entry:
                        if getattr(oracle, 'takes_entry', False):
                            oracle_score = oracle(entry)
                        else:
                            oracle_score = oracle(entry.smiles)
                        entry.score = oracle_score
                        entry.add_props["prompt"] = prompts[i]
                        current_entries.append(entry)
                        file.write(f"generated smiles: {entry.smiles}, score: {entry.score:.4f}\n")
                        if entry.score > max_score + 0.01:
                            max_score = entry.score
                            tol_level = 0
                        if oracle.finish or len(current_entries) >= config["num_gens_per_iter"]:
                            break

            # print(num_iter, len(current_entries))
            if oracle.finish:
                break

        if oracle.finish:
            break
        current_entries = list(np.unique(current_entries))[::-1]
        initial_num_iter = num_iter
        num_iter = len(oracle.mol_buffer) // config["num_gens_per_iter"]
        print("num_iter: ", num_iter)

        # diversity_score = 1 / (1 + math.log(1 + repeated_max_score) / math.log(10))
        molecule_pool.add(current_entries)
        file.write("Molecule pool\n")
        for i, mol in enumerate(molecule_pool.molecule_entries):
            file.write(f"\t{i} smiles: {mol.smiles}, score: {mol.score:.4f}\n")

        if "rej-sample" in config["strategy"]:
            # round_entries.extend(current_entries)
            # round_entries = list(np.unique(round_entries))[::-1]
            # top_k = int(len(all_entries) * config["rej_sample_config"]["rej_perc"])
            # if top_k >= config["rej_sample_config"]["num_samples_per_round"]:
            if tol_level >= 3 and num_iter > initial_num_iter:
                training_entries = molecule_pool.molecule_entries
                print(f"Num of train examples {len(training_entries)}.")
                file.write("Training entries\n")
                for i, mol in enumerate(training_entries):
                    file.write(f"\t{i} smiles: {mol.smiles}, score: {mol.score:.4f}\n")

                def create_training_sample(entry):
                    sample = entry.add_props["prompt"]
                    return sample + f"[START_SMILES]{entry.smiles}[END_SMILES]"
                
                train_dataset = Dataset.from_dict({
                    "sample": [
                        create_training_sample(entry)
                        for entry in training_entries
                    ]
                })
                train_dataset.shuffle(seed=42)
                config["rej_sample_config"]["formatting_func"] = lambda x: x["sample"]
                supervised_fine_tune(model, tokenizer, train_dataset, config["rej_sample_config"])
                gc.collect()
                torch.cuda.empty_cache()
                tol_level = 0
        if "pool-dump" in config["strategy"] and tol_level >= 10:
            num_to_dump = int(len(molecule_pool) * config["pool_dump_config"]["dump_perc"])
            molecule_pool.random_dump(num_to_dump)
            file.write(f"Dump {num_to_dump} random elements from pool, num pool mols {len(molecule_pool)}\n")
            tol_level = 0