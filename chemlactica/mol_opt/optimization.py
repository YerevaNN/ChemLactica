import torch
from datasets import Dataset
import multiprocessing
import gc
import math
import numpy as np
from collections import namedtuple
from chemlactica.mol_opt.utils import MoleculeEntry, MoleculePool, generate_random_number
from chemlactica.mol_opt.tunning import supervised_fine_tune


def create_optimization_prompts(num_prompts, molecule_pool, max_similars_in_prompt: int, sim_range):
    prompts = []
    for i in range(num_prompts):
        similars_in_prompt = molecule_pool.random_subset(max_similars_in_prompt)
        prompt = "</s>"
        for mol in similars_in_prompt:
            prompt += f"[SIMILAR]{mol.smiles} {generate_random_number(sim_range[0], sim_range[1]):.2f}[/SIMILAR]"
        prompt += "[START_SMILES]"
        prompts.append(prompt)
    return prompts


def create_molecule_entry(output_text):
    start_smiles_tag, end_smiles_tag = "[START_SMILES]", "[END_SMILES]"
    start_ind = output_text.find(start_smiles_tag)
    end_ind = output_text.find(end_smiles_tag)
    if start_ind == -1 or end_ind == -1:
        return None

    generated_smiles = output_text[start_ind+len(start_smiles_tag):end_ind]
    for similar in output_text.split("[SIMILAR]")[1:-1]:
        similar_smiles = similar.split(" ")[0]
        if generated_smiles == similar_smiles:
            return None
    try:
        return MoleculeEntry(
            smiles=generated_smiles,
        )
    except:
        return None


def query_molecule_properties(model, tokenizer, smiles, property_tag, prop_pred_kwargs):
    property_start_tag, property_end_tag = f"[{property_tag}]", f"[/{property_tag}]"
    prompts = [f"</s>[START_SMILES]{smiles}[END_SMILES][{property_tag}]"]
    data = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    del data["token_type_ids"]
    outputs = model.generate(
        **data,
        **prop_pred_kwargs
    )
    predicted_property_values = []
    output_texts = tokenizer.batch_decode(outputs)
    for output_text in output_texts:
        start_ind = output_text.find(property_start_tag)
        end_ind = output_text.find(property_end_tag)
        if start_ind != -1 and end_ind != -1:
            predicted_property_values.append(output_text[start_ind+len(property_start_tag):end_ind])
        else:
            predicted_property_values.append(None)
    return predicted_property_values


def optimize(
        model, tokenizer,
        oracle, config
    ):
    file = open(config["log_dir"], "w")
    print("config", config)
    # print("molecule generation arguments", config["generation_config"])
    molecule_pool = MoleculePool(config["molecule_pool_size"])

    if config["strategy"] == "rej-sample":
        round_entries = []

    max_score = 0
    tol_level = 0
    num_iter = 1
    while True:
        model.eval()
        current_entries = []
        while len(current_entries) < config["num_gens_per_iter"]:
            prompts = create_optimization_prompts(
                config["num_gens_per_iter"], molecule_pool,
                max_similars_in_prompt=config["max_similars_in_prompt"],
                sim_range=config["sim_range"]
            )
            output_texts = []
            generation_batch_size = 64
            for i in range(0, len(prompts), generation_batch_size):
                prompt_batch = prompts[i: min(len(prompts), i + generation_batch_size)]
                data = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(model.device)
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
                        entry.score = oracle(entry.smiles)
                        entry.additional_properties["prompt"] = prompts[i]
                        current_entries.append(entry)
                        file.write(f"generated smiles: {entry.smiles}, score: {entry.score:.4f}\n")
                        if entry.score > max_score + 0.01:
                            max_score = entry.score
                            tol_level = 0
                        if oracle.finish or len(current_entries) >= config["num_gens_per_iter"]:
                            break

            if oracle.finish:
                break

        current_entries = list(np.unique(current_entries))[::-1]
        tol_level += 1
        num_iter += 1
        if oracle.finish:
            break

        # print("tol_level", tol_level)
        if tol_level >= 5:
            num_to_dump = len(molecule_pool) // 2
            molecule_pool.random_dump(num_to_dump)
            file.write(f"Dump {num_to_dump} random elements from pool, num pool mols {len(molecule_pool)}\n")
            tol_level = 0
        if config["strategy"] == "rej-sample" and tol_level >= 5:
            round_entries.extend(current_entries)
            round_entries = list(np.unique(round_entries))[::-1]
            top_k = int(len(round_entries) * config["rej_sample_config"]["rej_perc"])
            if len(round_entries[:top_k]) >= config["rej_sample_config"]["num_samples_per_round"]:
                training_entries = round_entries[:top_k]
                print(f"Num of train examples {len(training_entries)}.")
                file.write("Training entries\n")
                for i, mol in enumerate(training_entries):
                    file.write(f"\t{i} smiles: {mol.smiles}, score: {mol.score:.4f}\n")
                train_dataset = Dataset.from_dict({
                    "sample": [
                        f"{entry.additional_properties['prompt']}{entry.smiles}[END_SMILES]</s>"
                        for entry in training_entries
                    ]
                })
                train_dataset.shuffle(seed=42)
                config["rej_sample_config"]["formatting_func"] = lambda x: x["sample"]
                supervised_fine_tune(model, tokenizer, train_dataset, config["rej_sample_config"])
                round_entries = []
                gc.collect()
                torch.cuda.empty_cache()

        # diversity_score = 1 / (1 + math.log(1 + repeated_max_score) / math.log(10))
        molecule_pool.add(current_entries)
        file.write("Molecule pool\n")
        for i, mol in enumerate(molecule_pool.molecule_entries):
            file.write(f"\t{i} smiles: {mol.smiles}, score: {mol.score:.4f}\n")


# def optimize_reinvent(
#         model, prior_model,
#         tokenizer, oracle,
#         config
#     ):
#     print("molecule pool size", config["molecule_pool_size"])
#     print("molecule generation arguments", config["mol_gen_kwargs"])
#     molecule_pool = MoleculePool(config["molecule_pool_size"])

#     num_iter = 1
#     while True:
#         generated_entries = []
#         while len(generated_entries) < config["num_gens_per_iter"]:
#             prompts = create_optimization_prompts(
#                 config["num_gens_per_iter"], molecule_pool,
#                 max_similars_in_prompt=config["max_similars_in_prompt"],
#                 sim_range=config["sim_range"]
#             )
#             output_texts = []
#             generation_batch_size = 200
#             for i in range(0, len(prompts), generation_batch_size):
#                 prompt_batch = prompts[i: min(len(prompts), i + generation_batch_size)]
#                 data = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(model.device)
#                 del data["token_type_ids"]
#                 output = model.generate(
#                     **data,
#                     **config["mol_gen_kwargs"]
#                 )
#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 output_texts.extend(tokenizer.batch_decode(output))

#             with multiprocessing.Pool(processes=config["num_processes"]) as pol:
#                 for entry in pol.map(create_molecule_entry, output_texts) if entry]:
#                     entry.score = oracle(entry.smiles)
#                     generated_entries.append(entry)
#                     if oracle.finish or len(generated_entries) >= config["num_gens_per_iter"]:
#                         break

#             if oracle.finish:
#                 break

#         num_iter += 1
#         if oracle.finish:
#             break
#         molecule_pool.add(generated_entries)