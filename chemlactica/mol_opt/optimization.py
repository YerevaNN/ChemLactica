import torch
from transformers import OPTForCausalLM, AutoTokenizer
import multiprocessing
import gc
from collections import namedtuple
from chemlactica.mol_opt.utils import MoleculeEntry, MoleculePool, generate_random_number


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
        oracle, oracle_estimator,
        config
    ):
    print("molecule pool size", config["molecule_pool_size"])
    print("molecule generation arguments", config["mol_gen_kwargs"])
    molecule_pool = MoleculePool(config["molecule_pool_size"])

    num_iter = 1
    while True:
        generated_entries = []
        oracle_estimator_error = 0
        while len(generated_entries) < config["num_gens_per_iter"]:
            prompts = create_optimization_prompts(
                config["num_gens_per_iter"], molecule_pool,
                max_similars_in_prompt=config["max_similars_in_prompt"],
                sim_range=config["sim_range"]
            )
            output_texts = []
            generation_batch_size = 200
            for i in range(0, len(prompts), generation_batch_size):
                prompt_batch = prompts[i: min(len(prompts), i + generation_batch_size)]
                data = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(model.device)
                del data["token_type_ids"]
                output = model.generate(
                    **data,
                    **config["mol_gen_kwargs"]
                )
                gc.collect()
                torch.cuda.empty_cache()
                output_texts.extend(tokenizer.batch_decode(output))

            candidate_entries = []
            with multiprocessing.Pool(processes=config["num_processes"]) as pol:
                candidate_entries.extend([entry for entry in pol.map(create_molecule_entry, output_texts) if entry])

            # take top-k using oracle estimator
            top_k = len(candidate_entries)
            if num_iter != 1 and oracle_estimator:
                score_estimates = oracle_estimator(candidate_entries)
                for score_est, entry in zip(score_estimates, candidate_entries):
                    entry.score_estimate = score_est
                candidate_entries.sort(key=lambda x: x.score_estimate, reverse=True)
                top_k //= 4

            for entry in candidate_entries[:top_k]:
                entry.score = oracle(entry.smiles)
                generated_entries.append(entry)
                if oracle_estimator and entry.score_estimate:
                    oracle_estimator_error += abs(entry.score - entry.score_estimate)
                if oracle.finish or len(generated_entries) >= config["num_gens_per_iter"]:
                    break

            if oracle.finish:
                break

        num_iter += 1
        if oracle_estimator:
            oracle_estimator_error = oracle_estimator_error / len(generated_entries)
            print(f"Oracle estimate mean absolute error: {oracle_estimator_error:.4f}")
            if not oracle_estimator.is_fit or oracle_estimator_error > 0.1:
                oracle_estimator.fit(generated_entries)
        if oracle.finish:
            break
        molecule_pool.add(generated_entries)


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