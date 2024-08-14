from typing import List
import torch
from datasets import Dataset
import gc
from functools import partial
import shutil
from trl import SFTTrainer
from transformers import OPTForCausalLM
from chemlactica.mol_opt.utils import OptimEntry, MoleculeEntry, Pool
from chemlactica.mol_opt.tunning import get_training_arguments, get_optimizer_and_lr_scheduler, CustomEarlyStopCallback, CustomModelSelectionCallback

def create_similar_mol_entries(pool, mol_entry, num_similars):
    similar_entries = [e.last_entry for e in pool.random_subset(num_similars)]
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


def create_molecule_entry(output_text, validate_smiles):
    start_smiles_tag, end_smiles_tag = "[START_SMILES]", "[END_SMILES]"
    start_ind = output_text.rfind(start_smiles_tag)
    end_ind = output_text.rfind(end_smiles_tag)
    if start_ind == -1 or end_ind == -1:
        return None
    generated_smiles = output_text[start_ind+len(start_smiles_tag):end_ind]
    if not validate_smiles(generated_smiles):
        return None
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
        additional_properties={},
        validate_smiles=lambda x:True
    ):
    file = open(config["log_dir"], "w")
    print("config", config)
    # print("molecule generation arguments", config["generation_config"])
    pool = Pool(config["pool_size"], validation_perc=config["validation_perc"])

    config["generation_config"]["temperature"] = config["generation_temperature"][0]

    if "rej-sample-v2" in config["strategy"]:
        training_args = get_training_arguments(config["rej_sample_config"])
        effective_batch_size = config["rej_sample_config"]["gradient_accumulation_steps"] * config["rej_sample_config"]["train_batch_size"]
        num_single_train_steps = config["rej_sample_config"]["num_train_epochs"] * ((1 - config["validation_perc"]) * config["pool_size"] / effective_batch_size)
        max_num_trains = oracle.max_oracle_calls / (config["rej_sample_config"]["train_tol_level"] * config["num_gens_per_iter"])
        max_num_train_steps = int(max_num_trains * num_single_train_steps)
        optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(model, config["rej_sample_config"], max_num_train_steps)
    max_score = 0
    tol_level = 0
    num_iter = 0
    prev_train_iter = 0
    while True:
        model.eval()
        new_best_molecule_generated = False
        iter_unique_optim_entries: List[OptimEntry] = {}
        while len(iter_unique_optim_entries) < config["num_gens_per_iter"]:
            optim_entries = create_optimization_entries(
                config["generation_batch_size"], pool,
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
                optim_entry.to_prompt(
                    is_generation=True, include_oracle_score=prev_train_iter != 0,
                    config=config, max_score=max_score
                )
                for optim_entry in optim_entries
            ]
            output_texts = []
            data = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
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

            current_unique_optim_entries = {}
            # with multiprocessing.Pool(processes=config["num_processes"]) as pol:
            for i, molecule in enumerate(map(partial(create_molecule_entry, validate_smiles=validate_smiles), output_texts)):
                if molecule and not optim_entries[i].contains_entry(molecule):
                    if molecule.smiles not in oracle.mol_buffer and molecule.smiles not in current_unique_optim_entries:
                        molecule.similar_mol_entries = optim_entries[i].last_entry.similar_mol_entries
                        for prop_name, prop_spec in additional_properties.items():
                            molecule.add_props[prop_name] = prop_spec
                            molecule.add_props[prop_name]["value"] = molecule.add_props[prop_name]["calculate_value"](molecule)
                        optim_entries[i].last_entry = molecule
                        current_unique_optim_entries[molecule.smiles] = optim_entries[i]

            num_of_molecules_to_score = min(len(current_unique_optim_entries), config["num_gens_per_iter"] - len(iter_unique_optim_entries))
            current_unique_smiles_list = list(current_unique_optim_entries.keys())[:num_of_molecules_to_score]
            current_unique_optim_entries = {smiles: current_unique_optim_entries[smiles] for smiles in current_unique_smiles_list}

            if getattr(oracle, "takes_entry", False):
                oracle_scores = oracle([current_unique_optim_entries[smiles].last_entry for smiles in current_unique_smiles_list])
            else:
                oracle_scores = oracle(current_unique_smiles_list)

            reports_component_scores = getattr(oracle, "reports_component_scores", False)

            for smiles, oracle_score in zip(current_unique_smiles_list, oracle_scores):
                component_scores_str = ""
                if reports_component_scores or isinstance(oracle_score, (list, tuple)):
                    oracle_score, component_scores = oracle_score
                    component_scores_str = ", ".join([f"{comp}: {score:.4f}" for comp, score in component_scores.items()])

                current_unique_optim_entries[smiles].last_entry.score = oracle_score
                iter_unique_optim_entries[smiles] = current_unique_optim_entries[smiles]
                
                file.write(f"generated smiles: {smiles}, aggregate score: {current_unique_optim_entries[smiles].last_entry.score:.4f}, {component_scores_str}\n")
                
                if max_score >= config["max_possible_oracle_score"] - 1e-2 or current_unique_optim_entries[smiles].last_entry.score > max_score:
                    max_score = max(max_score, current_unique_optim_entries[smiles].last_entry.score)
                    new_best_molecule_generated = True

            print(f"Iter unique optim entries: {len(iter_unique_optim_entries)}, budget: {len(oracle)}")

            if oracle.finish:
                break

        if oracle.finish:
            break
        initial_num_iter = num_iter
        num_iter = len(oracle.mol_buffer) // config["num_gens_per_iter"]
        if num_iter > initial_num_iter:
            tol_level += 1

        if new_best_molecule_generated:
            tol_level = 0

        print(f"num_iter: {num_iter}, tol_level: {tol_level}, prev_train_iter: {prev_train_iter}")
        if num_iter > initial_num_iter:
            config["generation_config"]["temperature"] += config["num_gens_per_iter"] / (oracle.budget - config["num_gens_per_iter"]) * (config["generation_temperature"][1] - config["generation_temperature"][0])
            print(f"Generation temperature: {config['generation_config']['temperature']}")

        # diversity_score = 1 / (1 + math.log(1 + repeated_max_score) / math.log(10))
        pool.add(list(iter_unique_optim_entries.values()))
        file.write("Pool\n")
        for i, optim_entry in enumerate(pool.optim_entries):
            file.write(f"\t{i} smiles: {optim_entry.last_entry.smiles}, score: {optim_entry.last_entry.score:.4f}\n")

        if "rej-sample-v2" in config["strategy"]:
            # round_entries.extend(current_entries)
            # round_entries = list(np.unique(round_entries))[::-1]
            # top_k = int(len(all_entries) * config["rej_sample_config"]["rej_perc"])
            # if top_k >= config["rej_sample_config"]["num_samples_per_round"]:
            if tol_level >= config["rej_sample_config"]["train_tol_level"]:
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
                        optim_entry.to_prompt(
                            is_generation=False, include_oracle_score=True,
                            config=config, max_score=config["max_possible_oracle_score"]
                        )
                        for optim_entry in train_entries
                    ]
                })
                validation_dataset = Dataset.from_dict({
                    "sample": [
                        optim_entry.to_prompt(
                            is_generation=False, include_oracle_score=True,
                            config=config, max_score=config["max_possible_oracle_score"]
                        )
                        for optim_entry in validation_entries
                    ]
                })
                train_dataset.shuffle(seed=42)
                validation_dataset.shuffle(seed=42)

                # early_stopping_callback = CustomEarlyStopCallback(
                #     early_stopping_patience=1,
                #     early_stopping_threshold=0.0001
                # )
                model_selection_callback = CustomModelSelectionCallback()

                model.train()
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=validation_dataset,
                    formatting_func=lambda x: x["sample"],
                    args=training_args,
                    packing=config["rej_sample_config"]["packing"],
                    tokenizer=tokenizer,
                    max_seq_length=config["rej_sample_config"]["max_seq_length"],
                    # data_collator=collator,
                    callbacks=[model_selection_callback],
                    optimizers=[optimizer, lr_scheduler],
                )
                trainer.train()
                print(f"Loading the best model state dict with validation loss {model_selection_callback.best_validation_loss}")
                model.load_state_dict(model_selection_callback.best_model_state_dict)
                del model_selection_callback.best_model_state_dict
                gc.collect()
                torch.cuda.empty_cache()
                tol_level = 0
                prev_train_iter = num_iter