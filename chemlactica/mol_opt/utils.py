from typing import List
import datetime
import os
import random
from pathlib import Path
import numpy as np
import torch
from chemlactica.mol_opt.metrics import top_auc
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

# Disable RDKit logs
RDLogger.DisableLog("rdApp.*")


def set_seed(seed_value):
    random.seed(seed_value)
    # Set seed for NumPy
    np.random.seed(seed_value)
    # Set seed for PyTorch
    torch.manual_seed(seed_value)


def get_short_name_for_ckpt_path(chpt_path: str, hash_len: int = 6):
    get_short_name_for_ckpt_path = Path(chpt_path)
    return (
        get_short_name_for_ckpt_path.parent.name[:hash_len]
        + "-"
        + get_short_name_for_ckpt_path.name.split("-")[-1]
    )


def get_morgan_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def get_maccs_fingerprint(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def tanimoto_dist_func(fing1, fing2, fingerprint: str = "morgan"):
    return DataStructs.TanimotoSimilarity(
        fing1 if fingerprint == "morgan" else fing1,
        fing2 if fingerprint == "morgan" else fing2,
    )


def generate_random_number(lower, upper):
    return lower + random.random() * (upper - lower)


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True)
    # return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), kekuleSmiles=True)


class MoleculeEntry:
    def __init__(self, smiles, score=0, **kwargs):
        self.smiles = smiles
        self.score = score
        self.similar_mol_entries = []
        if smiles:
            self.smiles = canonicalize(smiles)
            self.mol = Chem.MolFromSmiles(smiles)
            self.fingerprint = get_morgan_fingerprint(self.mol)
        self.add_props = kwargs

    def __eq__(self, other):
        return self.smiles == other.smiles

    def __lt__(self, other):
        if self.score == other.score:
            return self.smiles < other.smiles
        return self.score < other.score

    def __str__(self):
        return (
            f"smiles: {self.smiles}, "
            f"score: {round(self.score, 4) if self.score != None else 'none'}"
        )

    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(self.smiles)


class ConstraedTPSAOracle:
    def __init__(self, max_oracle_calls: int):
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = 100
        self.mol_buffer = {}
        self.max_possible_oracle_score = 1.0
        self.takes_entry = True

    def __call__(self, molecules):
        oracle_scores = []
        for molecule in molecules:
            if self.mol_buffer.get(molecule.smiles):
                oracle_scores.append(sum(self.mol_buffer[molecule.smiles][0]))
            else:
                try:
                    tpsa = rdMolDescriptors.CalcTPSA(molecule.mol)
                    tpsa_score = min(tpsa / 1000, 1)
                    weight = rdMolDescriptors.CalcExactMolWt(molecule.mol)
                    if weight <= 349:
                        weight_score = 1
                    elif weight >= 500:
                        weight_score = 0
                    else:
                        weight_score = -0.00662 * weight + 3.31125
                    
                    num_rings = rdMolDescriptors.CalcNumRings(molecule.mol)
                    if num_rings >= 2:
                        num_rights_score = 1
                    else:
                        num_rights_score = 0
                    # print(tpsa_score, weight_score, num_rights_score)
                    oracle_score = (tpsa_score + weight_score + num_rights_score) / 3
                except Exception as e:
                    print(e)
                    oracle_score = 0
                self.mol_buffer[molecule.smiles] = [oracle_score, len(self.mol_buffer) + 1]
                if len(self.mol_buffer) % 100 == 0:
                    self.log_intermediate()
                oracle_scores.append(oracle_score)
        return oracle_scores
    
    def log_intermediate(self):
        scores = [v[0] for v in self.mol_buffer.values()]
        scores_sorted = sorted(scores, reverse=True)[:100]
        n_calls = len(self.mol_buffer)

        score_avg_top1 = np.max(scores_sorted)
        score_avg_top10 = np.mean(scores_sorted[:10])
        score_avg_top100 = np.mean(scores_sorted)

        print(f"{n_calls}/{self.max_oracle_calls} | ",
                f"auc_top1: {top_auc(self.mol_buffer, 1, False, self.freq_log, self.max_oracle_calls)} | ",
                f"auc_top10: {top_auc(self.mol_buffer, 10, False, self.freq_log, self.max_oracle_calls)} | ",
                f"auc_top100: {top_auc(self.mol_buffer, 100, False, self.freq_log, self.max_oracle_calls)}")

        print(f'avg_top1: {score_avg_top1:.3f} | '
                f'avg_top10: {score_avg_top10:.3f} | '
                f'avg_top100: {score_avg_top100:.3f}')

    def __len__(self):
        return len(self.mol_buffer)

    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls



class Pool:
    def __init__(self, size, validation_perc: float):
        self.size = size
        self.optim_entries: List[OptimEntry] = []
        self.num_validation_entries = int(size * validation_perc + 1)

    # def random_dump(self, num):
    #     for _ in range(num):
    #         rand_ind = random.randint(0, num - 1)
    #         self.molecule_entries.pop(rand_ind)
    #     print(f"Dump {num} random elements from pool, num pool mols {len(self)}")

    def add(self, entries: List, diversity_score=1.0):
        assert type(entries) == list
        self.optim_entries.extend(entries)
        self.optim_entries.sort(key=lambda x: x.last_entry, reverse=True)

        # remove doublicates
        new_optim_entries = []
        for entry in self.optim_entries:
            insert = True
            for e in new_optim_entries:
                if (
                    entry.last_entry == e.last_entry
                    or tanimoto_dist_func(
                        entry.last_entry.fingerprint, e.last_entry.fingerprint
                    )
                    > diversity_score
                ):
                    insert = False
                    break
            if insert:
                new_optim_entries.append(entry)

        self.optim_entries = new_optim_entries[: min(len(new_optim_entries), self.size)]
        curr_num_validation_entries = sum([entry.entry_status == EntryStatus.valid for entry in self.optim_entries])

        i = 0
        while curr_num_validation_entries < self.num_validation_entries:
            if self.optim_entries[i].entry_status == EntryStatus.none:
                self.optim_entries[i].entry_status = EntryStatus.valid
                curr_num_validation_entries += 1
            i += 1
        
        for j in range(i, len(self.optim_entries)):
            if self.optim_entries[j].entry_status == EntryStatus.none:
                self.optim_entries[j].entry_status = EntryStatus.train
        
        curr_num_validation_entries = sum([entry.entry_status == EntryStatus.valid for entry in self.optim_entries])
        assert curr_num_validation_entries == min(len(self.optim_entries), self.num_validation_entries)

    def get_train_valid_entries(self):
        train_entries = []
        valid_entries = []
        for entry in self.optim_entries:
            if entry.entry_status == EntryStatus.train:
                train_entries.append(entry)
            elif entry.entry_status == EntryStatus.valid:
                valid_entries.append(entry)
            else:
                raise Exception(f"EntryStatus of an entry in pool cannot be {entry.entry_status}.")
        assert min(len(self.optim_entries), self.num_validation_entries) == len(valid_entries)
        return train_entries, valid_entries

    def random_subset(self, subset_size):
        rand_inds = np.random.permutation(min(len(self.optim_entries), subset_size))
        # rand_inds = np.random.permutation(len(self.optim_entries))
        # rand_inds = rand_inds[:subset_size]
        return [self.optim_entries[i] for i in rand_inds]

    def __len__(self):
        return len(self.optim_entries)


def make_output_files_base(input_path, results_dir, run_name, config):
    formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
    base = os.path.join(input_path, run_name, formatted_date_time)
    os.makedirs(base, exist_ok=True)
    v = 0
    strategy = "+".join(config["strategy"])
    while os.path.exists(os.path.join(base, f"{strategy}-{v}")):
        v += 1
    output_dir = os.path.join(base, f"{strategy}-{v}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_prompt_with_similars(mol_entry: MoleculeEntry, sim_range=None):
    prompt = ""
    for sim_mol_entry in mol_entry.similar_mol_entries:
        if sim_range:
            prompt += f"[SIMILAR]{sim_mol_entry.smiles} {generate_random_number(sim_range[0], sim_range[1]):.2f}[/SIMILAR]"  # noqa
        else:
            prompt += f"[SIMILAR]{sim_mol_entry.smiles} {tanimoto_dist_func(sim_mol_entry.fingerprint, mol_entry.fingerprint):.2f}[/SIMILAR]"  # noqa
    return prompt


class EntryStatus:
    none = 0
    train = 1
    valid = 2


class OptimEntry:
    def __init__(self, last_entry, mol_entries):
        self.last_entry: MoleculeEntry = last_entry
        self.mol_entries: List[MoleculeEntry] = mol_entries
        self.entry_status: EntryStatus = EntryStatus.none

    def to_prompt(
            self, is_generation: bool,
            include_oracle_score: bool, config,
            max_score=None
        ):
        prompt = ""
        # prompt = config["eos_token"]
        for mol_entry in self.mol_entries:
            prompt += config["eos_token"]
            prompt += create_prompt_with_similars(mol_entry=mol_entry)

            for prop_name, prop_spec in mol_entry.add_props.items():
                prompt += f"{prop_spec['start_tag']}{prop_spec['value']}{prop_spec['end_tag']}"
                
            if "default" in config["strategy"]:
                pass
            elif "rej-sample-v2" in config["strategy"]:
                if include_oracle_score:
                    prompt += f"[PROPERTY]oracle_score {mol_entry.score:.2f}[/PROPERTY]"
            else:
                raise Exception(f"Strategy {config['strategy']} not known.")
            prompt += f"[START_SMILES]{mol_entry.smiles}[END_SMILES]"

        assert self.last_entry
        prompt += config["eos_token"]
        if is_generation:
            prompt_with_similars = create_prompt_with_similars(
                self.last_entry, sim_range=config["sim_range"]
            )
        else:
            prompt_with_similars = create_prompt_with_similars(self.last_entry)

        prompt += prompt_with_similars

        for prop_name, prop_spec in self.last_entry.add_props.items():
            prompt += prop_spec["start_tag"] + prop_spec["infer_value"](self.last_entry) + prop_spec["end_tag"]

        if "default" in config["strategy"]:
            pass
        elif "rej-sample-v2" in config["strategy"]:
            if is_generation:
                # oracle_scores_of_mols_in_prompt = [e.score for e in self.mol_entries]
                # q_0_9 = (
                #     np.quantile(oracle_scores_of_mols_in_prompt, 0.9)
                #     if oracle_scores_of_mols_in_prompt
                #     else 0
                # )
                # desired_oracle_score = generate_random_number(
                #     q_0_9, config["max_possible_oracle_score"]
                # )
                desired_oracle_score = max_score
                oracle_score = desired_oracle_score
            else:
                oracle_score = self.last_entry.score
            if include_oracle_score:
                prompt += f"[PROPERTY]oracle_score {oracle_score:.2f}[/PROPERTY]"
        else:
            raise Exception(f"Strategy {config['strategy']} not known.")

        if is_generation:
            prompt += "[START_SMILES]"
        else:
            prompt += f"[START_SMILES]{self.last_entry.smiles}[END_SMILES]"
            prompt += config["eos_token"]

        return prompt

    def contains_entry(self, mol_entry: MoleculeEntry):
        for entry in self.mol_entries:
            if mol_entry == entry:
                return True
            for sim_entry in entry.similar_mol_entries:
                if mol_entry == sim_entry:
                    return True

        return False
