from typing import List, Dict
import argparse
import datetime
import random
import glob
import math
import itertools as it
import os
import tqdm
from dataclasses import dataclass
from enum import Enum

from utils import get_tokenizer
from config.create_train_config import model_fine_tune_configs
from model_utils import load_model
from generation.generation import generate
from rejection_sampling_configs import sample_gen_args, rej_sample_args

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem import MACCSkeys

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_inchi(smiles: str):
    return Chem.MolToInchi(Chem.MolFromSmiles(smiles))


def get_morgan_fingerprint(smiles: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)


def get_maccs_fingerprint(smiles: str):
    return MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))


class FingerprintType(Enum):
    MORGAN=1,
    MACCS=2


def tanimoto_dist_func(smiles1: str, smiles2: str, fingerprint: FingerprintType=FingerprintType.Morgan):
    return DataStructs.TanimotoSimilarity(
        get_morgan_fingerprint(smiles1) if fingerprint == FingerprintType.Morgan else get_maccs_fingerprint(smiles1),
        get_morgan_fingerprint(smiles2) if fingerprint == fingerprint == FingerprintType.Morgan else get_maccs_fingerprint(smiles2),
    )


def compute_qed(smiles: str):
    return qed(Chem.MolFromSmiles(smiles))


def find(string: str, start: str, end: str="\n"):
    s = string.find(start)
    e = string.find(end)
    if e != -1:
        return string[s + len(start):e]


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


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_dataset(
    checkpoint_path,
    run_hash,
    round,
    num_samples,
    max_similars_in_prompt,
    use_flash_attn,
    seed,
    lead_molecule,
    device
):
    set_seed(seed)
    formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M")
    model_name = checkpoint_path.split(r'/')[-2:]
    base_path = f"/nfs/dgx/raid/chem/data/rej_sampling_data/{run_hash}"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    ds_file_name = f'{base_path}/round:{round}_hash:{model_name[0][:6]}_step:{model_name[1].split("-")[-1]}_date:{formatted_date_time}.csv'

    tokenizer = get_tokenizer()
    
    lead_molecule = Chem.MolToSmiles(Chem.MolFromSmiles(lead_molecule), canonical=True)
    lead_molecule = Chem.MolToSmiles(Chem.MolFromSmiles(lead_molecule), kekuleSmiles=True)
    print("some molecule", lead_molecule)

    
    generator_model = load_model(checkpoint_path, use_flash_attn=use_flash_attn, dtype=torch.bfloat16).to(device)
    def next_input_sample(lead_molecule: str):
        num_of_similar = random.randint(0, max_similars_in_prompt)
        # num_of_similar = 5
        input_text = "</s>"
        try:
            lead_molecule_entry = MoleculeEntry(
                smiles=lead_molecule, qed=compute_qed(lead_molecule),
                morgan_sim_to_lead=random.uniform(0.9, 0.99),
                inchi=get_inchi(lead_molecule)
            )
        except Exception:
            return None
        cand_similar_molecules_in_prompt = []
        if num_of_similar:
            sample_gen_args["num_return_sequences"] = num_of_similar
            for outputs in generate(
                    prompts=[
                        f"</s>[SIMILAR]{lead_molecule_entry.smiles} {random.uniform(0.9, 0.99):.2f}[/SIMILAR][QED]{random.uniform(0.9, 0.99):.2f}[/QED][START_SMILES]"
                        for i in range(num_of_similar)
                    ],
                    model=generator_model,
                    **sample_gen_args
                ).values():
                for out in outputs:
                    smiles = find(out, "[START_SMILES]", "[END_SMILES]")
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        gen_molecule_entry = MoleculeEntry(
                            smiles=smiles,
                            qed=compute_qed(smiles),
                            morgan_sim_to_lead=tanimoto_dist_func(smiles, lead_molecule_entry.smiles),
                            maccs_sim_to_lead=tanimoto_dist_func(smiles, lead_molecule_entry.smiles, FingerprintType.MACCS),
                            inchi=get_inchi(smiles)
                        )
                        if gen_molecule_entry != lead_molecule_entry:
                            cand_similar_molecules_in_prompt.append(gen_molecule_entry)
                    except Exception as e:
                        # print(e)
                        pass

        # cand_similar_molecules_in_prompt = np.unique(cand_similar_molecules_in_prompt)
        # three_times_num_of_similar = num_of_similar * 3
        # cand_similar_molecules_in_prompt = cand_similar_molecules_in_prompt[-three_times_num_of_similar:]

        # pairwise_distance_matrix = np.ones(three_times_num_of_similar, three_times_num_of_similar)
        # for i in range(three_times_num_of_similar):
        #     for j in range(i):
        #         pairwise_distance_matrix[i][j] = tanimoto_dist_func(cand_similar_molecules_in_prompt[i], cand_similar_molecules_in_prompt[[j]])
        #         pairwise_distance_matrix[j][i] = pairwise_distance_matrix[i][j]

        # best_sum_pairwise_distance = math.inf
        # best_combination = None
        # for comb in it.combinations(list(range(three_times_num_of_similar)), num_of_similar):
        #     cur_sum_pairwise_distance = 0
        #     for i in comb:
        #         for j in comb:
        #             cur_sum_pairwise_distance += pairwise_distance_matrix[i][j]
        #     if cur_sum_pairwise_distance < best_sum_pairwise_distance:
        #         best_combination = comb

        # similar_molecules_in_prompt = [cand_similar_molecules_in_prompt[i] for i in best_combination]

        cand_similar_molecules_in_prompt = list(np.unique(cand_similar_molecules_in_prompt)[-max_similars_in_prompt:])
        similar_molecules_in_prompt = cand_similar_molecules_in_prompt.copy()
        similar_molecules_in_prompt.append(lead_molecule_entry)
        random.shuffle(similar_molecules_in_prompt)
        for mol in similar_molecules_in_prompt:
            input_text += f"[SIMILAR]{mol.smiles} {mol.morgan_sim_to_lead:.2f}[/SIMILAR]"
        input_text += f"[QED]{random.uniform(0.9, 0.99):.2f}[/QED]"
        candidate_target_molecules = []
        for outputs in generate(
                input_text,
                model=generator_model,
                **rej_sample_args
            ).values():
            for out in outputs:
                smiles = find(out, "[START_SMILES]", "[END_SMILES]")
                try:
                    gen_mol = MoleculeEntry(
                        smiles=smiles,
                        qed=compute_qed(smiles),
                        morgan_sim_to_lead=tanimoto_dist_func(smiles, lead_molecule_entry.smiles),
                        maccs_sim_to_lead=tanimoto_dist_func(smiles, lead_molecule_entry.smiles, FingerprintType.MACCS),
                        inchi=get_inchi(smiles)
                    )
                    candidate_target_molecules.append(gen_mol)
                    cand_similar_molecules_in_prompt.append(gen_mol)
                except Exception as e:
                    pass

        for target_mol in np.unique(candidate_target_molecules)[::-1]:
            if target_mol in similar_molecules_in_prompt:
                continue
            sample = "</s>"
            for mol in similar_molecules_in_prompt:
                sample += f"[SIMILAR]{mol.smiles} {tanimoto_dist_func(mol.smiles, target_mol.smiles):.2f}[/SIMILAR]"
            sample += f"[QED]{target_mol.qed:.2f}[/QED][START_SMILES]{target_mol.smiles}[END_SMILES]</s>"
            if len(tokenizer(sample)["input_ids"]) > 500:
                continue
            # yield sample, target_mol
            return [(sample, target_mol)]

    list_of_entries = {
        "samples":[],
        "smiles":[],
        "qed":[],
        "morgan_sim_to_lead":[]
    }
    progress_bar = tqdm.tqdm(total=num_samples)
    while num_samples > 0:
        samples = next_input_sample(lead_molecule)
        if samples:
            for sample, target_mol in samples:
                # sample, target_mol = samples
                list_of_entries["samples"].append(sample)
                list_of_entries["smiles"].append(target_mol.smiles)
                list_of_entries["qed"].append(target_mol.qed)
                list_of_entries["morgan_sim_to_lead"].append(target_mol.morgan_sim_to_lead)
        num_samples -= 1
        progress_bar.update(1)
        if num_samples <= 0:
            break
    
    pd.DataFrame(list_of_entries).to_csv(ds_file_name)
    return ds_file_name


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
        "--num_samples",
        type=int,
        dest="num_samples",
        required=True,
    )
    parser.add_argument(
        "--generation_name",
        type=str,
        metavar="EN",
        dest="generation_name",
        required=False,
        help="the name of the experiment",
        default="none",
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
        "--seed",
        type=int,
        required=False,
        default=42
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True
    )
    parser.add_argument(
        "--lead_molecule",
        type=str,
        required=True
    )
    
    args = parser.parse_args()
    generate_dataset(**args.__dict__)