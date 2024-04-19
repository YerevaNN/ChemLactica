from typing import List
import random
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.QED import qed

# Disable RDKit logs
RDLogger.DisableLog('rdApp.*')


def set_seed(seed_value):
    random.seed(seed_value)
    # Set seed for NumPy
    np.random.seed(seed_value)
    # Set seed for PyTorch
    torch.manual_seed(seed_value)


def get_short_name_for_ckpt_path(chpt_path: str, hash_len: int=6):
    get_short_name_for_ckpt_path = Path(chpt_path)
    return get_short_name_for_ckpt_path.parent.name[:hash_len] + '-' + get_short_name_for_ckpt_path.name.split("-")[-1]


def get_morgan_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def get_maccs_fingerprint(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def tanimoto_dist_func(mol1, mol2, fingerprint: str="morgan"):
    return DataStructs.TanimotoSimilarity(
        get_morgan_fingerprint(mol1) if fingerprint == 'morgan' else get_maccs_fingerprint(mol1),
        get_morgan_fingerprint(mol2) if fingerprint == 'morgan' else get_maccs_fingerprint(mol2),
    )


def generate_random_number(lower, upper):
    return lower + random.random() * (upper - lower)


def canonicalize(smiles):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), kekuleSmiles=True)


class MoleculeEntry:

    def __init__(self, smiles, score=None, score_estimate=None, **kwargs):
        self.smiles = canonicalize(smiles)
        self.mol = Chem.MolFromSmiles(smiles)
        self.inchi = Chem.MolToInchi(self.mol)
        self.fingerprint = get_morgan_fingerprint(self.mol)
        self.score = score
        self.score_estimate = score_estimate
        self.additional_properties = kwargs

    def __eq__(self, other):
        return self.inchi == other.inchi

    def __lt__(self, other):
        return self.score < other.score
    
    def __str__(self):
        return f"smiles: {self.smiles}, " \
            f"score: {round(self.score, 4) if self.score else 'none'}, " \
            f"score_estimate: {round(self.score_estimate, 4) if self.score_estimate else 'none'}"
    
    def __repr__(self):
        return str(self)
    

class MoleculePool:

    def __init__(self, size):
        self.size = size
        self.molecule_entries: List[MoleculeEntry] = []

    def add(self, entries: List[MoleculeEntry]):
        assert type(entries) == list
        self.molecule_entries.extend(entries)
        self.molecule_entries.sort(reverse=True)

        # remove doublicates
        new_molecule_list = []
        for mol in self.molecule_entries:
            if len(new_molecule_list) == 0 or new_molecule_list[-1] != mol:
                new_molecule_list.append(mol)

        self.molecule_entries = new_molecule_list[:min(len(new_molecule_list), self.size)]

    def random_subset(self, subset_size):
        rand_inds = np.random.permutation(min(len(self.molecule_entries), subset_size))
        return [self.molecule_entries[i] for i in rand_inds]