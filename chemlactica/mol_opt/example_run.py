from typing import List
import yaml
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from rdkit.Chem import rdMolDescriptors
from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import set_seed, MoleculeEntry


class TPSA_Weight_Oracle:
    def __init__(self, max_oracle_calls: int):
        # maximum number of oracle calls to make
        self.max_oracle_calls = max_oracle_calls

        # the frequence with which to log
        self.freq_log = 100

        # the buffer to keep track of all unique molecules generated
        self.mol_buffer = {}

        # the maximum possible oracle score or an upper bound
        self.max_possible_oracle_score = 800

        # if True the __call__ function takes list of MoleculeEntry objects
        # if False (or unspecified) the __call__ function takes list of SMILES strings
        self.takes_entry = True

    def __call__(self, molecules: List[MoleculeEntry]):
        """
            Evaluate and return the oracle scores for molecules. Log the intermediate results if necessary.
        """
        oracle_scores = []
        for molecule in molecules:
            if self.mol_buffer.get(molecule.smiles):
                oracle_scores.append(sum(self.mol_buffer[molecule.smiles][0]))
            else:
                try:
                    tpsa = rdMolDescriptors.CalcTPSA(molecule.mol)
                    oracle_score = tpsa
                    weight = rdMolDescriptors.CalcExactMolWt(molecule.mol)
                    num_rings = rdMolDescriptors.CalcNumRings(molecule.mol)
                    if weight >= 350:
                        oracle_score = 0
                    if num_rings < 2:
                        oracle_score = 0

                except Exception as e:
                    print(e)
                    oracle_score = 0
                self.mol_buffer[molecule.smiles] = [oracle_score, len(self.mol_buffer) + 1]
                if len(self.mol_buffer) % 100 == 0:
                    self.log_intermediate()
                oracle_scores.append(oracle_score)
        return oracle_scores
    
    def log_intermediate(self):
        scores = [v[0] for v in self.mol_buffer.values()][-self.max_oracle_calls:]
        scores_sorted = sorted(scores, reverse=True)[:100]
        n_calls = len(self.mol_buffer)

        score_avg_top1 = np.max(scores_sorted)
        score_avg_top10 = np.mean(scores_sorted[:10])
        score_avg_top100 = np.mean(scores_sorted)

        print(f"{n_calls}/{self.max_oracle_calls} | ",
                f'avg_top1: {score_avg_top1:.3f} | '
                f'avg_top10: {score_avg_top10:.3f} | '
                f'avg_top100: {score_avg_top100:.3f}')

    def __len__(self):
        return len(self.mol_buffer)

    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        # the stopping condition for the optimization process
        return len(self.mol_buffer) >= self.max_oracle_calls


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_default", type=str, required=True)
    parser.add_argument("--n_runs", type=int, required=False, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(open(args.config_default))

    model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for i in range(args.n_runs):
        set_seed(seeds[i])
        oracle = TPSA_Weight_Oracle(max_oracle_calls=1000)
        config["log_dir"] = os.path.join(args.output_dir, f"results_chemlactica_tpsa+weight+num_rungs_{seeds[i]}.log")
        config["max_possible_oracle_score"] = oracle.max_possible_oracle_score
        optimize(
            model, tokenizer,
            oracle, config
        )