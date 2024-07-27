from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
import datetime
import argparse
import os
from utils import ConstrainedTPSAOracle
from typing import List
from chemlactica.mol_opt.optimization import optimize

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_default", type=str, required=False, default="chemlactica/chemlactica_125m_hparams.yaml")
    parser.add_argument("--n_runs", type=int, required=False, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(open(args.config_default))

    model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    oracle = ConstrainedTPSAOracle(max_oracle_calls=5000)
    config["log_dir"] = os.path.join(args.output_dir, "results_tpsa+weight+num_rungs.log")
    optimize(
        model, tokenizer,
        oracle, config
    )