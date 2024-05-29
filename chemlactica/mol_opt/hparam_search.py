from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
import datetime
import argparse
import os
from utils import ConstraedTPSAOracle
from typing import List
from chemlactica.mol_opt.optimization import optimize

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def default_train_condition(num_iter, tol_level, prev_train_iter):
    return num_iter - prev_train_iter >= 3


def tolerance_train_condition(cur_tol_level, train_tol_level):
    return cur_tol_level >= train_tol_level


def choose_train_condition(name):
    return {
        "default" : default_train_condition,
        "tolerance": tolerance_train_condition
    }[name]


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
    print(config)

    model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    oracle = ConstraedTPSAOracle(max_oracle_calls=5000)
    for seed in seeds[:args.n_runs]:
        config["log_dir"] = os.path.join(args.output_dir, "results_tpsa+weight+num_rungs.log")
        config["rej_sample_config"]["should_train"] = choose_train_condition("tolerance")
        optimize(
            model, tokenizer,
            oracle, config
        )