import os
import json
import yaml
from transformers import AutoTokenizer
from functools import cache
from chemlactica.config.default_train_config import TrainConfig, ModelConfig

default_tokenizer_path = (
    "/auto/home/menuab/code/ChemLactica/chemlactica/tokenizer/ChemLacticaTokenizer66"
)


@cache
def get_start2end_tags_map(tokenizer_path: str = default_tokenizer_path):
    with open(os.path.join(tokenizer_path, "special_tokens_map.json"), "r") as _f:
        special_tokens_map = json.load(_f)
    additional_tokens = special_tokens_map.get("additional_special_tokens", None)
    n = len(additional_tokens)
    assert (n & 1) == 0  # should be even, every opening tag needs a closing tag
    return {
        additional_tokens[i]: additional_tokens[n // 2 + i] for i in range(n // 2)
    } | {"[START_SMILES]": "[END_SMILES]"}


# def get_tokenizer_special_tokens(tokenizer_path: str = default_tokenizer_path):
#     with open(os.path.join(tokenizer_path, "special_tokens_map.json"), "r") as _f:
#         special_tokens_json = json.load(_f)
#     return special_tokens_json["additional_special_tokens"]


def get_tokenizer_length(model_config):
    tokenizer = get_tokenizer(model_config.tokenizer_path)
    tokenizer_len = len(tokenizer)
    del tokenizer
    return tokenizer_len


@cache
def get_tokenizer(tokenizer_path):
    return create_tokenizer(tokenizer_path)


def create_tokenizer(tokenizer_path):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.bos_token = "<s>"
    tok.bos_token_id = 0
    tok.pad_token = "<pad>"
    tok.pad_token_id = 1
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    print(f"Process {os.getpid()} created a tokenizer")
    return tok


class ForcedStop(RuntimeError):
    def __init__(self, message="Forced stop occurred"):
        super().__init__(message)


def signal_handler(sig, frame):
    raise ForcedStop


def get_called_command(args):
    if args.slurm_eval:
        print("slurm eval")
        command = [
            "python3",
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            "chemlactica/config/eval_config.yaml",
            "chemlactica/train.py",
        ]
        for arg, value in vars(args).items():
            if isinstance(value, list):
                list_vals_str = str(" ".join(map(str, value)))
                command.extend([f"--{arg}", list_vals_str])
            elif value is not None:
                if isinstance(value, bool) and value:
                    command.extend([f"--{arg}"])
                elif isinstance(value, bool) and not value:
                    pass
                else:
                    command.extend([f"--{arg}", str(value)])
    else:
        command = None
    return command


def remove_extraneous_args(args):
    if (
        hasattr(args, "accelerate_eval_config_file")
        and args.accelerate_eval_config_file
    ):
        delattr(args, "accelerate_eval_config_file")


def get_model_train_config(train_config_name):
    model_config = ModelConfig()
    train_config = TrainConfig()
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "config_yamls",
        f"{train_config_name}_config.yaml",
    )
    with open(config_path, "r") as infile:
        custom_config = yaml.full_load(infile)
        for k, v in custom_config["model_config"].items():
            setattr(model_config, k, v)
        for k, v in custom_config["train_config"].items():
            setattr(train_config, k, v)
    return model_config, train_config


if __name__ == "__main__":
    # import sys
    import glob

    # from utils import CustomTokenizer
    from config.create_train_config import model_train_configs
    from datasets import load_dataset
    from dataset_utils import process_dataset

    train_config = model_train_configs["125m"]
    train_config["block_size"] = 2048

    # CustomTokenizer.set_model_size("125m")
    # tokenizer = CustomTokenizer.get_instance()
    # tokenizer.save_pretrained("ChemLacticaTokenizer")
    training_data_dir = ".small_data/valid"

    # training_data_files = glob.glob(training_data_dir + "/xae_shuf.jsonl")
    training_data_files = glob.glob(training_data_dir + "/*.jsonl")

    dataset = load_dataset(
        "text",
        data_files={"train": training_data_files, "validation": training_data_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset, train_config=train_config, process_batch_sizes=(50, 50)
    )

    sample = next(iter(processed_dataset["train"]))
    prompt = "[START_SMILES] CCCCN [END_SMILES][CLOGP 0.00][SAS 123][QED]"
    # print(tokenizer.decode(sample["input_ids"]))
    # print(*sample["input_ids"].numpy())
    # print(len(tokenizer.decode(sample["input_ids"])), len(sample["input_ids"].numpy()))
    # print('*'*20)
    # print(prompt)
    # print(tokenizer.encode(prompt))
