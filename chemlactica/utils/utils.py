import os
import json
import yaml
from transformers import AutoTokenizer
from functools import cache
from chemlactica.config.default_train_config import TrainConfig, ModelConfig
from sklearn.metrics import root_mean_squared_error

# from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import torch

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
    tok.add_bos_token = False
    tok.padding_side = "right"
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
        f"{train_config_name}-config.yaml",
    )
    with open(config_path, "r") as infile:
        custom_config = yaml.full_load(infile)
        for k, v in custom_config["model_config"].items():
            setattr(model_config, k, v)
        for k, v in custom_config["train_config"].items():
            setattr(train_config, k, v)
    return model_config, train_config


def get_numerical_validation(model, tokenizer, dataset, separator_token):
    # uncomment for sft classification tasks
    # model.eval()
    # ground_truths, preds, probs = [], [], []
    # eos_token_id = tokenizer.encode("[/PROPERTY]")[0]
    # for sample in self.dataset["validation"]:
    #     ground_truth = round(sample["activity"], 2)
    #     prompt = (
    #                 f"{self.separator_token}[START_SMILES]{sample['smiles']}"
    #                 "[END_SMILES][PROPERTY]activity"
    #             )
    #     texts = [" 0.0[/PROPERTY]", " 1.0[/PROPERTY]"]
    #     #uncomment the section below to fine tune a galactica model
    #     # prompt = f"Here is a SMILES formula: [START_I_SMILES]{sample['smiles']}[END_I_SMILES]"
    #                f"\n\nQuestion: Will the chemical compound penetrate the blood-brain barrier?"
    #                f"\n\nAnswer:"
    #     # texts = [" No</s>", " Yes</s>"]
    #     scores = []
    #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    #     for text in texts:
    #         text_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    #         sequence_ids = torch.cat([input_ids, text_ids], dim=-1).to(model.device)
    #         labels = torch.full(sequence_ids.shape, -100)
    #         labels[:, -text_ids.size(1):] = text_ids
    #         labels = labels.to(model.device)
    #         with torch.no_grad():
    #             outputs = model(sequence_ids, attention_mask=torch.ones_like(sequence_ids),\
    #                       labels=labels, return_dict = True)
    #             perplexity = -torch.exp(outputs.loss).item()
    #             scores.append(perplexity)
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #     scores = [(score - scipy.special.logsumexp(scores)) for score in scores]
    #     probs_ = list(np.exp(scores))
    #     pred = np.argmax(probs_)
    #     probs.append(probs_[1])
    #     preds.append(pred)
    #     ground_truths.append(ground_truth)
    # try:
    #     rmse = root_mean_squared_error(ground_truths, preds) if preds else 10
    #     roc = roc_auc_score(ground_truths, probs)
    # except ValueError:
    #     rmse, roc = 10, 0
    # model.train()
    # return rmse, roc
    model.eval()
    ground_truths, gens, diffs = [], [], []
    eos_token_id = tokenizer.encode("[/PROPERTY]")[0]
    for sample in dataset:
        ground_truth = round(sample["activity"], 2)
        prompt = (
            f"{separator_token}[START_SMILES]{sample['smiles']}"
            "[END_SMILES][PROPERTY]activity"
        )
        prompt = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                prompt.input_ids,
                do_sample=False,
                eos_token_id=eos_token_id,
                max_new_tokens=100,
            )
        out = tokenizer.batch_decode(out)[0]
        try:
            gen = out[
                out.find("activity ")
                + len("activity ") : out.find("[/PROPERTY]")  # noqa
            ]
            gen = float(gen)
            diff = abs(ground_truth - gen)
            ground_truths.append(ground_truth)
            gens.append(gen)
            diffs.append(diff)
        except ValueError:
            print(f"could not generate for {sample['smiles']}")
            pass
    try:
        rmse = root_mean_squared_error(ground_truths, gens) if gens else 10
        r, _ = pearsonr(ground_truths, gens)
    except ValueError:
        rmse, r = 10, 0
    model.train()
    return rmse, r


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
