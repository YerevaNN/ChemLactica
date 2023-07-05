import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from aim.hugging_face import AimCallback
from text_format_utils import generate_formatted_string, delete_empty_tags
import json
import yaml
import argparse
import glob
import sys


def process_str(str):
    # it's wierd workaround but works for now
    # st = str["text"].replace("\\", "")
    # print('ST IS    :   ',st)
    compound = json.loads(json.loads((str["text"])))
    str["text"] = delete_empty_tags(compound)
    str["text"] = generate_formatted_string(compound)
    # print(str['text'])
    # print('***************')
    # print(type(str['text']))
    return str


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder,
    # we could add padding if the model supported it instead of this drop.
    total_length = (total_length // train_config["block_size"]) * train_config[
        "block_size"
    ]
    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + train_config["block_size"]]  # noqa
            for i in range(0, total_length, train_config["block_size"])
        ]  # noqa
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result[
        "input_ids"
    ].copy()  # TODO: are not the targets shifted one to the right?
    return result


def load_model(load_small_opt: bool):
    if load_small_opt:
        return transformers.OPTForCausalLM(
            transformers.OPTConfig(
                vocab_size=train_config["vocab_size"],
                hidden_size=16,
                num_hidden_layers=1,
                ffn_dim=16,
                max_position_embeddings=2048,
                num_attention_heads=1,
                word_embed_proj_dim=16,
            )
        )
    return AutoModelForCausalLM.from_pretrained(model_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="none")

    parser.add_argument(
        "--model_type",
        type=str,
        metavar="MT",
        dest="model_type",
        required=True,
        help="the type of the model (depending on param size)",
    )

    parser.add_argument(
        "--training_data_dir",
        type=str,
        metavar="DT",
        dest="training_data_dir",
        required=True,
        help="path to directory containing training data",
    )
    parser.add_argument(
        "--valid_data_dir",
        type=str,
        metavar="VD",
        dest="valid_data_dir",
        required=True,
        help="path to directory containing validation data",
    )

    # parser.add_argument(
    #     "--n_epochs",
    #     type=int,
    #     metavar="EP",
    #     dest="n_epochs",
    #     required=True,
    #     help="the number of epochs to train",
    # )

    parser.add_argument(
        "--max_steps",
        type=int,
        metavar="MS",
        dest="max_steps",
        required=True,
        help="the number of steps to train (overrides the n_epochs)",
    )

    parser.add_argument(
        "--load_small_opt",
        type=bool,
        dest="load_small_opt",
        required=False,
        help="load small opt instead of Galactica (should be used in precommit_test.py only)",
        default=False,
    )

    args = parser.parse_args()
    model_type = args.model_type
    training_data_dir = args.training_data_dir
    valid_data_dir = args.valid_data_dir
    training_data_files = glob.glob(training_data_dir + "/*.jsonl")
    valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
    print(training_data_files)
    print(valid_data_files)
    # n_epochs = args.n_epochs
    max_steps = args.max_steps

    model_checkpoint = f"facebook/galactica-{model_type}"
    load_small_opt = args.load_small_opt

    with open("models_train_config.yaml", "r") as f_:
        train_config = yaml.full_load(f_)[model_type]

    model_checkpoint = f"facebook/galactica-{model_type}"

    model = load_model(load_small_opt)
    # print("number of parameters:", sum(p.numel() for p in model.parameters()))
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir=f"{model_checkpoint.split('/')[-1]}-finetuned-pubchem",
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        learning_rate=train_config["max_learning_rate"],
        lr_scheduler_type="linear",
        weight_decay=train_config["weight_decay"],
        adam_beta1=train_config["adam_beta1"],
        adam_beta2=train_config["adam_beta2"],
        warmup_steps=train_config["warmup_steps"],
        max_grad_norm=train_config["global_gradient_norm"],
        evaluation_strategy="steps",
        eval_steps=32,
        max_steps=max_steps,
    )

    dataset = load_dataset(
        "text",
        data_files={"train": training_data_files, "validation": valid_data_files},
        streaming=True,
    )
    dataset = dataset.map(process_str)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000)

    aim_callback = AimCallback(
        repo="/mnt/sxtn/chem/chemlactica_metadata/",
        experiment="experiment",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        # callbacks=[aim_callback],
    )

    trainer.train()

    sys.exit(0)  # explositly set exit code to 0 when succesfully termitating
