from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# from eval_metrics import compute_metrics
from aim.hugging_face import AimCallback
from text_format_utils import generate_formatted_string, delete_empty_tags
import json
import yaml
import argparse
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

    args = parser.parse_args()
    model_type = args.model_type
    # n_epochs = args.n_epochs
    max_steps = args.max_steps

    model_checkpoint = f"facebook/galactica-{model_type}"

    with open("models_train_config.yaml", "r") as f_:
        train_config = yaml.full_load(f_)[model_type]

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
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
        eval_steps=1,
        max_steps=max_steps,
    )

    dataset = load_dataset(
        "text",
        data_files={"train": "./train.jsonl", "validation": "./evaluation.jsonl"},
        streaming=True,
    )
    dataset = dataset.map(process_str)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000)

    aim_callback = AimCallback(
        repo="/mnt/sxtn/chem/ChemLactica/metadata/aim",
        experiment="experiment",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        # callbacks=[aim_callback]
    )

    trainer.train()

    sys.exit(0)  # explositly set exit code to 0 when succesfully termitating
