import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from eval_metrics import compute_metrics
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


def load_model(model_type: str):
    if model_type == "small_opt":
        return transformers.OPTForCausalLM(
            transformers.OPTConfig(
                vocab_size=train_config["vocab_size"],
                hidden_size=train_config["hidden_size"],
                num_hidden_layers=train_config["num_hidden_layers"],
                ffn_dim=train_config["ffn_dim"],
                max_position_embeddings=train_config["max_position_embeddings"],
                num_attention_heads=train_config["num_attention_heads"],
                word_embed_proj_dim=train_config["word_embed_proj_dim"],
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
    parser.add_argument(
        "--max_steps",
        type=int,
        metavar="MS",
        dest="max_steps",
        required=True,
        help="the number of steps to train (overrides the n_epochs)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        metavar="ES",
        dest="eval_steps",
        required=False,
        help="the number of training steps after which to evaluate the model",
        default=32,
    )
    parser.add_argument(
        "--track",
        type=bool,
        metavar="TR",
        dest="track",
        required=False,
        help="weather or not track the trainig using aim",
        default=False,
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        metavar="EAS",
        dest="eval_accumulation_steps",
        required=False,
        help="the number of steps after which to move \
            the prediction tensor from GPU to CPU during the evaluation \
            (specified to avoid OOM errors)",
        default=2,
    )

    args = parser.parse_args()
    model_type = args.model_type
    training_data_dir = args.training_data_dir
    valid_data_dir = args.valid_data_dir
    max_steps = args.max_steps
    eval_steps = args.eval_steps
    track = args.track
    eval_accumulation_steps = args.eval_accumulation_steps

    training_data_files = glob.glob(training_data_dir + "/*.jsonl")
    valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")

    with open("models_train_config.yaml", "r") as f_:
        train_config = yaml.full_load(f_)[model_type]

    model_checkpoint = f"facebook/galactica-{train_config['name_suffix']}"

    model = load_model(model_type)
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
        eval_accumulation_steps=eval_accumulation_steps,
        eval_steps=eval_steps,
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

    trainer_callback_list = []
    if track:
        trainer_callback_list.append(
            AimCallback(
                repo="/mnt/sxtn/chem/chemlactica_metadata/",
                experiment="experiment",
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        callbacks=trainer_callback_list,
    )

    trainer.train()

    sys.exit(0)  # explositly set exit code to 0 when succesfully termitating
