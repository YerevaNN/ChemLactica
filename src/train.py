import transformers
from typing import Optional
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from eval_metrics import compute_metrics, preprocess_logits_for_metrics
from text_format_utils import generate_formatted_string, delete_empty_tags
import json
import yaml
import argparse
import glob
import sys
from callbacks import CustomAimCallback
import os


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
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
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
        required=True,
        help="the number of training steps after which to evaluate the model",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        metavar="SS",
        dest="save_steps",
        required=True,
        help="the number of steps to save a model checkpoint",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        dest="blocksize",
        required=False,
        default=2048,
        help="the number of count of checkpoint a model checkpoint",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        metavar="EN",
        dest="experiment_name",
        required=False,
        help="the name of the experiment",
        default="none",
    )
    parser.add_argument(
        "--track_dir",
        type=str,
        metavar="ATD",
        dest="track_dir",
        required=False,
        help="aim track directory",
        default="/mnt/sxtn/chem/ChemLactica/metadata/aim",
    )
    parser.add_argument(
        "--checkpoints_root_dir",
        type=str,
        metavar="CSD",
        dest="checkpoints_root_dir",
        required=False,
        help="directory where to save checkpoints",
        default="/mnt/sxtn/chem/ChemLactica/checkpoints",
    )
    parser.add_argument(
        "--track",
        type=bool,
        metavar="TR",
        dest="track",
        required=False,
        help="weather or not track the trainig using aim",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    model_type = args.model_type
    training_data_dir = args.training_data_dir
    valid_data_dir = args.valid_data_dir
    max_steps = args.max_steps
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    experiment_name = args.experiment_name
    track = args.track
    blocksize = args.blocksize
    track_dir = args.track_dir
    checkpoints_root_dir = args.checkpoints_root_dir

    training_data_files = glob.glob(training_data_dir + "/*.jsonl")
    valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(absolute_path)
    relative_path = "config/models_train_config.yaml"

    full_path = os.path.join(absolute_path, relative_path)

    with open(full_path, "r") as f_:
        train_config = yaml.full_load(f_)[model_type]

    experiment_hash = "none"

    model_checkpoint = f"facebook/galactica-{train_config['name_suffix']}"
    model = load_model(model_type)
    model = (
        model.to_bettertransformer()
    )  # Converts the model to use PyTorch’s native attention implementation

    trainer_callback_list = []
    if track:
        aim_callback = CustomAimCallback(
            checkpoints_dict_name="checkpoints_hashes",
            repo=track_dir,
            experiment=experiment_name,
            model=model,
            blocksize=2048,
        )

        experiment_hash = aim_callback._run_hash
        trainer_callback_list.append(aim_callback)

    checkpoints_dir = os.path.join(
        checkpoints_root_dir, f"galactica-{model_type}/{experiment_hash}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
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
        eval_steps=eval_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        dataloader_pin_memory=True,
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

    # train_count = 0
    # for sample in lm_datasets["train"]:
    #     train_count += 1
    # valid_count = 0
    # for sample in lm_datasets["validation"]:
    #     valid_count += 1

    # print("train", train_count, "valid", valid_count)
    class CustomTrainer(Trainer):
        def save_model(
            self, output_dir: Optional[str] = None, _internal_call: bool = False
        ):
            self.model = self.model.reverse_bettertransformer()
            super().save_model(output_dir)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        callbacks=trainer_callback_list,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    sys.exit(0)  # explositly set exit code to 0 when succesfully termitating
