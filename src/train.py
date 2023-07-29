import transformers
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from eval_metrics import compute_metrics, preprocess_logits_for_metrics
import yaml
import argparse
import glob
import sys
from callbacks import CustomAimCallback
import os
from custom_trainer import CustomTrainer
from dataset_utils import process_dataset
from utils import CustomTokenizer


def load_model(from_pretrained: str):
    if from_pretrained == "small_opt":
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
    return AutoModelForCausalLM.from_pretrained(from_pretrained)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="none")

    parser.add_argument(
        "--from_pretrained",
        type=str,
        metavar="FP",
        dest="from_pretrained",
        required=True,
        help="the path to the model dir",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        metavar="MC",
        dest="model_config",
        required=True,
        help="the galactica configuration to use",
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
        "--no_track",
        type=bool,
        metavar="NT",
        dest="no_track",
        required=False,
        help="whether or not track the training using aim",
        default=False,
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        metavar="TC",
        dest="tokenizer_checkpoint",
        required=False,
        help="tokenizer checkpoint name",
        default=None,
    )

    args = parser.parse_args()
    from_pretrained = args.from_pretrained
    model_config = args.model_config
    training_data_dir = args.training_data_dir
    valid_data_dir = args.valid_data_dir
    max_steps = args.max_steps
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    experiment_name = args.experiment_name
    no_track = args.no_track
    blocksize = args.blocksize
    track_dir = args.track_dir
    checkpoints_root_dir = args.checkpoints_root_dir
    tokenizer_checkpoint = (
        args.tokenizer_checkpoint if args.tokenizer_checkpoint else from_pretrained
    )

    training_data_files = glob.glob(training_data_dir + "/*.jsonl")
    valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(absolute_path)
    relative_path = "config/models_train_config.yaml"

    full_path = os.path.join(absolute_path, relative_path)

    with open(full_path, "r") as f_:
        full_file = yaml.full_load(f_)
        train_config = full_file[model_config]

    experiment_hash = "none"

    model = load_model(from_pretrained)
    tokenizer = CustomTokenizer(
        instance=AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    ).get_instance()
    model = (
        model.to_bettertransformer()
    )  # Converts the model to use PyTorch’s native attention implementation

    trainer_callback_list = []
    if not no_track:
        aim_callback = CustomAimCallback(
            checkpoints_dict_name="checkpoints_hashes",
            repo=track_dir,
            experiment=experiment_name,
            model=model,
            blocksize=train_config["block_size"],
        )

        experiment_hash = aim_callback._run_hash
        trainer_callback_list.append(aim_callback)

    checkpoints_dir = os.path.join(
        checkpoints_root_dir, from_pretrained, experiment_hash
    )

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
        torch_compile=True,
        dataloader_num_workers=8,
    )

    dataset = load_dataset(
        "text",
        data_files={"train": training_data_files, "validation": valid_data_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset, tokenizer=tokenizer, train_config=train_config
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        callbacks=trainer_callback_list,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    sys.exit(0)  # explositly set exit code to 0 when succesfully termitating
