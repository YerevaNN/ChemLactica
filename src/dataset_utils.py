import json
from text_format_utils import generate_formatted_string, delete_empty_tags
import torch
import os

from utils import get_tokenizer


def tokenize_function(examples):
    tokenizer = get_tokenizer()
    # print(f"Process id: {os.getpid()}, {tokenizer}")
    return tokenizer(examples["text"])


def process_str(str):
    # it's wierd workaround but works for now
    try:
        compound = json.loads(json.loads((str["text"])))
    except Exception as e:
        print(e)
        return ""
    str["text"] = delete_empty_tags(compound)
    str["text"] = generate_formatted_string(compound)
    return str


def group_texts(examples, train_config):
    # Concatenate all texts.
    concatenated_examples = {
        "input_ids": torch.as_tensor(
            sum(examples["input_ids"], [get_tokenizer().eos_token_id])
        ),
        "token_type_ids": torch.as_tensor(sum(examples["token_type_ids"], [0])),
        "attention_mask": torch.as_tensor(sum(examples["attention_mask"], [1])),
    }

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
        # k : t[:total_length].view(-1, train_config["block_size"])
        # for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def process_dataset(dataset, train_config, process_batch_sizes: tuple):
    dataset = dataset.map(process_str)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=process_batch_sizes[0],
        remove_columns=["text"],
    )
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=process_batch_sizes[1],
        fn_kwargs={"train_config": train_config},
    )
    return lm_datasets
