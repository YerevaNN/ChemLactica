# import ujson

import orjson
import numpy as np
import json
from .text_format_utils import generate_formatted_string, delete_empty_tags

# import torch

from .utils import get_tokenizer
from .assay_doc_utils import get_compound_assay_docs, process_incomplete_docs

from accelerate import PartialState

state = PartialState()

DIR_DATA_TYPES = {"computed", "assay"}


def load_jsonl_line(jsonl_line):
    try:
        _maybe_compound_dict = orjson.loads(jsonl_line)
        if isinstance(_maybe_compound_dict, dict):
            return _maybe_compound_dict
        else:
            return orjson.loads(_maybe_compound_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")


def generate_assay_docs(examples, train_config, model_config):
    tokenizer = get_tokenizer(model_config.tokenizer_path)
    MODEL_CONTEXT_LENGTH = model_config.block_size
    final = {
        "input_ids": [],
        # "token_type_ids": [],
        "attention_mask": [],
    }
    incomplete_docs = []
    for compound_str in examples["text"]:
        try:
            compound = load_jsonl_line(compound_str)
            result, incomplete_doc = get_compound_assay_docs(
                tokenizer, compound, MODEL_CONTEXT_LENGTH
            )
            if incomplete_doc:
                incomplete_docs.append(incomplete_doc)
            if len(result["input_ids"]) == 0:
                raise ValueError
            final["input_ids"].extend(result["input_ids"])
            # final["token_type_ids"].extend(result["token_type_ids"])
            final["attention_mask"].extend(result["attention_mask"])
        except Exception:
            continue
    patched_documents = process_incomplete_docs(
        incomplete_docs, tokenizer, MODEL_CONTEXT_LENGTH
    )
    final["input_ids"].extend(patched_documents["input_ids"])
    # final["token_type_ids"].extend(patched_documents["token_type_ids"])
    final["attention_mask"].extend(patched_documents["attention_mask"])

    final["labels"] = final["input_ids"].copy()
    return final


def tokenize_function(examples, model_config, tokenizer):
    tokenizer = get_tokenizer(model_config.tokenizer_path)
    # print(f"Process id: {os.getpid()}, {tokenizer}")
    return tokenizer(examples["text"], return_token_type_ids=False)


def process_str(str, random_number_generator, model_config):
    # it's wierd workaround but works for now
    try:
        compound = load_jsonl_line(str["text"])
    except Exception as e:
        print(e)
        return ""
    compound = delete_empty_tags(compound)
    str["text"] = generate_formatted_string(
        compound, random_number_generator, model_config
    )
    return str


def group_texts(examples, model_config, eos_token_id):
    # Concatenate all texts.
    concatenated_examples = {
        "input_ids": sum(
            examples["input_ids"],
            [eos_token_id],
        ),
        # "token_type_ids": torch.as_tensor(sum(examples["token_type_ids"], [0])),
        "attention_mask": sum(examples["attention_mask"], [1]),
    }

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder,
    # we could add padding if the model supported it instead of this drop.
    total_length = (total_length // model_config.block_size) * model_config.block_size
    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + model_config.block_size]  # noqa
            for i in range(0, total_length, model_config.block_size)
        ]
        for k, t in concatenated_examples.items()
        # k : t[:total_length].view(-1, train_config["block_size"])
        # for k, t in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


def process_dataset(
    dataset,
    train_config,
    model_config,
    process_batch_sizes: tuple,
    is_eval=False,
    assay=True,
):
    tokenizer = get_tokenizer(model_config.tokenizer_path)
    eos_token_id = tokenizer.eos_token_id
    rng = np.random.default_rng()

    if assay:
        if is_eval:
            lm_datasets = dataset.map(
                generate_assay_docs,
                batched=True,
                fn_kwargs={"train_config": train_config, "model_config": model_config},
                remove_columns=["text"],
                batch_size=30,
                num_proc=4,
            )
        else:
            lm_datasets = dataset.map(
                generate_assay_docs,
                batched=True,
                fn_kwargs={"train_config": train_config, "model_config": model_config},
                remove_columns=["text"],
                batch_size=30,
            )
    else:
        if is_eval:
            dataset = dataset.map(
                process_str,
                num_proc=8,
                fn_kwargs={
                    "random_number_generator": rng,
                    "model_config": model_config,
                },
            )
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=False,
                fn_kwargs={"model_config": model_config, "tokenizer": tokenizer},
                remove_columns=["text"],
                batch_size=process_batch_sizes[0],
                num_proc=4,
            )
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                fn_kwargs={"model_config": model_config, "eos_token_id": eos_token_id},
                num_proc=4,
            )
        else:
            # with state.main_process_first():
            dataset = dataset.map(
                process_str,
                fn_kwargs={
                    "random_number_generator": rng,
                    "model_config": model_config,
                },
            )
            # with state.main_process_first():
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                fn_kwargs={"model_config": model_config, "tokenizer": tokenizer},
                batch_size=process_batch_sizes[0],
                remove_columns=["text"],
            )
            # with state.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                batch_size=process_batch_sizes[1],
                fn_kwargs={
                    "model_config": model_config,
                    "eos_token_id": eos_token_id,
                },
            )

    return lm_datasets


def sft_formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["smiles"])):
        text = (
            f"[START_SMILES]{example['smiles'][i]}[END_SMILES]"
            f"[PROPERTY]activity {round(example['activity'][i], 2)}[/PROPERTY]"
        )
        output_texts.append(text)
    return output_texts
