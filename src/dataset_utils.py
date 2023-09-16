import json
from text_format_utils import generate_formatted_string, delete_empty_tags
import torch

from utils import CustomTokenizer
from multiprocessing import Process, Queue
from multiprocessing import set_start_method
from typing import Dict
from datasets import IterableDataset
set_start_method("fork")


class JsonlDataset:

    def __init__(self, _file_path):
        self._f = open(_file_path, "r")
        self._line_number = 0
        self._exhausted = False

    def __next__(self):
        if self._exhausted:
            return None
        
        line = self._f.readline()
        self._line_number += 1
        if not line:
            self._f.close()
            self._exhausted = True
            return None
        
        return {"text": line} # this is done to be compatible with the old code

    def get_read_position(self):
        """ return the read position in bytes """
        return self._f.tell()

    def set_read_position(self, position):
        self._f.seek(position)


# class CustomIterableDataset(IterableDataset):

#     def __init__(self, *args, **kwargs):
#         self.generator = None

#     @staticmethod
#     def from_generator(generator, *args, **kwargs):
#         return super().from_generator(generator, *args, **kwargs)


# class CustomSamplesGenerator:

#     def __init__(self, jsonl_datasets: Dict[str, JsonlGeneratorDataset]):
#         self.jsonl_datasets = jsonl_datasets
#         self.iter = self.__iter__()

#     def __call__(self):
#         return next(self.iter)
    
#     def __iter__(self):
#         for ds in self.jsonl_datasets.values():
#             for sample in ds.generator():
#                 if sample:
#                     yield sample

def samples_generator(jsonl_datasets: Dict[str, JsonlDataset]):
    def inner_func():
        while True:
            _returned = False
            for ds in jsonl_datasets.values():
                sample = next(ds)
                if sample:
                    _returned = True
                    yield sample
            
            if not _returned:
                break

    return inner_func


def tokenize_function(examples):
    tokenizer = CustomTokenizer.get_instance()
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
            sum(examples["input_ids"], [CustomTokenizer.eos_token_id])
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
            t[i : i + train_config["block_size"]] # noqa
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
