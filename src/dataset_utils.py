import json
from text_format_utils import generate_formatted_string, delete_empty_tags
import torch

from utils import CustomTokenizer
from typing import Dict, List
import pickle
import os
import subprocess


class JsonlDataset:

    def __init__(self, file_path):
        self._file_path = file_path
        self._f = open(file_path, "r")
        # for some mysterious reasons the line_number cannot be kept in the json file,
        # it does not work properly
        self.line_number = 0

    def __next__(self):
        line = self._f.readline()[:-1] # exclude the newline character
        if not line:
            if not self._f.closed:
                self._f.close()
            return None
        print("Process:", torch.distributed.get_rank(), self.line_number, "line number id:", id(self.line_number), "dataloader id:", id(self), self.get_read_position())
        self.line_number += 1
        return line
    
    def get_read_position(self):
        """
            returns an integer giving the file object’s current position in the file
            represented as number of bytes from the beginning of the file
        """
        return self._f.tell()

    def set_read_position(self, position):
        self._f.seek(position, 0)

    def get_state(self):
        return {
            "position": self.get_read_position(),
            "line_number": self.line_number
        }
    
    def load_state(self, state: Dict):
        self.set_read_position(state["position"])
        self.line_number = state["line_number"]
        print("loaded jsonl state:", {state: self.get_state()})


def samples_generator(jsonl_datasets_dict: Dict[str, JsonlDataset], pickle_states_path: str):
    jsonl_states = {}
    if os.path.exists(pickle_states_path):
        with open(pickle_states_path, "rb") as file:
            jsonl_states = pickle.load(file) # load the states
            for name, state in jsonl_states.items():
                jsonl_datasets_dict[name].load_state(state)

    while True:
        returned = False
        for name, ds in jsonl_datasets_dict.items():
            sample = next(ds)
            jsonl_states[name] = ds.get_state()
            with open(pickle_states_path, "wb") as file:
                pickle.dump(jsonl_states, file) # save the states
            if sample:
                returned = True
                yield {"text" : sample} # this is done to be compatible with the old code

        if not returned:
            break


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
