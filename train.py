import transformers
import random
from transformers import AutoTokenizer
from datasets import load_dataset
import json

model_name = "facebook/galactica-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_data(compound_data):
    decoder = json.JSONDecoder()
    compound_data = convert_to_json(compound_data, decoder)
    return tokenizer(compound_data["text"], padding="max_length", truncation=True)


def convert_to_json(compound_data, decoder):
    # Decode and load json strings
    compound_data["text"] = [
        generate_formatted_string(json.loads(decoder.decode(compound_string)))
        for compound_string in compound_data["text"]
    ]

    return compound_data


def generate_formatted_string(compound_json):
    # Shuffle the keys
    keys = list(compound_json.keys())
    random.shuffle(keys)

    # Convert keys to uppercase and create the key-value string
    key_value_pairs = []
    for key in keys:
        upper_key = key.upper()
        value = compound_json[key]
        key_value_pairs.append(f"{upper_key}{value}")

    # Join the key-value pairs into a string
    compound_formatted_string = "".join(key_value_pairs)
    return compound_formatted_string


dataset = load_dataset(path="data")
tokenized_datasets = dataset.map(preprocess_data, batched=True)
subset_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
