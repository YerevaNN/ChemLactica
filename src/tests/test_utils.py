from datasets import load_dataset
from dataset_utils import process_dataset
import logging


def generate_batches_from_jsonls(jsonl_files, count):
    dataset = load_dataset(
        "text",
        data_files={"data": jsonl_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset,
        train_config={"block_size": 2048},
        process_batch_sizes=(100, 100),
    )

    batches = []
    for i, inp in enumerate(processed_dataset["data"]):
        if i == count: break
        del inp["token_type_ids"]
        inp = {k: inp[k].unsqueeze(0) for k in inp.keys()}
        batches.append(inp)
    return batches