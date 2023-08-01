import time
import argparse
import glob
import logging

from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset_utils import process_dataset
from transformers import TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers.trainer_pt_utils import IterableDatasetShard


logging.basicConfig(
    filename="dataloader_benchmark.log",
    level=logging.INFO,
    format="%(levelname)s: " "%(message)s",
)


def test_dataloader_speed(dl: DataLoader, config, max_num_of_samples: int = None):
    logging.info("\tTesting the dataloader speed.")
    total_time = 0
    start_time = time.time()
    num_of_batches = 0

    for _ in dl:
        if max_num_of_samples is not None and num_of_batches == max_num_of_samples:
            break
        num_of_batches += 1
        print(num_of_batches, f"speed {time.time() - start_time}s")
        total_time += time.time() - start_time
        start_time = time.time()

    logging.info(
        f"\tTotal time for {max(1, num_of_batches)} "
        + "batches (batch_size: {config['batch_size']}) is {total_time}s, "
        + "the average time for a batch "
        + f"(calculated over {max(1, num_of_batches)} batches) "
        + f"is {total_time / max(1, num_of_batches)}s."
    )


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="dataloader speed testing parser")

    parser.add_argument(
        "--data_dir",
        type=str,
        metavar="DT",
        dest="data_dir",
        required=True,
        help="path to directory containing *.jsonl files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        metavar="NW",
        dest="num_workers",
        required=True,
        help="number of processes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        metavar="BS",
        dest="batch_size",
        required=True,
        help="batch size",
    )
    parser.add_argument(
        "--process_batch_size",
        type=int,
        metavar="PBS",
        dest="process_batch_size",
        required=True,
        help="process batch size",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    num_workers = args.num_workers
    batch_size = args.batch_size
    process_batch_size = args.process_batch_size

    config = {
        "block_size": 2048,
        "batch_size": batch_size,
        "dataloader_pin_memory": True,
        "torch_compile": True,
        "num_workers": num_workers,
        "process_batch_size": process_batch_size,
    }

    logging.info("Params:")
    for key, value in config.items():
        logging.info(f"\t{key}: {value}")

    data_collator = default_data_collator

    data_files = glob.glob(data_dir + "/*.jsonl")

    dataset = load_dataset(
        "text",
        data_files={"data": data_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset,
        train_config=config,
        process_batch_sizes=(
            config["process_batch_size"],
            config["process_batch_size"],
        ),
    )

    args = TrainingArguments(
        output_dir="./",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        dataloader_pin_memory=config["dataloader_pin_memory"],
        torch_compile=config["torch_compile"],
        dataloader_num_workers=config["num_workers"],
        dataloader_drop_last=True,
    )

    if args.world_size > 1:
        processed_dataset["data"] = IterableDatasetShard(
            processed_dataset["data"],
            batch_size=config["process_batch_size"],
            drop_last=args.dataloader_drop_last,
            num_processes=args.world_size,
            process_index=args.process_index,
        )

    dataloader = DataLoader(
        processed_dataset["data"],
        batch_size=config["batch_size"],
        drop_last=args.dataloader_drop_last,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=args.dataloader_pin_memory,
    )

    # num_samples = 0
    # for s in processed_dataset["data"]:
    #     # print(s["input_ids"].shape)
    #     num_samples += 1

    # print(num_samples)

    test_dataloader_speed(dataloader, config, max_num_of_samples=20)
    logging.info(f"finished (took {time.time() - start_time}).")
