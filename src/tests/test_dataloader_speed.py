import time
import argparse
import glob

from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset_utils import process_dataset
from utils import CustomTokenizer
from transformers import AutoTokenizer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_pt_utils import IterableDatasetShard


def test_dataloader_speed(dl: DataLoader, config, max_num_of_samples: int = None):
    print("Testing the dataloader speed.")
    total_time = 0
    start_time = time.time()
    num_of_samples = 0

    for _ in dl:
        if max_num_of_samples is not None and num_of_samples == max_num_of_samples:
            break
        num_of_samples += 1
        total_time += time.time() - start_time
        start_time = time.time()

    print(
        f"Total time for {num_of_samples} is {total_time}s, "
        + "the average time for a batch "
        + f"(calculated over {num_of_samples} samples) "
        + f"is {total_time / num_of_samples}s."
    )


if __name__ == "__main__":
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

    args = parser.parse_args()
    data_dir = args.data_dir
    num_workers = args.num_workers
    batch_size = args.batch_size

    config = {
        "block_size": 2048,
        "batch_size": batch_size,
        "dataloader_pin_memory": True,
        "torch_compile": True,
        "num_workers": num_workers,
    }

    tokenizer = CustomTokenizer(
        instance=AutoTokenizer.from_pretrained("facebook/galactica-125m")
    ).get_instance()
    data_collator = DataCollatorWithPadding(tokenizer)

    data_files = glob.glob(data_dir + "/*.jsonl")

    dataset = load_dataset(
        "text",
        data_files={"data": data_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset, train_config=config, process_batch_sizes=(10000, 10000)
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

    processed_dataset["data"] = IterableDatasetShard(
        processed_dataset["data"],
        batch_size=config["batch_size"],
        drop_last=args.dataloader_drop_last,
        num_processes=args.world_size,
        process_index=args.process_index,
    )

    dataloader = DataLoader(
        processed_dataset["data"],
        batch_size=config["batch_size"],
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=args.dataloader_pin_memory,
    )

    test_dataloader_speed(dataloader, config, max_num_of_samples=10000)
