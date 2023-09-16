import time
import glob
import logging
import sys

sys.path.append("/home/tigranfahradyan/ChemLactica/ChemLactica/src")

from torch.utils.data import DataLoader
import torch
import numpy
import random
from dataset_utils import process_dataset, samples_generator, JsonlDataset
from transformers import TrainingArguments
from transformers.data.data_collator import default_data_collator
from datasets.iterable_dataset import IterableDataset
from datasets.dataset_dict import IterableDatasetDict
from utils import CustomTokenizer
from datasets import load_dataset, IterableDataset
from datasets.download.streaming_download_manager import FilesIterable

logging.basicConfig(
    filename="dataloader_benchmark.log",
    level=logging.INFO,
    format="%(levelname)s: " "%(message)s",
)

torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)

if __name__ == "__main__":
    start_time = time.time()

    # training_data_files = glob.glob("/home/tigranfahradyan/single_data/train" + "/*.jsonl")
    # valid_data_files = glob.glob("/home/tigranfahradyan/single_data/valid" + "/*.jsonl")
    training_data_files = glob.glob("/home/tigranfahradyan/ChemLactica/ChemLactica/.small_data/train" + "/*.jsonl")
    validation_data_files = glob.glob("/home/tigranfahradyan/ChemLactica/ChemLactica/.small_data/valid" + "/*.jsonl")
    print(training_data_files)
    # dataset = load_dataset(
    #     "text",
    #     data_files={"train": training_data_files, "validation": valid_data_files},
    #     streaming=True
    # )
    train_jsonl_datasets = {_file: JsonlDataset(_file) for _file in training_data_files}
    valid_jsonl_datasets = {_file: JsonlDataset(_file) for _file in validation_data_files}
    dataset = IterableDatasetDict({
        "train": IterableDataset.from_generator(samples_generator(train_jsonl_datasets)),
        "validation": IterableDataset.from_generator(samples_generator(valid_jsonl_datasets))
    })

    # dataset = process_dataset(
    #     dataset=dataset, train_config={"block_size": 2048}, process_batch_sizes=(50, 50)
    # )

    CustomTokenizer.set_model_size("125m")

    # for s in samples_generator(train_jsonl_datasets)():
    #     print(s)

    # _samples = []
    samples = 0
    for s in dataset["train"]:
        samples += 1
        print(s)
        print(samples)
        # _samples.append(s)
        # if len(_samples) == 2:
        #     print(_samples[0] == _samples[1])
        #     _samples.clear()
    
    logging.info(f"finished (took {time.time() - start_time}).")


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


# from multiprocessing import Process, Queue
# def writer(i,q):
#     message = f'I am Process {i}'
#     q.put(message)
# if __name__ ==  '__main__':
#     # Create multiprocessing queue
#     q = Queue()
    
#     # Create a group of parallel writers and start them
#     for i in range(10):
#         Process(target=writer, args=(i,q,)).start()
#     # Read the queue sequentially
#     for i in range(10):
#         message = q.get()
#         print(message)