import os
import multiprocessing
import glob
import shutil
from datetime import timedelta
import torch
import signal
import random
import numpy
from accelerate import Accelerator, logging, InitProcessGroupKwargs
import hashlib
from datasets.iterable_dataset import IterableDataset
from chemlactica.jsonl_dataset import samples_generator
from chemlactica.utils.utils import (
    signal_handler,
)

from accelerate.state import PartialState

distributed_state = PartialState()
torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)
logger = logging.get_logger("transformers")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run():
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        kwargs_handlers=[kwargs], log_with="all", project_dir=None
    )

    directory_name = ".tmp"
    if distributed_state.is_main_process:
        if os.path.exists(directory_name):
            print(f"test directory '{directory_name}' already exists. Clearing it now")
            shutil.rmtree(directory_name)
        os.makedirs(directory_name)
        print(f"test directory '{directory_name}' created successfully.")
    num_files = 5
    num_lines = 100

    for i in range(num_files):
        file_name = os.path.join(directory_name, f"test_file_{i}.jsonl")
        with open(file_name, "w") as file:
            for j in range(num_lines):
                sha3_hash = hashlib.sha3_256(
                    str.encode(f"test_file_{i}.jsonl - Line {j}")
                ).hexdigest()
                file.write(f"{sha3_hash}\n")
    accelerator.wait_for_everyone()
    with multiprocessing.Manager() as manager:
        shared_jsonl_files = manager.dict()

        training_data_files = glob.glob(directory_name + "/*.jsonl")
        training_data_files = [os.path.abspath(path) for path in training_data_files]

        print(training_data_files)

        accelerator.wait_for_everyone()
        dataset = IterableDataset.from_generator(
            samples_generator,
            gen_kwargs={
                "files": training_data_files,
                "shared_jsonl_files": shared_jsonl_files,
            },
        )

        file_name_mapping = {}
        for process_index in range(distributed_state.num_processes):
            file_name_mapping[process_index] = f"dataload_proc{process_index}.jsonl"

        for example in dataset:
            file_to_write = file_name_mapping[distributed_state.process_index]
            with open(file_to_write, "a") as f:
                f.write(example["text"] + "\n")
    accelerator.wait_for_everyone()
    file_line_sets = []

    # for process_index in range(distributed_state.num_processes):
    if distributed_state.is_main_process:
        for process_index, file_to_check in file_name_mapping.items():
            file_lines = load_file_contents(file_to_check)
            file_line_set = set(file_lines)
            file_line_sets.append(file_line_set)
            print(f"file line set length {len(file_line_set)}")
            print(f"file line length {len(file_lines)}")
            assert len(file_lines) == len(file_line_set)

    num_sets = len(file_line_sets)
    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            set1 = file_line_sets[i]
            set2 = file_line_sets[j]
            assert set1.isdisjoint(set2)

    accelerator.wait_for_everyone()
    if distributed_state.is_main_process:
        for process_index in file_name_mapping:
            file_to_check = file_name_mapping[process_index]
            os.remove(file_to_check)


def load_file_contents(file):
    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


if __name__ == "__main__":
    run()
