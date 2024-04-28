import os
from chemlactica.utils.utils import get_tokenizer
from chemlactica.utils.dataset_utils import process_dataset
import tqdm
from chemlactica.utils.utils import get_model_train_config
import multiprocessing
import glob
from datetime import timedelta
import torch
import signal
import random
import numpy
from accelerate import Accelerator, logging, InitProcessGroupKwargs
from datasets.iterable_dataset import IterableDataset

# from datasets import load_dataset
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
    directory_name = "/mnt/sxtn/phil/vsmallfilewithindices"
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
        # dataset = load_dataset(
        #     "text",
        #     data_files={"arbitrary": training_data_files},
        #     streaming=False,
        # )
        is_assay_split = False
        model_config, train_config = get_model_train_config("galactica_125m_pretrain")

        dataset = process_dataset(
            dataset=dataset,
            train_config=train_config,
            model_config=model_config,
            process_batch_sizes=(50, 50),
            is_eval=False,
            assay=is_assay_split,
        )
        counter = 0
        for sample in tqdm.tqdm(dataset):
            if counter == 3444672:
                print(f"step {counter}----------------------------")
                print(shared_jsonl_files)
            if counter > 5000000:
                print(f"reached step {counter}----------------------------")
                print(shared_jsonl_files)
                break
    print(len(sample["input_ids"]))
    tokenizer = get_tokenizer(model_config.tokenizer_path)
    tokenizer.decode(sample["input_ids"])
    print("done")


if __name__ == "__main__":
    run()
