import unittest
import gc
import glob
import multiprocessing

import torch
from torch.utils.data import DataLoader
from datasets.iterable_dataset import IterableDataset
from datasets.dataset_dict import IterableDatasetDict
from transformers.trainer_utils import seed_worker

from jsonl_dataset import samples_generator


class TestDataloader(unittest.TestCase):

    def setUp(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

    def get_train_dataloader(
            self, train_dataset, batch_size,
            num_workers, pin_memory
        ) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_params = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = False
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)

    def test_dataloader(self):
        """
            The following code replicates the resumed training

            The test works the following way
            1. We return line information along with each read line by the dataset
            and check if the read line corresponds to the line in the file
            2. We do 2 stages, the first one is a dataset starting from scratch
            and the second one a dataset (fast) resumed
        """
        training_data_dir = ".small_data/train"
        valid_data_dir = ".small_data/valid"

        with multiprocessing.Manager() as manager:
            shared_jsonl_files = manager.dict()
            training_data_files = glob.glob(training_data_dir + "/*.jsonl")
            # combine small train and valid to have 2 files to test
            training_data_files.extend(glob.glob(valid_data_dir + "/*.jsonl"))

            initial_train_dataset = IterableDatasetDict({
                "train": IterableDataset.from_generator(
                    samples_generator,
                    gen_kwargs={
                        "files": training_data_files,
                        "shared_jsonl_files": shared_jsonl_files,
                        "return_line_info": True,
                    }
                )
            })

            """
                get_train_dataloader is a function similar to (Trainer.get_train_dataloader)
                what we do in training (we exclude accelerate.prepare operation, because that is for distributed training)
                Use get_train_dataloader to replicate our dataloader during the training
                and use and test our dataset with multiprocessing
            """
            initial_train_dataloader = self.get_train_dataloader(
                initial_train_dataset["train"], 16,
                num_workers=2, pin_memory=True
            )

            """
                keep all the lines in the memory to check if the read line by dataset matches the actual line in the file
                (this is kind of the "ground truth")
            """
            loaded_files = {}
            for file in training_data_files:
                with open(file, "r") as _f:
                    loaded_files[file] = [{
                            "text": line.rstrip("\n"),
                            "is_read": False
                        } for line in _f.readlines()]

            sample_to_pass = 10
            for i, samples in enumerate(initial_train_dataloader):
                for text, file, line_number in zip(
                                        samples["text"],
                                        samples["line_info"]["file"],
                                        samples["line_info"]["line_number"]
                                    ):
                    # check if the line matches with what is actually in the file
                    assert loaded_files[file][line_number - 1]["text"] == text
                    assert not loaded_files[file][line_number - 1]["is_read"]
                    loaded_files[file][line_number - 1]["is_read"] = True
                    print(f'{file} {line_number} passed')
                if i == sample_to_pass:
                    break

            resumed_train_dataset = IterableDatasetDict({
                "train": IterableDataset.from_generator(
                    samples_generator,
                    gen_kwargs={
                        "files": training_data_files,
                        "shared_jsonl_files": shared_jsonl_files,
                        "return_line_info": True,
                    }
                )
            })
            resumed_train_dataloader = self.get_train_dataloader(
                resumed_train_dataset["train"], 16,
                num_workers=2, pin_memory=True
            )

            for samples in resumed_train_dataloader:
                for text, file, line_number in zip(
                                        samples["text"],
                                        samples["line_info"]["file"],
                                        samples["line_info"]["line_number"]
                                    ):
                    # check if the line matches with what is actually in the file
                    assert loaded_files[file][line_number - 1]["text"] == text
                    assert not loaded_files[file][line_number - 1]["is_read"]
                    loaded_files[file][line_number - 1]["is_read"] = True
                    print(f'{file} {line_number} passed')

            for file_name, lines in loaded_files.items():
                for i, line in enumerate(lines, start=1):
                    assert line["is_read"], f"'{file_name}' line {i} is not read."
            print("All lines are read axactly once.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
