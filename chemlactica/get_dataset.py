import glob

import deepchem as dc
import polaris as po
from sklearn.model_selection import train_test_split


from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict
from datasets.iterable_dataset import IterableDataset

from chemlactica.utils.dataset_utils import (
    process_dataset,
    DIR_DATA_TYPES,
    make_canonical,
)
from chemlactica.jsonl_dataset import samples_generator


def get_dataset(
    train_type,
    training_args,
    training_data_dirs,
    valid_data_dir,
    dir_data_types,
    train_config,
    model_config,
    shared_jsonl_files,
    evaluate_only,
    slurm_eval,
    shuffle_buffer_size,
    random_seed,
):
    if train_type == "pretrain":
        assert len(training_data_dirs) == len(dir_data_types)
        train_dataset_dict = {}
        print("---Training dataset names---")
        for i, (training_data_dir, dir_data_type) in enumerate(
            zip(training_data_dirs, dir_data_types)
        ):
            if dir_data_type.lower() not in DIR_DATA_TYPES:
                raise ValueError(
                    f"""Unknown data type {dir_data_type},
                    the following data types are supported: {DIR_DATA_TYPES}"""
                )
            training_data_files = glob.glob(training_data_dir + "/*.jsonl")
            ds_name = f"{dir_data_type}_{i}"
            is_assay_split = "assay" in dir_data_type
            dataset = IterableDataset.from_generator(
                samples_generator,
                gen_kwargs={
                    "files": training_data_files,
                    "shared_jsonl_files": shared_jsonl_files,
                },
            )
            dataset = process_dataset(
                dataset=dataset,
                train_config=train_config,
                model_config=model_config,
                process_batch_sizes=(50, 50),
                is_eval=False,
                assay=is_assay_split,
            )
            if is_assay_split:
                dataset.shuffle(buffer_size=shuffle_buffer_size)
            print(f"Dataset {i}: {ds_name}")
            train_dataset_dict[ds_name] = dataset

        valid_data_files = glob.glob(valid_data_dir + "/*.jsonl")

        train_dataset = list(train_dataset_dict.values())
        if len(train_dataset) > 1:
            train_dataset = interleave_datasets(train_dataset)
        else:
            train_dataset = train_dataset[0]

        if evaluate_only or not slurm_eval:
            eval_dataset = load_dataset(
                "text", data_files={"validation": valid_data_files}, streaming=False
            )
            processed_eval_dataset = process_dataset(
                dataset=eval_dataset["validation"],
                train_config=train_config,
                model_config=model_config,
                process_batch_sizes=(50, 50),
                is_eval=True,
                assay=False,
            )
        else:
            processed_eval_dataset = None
        dataset = {"train": train_dataset, "validation": processed_eval_dataset}

    elif train_type == "sft":
        benchmark = None
        # dataset = load_dataset(training_data_dirs[0])
        # dataset = load_dataset(training_data_dirs[0]).shuffle(seed=random_seed)

        polaris_benchmarks = {
            "HLM": "polaris/adme-fang-HCLint-1",
            "RLM": "polaris/adme-fang-RCLint-1",
            "SOL": "polaris/adme-fang-SOLU-1",
            "RPPB": "polaris/adme-fang-RPPB-1",
            "HPPB": "polaris/adme-fang-HPPB-1",
            "MDR": "polaris/adme-fang-PERM-1",
        }

        if training_data_dirs[0].split("/")[-1] in ["freesolv", "delaney", "lipo"]:
            _, datasets, _ = getattr(
                dc.molnet, f"load_{training_data_dirs[0].split('/')[-1]}"
            )(featurizer="raw", splitter="random")
            train, valid, test = datasets
            dataset = DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {"smiles": train.ids, "activity": train.y.reshape(-1)}
                    ),
                    "validation": Dataset.from_dict(
                        {"smiles": valid.ids, "activity": valid.y.reshape(-1)}
                    ),
                    "test": Dataset.from_dict(
                        {"smiles": test.ids, "activity": test.y.reshape(-1)}
                    ),
                }
            )
            dataset = dataset.map(make_canonical)
            dataset = dataset.shuffle(seed=random_seed)
            print("data loaded by deepchem 1")

        elif training_data_dirs[0].split("/")[-1] == "bbbp":
            _, datasets, _ = dc.molnet.load_bbbp(featurizer="raw", splitter="scaffold")
            train, valid, test = datasets
            dataset = DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {"smiles": train.ids, "activity": train.y.reshape(-1)}
                    ),
                    "validation": Dataset.from_dict(
                        {"smiles": valid.ids, "activity": valid.y.reshape(-1)}
                    ),
                    "test": Dataset.from_dict(
                        {"smiles": test.ids, "activity": test.y.reshape(-1)}
                    ),
                }
            )
            dataset = dataset.map(make_canonical)
            dataset = dataset.shuffle(seed=random_seed)
            print("data loaded by deepchem 2")

        elif training_data_dirs[0].split("/")[-2] in [
            "RLM",
            "HLM",
            "MDR",
            "SOL",
            "HPPB",
            "RPPB",
        ]:
            benchmark = po.load_benchmark(
                polaris_benchmarks[training_data_dirs[0].split("/")[-2]]
            )
            train, test = benchmark.get_train_test_split()
            X_train, X_valid, y_train, y_valid = train_test_split(
                train.X, train.y, test_size=0.1, random_state=42
            )
            if training_data_dirs[0].split("/")[-1] == "full":
                dataset = DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {"smiles": train.X, "activity": train.y}
                        ),
                        "validation": Dataset.from_dict(
                            {"smiles": X_valid, "activity": y_valid}
                        ),
                        "test": Dataset.from_dict({"smiles": test.X}),
                    }
                )
            else:
                dataset = DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {"smiles": X_train, "activity": y_train}
                        ),
                        "validation": Dataset.from_dict(
                            {"smiles": X_valid, "activity": y_valid}
                        ),
                        "test": Dataset.from_dict({"smiles": test.X}),
                    }
                )
            dataset["train"] = dataset["train"].shuffle(seed=random_seed)
            # dataset = dataset.shuffle(seed=random_seed)
            dataset = dataset.map(make_canonical)
            print("data loaded by polaris")

        else:
            dataset = load_dataset(training_data_dirs[0])

        training_args.per_device_eval_batch_size = min(
            training_args.per_device_eval_batch_size, dataset["validation"].num_rows
        )
        steps_per_epoch = (
            dataset["train"].num_rows // training_args.per_device_train_batch_size
        )
        training_args.warmup_steps *= steps_per_epoch
        training_args.eval_steps *= steps_per_epoch
        # print(f"{steps_per_epoch=},{training_args.per_device_train_batch_size},{dataset['train'].num_rows}")
        print(dataset)

    return dataset, benchmark
