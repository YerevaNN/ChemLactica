import glob

from datasets import load_dataset, interleave_datasets
from datasets.iterable_dataset import IterableDataset

from chemlactica.utils.dataset_utils import process_dataset, DIR_DATA_TYPES
from chemlactica.jsonl_dataset import samples_generator


def get_dataset(
    train_type,
    training_data_dirs,
    valid_data_dir,
    dir_data_types,
    train_config,
    model_config,
    shared_jsonl_files,
    evaluate_only,
    slurm_eval,
    shuffle_buffer_size,
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
                dataset=eval_dataset,
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
        dataset = load_dataset(training_data_dirs[0])

    return dataset
