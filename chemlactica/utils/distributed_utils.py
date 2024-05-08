import os
from typing import Callable, List, Optional, Union
from accelerate.state import (
    AcceleratorState,
    DistributedType,
)
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from accelerate.data_loader import (
    DataLoaderShard,
    DataLoaderDispatcher,
    SeedableRandomSampler,
    BatchSamplerShard,
    # IterableDatasetShard,
    # SkipBatchSampler,
    # SkipDataLoader,
    # DataLoaderStateMixin,
)

import torch

from accelerate.logging import get_logger

from accelerate.utils import (
    RNGType,
    is_torch_version,
)


logger = get_logger(__name__)

# kwargs of the DataLoader in min version 1.4.0.
_PYTORCH_DATALOADER_KWARGS = {
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
    "generator": None,
    "prefetch_factor": 2,
    "persistent_workers": False,
}

# kwargs added after by version
_PYTORCH_DATALOADER_ADDITIONAL_KWARGS = {}

for v, additional_kwargs in _PYTORCH_DATALOADER_ADDITIONAL_KWARGS.items():
    if is_torch_version(">=", v):
        _PYTORCH_DATALOADER_KWARGS.update(additional_kwargs)


def get_experiment_hash(from_pretrained, train_type="pretrain"):
    if os.path.isdir(from_pretrained) and train_type == "pretrain":
        return str(from_pretrained.split(os.path.sep)[-2])
    else:
        return "none"


def custom_prepare_data_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    split_batches: bool = False,
    put_on_device: bool = False,
    rng_types: Optional[List[Union[str, RNGType]]] = None,
    dispatch_batches: Optional[bool] = None,
    even_batches: bool = True,
    slice_fn_for_dispatch: Optional[Callable] = None,
    use_seedable_sampler: bool = False,
) -> DataLoader:
    if dispatch_batches is None:
        if not put_on_device:
            dispatch_batches = False
        else:
            dispatch_batches = isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")
    # Grab defaults from AcceleratorState
    state = AcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    # Sanity check
    if split_batches:
        if dataloader.batch_size is not None:
            batch_size_for_check = dataloader.batch_size
        else:
            # For custom batch_sampler
            if hasattr(dataloader.batch_sampler, "batch_size"):
                batch_size_for_check = dataloader.batch_sampler.batch_size
            else:
                raise ValueError(
                    "In order to use `split_batches==True` you must pass `batch_size`"
                    "to `dataloader` or `dataloader.batch_sampler` objects"
                    "and it has to return a natural number. "
                    "Your `dataloader.batch_size` is None and"
                    "`dataloader.batch_sampler` "
                    f"(`{type(dataloader.batch_sampler)}`) does not have"
                    "the `batch_size` attribute set."
                )

        if batch_size_for_check > 1 and batch_size_for_check % num_processes != 0:
            raise ValueError(
                f"To use a `DataLoader` in `split_batches` mode:"
                "the batch size ({dataloader.batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = (
        dataloader.batch_sampler
        if not isinstance(new_dataset, IterableDataset)
        else None
    )
    sampler_is_batch_sampler = False
    synchronized_generator = None
    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    if sampler_is_batch_sampler:
        sampler = getattr(dataloader.sampler, "sampler", None)
    else:
        sampler = getattr(dataloader.batch_sampler, "sampler", None)
    if isinstance(sampler, RandomSampler) and use_seedable_sampler:
        # When iterating through the dataloader during distributed processes
        # we want to ensure that on each process we are iterating through the same
        # samples in the same order if a seed is set. This requires a tweak
        # to the `torch.utils.data.RandomSampler` class (if used).
        sampler = SeedableRandomSampler(
            data_source=sampler.data_source,
            replacement=sampler.replacement,
            num_samples=sampler._num_samples,
            generator=getattr(sampler, "generator", torch.Generator()),
        )

    # No change if no multiprocess
    if (
        num_processes != 1 or state.distributed_type == DistributedType.MEGATRON_LM
    ) and not dispatch_batches:
        if isinstance(new_dataset, IterableDataset):
            # if getattr(dataloader.dataset, "generator", None) is not None:
            #     synchronized_generator = dataloader.dataset.generator
            # new_dataset = IterableDatasetShard(
            #     new_dataset,
            #     batch_size=dataloader.batch_size,
            #     drop_last=dataloader.drop_last,
            #     num_processes=num_processes,
            #     process_index=process_index,
            #     split_batches=split_batches,
            # )
            pass
        else:
            batch_sampler = (
                dataloader.sampler
                if sampler_is_batch_sampler
                else dataloader.batch_sampler
            )
            new_batch_sampler = BatchSamplerShard(
                batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
                even_batches=even_batches,
            )

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    if (
        rng_types is not None
        and synchronized_generator is None
        and "generator" in rng_types
    ):
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = (
            dataloader.batch_size // num_processes
            if split_batches and not dispatch_batches
            else dataloader.batch_size
        )
    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = DataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            slice_fn=slice_fn_for_dispatch,
            **kwargs,
        )
    elif sampler_is_batch_sampler:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device
            if put_on_device and state.distributed_type != DistributedType.XLA
            else None,
            sampler=new_batch_sampler,
            batch_size=dataloader.batch_size,
            rng_types=rng_types,
            _drop_last=dataloader.drop_last,
            synchronized_generator=synchronized_generator,
            **kwargs,
        )
    else:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device
            if put_on_device and state.distributed_type != DistributedType.XLA
            else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            _drop_last=dataloader.drop_last,
            **kwargs,
        )

    if isinstance(sampler, SeedableRandomSampler) and use_seedable_sampler:
        if sampler_is_batch_sampler:
            dataloader.sampler.sampler = sampler
        else:
            dataloader.batch_sampler.sampler = sampler

    return dataloader
