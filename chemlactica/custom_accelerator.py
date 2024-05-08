from accelerate.state import (
    DistributedType,
)
import torch
from accelerate import optimizer, accelerator
import inspect
from chemlactica.utils.distributed_utils import custom_prepare_data_loader


class CustomAcceleratedOptimizer(optimizer.AcceleratedOptimizer):
    def zero_grad(self, set_to_none=None):
        if self.gradient_state.sync_gradients:
            accept_arg = (
                "set_to_none" in inspect.signature(self.optimizer.zero_grad).parameters
            )
            if accept_arg:
                if set_to_none is None:
                    set_to_none = False
                else:
                    set_to_none = True
                self.optimizer.zero_grad(set_to_none=set_to_none)
            else:
                if set_to_none is not None:
                    raise ValueError(
                        "`set_to_none` for Optimizer.zero_grad` is not supported by this optimizer."
                    )
                self.optimizer.zero_grad()


class CustomAccelerator(accelerator.Accelerator):
    def prepare_optimizer(
        self, optimizer: torch.optim.Optimizer, device_placement=None
    ):
        if getattr(optimizer, "_is_accelerate_prepared", False):
            if optimizer not in self._optimizers:
                self._optimizers.append(optimizer)
            return optimizer
        if device_placement is None:
            device_placement = self.device_placement
        optimizer = CustomAcceleratedOptimizer(
            optimizer, device_placement=device_placement, scaler=self.scaler
        )
        self._optimizers.append(optimizer)
        return optimizer

    def prepare_data_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        device_placement=None,
        slice_fn_for_dispatch=None,
    ):
        # Ensure we can't double wrap a DataLoader due to `find_batch_size`
        if getattr(data_loader, "_is_accelerate_prepared", False):
            if data_loader not in self._dataloaders:
                self._dataloaders.append(data_loader)
            return data_loader
        if device_placement is None:
            device_placement = (
                self.device_placement
                if self.distributed_type != DistributedType.XLA
                else False
            )
        prepared_data_loader = custom_prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
            even_batches=self.even_batches,
            slice_fn_for_dispatch=slice_fn_for_dispatch,
            use_seedable_sampler=self.use_seedable_sampler,
        )
        self._dataloaders.append(prepared_data_loader)
        return prepared_data_loader
