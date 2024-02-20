from accelerate import accelerator
import torch
from accelerate import optimizer
import inspect


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
