from transformers import TrainingArguments
import aim

aim_run = aim.Run(repo=".", experiment="sample expermient")


class AimTrackerCallback(TrainerCallback):
    def __init__(self, aim_run, *args, **kwargs):
        super(AimTrackerCallback, self).__init__(*args, **kwargs)
        self._aim_run = aim_run

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        print("aim tracker train begin")
        _hyper_params = {
            name: value
            for name, value in zip(
                [
                    "leadning_rate",
                    "batch_szie",
                    "weight_decay",
                    "max_steps",
                    "evaluation_strategy",
                ],
                [
                    args.leadning_rate,
                    args.batch_size,
                    args.weight_decay,
                    args.max_steps,
                ],
            )
        }
        self._aim_run["hyper_params"] = _hyper_params

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("aim tracker epoch begin")

    def on_epoch_end(self, args, state, control, **kwargs):
        print("aim tracker epoch end")
