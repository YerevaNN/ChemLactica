from transformers import TrainingArguments
import transformers
import aim

"""
    this can be an alternative intagration to huggingface.AimCallback,
    which is not used for now, because huggingface.AimCallback serves our purpose well so far
"""


class AimTrackerCallback(transformers.TrainerCallback):
    def __init__(self, *args, **kwargs):
        super(AimTrackerCallback, self).__init__(*args, **kwargs)
        self._aim_run = aim.Run(repo=".", experiment="sample exp")

    def on_log(self, args, state, contro, model, logs=None, **kwargs):
        for name, value in logs.items():
            print(name, value)

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        print("ATTENTION! aim tracker train begin")
        _hyper_params = {
            name: value
            for name, value in zip(
                ["learning_rate", "weight_decay"],
                [args.learning_rate, args.weight_decay],
            )
        }
        print(_hyper_params)
        self._aim_run["hparams"] = _hyper_params

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("aim tracker epoch begin")

    def on_epoch_end(self, args, state, control, **kwargs):
        print("aim tracker epoch end")
