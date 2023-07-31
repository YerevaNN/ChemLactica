from transformers import Trainer
from typing import Optional


class CustomTrainer(Trainer):
    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        self.model = self.model.reverse_bettertransformer()
        super().save_model(output_dir)
        self.model = self.model.to_bettertransformer()
