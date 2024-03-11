import yaml
import os

absolute_path = os.path.dirname(os.path.abspath(__file__))
relative_path = "models_train_config.yaml"
full_path = os.path.join(absolute_path, relative_path)


# read static fine-tine config
with open(os.path.join(absolute_path, "models_fine-tune_config.yaml"), "r") as f_:
    model_fine_tune_configs = yaml.full_load(f_)

model_fine_tune_configs["125m"]["max_learning_rate"] = 1e-5
model_fine_tune_configs["125m"]["adam_beta1"] = 0.9
model_fine_tune_configs["125m"]["adam_beta2"] = 0.95
model_fine_tune_configs["125m"]["warmup_steps"] = 0