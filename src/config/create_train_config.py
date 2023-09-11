import yaml
import os

absolute_path = os.path.dirname(os.path.abspath(__file__))
relative_path = "models_train_config.yaml"
full_path = os.path.join(absolute_path, relative_path)

# read static train config
with open(full_path, "r") as f_:
    model_train_configs = yaml.full_load(f_)

for key in model_train_configs.keys():
    model_train_configs[key]["max_learning_rate"] *= 0.08

model_train_configs["1.3b"]["warmup_steps"] = 2000
model_train_configs["1.3b"]["max_learning_rate"] = 1.0e-5
model_train_configs["1.3b"]["global_gradient_norm"] = 0.1
