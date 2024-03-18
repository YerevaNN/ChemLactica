import os
import sys
import yaml
from datetime import datetime
import submitit

use_accelerate = True
num_gpus = 2
model_name = "galactica"
model_size = "125m"
train_type = "sft"

slurm_params = {
    "slurm_job_name": "submitit_test",
    "timeout_min": 60,
    "nodes": 1,
    "tasks_per_node": 1,
    "gpus_per_node": num_gpus,
    "cpus_per_task": num_gpus * 8,
    "mem_gb": num_gpus * 25.0 + 20.0,
    "stderr_to_stdout": True,
}

accelerate_config = {"num_processes": num_gpus}

env_variables = {
    "TOKENIZERS_PARALLELISM": "true",
    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3, 4, 5, 6, 7",
}

cli_arguments = {
    "train_type": train_type,
    "from_pretrained": "facebook/galactica-125m",
    "model_config": "125m",
    "dir_data_types": "computed",
    "training_data_dirs": "/auto/home/menuab/code/sft_data/ADME_HLM",
    "valid_data_dir": "",
    # "max_steps":120000,
    "num_train_epochs": 2,
    "eval_steps": 2048,
    "save_steps": 2048,
    "train_batch_size": 16,
    # "valid_batch_size":,
    "dataloader_num_workers": 30,
    "experiment_name": "testing_submitit",
    "checkpoints_root_dir": "/raid/chem/checkpoints/",
    "flash_attn": False,
    "track": True,
    "track_dir": "/raid/chem/aim/",
    # "profile":,
    # "profile_dir":,
    # "gradient_accumulation_steps":,
    # "gradient_checkpointing":,
    # "evaluate_only":,
    # "check_reproducability":,
}


def get_accelerate_config_file(root_path):
    relative_path = "chemlactica/config/accelerate_config.yaml"
    defaults_full_path = os.path.join(root_path, relative_path)
    custom_relative_path = "chemlactica/config/accelerate_config_custom.yaml"
    custom_full_path = os.path.join(root_path, custom_relative_path)

    with open(defaults_full_path, "r") as infile:
        accelerate_config_defaults = yaml.full_load(infile)

    for k, v in accelerate_config.items():
        accelerate_config_defaults[k] = v

    with open(custom_full_path, "w") as outfile:
        yaml.dump(accelerate_config_defaults, outfile, default_flow_style=False)

    return custom_full_path


if __name__ == "__main__":
    snapshot_path = (
        "/raid/chem/rsyncsnapshots/"
        f"{model_name}-{model_size}-{train_type}-{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    )
    print("snapshot path: ", snapshot_path)

    with submitit.helpers.RsyncSnapshot(snapshot_dir=snapshot_path):
        python_executable = sys.executable
        command = [python_executable]

        if use_accelerate:
            accelerate_path = get_accelerate_config_file(snapshot_path)
            command.extend(
                f"-m accelerate.commands.launch --config_file {accelerate_path}".split(
                    " "
                )
            )

        command.append("chemlactica/train.py")
        for x, y in cli_arguments.items():
            if isinstance(y, bool):
                if y:
                    command.append(f"--{x}")
            else:
                command.append(f"--{x}={y}")
        print(f'command being executed: {" ".join(command)}')

        executor = submitit.AutoExecutor(folder="log_test/%j")
        executor.update_parameters(**slurm_params)
        function = submitit.helpers.CommandFunction(command, env=env_variables)
        job = executor.submit(function)
        print(job.result())
