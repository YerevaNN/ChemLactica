import os
import sys
from contextlib import contextmanager
from datetime import datetime
import submitit

use_accelerate = True
rsync_enabled = True
executor_name = "slurm"  # options are ["slurm", "local"]
root_path = ""
num_gpus = 6
model_name = "galactica"
model_size = "125m"
train_type = "pretrain"
train_name = "_".join([model_name, model_size, train_type])
job_name = "gal_relform2"

slurm_params = {
    "slurm_job_name": job_name,
    "timeout_min": 60 * 24 * 2,
    "nodes": 1,
    "tasks_per_node": 1,
    "gpus_per_node": num_gpus,
    "cpus_per_task": num_gpus * 20,
    "mem_gb": num_gpus * 40.0 + 20.0,
    "stderr_to_stdout": True,
}

accelerate_config = {"num_processes": num_gpus}

env_variables = {
    "HF_HOME": "/auto/home/menuab/",
    "TOKENIZERS_PARALLELISM": "true",
    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3, 4, 5, 6, 7",
}

cli_arguments = {
    "train_type": train_type,
    "from_pretrained": "facebook/galactica-125m",
    "model_config": train_name,
    "dir_data_types": "computed",
    "training_data_dirs": "/nfs/ap/mnt/sxtn/rdkit_computed_rel+form/train_rdkit_computed_rel+form",
    "valid_data_dir": "/nfs/ap/mnt/sxtn/rdkit_computed_rel+form/valid_rdkit_computed_rel+form",
    "max_steps": 20000,
    # "num_train_epochs": 24,
    "eval_steps": 2000,
    "save_steps": 2000,
    "train_batch_size": 16,
    "valid_batch_size": 16,
    "dataloader_num_workers": 1,
    "experiment_name": job_name,
    "checkpoints_root_dir": "/nfs/dgx/raid/chem/checkpoints/",
    "flash_attn": True,
    "track": True,
    "track_dir": "/nfs/dgx/raid/chem/aim/",
    # "profile":,
    # "profile_dir":,
    "gradient_accumulation_steps": 11,
    # "gradient_checkpointing":,
    # "evaluate_only":,
    # "check_reproducability":,
}


def get_command(use_accelerate, repo_path):
    python_executable = sys.executable
    command = [python_executable]
    if use_accelerate:
        accelerate_path = f"chemlactica/config/{model_name}_accelerate_config.yaml"
        command.extend(
            f"-m accelerate.commands.launch --config_file {accelerate_path}".split(" ")
        )
        for k, v in accelerate_config.items():
            command.append(f"--{k}={v}")
    command.append(os.path.join(repo_path, "chemlactica/train.py"))
    for x, y in cli_arguments.items():
        if isinstance(y, bool):
            if y:
                command.append(f"--{x}")
        else:
            command.append(f"--{x}={y}")

    print(f'command being executed: {" ".join(command)}')
    return command


@contextmanager
def conditional_context_manager(rsync_enabled, repo_path):
    if rsync_enabled:
        with submitit.helpers.RsyncSnapshot(repo_path) as cm:
            yield cm
    else:
        yield None


def get_executor(executor_name, logs_path):
    if executor_name == "slurm":
        executor = submitit.AutoExecutor(folder=logs_path)
    elif executor_name == "local":
        executor = submitit.local.local.LocalExecutor(folder=logs_path)
    return executor


if __name__ == "__main__":
    train_name = "_".join([model_name, model_size, train_type])
    current_path = os.getcwd()
    logs_path = "submitit_logs/%j"
    logs_path = "/nfs/dgx/raid/chem/" + logs_path if rsync_enabled else logs_path
    repo_path = (
        (
            "/nfs/dgx/raid/chem/rsyncsnapshots/"
            f"{train_name}-{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
        )
        if rsync_enabled
        else current_path
    )

    with conditional_context_manager(rsync_enabled, repo_path):
        command = get_command(use_accelerate, repo_path)
        executor = get_executor(executor_name, logs_path)
        executor.update_parameters(**slurm_params)
        print("train_name: ", train_name)
        print("logs_path: ", logs_path)
        print("repo path: ", repo_path)
        function = submitit.helpers.CommandFunction(command, env=env_variables)
        job = executor.submit(function)
        # print(job.result())
