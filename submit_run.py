import sys
from contextlib import nullcontext
from datetime import datetime
import submitit

use_accelerate = True
rsync_enabled = False
executor_name = "slurm"  # options are ["slurm", "local"]
root_path = ""
num_gpus = 2
model_name = "galactica"
model_size = "125m"
train_type = "sft"

slurm_params = {
    "slurm_job_name": "submitit_test",
    "timeout_min": 10,
    "nodes": 1,
    "tasks_per_node": 1,
    "gpus_per_node": num_gpus,
    "cpus_per_task": num_gpus * 8,
    "mem_gb": num_gpus * 15.0 + 20.0,
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
    "model_config": "galactica_125m_sft",
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
    "checkpoints_root_dir": "/nfs/dgx/raid/chem/checkpoints/",
    "flash_attn": False,
    "track": True,
    "track_dir": "/nfs/dgx/raid/chem/aim/",
    # "profile":,
    # "profile_dir":,
    # "gradient_accumulation_steps":,
    # "gradient_checkpointing":,
    # "evaluate_only":,
    # "check_reproducability":,
}


def get_command(use_accelerate):
    python_executable = sys.executable
    command = [python_executable]
    if use_accelerate:
        accelerate_path = "chemlactica/config/accelerate_config.yaml"
        command.extend(
            f"-m accelerate.commands.launch --config_file {accelerate_path}".split(" ")
        )
        for k, v in accelerate_config.items():
            command.append(f"--{k}={v}")
    command.append("chemlactica/train.py")
    for x, y in cli_arguments.items():
        if isinstance(y, bool):
            if y:
                command.append(f"--{x}")
        else:
            command.append(f"--{x}={y}")

    print(f'command being executed: {" ".join(command)}')
    return command


def get_conditional_context_manager(rsync_enabled, snapshot_path):
    if rsync_enabled:
        yield submitit.helpers.RsyncSnapshot(snapshot_path)
    else:
        yield nullcontext()


def get_executor(executor_name, logs_path):
    if executor_name == "slurm":
        executor = submitit.AutoExecutor(folder=logs_path)
    elif executor_name == "local":
        executor = submitit.local.local.LocalExecutor(folder=logs_path)
    return executor


if __name__ == "__main__":
    train_name = "_".join([model_name, model_size, train_type])
    logs_path = "/nfs/dgx/raid/chem/submitit_logs/%j"
    snapshot_path = (
        "/nfs/dgx/raid/chem/rsyncsnapshots/"
        f"{train_name}-{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    )
    print("train_name: ", train_name)
    print("logs_path: ", logs_path)
    print("snapshot path: ", snapshot_path)

    context_manager = get_conditional_context_manager(
        rsync_enabled=rsync_enabled, path=snapshot_path
    )
    with context_manager:
        command = get_command(snapshot_path, use_accelerate)
        executor = get_executor(executor_name, logs_path)
        executor.update_parameters(**slurm_params)
        function = submitit.helpers.CommandFunction(command, env=env_variables)
        job = executor.submit(function)
        print(job.result())
