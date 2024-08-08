import sys
from contextlib import contextmanager
from datetime import datetime
import submitit
from datasets import load_dataset

use_accelerate = False
rsync_enabled = False
executor_name = "slurm"  # options are ["slurm", "local"]
root_path = ""
num_gpus = 1
task_name = "freesolv"
training_data_dirs = f"gayane/{task_name}"
model_name = "chemlactica-125m"
train_type = "sft"
job_name = f"{task_name}_{model_name}"
batch_size = 32
n_steps = int(
    load_dataset(f"{training_data_dirs}", split="train").num_rows / batch_size
)


checkpoints = {
    "chemlactica-125m": "/nfs/dgx/raid/chem/checkpoints/facebook/"
    "galactica-125m/1f289ff103034364bd27e1c3/checkpoint-18000/",
    "chemlactica-1b": "/nfs/dgx/raid/chem/checkpoints/h100/facebook/"
    "galactica-1.3b/6d68b252d53647a99cf2fa8b/checkpoint-19000",
    "chemma-2b-12k": "/nfs/dgx/raid/chem/checkpoints/h100/"
    "google/gemma-2b/0717d445bcf44e31b2887892/checkpoint-12000",
    "chemma-2b": "/nfs/dgx/raid/chem/checkpoints/h100/"
    "google/gemma-2b/0717d445bcf44e31b2887892/checkpoint-18000",
}

slurm_params = {
    "slurm_job_name": job_name,
    "timeout_min": 60 * 3,
    "nodes": 1,
    "tasks_per_node": 1,
    "gpus_per_node": num_gpus,
    "cpus_per_task": num_gpus * 2,
    "mem_gb": num_gpus * 40.0 + 20.0,
    "stderr_to_stdout": True,
}

accelerate_config = {"num_processes": num_gpus}

env_variables = {
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3, 4, 5, 6, 7",
    # "CUDA_VISIBLE_DEVICES": "1",
}

cli_arguments = {
    "train_type": train_type,
    "from_pretrained": checkpoints[model_name],
    "model_config": model_name + "-" + train_type,
    "dir_data_types": "computed",
    "training_data_dirs": "gayane/freesolv",
    "valid_data_dir": "",
    # "max_steps":120000,
    "num_train_epochs": 15,
    "warmup": 3 * n_steps,
    "learning_rate": 0.0002,
    "eval_steps": 1 * n_steps,
    "save_steps": 600,
    "train_batch_size": batch_size,
    "valid_batch_size": batch_size,
    "dataloader_num_workers": 1,
    "experiment_name": job_name,
    "checkpoints_root_dir": "/nfs/ap/mnt/sxtn2/chem/experiments/checkpoints/",
    "flash_attn": False,
    "track": True,
    "save_final_model": False,
    "track_dir": "/nfs/ap/mnt/sxtn2/chem/experiments/aim/",
    "neftune_noise": 5,
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
    logs_path = "submitit_logs/%j"
    logs_path = "/nfs/dgx/raid/chem/" + logs_path if rsync_enabled else logs_path
    repo_path = (
        "/nfs/dgx/raid/chem/rsyncsnapshots/"
        f"{job_name}-{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    )

    with conditional_context_manager(rsync_enabled, repo_path):
        command = get_command(use_accelerate)
        executor = get_executor(executor_name, logs_path)
        executor.update_parameters(**slurm_params)
        print("train_name: ", job_name)
        print("logs_path: ", logs_path)
        print("repo path: ", repo_path)
        jobs = []

        # hyperparameter search loop
        with executor.batch():
            for lr in [0.00001, 0.00005, 0.0001, 0.0002]:
                for wu in [0, 1, 2, 3]:
                    for ep in [10, 15, 20]:
                        for nfn in [0.0, 5.0, 10.0]:
                            cli_arguments["learning_rate"] = lr
                            cli_arguments["warmup"] = wu * n_steps
                            cli_arguments["num_train_epochs"] = ep
                            cli_arguments["neftune_noise"] = nfn
                            cli_arguments[
                                "experiment_name"
                            ] = f"{job_name}_lr{lr}_wu{wu}_epoch{ep}_nef{nfn}"
                            command = get_command(use_accelerate)
                            function = submitit.helpers.CommandFunction(
                                command, env=env_variables
                            )
                            job = executor.submit(function)
                            jobs.append(job)

        # random state runs loop
        # with executor.batch():
        #     for rs in range(42,72,10):
        #         cli_arguments['seed'] = rs
        #         cli_arguments['experiment_name'] = f"{job_name}_seed{rs}"
        #         command = get_command(use_accelerate)
        #         function = submitit.helpers.CommandFunction(command, env=env_variables)
        #         job = executor.submit(function)
        #         jobs.append(job)

        # single submissions
        # function = submitit.helpers.CommandFunction(command, env=env_variables)
        # job = executor.submit(function)
        # jobs.append(job)