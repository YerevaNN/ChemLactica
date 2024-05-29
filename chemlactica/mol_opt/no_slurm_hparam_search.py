import submitit
import subprocess
import itertools as it
import datetime
import yaml
import os
import copy
import time
import torch


def is_gpu_being_used(gpu_id):
    try:
        # Run the nvidia-smi command
        cmd = ['nvidia-smi','-i',f"{gpu_id}"]
        output = subprocess.check_output(cmd)
        output = output.decode('utf-8')
        if "No running processes found" in output:
            return False
        else:
            return True

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi command: {e}")


def create_hparam_configs(config_file_path):
    config_tune = yaml.safe_load(open("hparams_tune.yaml"))
    config_merged = {}
    for key, value in config_tune["parameters"].items():
        if type(value) == list:
            config_merged[key] = value
        else:
            for k, v in value.items():
                config_merged[key+'+'+k] = v
    
    config_default = yaml.safe_load(open(config_file_path))
    hparam_names = list(config_merged.keys())
    all_configs = []
    for params in it.product(*config_merged.values()):
        # pprint(params)
        # pprint(hparam_names)
        config = copy.deepcopy(config_default)
        for i, p in enumerate(params):
            if '+' in hparam_names[i]:
                a, b = hparam_names[i].split("+")
                config[a][b] = p
            else:
                config[hparam_names[i]] = p
        # pprint(params)
        # pprint(config)
        all_configs.append(config)
        # print(config)
    return all_configs


if __name__ == "__main__":
    n_runs = 3

    config_file_path = "chemlactica_125m_hparams.yaml"
    # config_file_path = "main/chemlactica/chemma_2b_hparams.yaml"
    hparam_configs = create_hparam_configs(config_file_path)
    # infer_config = [yaml.safe_load(open(config_file_path))]
    model_name = "-".join(config_file_path.split("/")[-1].split("_")[:2])
    gpu_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    index = 0
    while index < len(hparam_configs):
        free_gpu_index = None
        for gpu_index in gpu_indices:
            gpu_is_free = True
            print(f"Checking gpu: {gpu_index}")
            for _ in range(10):
                if is_gpu_being_used(gpu_index):
                    gpu_is_free = False
                    break
                time.sleep(1)
            if gpu_is_free:
                free_gpu_index = gpu_index
                print(f"gpu: {gpu_index} is free")
                break
            else:
                print(f"gpu: {gpu_index} is being used")
        if free_gpu_index is not None:
            print(f"found a free gpu {free_gpu_index}, putting a job")
            executor = submitit.LocalExecutor(folder="/home/admin/tigran/slurm_jobs/PMO/job_%j")
            executor.update_parameters(
                name="chemlactica-pmo", timeout_min=n_runs * 12 * 60,
                visible_gpus=[free_gpu_index],
                gpus_per_node=1, nodes=1, mem_gb=80, cpus_per_task=8, 
                slurm_array_parallelism=10
            )
            jobs = []
            with executor.batch():
                current_hparams = [hparam_configs[index]]
                for config in current_hparams:
                    formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
                    base = f"results/{formatted_date_time}"
                    os.makedirs(base, exist_ok=True)
                    v = 0
                    name = model_name + "-" + "+".join(config["strategy"])
                    while os.path.exists(os.path.join(base, f"{name}-{v}-hparam-search")):
                        v += 1
                    output_dir = os.path.join(base, f"{name}-{v}-hparam-search")
                    os.makedirs(output_dir, exist_ok=True)
                    yaml.safe_dump(config, open(os.path.join(output_dir, "hparams.yaml"), "w"))
                    function = submitit.helpers.CommandFunction([
                        'python3', 'hparam_search.py',
                        '--config_default', os.path.join(output_dir, "hparams.yaml"),
                        '--output_dir', output_dir,
                        '--n_runs', str(n_runs),
                    ])
                    print(' '.join(function.command))
                    job = executor.submit(function)
                    jobs.append(job)
            for job in jobs:
                print(job.job_id)
            index += 1
            free_gpu_index = None