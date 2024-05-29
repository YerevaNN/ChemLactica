import submitit
import subprocess
import itertools as it
import datetime
import yaml
import os
import copy


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

    executor = submitit.AutoExecutor(folder="/auto/home/tigranfahradyan/slurm_jobs/PMO/job_%j")
    executor.update_parameters(
        name="chemlactica-pmo", timeout_min=n_runs * 3 * 60,
        gpus_per_node=1, nodes=1, mem_gb=50, cpus_per_task=8, 
        slurm_array_parallelism=10
    )
    jobs = []
    with executor.batch():
        for config in hparam_configs[:1]:
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = f"results/{formatted_date_time}"
            os.makedirs(base, exist_ok=True)
            v = 0
            name = model_name + "-" + "+".join(config["strategy"])
            while os.path.exists(os.path.join(base, f"{name}-{v}")):
                v += 1
            output_dir = os.path.join(base, f"{name}-{v}")
            output_dir += "tune"
            # output_dir = "main/chemlactica/results/2024-05-11/chemlactica-125m-rej-sample-4"
            os.makedirs(output_dir, exist_ok=True)
            yaml.safe_dump(config, open(os.path.join(output_dir, "hparams.yaml"), "w"))
            function = submitit.helpers.CommandFunction([
                'python3', 'hparam_search.py',
                '--config_default', os.path.join(output_dir, "hparams.yaml"),
                '--output_dir', output_dir,
                '--n_runs', str(n_runs),
            ])
            print(' '.join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            jobs.append(job)
