def create_train_command(module, module_args, script, script_args):
    train_command = "python3 -m "
    train_command += (
        f"{module} {''.join([f'--{arg} {val} ' for arg, val in module_args.items()])}"
    )
    train_command += (
        f"{script} {''.join([f'--{arg} {val} ' for arg, val in script_args.items()])}"
    )
    return train_command


# def setUp():
#     file_path = os.path.dirname(os.path.abspath(__file__))
#     print("FILE_PATH", file_path)
#     os.environ["PYTHONPATH"] = os.path.join(file_path, "src")
#     os.environ["PYTHONPATH"] += os.path.join(file_path, "tests")


def create_vs_code_launch(module, module_args, script, script_args):
    """
        this is for creating a launch file config for vs code editor to be able 
        to easily debug the command running in the test.
    """
    pass


# def generate_batches_from_jsonls(jsonl_files, count):
#     dataset = load_dataset(
#         "text",
#         data_files={"data": jsonl_files},
#         streaming=True,
#     )

#     processed_dataset = process_dataset(
#         dataset=dataset,
#         train_config={"block_size": 2048},
#         process_batch_sizes=(100, 100),
#     )

#     batches = []
#     for i, inp in enumerate(processed_dataset["data"]):
#         if i == count: break
#         del inp["token_type_ids"]
#         inp = {k: inp[k].unsqueeze(0) for k in inp.keys()}
#         batches.append(inp)
#     return batches