from typing import List
import torch

# from io import StringIO
import os


def generator_init_print(shared_jsonl_files, files):
    print("sharded_jsonl_files", shared_jsonl_files)
    print(f"TOK_PAR: {os.environ['TOKENIZERS_PARALLELISM']}")
    print("process id", os.getpid(), files)


def setup_generator(shared_jsonl_files, files):
    generator_init_print(shared_jsonl_files, files)

    file_states = {f: {"position": 0, "line_number": 0} for f in files}
    for file in file_states.keys():
        if shared_jsonl_files.get(file):
            jsonl_state = shared_jsonl_files[file]
            file_states[file] = jsonl_state
            print(f"loaded {file}: {jsonl_state['position']}")
    return file_states


def get_batch(file, state, chunk_size):
    with open(file) as f:
        f.seek(state["position"])
        batch = f.read(chunk_size)
        if not batch:
            raise StopIteration

        batch += f.readline()
        batch = batch.splitlines()

        # batch = [line.rstrip("\n") for line in batch]
        state["position"] = f.tell()
        batch_len = len(batch)
        state["line_number"] += batch_len
    return batch, batch_len, state


def format_sample(sample, return_line_info, batch_len, file, state, i):
    ret = {"text": sample}
    if return_line_info:
        ret["line_info"] = {
            "file": file,
            "line_number": state["line_number"] - batch_len + i,
        }
    return ret


def samples_generator(
    files: List[str], shared_jsonl_files, chunk_size=25000, return_line_info=False
):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        file_states = setup_generator(shared_jsonl_files, files)

        returned = True
        while returned:
            returned = False
            for file, state in file_states.items():
                try:
                    batch, batch_len, state = get_batch(file, state, chunk_size)
                except StopIteration:
                    break
                for i, sample in enumerate(batch, start=1):
                    returned = True
                    ret = format_sample(
                        sample, return_line_info, batch_len, file, state, i
                    )
                    yield ret
                shared_jsonl_files[file] = state
