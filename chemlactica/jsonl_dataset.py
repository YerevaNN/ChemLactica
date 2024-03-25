from typing import List

import os
from accelerate.state import PartialState

distributed_state = PartialState()


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


def should_yield_on_current_rank(i, num_processes, process_index):
    return i % num_processes == process_index


def format_sample(line):
    sample = line.strip()
    ret = {"text": sample}
    return ret


def samples_generator(
    files: List[str], shared_jsonl_files, chunk_size=25000, return_line_info=False
):
    file_states = setup_generator(shared_jsonl_files, files)

    returned = True
    while returned:
        returned = False
        for file, state in file_states.items():
            with open(file) as f:
                f.seek(state["position"])
                line = f.readline()
                counter = 0
                while line:
                    state["position"] = f.tell()
                    if should_yield_on_current_rank(
                        counter,
                        distributed_state.num_processes,
                        distributed_state.process_index,
                    ):
                        returned = True
                        ret = format_sample(line)
                        yield ret
                    counter = counter + 1
                    shared_jsonl_files[file] = state
                    line = f.readline()
