from multiprocessing import Manager, Pool
from typing import List
import torch
from io import StringIO
import os


def samples_generator(files: List[str], shared_jsonl_files, chunk_size=25000):
    if torch.distributed.get_rank() == 0:
        print(f"TOK_PAR: {os.environ['TOKENIZERS_PARALLELISM']}")
        print("process id", os.getpid(), files)

        file_states = {file: {"position": 0} for file in files}
        for file in file_states.keys():
            if shared_jsonl_files.get(file):
                jsonl_state = shared_jsonl_files[file]
                print(f"loaded {file}: {jsonl_state['position']}")

        while True:
            for file, state in file_states.items():
                with open(file) as f:
                    f.seek(state["position"])
                    batch = f.read(chunk_size)
                    if not batch:
                        break
                    batch += f.readline()
                    batch = StringIO(batch).readlines()
                    batch = [line.rstrip("\n") for line in batch]
                    state["position"] = f.tell()
                    for sample in batch:
                        yield {"text": sample} # this is done to be compatible with the old code

                shared_jsonl_files[file] = state