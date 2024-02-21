from typing import List
import torch
from io import StringIO
import os


def samples_generator(
    files: List[str], shared_jsonl_files, chunk_size=25000, return_line_info=False
):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("sharded_jsonl_files", shared_jsonl_files)
        print(f"TOK_PAR: {os.environ['TOKENIZERS_PARALLELISM']}")
        print("process id", os.getpid(), files)

        file_states = {f: {"position": 0, "line_number": 0} for f in files}
        for file in file_states.keys():
            if shared_jsonl_files.get(file):
                jsonl_state = shared_jsonl_files[file]
                file_states[file] = jsonl_state
                print(f"loaded {file}: {jsonl_state['position']}")

        returned = True
        while returned:
            returned = False
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
                    state["line_number"] += len(batch)
                    for i, sample in enumerate(batch, start=1):
                        returned = True
                        ret = {"text": sample}
                        if return_line_info:
                            ret["line_info"] = {
                                "file": file,
                                "line_number": state["line_number"] - len(batch) + i,
                            }
                        yield ret
                shared_jsonl_files[file] = state
