from multiprocessing import Manager, Pool
from typing import List
import torch
from io import StringIO


global shared_jsonl_files
manager = Manager()
shared_jsonl_files = manager.dict()


def get_samples(entry, chunk_size: int=25000):
    file, position = entry
    with open(file) as f:
        f.seek(position)
        batch = f.read(chunk_size)
        if not batch:
            return file, None, f.tell()
        batch += f.readline()
        batch = StringIO(batch).readlines()
        batch = [line.rstrip("\n") for line in batch]
        return file, batch, f.tell()


def samples_generator(jsonl_files: List[str]):
    if torch.distributed.get_rank() == 0:
        
        jsonl_files = {file: 0 for file in jsonl_files}
        for file, pos in shared_jsonl_files.items():
            jsonl_files[file] = pos
            print(f"loaded {file}: {pos} (process {torch.distributed.get_rank()})")
        with Pool(len(jsonl_files)) as pol:
            while True:
                returned = False
                for file, batch, pos in pol.map(get_samples, jsonl_files.items()):
                    jsonl_files[file] = pos
                    if batch:
                        for sample in batch:
                            if sample:
                                returned = True
                                yield {"text": sample} # this is done to be compatible with the old code

                for key, value in jsonl_files.items():
                    shared_jsonl_files[key] = value

                if not returned:
                    break