import multiprocessing
from typing import Dict, Any


global shared_jsonl_states
shared_jsonl_states = multiprocessing.Queue()


class JsonlDataset:

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._f = open(file_path, "r")
        self.line_number = 0

    def __next__(self) -> str:
        line = self._f.readline()[:-1] # exclude the newline character
        if not line:
            if not self._f.closed:
                self._f.close()
            return None
        # print("Process:", torch.distributed.get_rank(), self.line_number, "line number id:", id(self.line_number), "dataloader id:", id(self), self.get_read_position())
        self.line_number += 1
        return line
    
    def get_read_position(self) -> int:
        """
            returns an integer giving the file object’s current position in the file
            represented as number of bytes from the beginning of the file
        """
        return self._f.tell()

    def set_read_position(self, position: int):
        self._f.seek(position, 0)

    def get_state(self) -> Dict[str, Any]:
        return {
            "position": self.get_read_position(),
            "line_number": self.line_number
        }
    
    def load_state(self, state: Dict[str, Any]):
        self.set_read_position(state["position"])
        self.line_number = state["line_number"]
        print(f"loaded jsonl state: {state}")


def samples_generator(jsonl_datasets_dict: Dict[str, JsonlDataset]):
    if shared_jsonl_states.empty():
        jsonl_states = {}
    else:
        jsonl_states = shared_jsonl_states.get()
        print(f"Jsonl states loaded.")
    while True:
        returned = False
        for name, ds in jsonl_datasets_dict.items():
            sample = next(ds)
            jsonl_states[name] = ds.get_state()

            # empty the queue and put the latest states
            while not shared_jsonl_states.empty():
                shared_jsonl_states.get()   
            shared_jsonl_states.put(jsonl_states)

            if sample:
                returned = True
                yield {"text": sample} # this is done to be compatible with the old code

        if not returned:
            break