import tqdm
import torch


class CustomTokenizer:
    __instance = None
    precomuted_ids = {}
    bos_token = "<s>"
    bos_token_id = 0
    eos_token = "</s>"
    eos_token_id = 2

    def __init__(self, instance):
        if CustomTokenizer.__instance is not None:
            raise Exception(f"There can only be one instance of {__class__.__name__}")
        CustomTokenizer.__instance = instance

        CustomTokenizer.precomuted_ids = {
            v: torch.tensor(CustomTokenizer.__instance.encode(v), dtype=torch.int32)
            for v in ["[START_SMILES]", "[END_SMILES]", "[", "]", "<pad>"]
        }

    @staticmethod
    def get_instance():
        return CustomTokenizer.__instance


class ProgressBar(tqdm.tqdm):
    __instance = None
    __total = None

    def __init__(self, total=None, *args, **kwargs):
        new_total = ProgressBar.__total if total is None else total
        super().__init__(total=new_total, *args, **kwargs)
        if ProgressBar.__instance is not None:
            raise Exception(f"There can only be one instance of {__class__.__name__}")

        ProgressBar.__instance = self
        ProgressBar.__total = new_total

    @staticmethod
    def set_total(total):
        ProgressBar.__total = total

    @staticmethod
    def delete_instance():
        ProgressBar.__instance = None

    @staticmethod
    def get_instance():
        return ProgressBar.__instance
