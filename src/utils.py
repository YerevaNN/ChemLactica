import tqdm


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
