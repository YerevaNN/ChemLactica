import tqdm


class ProgressBar(tqdm.tqdm):
    __instance = None

    def __init__(self, *args, **kwards):
        if ProgressBar.__instance is not None:
            raise Exception(f"{__class__.__name__} is a singleton class.")
        super().__init___(*args, **kwards)
        self._finished = False

    def get_instance():
        return ProgressBar.__instance
