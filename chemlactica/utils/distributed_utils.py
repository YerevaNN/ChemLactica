import os


def get_experiment_hash(from_pretrained):
    if os.path.isdir(from_pretrained):
        return str(from_pretrained.split(os.path.sep)[-2])
    else:
        return "none"
