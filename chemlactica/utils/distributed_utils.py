import os


def get_experiment_hash(from_pretrained, train_type="pretrain"):
    if os.path.isdir(from_pretrained) and train_type == "pretrain":
        return str(from_pretrained.split(os.path.sep)[-2])
    else:
        return "none"
