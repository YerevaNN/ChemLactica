import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="none")
    parser.add_argument(
        "--train_type",
        type=str,
        metavar="TT",
        dest="train_type",
        choices=["pretrain", "sft", "isft", "dpo"],
        required=False,
        default="pretrain",
        help="determining the type of training",
    )
    parser.add_argument(
        "--from_pretrained",
        type=str,
        metavar="FP",
        dest="from_pretrained",
        required=True,
        help="the path to the model dir",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        metavar="MC",
        dest="model_config",
        required=True,
        help="the model configuration to use",
    )
    parser.add_argument(
        "--training_data_dirs",
        metavar="DT",
        nargs="*",
        dest="training_data_dirs",
        required=True,
        help="path to directory containing training data",
    )
    parser.add_argument(
        "--dir_data_types",
        metavar="DD",
        nargs="*",
        dest="dir_data_types",
        required=True,
        help="corresponding type of data for directory (in same order)",
    )
    parser.add_argument(
        "--valid_data_dir",
        type=str,
        metavar="VD",
        dest="valid_data_dir",
        required=True,
        help="path to directory containing validation data",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        metavar="MS",
        dest="max_steps",
        required=False,
        default=-1,
        help="the number of steps to train (overrides the n_epochs)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        metavar="MS",
        dest="num_train_epochs",
        required=False,
        help="the number of epochs to train",
    )
    parser.add_argument(
        "--scheduler_max_steps",
        type=int,
        metavar="SMS",
        dest="scheduler_max_steps",
        required=False,
        default=None,
        help="the number of steps the scheduler should run",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        metavar="ES",
        dest="eval_steps",
        required=True,
        help="the number of training steps after which to evaluate the model",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        metavar="SS",
        dest="save_steps",
        required=True,
        help="the number of steps to save a model checkpoint",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        metavar="TBS",
        required=True,
        help="train batch size (per GPU when using dist training)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        metavar="TBS",
        required=False,
        help="valid batch size (per GPU when using dist validation)",
        default=None,
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        metavar="SBS",
        dest="shuffle_buffer_size",
        required=False,
        help="the buffer size of the buffered shuffle",
        default=4,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        metavar="EN",
        dest="experiment_name",
        required=False,
        help="the name of the experiment",
        default="none",
    )
    parser.add_argument(
        "--checkpoints_root_dir",
        type=str,
        metavar="CRD",
        dest="checkpoints_root_dir",
        required=True,
        help="directory where to save checkpoints",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        metavar="NW",
        dest="dataloader_num_workers",
        required=False,
        help="number of processes to use for dataloading",
        default=0,
    )
    parser.add_argument(
        "--track",
        action="store_true",
        dest="track",
        help="whether or not track the training using aim",
    )
    parser.add_argument(
        "--no_track",
        action="store_false",
        dest="track",
        help="the directory to save the aim tracking information",
    )
    parser.set_defaults(track=True)
    parser.add_argument(
        "--track_dir",
        type=str,
        metavar="TD",
        dest="track_dir",
        required=False,
        help="aim track directory",
        default=None,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        dest="profile",
        help="whether or not profile the training",
    )
    parser.add_argument(
        "--no_profile",
        action="store_false",
        dest="profile",
        help="whether or not profile the training",
    )
    parser.set_defaults(profile=False)
    parser.add_argument(
        "--profile_dir",
        type=str,
        metavar="PD",
        dest="profile_dir",
        required=False,
        help="profiling directory",
        default=None,
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        dest="flash_attn",
        help="whether or not to use flash attn)",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_false",
        dest="flash_attn",
        help="whether or not to use flash attn",
    )
    parser.set_defaults(flash_attn=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        metavar="GA",
        dest="gradient_accumulation_steps",
        required=False,
        help="the number of steps to over which to accumulate gradients",
        default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        dest="gradient_checkpointing",
        default=False,
        help="whether or not to use gradient_checkpointing",
    )
    parser.add_argument(
        "--check_reproducability",
        action="store_true",
        dest="check_reproducability",
        help="whether or not check reproducability (should only be use for testing)",
    )
    parser.add_argument(
        "--no_check_reproducability",
        action="store_false",
        dest="check_reproducability",
        help="whether or not check reproducability (should only be use for testing)",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        dest="evaluate_only",
        help="Whether to only call evaluation, this is for slurm use",
    )
    parser.add_argument(
        "--slurm_eval",
        action="store_true",
        required=False,
        dest="slurm_eval",
    )
    parser.set_defaults(profile=False)
    return parser
