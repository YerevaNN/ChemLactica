import unittest
import subprocess
from test_utils import create_train_command


class TestLineByLineDataloader(unittest.TestCase):
    def test_line_by_line_dataloader(self):
        command = create_train_command(
            module="accelerate.commands.launch",
            module_args={
                "config_file": "chemlactica/config/test_configs/fsdp_config.yaml"
            },
            script="tests/dataset/distributed_dataset_iter.py",
            script_args={},
        )

        print(f"Running command: {command}")
        out = subprocess.run(command, shell=True, capture_output=False)
        if out.returncode != 0:
            raise Exception(out.stderr.decode())
