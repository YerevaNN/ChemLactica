import subprocess
import unittest


class TestNetworkTraining(unittest.TestCase):
    def test_small_opt_model(self):
        executed_prog = subprocess.run(
            "python3 train.py --model_type small_opt \
                --training_data_dir .small_data/train \
                --valid_data_dir .small_data/valid \
                --max_steps 1 \
                --eval_steps 1 \
                --save_steps 1 \
                --checkpoints_root_dir ./",
            shell=True,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
            capture_output=True,
        )
        if executed_prog.returncode != 0:
            raise Exception(
                f"\n\tExit code: {executed_prog.returncode} \
                  \n\tError output: {executed_prog.stderr}."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
