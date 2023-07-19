import subprocess
import unittest
import os


class TestNetworkTraining(unittest.TestCase):
    def test_small_opt_model(self):
        script_path = os.path.dirname(os.path.abspath(__file__))

        # Building absolute paths
        train_script_path = os.path.join(script_path, "..", "train.py")
        training_data_dir_path = os.path.join(
            script_path, "..", "..", ".small_data", "train"
        )
        valid_data_dir_path = os.path.join(
            script_path, "..", "..", ".small_data", "valid"
        )
        executed_prog = subprocess.run(
            f"python3 {train_script_path} --model_type small_opt \
                --training_data_dir {training_data_dir_path} \
                --valid_data_dir {valid_data_dir_path} \
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
