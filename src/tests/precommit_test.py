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
        train_command = f"python3 {train_script_path} \
                --from_pretrained small_opt \
                --model_config small_opt \
                --training_data_dir {training_data_dir_path} \
                --valid_data_dir {valid_data_dir_path} \
                --max_steps 10 \
                --eval_steps 5 \
                --save_steps 5 \
                --tokenizer 125m \
                --checkpoints_root_dir ./ \
                --tokenizer_checkpoint facebook/galactica-125m \
                --track \
                --track_dir ./aim"
        # --profile \
        # --profile_dir ./profiling"

        print(train_command)
        subprocess.run("mkdir profiling", shell=True)
        executed_prog = subprocess.run(
            train_command,
            shell=True,
        )
        if executed_prog.returncode != 0:
            raise Exception(f"\n\tExit code: {executed_prog.returncode}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
