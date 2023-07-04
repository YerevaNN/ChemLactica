import subprocess
import unittest


class TestNetworkTraining(unittest.TestCase):
    def test_125m_model(self):
        executed_prog = subprocess.run(
            "python3 train.py --model_type 125m --max_steps 1 --load_small_opt true",
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
