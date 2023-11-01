import unittest
import subprocess


class TestEnv(unittest.TestCase):
    def test_env_file_vs_current_env_depenedency_match(self):
        compare_output = subprocess.check_output(["conda", "compare", "environment.yml"]).decode().strip()
        success_output = "Success. All the packages in the specification file are present in the environment with matching version and build string."
        print(compare_output)
        assert compare_output == success_output


if __name__ == "__main__":
    unittest.main(verbosity=2)