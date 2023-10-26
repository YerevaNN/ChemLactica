import unittest
import subprocess


class TestEnv(unittest.TestCase):

    def test_env_file_vs_current_env_depenedency_match(self):
        env_file_path = "environment.yml"
        conda_deps_str = subprocess.check_output(["conda", "env", "export"]).decode().strip().split("\n")
        with open(env_file_path) as _f:
            env_file_deps_str = [line.rstrip("\n") for line in _f.readlines()]

        conda_deps_str = conda_deps_str[:-1]
        env_file_deps_str = env_file_deps_str[:-1]
        for i, (l1, l2) in enumerate(zip(conda_deps_str, env_file_deps_str), start=1):
            assert l1 == l2, f"'{l1}' != '{l2}' on line {i}"


if __name__ == "__main__":
    unittest.main(verbosity=2)