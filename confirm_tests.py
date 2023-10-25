import subprocess
import argparse
import unittest
import yaml


def write_test_status(git_commit_hash: str, status: str="FAIL", file_name: str="test_status"):
    data = {git_commit_hash: status}
    with open(f"{file_name}.yaml", "w") as _f:
        yaml.dump(data, _f)


def read_test_status(git_commit_hash: str, file_name: str="test_status"):
    with open(f"{file_name}.yaml", "r") as _f:
        data = yaml.full_load(_f)
    return data.get(git_commit_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        dest="run",
        help="whether or not profile the training",
    )
    parser.add_argument(
        "--no_run",
        action="store_false",
        dest="run",
        help="whether or not profile the training",
    )
    parser.set_defaults(run=False)
    parser.add_argument(
        "--confirm",
        action="store_true",
        dest="confirm",
        help="whether or not profile the training",
    )
    parser.add_argument(
        "--no_confirm",
        action="store_false",
        dest="confirm",
        help="whether or not profile the training",
    )
    parser.set_defaults(confirm=False)
    args = parser.parse_args()
    run = args.run
    confirm = args.confirm
    git_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    if run:
        loader = unittest.TestLoader()
        tests = loader.discover("tests", pattern="test_*.py")
        testRunner = unittest.runner.TextTestRunner(verbosity=2)
        test_results = testRunner.run(tests)
        if len(test_results.errors) == 0 and len(test_results.failures) == 0 and test_results.wasSuccessful:
            status = "PASS"
        else:
            status = "FAIL"
        write_test_status(git_commit_hash, status=status)
    elif confirm:
        status = read_test_status(git_commit_hash)  
        if status == "FAIL":
            raise Exception(f"Commit '{git_commit_hash}' failed.")
        elif status == "PASS":
            print(f"Commit '{git_commit_hash}' passed.")
        else:
            raise Exception(f"Commit '{git_commit_hash}' has an unexpected status '{status}'.")
    else:
        raise Exception("Please pass the proper option in command line.")