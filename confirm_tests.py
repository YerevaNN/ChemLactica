import subprocess
import argparse
import unittest
import yaml
import os

# import sys
import enum


class TestType(enum.Enum):
    UNIT = "unit"
    INTEGRATION = "integration"


def print_test_details(unit_test_result):
    print(f"\nTotal Tests: {unit_test_result.testsRun}")
    print(f"Failures: {len(unit_test_result.failures)}")
    print(f"Errors: {len(unit_test_result.errors)}")
    print(f"Skipped: {len(unit_test_result.skipped)}")
    print(f"Successful: {unit_test_result.wasSuccessful()}")
    if unit_test_result.failures or unit_test_result.errors:
        print("\nDetails about failures and errors:")
        for failure in unit_test_result.failures + unit_test_result.errors:
            print(f"\nTest: {failure[0]}")
            print(f"Details: {failure[1]}")


def write_test_status(
    git_commit_hash: str, status: str = "FAIL", file_name: str = "test_status"
):
    data = {git_commit_hash: status}
    with open(f"{file_name}.yaml", "w") as _f:
        yaml.dump(data, _f)


def read_test_status(git_commit_hash: str, file_name: str = "test_status"):
    with open(f"{file_name}.yaml", "r") as _f:
        data = yaml.full_load(_f)
    return data.get(git_commit_hash)


def run_unit_tests():
    loader = unittest.TestLoader()
    # Discover and load unit tests
    unit_test_suite = loader.discover("unit_tests", pattern="*test*")

    # Run the unit tests
    runner = unittest.TextTestRunner(failfast=False, verbosity=2)
    result = runner.run(unit_test_suite)
    print_test_details(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=TestType,
        dest="run",
        choices=list(TestType),
        default=None,
        help="specify the type of test to run, will default to none if not specified",
    )
    parser.add_argument(
        "--no_run",
        action="store_false",
        dest="run",
        help="whether or not run tests",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        dest="gpus",
        help="comma seperated string of gpus indices to use for testing \
              (please choose at least 2 for proper testing, default is '0, 1').",
        required=False,
        default="0, 1",
    )
    parser.set_defaults(run=False)
    parser.add_argument(
        "--confirm",
        action="store_true",
        dest="confirm",
        help="whether or not confirm already run tests",
    )
    parser.add_argument(
        "--no_confirm",
        action="store_false",
        dest="confirm",
        help="whether or not confirm already run tests",
    )
    parser.set_defaults(confirm=False)
    args = parser.parse_args()
    run = args.run
    confirm = args.confirm
    gpus = args.gpus
    if run is not None:
        match (run):
            case TestType.UNIT:
                run_unit_tests()
            case TestType.INTEGRATION:
                git_commit_hash = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .decode()
                    .strip()
                )
                assert git_commit_hash
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus
                print(f"NOTE: Using GPU(s) '{gpus}' for testing.")
                loader = unittest.TestLoader()
                tests = loader.discover("tests", pattern="test_*.py")
                testRunner = unittest.runner.TextTestRunner(verbosity=2)
                test_results = testRunner.run(tests)
                if (
                    len(test_results.errors) == 0
                    and len(test_results.failures) == 0
                    and test_results.wasSuccessful
                ):
                    status = "PASS"
                else:
                    status = "FAIL"
                write_test_status(git_commit_hash, status=status)
            case _:
                pass
    elif confirm:
        git_commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD~1"]).decode().strip()
        )
        assert git_commit_hash
        status = read_test_status(git_commit_hash)
        if status == "FAIL":
            raise Exception(f"Commit '{git_commit_hash}' failed.")
        elif status == "PASS":
            print(f"Commit '{git_commit_hash}' passed.")
        else:
            raise Exception(
                f"Commit '{git_commit_hash}' has an unexpected status '{status}'."
            )
    else:
        raise Exception("Please pass the proper option in command line.")
