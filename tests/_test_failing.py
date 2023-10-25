import unittest


class FailingTest(unittest.TestCase):
    def test_failing(self):
        raise Exception("Failed")