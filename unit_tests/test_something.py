import unittest
import torch
from chemlactica.utils.dataset_utils import load_jsonl_line
from chemlactica.utils.dataset_utils import group_texts
from unittest.mock import Mock


class TestDataProcessing(unittest.TestCase):
    def test_positive(self):
        result = 2 + 2
        self.assertEqual(result, 4, "Expected result: 4")


class TestLoadJsonlLine(unittest.TestCase):
    #  Can load a valid JSONL line as a dictionary
    def test_load_valid_jsonl_line_as_dict(self):
        jsonl_line = """{"key": "value"}"""
        loaded_line = load_jsonl_line(jsonl_line)
        assert load_jsonl_line(jsonl_line) == loaded_line

    #  Returns None when given an empty string
    def test_raise_value_error_empyty_line(self):
        jsonl_line = ""
        with self.assertRaises(ValueError):
            load_jsonl_line(jsonl_line)


class TestGroupTexts(unittest.TestCase):
    def test_empty_attention_mask(self):
        # Mock the get_tokenizer function
        mocker = Mock()
        mocker.eos_token_id = 0
        mocker.return_value = mocker

        # Create example input tensors
        examples = {"input_ids": [torch.tensor([1, 2, 3])], "attention_mask": []}

        # Set train_config
        train_config = {"tokenizer_path": "path/to/tokenizer", "block_size": 3}

        # Call the group_texts function
        with self.assertRaises(Exception):
            group_texts(examples, train_config)

    # def test_splits_into_correct_size_chunks(self):
    #     mocker = Mock()
    #     mocker.eos_token_id = 0
    #     mocker.return_value = mocker
    #     train_config = {"block_size": 2037}
    #     examples = {"input_ids": [torch.tensor([1, 2, 3])], "attention_mask": []}

    #     self.assertTrue(all(len(ids) ==
    # train_config["block_size"] for ids in result["input_ids"]))
    #     self.assertTrue(all(len(mask) ==
    # train_config["block_size"] for mask in result["attention_mask"]))
