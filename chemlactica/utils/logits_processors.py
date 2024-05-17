from transformers import LogitsProcessor
import torch
from typing import List, Union


class OneOccurenceLogitsProcessor(LogitsProcessor):
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for token_id in self.suppress_tokens:
            if token_id in input_ids:
                scores[:, token_id] = -float("inf")
        return scores


class TunableExponentialDecayLengthPenalty(LogitsProcessor):
    def __init__(
        self,
        exponential_decay_factors: List[float],
        regulation_starts: List[int],
        decay_token_ids: Union[int, List[int]],
        input_ids_seq_length: int,
    ):
        self.regulation_starts = regulation_starts
        self.regulation_list = exponential_decay_factors
        if isinstance(decay_token_ids, int):
            decay_token_ids = [decay_token_ids]
        self.decay_token_ids = decay_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        penalties = torch.zeros_like(scores)
        scores_processed = scores
        for token_id, regulation_factor, regulation_start in zip(
            self.decay_token_ids, self.regulation_list, self.regulation_starts
        ):
            if cur_len > regulation_start:
                penalty_idx = cur_len - regulation_start
                penalty = torch.abs(scores[:, token_id]) * (
                    pow(regulation_factor, penalty_idx) - 1
                )
                penalties[:, token_id] = penalty
                scores_processed = scores + penalties
        return scores_processed


class SequentialDecayProcessor(LogitsProcessor):
    def __init__(
        self,
        exponential_decay_factors: List[float],
        decay_token_ids: Union[int, List[int]],
        input_ids_seq_length: int,
    ):
        self.regulation_list = exponential_decay_factors
        if isinstance(decay_token_ids, int):
            decay_token_ids = [decay_token_ids]

        self.decay_token_ids = decay_token_ids
        self.regulation_starts = 99999999999999999 * len(self.decay_token_ids)
        self.regulation_starts[0] = 0
        self.regulation_list = exponential_decay_factors

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        penalties = torch.zeros_like(scores)
        scores_processed = scores
        for i, (token_id, regulation_factor) in enumerate(
            zip(self.decay_token_ids, self.regulation_list)
        ):
            if token_id in input_ids:
                self.regulation_starts[i] = cur_len

            reg_start = self.regulation_starts[i]
            if cur_len > reg_start:
                penalty_idx = cur_len - reg_start
                penalty = torch.abs(scores[:, token_id]) * (
                    pow(regulation_factor, penalty_idx) - 1
                )
                penalties[:, token_id] = penalty
                scores_processed = scores + penalties
        return scores_processed
