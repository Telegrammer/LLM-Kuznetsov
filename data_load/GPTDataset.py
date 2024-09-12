import itertools
from typing import Callable

import tiktoken
import torch

from .AbstractTransformerTextDataset import AbstractTransformerTextDataset

__all__ = ["GPTTextDataset"]


class GPTTextDataset(AbstractTransformerTextDataset):

    def __init__(self,
                 path: str,
                 tokenization_method: str,
                 segmentation_method: str,
                 sequence_length: int):
        AbstractTransformerTextDataset.__init__(self)
        self._tokenization_methods: dict[str, Callable[[list[str]], list[list[int]]]] = {
            "custom": self._custom_tokenize,
            "tiktoken": self._tiktoken_tokenize
        }

        self._segmentation_methods: dict[str, Callable[[list[list[int]], int], None]] = {
            "shifted_tokens": self._split_by_shifted_tokens,
            "tokens": self._split_by_tokens,
        }

        self._len_dataset: int = 0
        self._data_list: list[torch.LongTensor] = []

        with open(path, mode="r", encoding="utf-8") as file:
            text_parts: list[str] = [part.replace("\n", " ")[:] for part in file.read().split("\n\n")]
        chunks: list[list[int]] = self._tokenization_methods[tokenization_method](text_parts)

        self._segmentation_methods[segmentation_method](chunks, sequence_length)
        self._make_dataset()

    @staticmethod
    def _custom_tokenize(text_parts: list[str]) -> list[list[str]]:
        return []

    @staticmethod
    def _tiktoken_tokenize(text_parts: list[str]) -> list[list[int]]:
        encoding = tiktoken.get_encoding("p50k_base")
        chunks: list[list[int]] = [torch.LongTensor(encoding.encode(part)) for part in text_parts]
        return chunks

    def _split_by_tokens(self, chunks: list[list[int]], sequence_length: int) -> None:
        tokens: list[int] = list(itertools.chain(*chunks))
        self._data_list: list[torch.LongTensor] = [torch.LongTensor(tokens[i:i + sequence_length]) for i in
                                                   range(0, len(tokens), sequence_length)]

    def _split_by_shifted_tokens(self, chunks: list[list[int]], sequence_length: int) -> None:
        tokens = list(itertools.chain(*chunks))
        for i in range(0, len(tokens) - 1, sequence_length):
            self._data_list.append(torch.LongTensor(tokens[i:i + sequence_length]))
            self._data_list.append(torch.LongTensor(tokens[i + 1:i + sequence_length + 1]))

    def __str__(self):
        return "class GPTDataset"
