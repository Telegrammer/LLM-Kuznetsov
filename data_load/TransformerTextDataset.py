import itertools
from typing import Callable

import spacy
import tiktoken

from .AbstractTransformerTextDataset import AbstractTransformerTextDataset

__all__ = ["TransformerTextDataset"]


class TransformerTextDataset(AbstractTransformerTextDataset):

    def __init__(self,
                 path: str,
                 tokenization_method: str,
                 segmentations_method: str,
                 sequence_length: int):
        AbstractTransformerTextDataset.__init__(self)
        self._tokenization_methods: dict[str, Callable[[list[str]], list[list[str]]]] = {
            "spacy": self._spacy_tokenize,
            "tiktoken": self._byte_pair_tokenize
        }

        self._segmentation_methods: dict[str, Callable[[list[list[str]], int], list[tuple[str]]]] = {
            "shifted_tokens": self._split_by_shifted_tokens,
            "tokens": self._split_by_tokens,
            "chunks": self._split_by_chunks
        }

        self._len_dataset: int = 0
        self._data_list: list[list[str]] = []

        with open(path, mode="r", encoding="utf-8") as file:
            text_parts = [part.replace("\n", " ")[:] for part in file.read().split("\n\n")]

        chunks: list[list[str]] = self._tokenization_methods[tokenization_method](text_parts)

        self.__bag_of_words = set(itertools.chain(*chunks))
        self.__bag_of_words.add("<sos>")
        self.__bag_of_words.discard("\n")
        self.__bag_of_words.discard("\n\n")
        self.__bag_of_words = list(self.__bag_of_words)
        self.__bag_of_words.sort()

        self._segmentation_methods[segmentations_method](chunks, sequence_length)
        self._make_dataset()

    @staticmethod
    def _spacy_tokenize(text_parts: list[str]) -> list[list[str]]:
        tokenizer = spacy.load("ru_core_news_sm")
        text_parts = [tokenizer(part) for part in text_parts]
        chunks: list[list[str]] = [[token.text for token in part] for part in text_parts]
        return chunks

    @staticmethod
    def _byte_pair_tokenize(text_parts: list[str]) -> list[list[str]]:
        encoding = tiktoken.get_encoding("p50k_base")
        chunks: list[list[str]] = [
            [encoding.decode_single_token_bytes(token).decode("utf-8") for token in encoding.encode(part)] for
            part in text_parts]
        return chunks

    def _split_by_tokens(self, chunks: list[list[int]], sequence_length: int) -> None:
        tokens: list[int] = list(itertools.chain(*chunks))
        self._data_list = [tokens[i:i + sequence_length] for i in range(0, len(tokens), sequence_length)]

    def _split_by_shifted_tokens(self, chunks: list[list[int]], sequence_length: int) -> None:
        tokens = list(itertools.chain(*chunks))
        for i in range(0, len(tokens) - 1, sequence_length):
            self._data_list.append(tokens[i:i + sequence_length])
            self._data_list.append(tokens[i + 1:i + sequence_length + 1])

    def _split_by_chunks(self, chunks: list[list[str]], sequence_length: int) -> None:
        self.__bag_of_words.extend(["<pad>", "<eos>"])
        self.__data_list = [["<sos>"] + chunk[:min(len(chunk), sequence_length - 2) + 1] + ["<eos>", "<sos>"] for chunk
                            in chunks]
        # self.__data_list = [chunk + ["<pad>" for i in range(batch_size - len(chunk))] for chunk in self.__data_list]

    def get_bag_of_words_once(self):
        bag_of_words = self.__bag_of_words[:]
        del self.__bag_of_words
        return bag_of_words

    def __str__(self):
        return "class TransformerTextDataset"
