import itertools
from typing import Callable

import spacy
import tiktoken
from torch.utils.data import Dataset

__all__ = ["TransformerTextDataset"]


class TransformerTextDataset(Dataset):

    def __init__(self,
                 path: str,
                 tokenization_method: str,
                 batching_method: str,
                 batch_size: int = 32):

        self.__tokenization_methods: dict[str, Callable] = {
            "spacy": self._spacy_tokenize,
            "tiktoken": self._byte_pair_tokenize
        }

        self.__batching_methods: dict[str, Callable] = {
            "shifted_tokens": self._batch_by_shifted_tokens,
            "tokens": self._batch_by_tokens,
            "chunks": self._batch_by_chunks
        }

        self.__len_dataset = 0
        self.__data_list: list = []

        with open(path, mode="r", encoding="utf-8") as file:
            text_parts = [part.replace("\n", " ")[:] for part in file.read().split("\n\n")]
        chunks = self.__tokenization_methods[tokenization_method](text_parts)

        self.__bag_of_words = set(itertools.chain(*chunks))
        self.__bag_of_words.add("<sos>")
        self.__bag_of_words.discard("\n")
        self.__bag_of_words.discard("\n\n")
        self.__bag_of_words = list(self.__bag_of_words)
        self.__bag_of_words.sort()

        self.__batching_methods[batching_method](chunks, batch_size)
        self._make_dataset()

    @staticmethod
    def _spacy_tokenize(text_parts: list[str]) -> list[str]:
        tokenizer = spacy.load("ru_core_news_sm")
        text_parts = [tokenizer(part) for part in text_parts]
        chunks = [[token.text for token in part] for part in text_parts]
        return chunks

    @staticmethod
    def _byte_pair_tokenize(text_parts: list[str]):
        encoding = tiktoken.get_encoding("p50k_base")
        chunks = [[encoding.decode_single_token_bytes(token).decode("utf-8") for token in encoding.encode(part)] for
                   part in text_parts]
        return chunks

    def _batch_by_tokens(self, chunks: list, batch_size: int):
        tokens = list(itertools.chain(*chunks))
        self.__data_list = [tokens[i:i + batch_size] for i in
                            range(0, len(tokens), batch_size)]

    def _batch_by_shifted_tokens(self, chunks: list, batch_size: int):
        tokens = list(itertools.chain(*chunks))
        for i in range(0, len(tokens) - 1, batch_size):
            self.__data_list.append(tokens[i:i + batch_size])
            self.__data_list.append(tokens[i + 1:i + batch_size + 1])

    def _batch_by_chunks(self, chunks: list[str], batch_size: int):
        self.__bag_of_words.extend(["<pad>", "<eos>"])
        self.__data_list = [["<sos>"] + chunk[:min(len(chunk), batch_size - 2) + 1] + ["<eos>", "<sos>"] for chunk in
                            chunks]
        # self.__data_list = [chunk + ["<pad>" for i in range(batch_size - len(chunk))] for chunk in self.__data_list]

    def _make_dataset(self):
        if len(self.__data_list) % 2 != 0:
            self.__data_list = self.__data_list[:-1]

        self.__len_dataset = len(self.__data_list) // 2
        self.__data_list = [[self.__data_list[i], self.__data_list[i + 1]] for i in
                            range(0, len(self.__data_list), 2)]
        self.__len_dataset = len(self.__data_list)

    def __len__(self):
        return self.__len_dataset

    def __getitem__(self, index):
        return self.__data_list[index]

    def __setitem__(self, index, value):
        self.__data_list[index] = value

    def get_bag_of_words_once(self):
        bag_of_words = self.__bag_of_words[:]
        del self.__bag_of_words
        return bag_of_words
