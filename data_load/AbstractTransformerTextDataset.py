from abc import ABC, abstractmethod
from typing import Callable
from typing import TypeVar, Generic

import pandas as pd
from torch.utils.data import Dataset

_T = TypeVar('_T')

__all__ = ["AbstractTransformerTextDataset", "Callable", "TypeVar", "RawTextDataset"]


class T(Generic[_T]):
    pass


class U(Generic[_T]):
    pass


class V(Generic[_T]):

    def __getitem__(self, item):
        pass

    def __delitem__(self, key):
        pass


class RawTextDataset:
    def __init__(self, path: str):
        print("constructor called")
        self.__load_methods = {"txt": self.__load_text_file, "parquet": self.__load_parquet}
        file_format: str = path[path.find(".") + 1:]
        self.__current_index: int = 0
        self.__length: int = 0
        self.__text_parts: V = None
        self.__load_methods[file_format](path)

    def __iter__(self):
        print("__iter__ check")
        return self

    def __load_text_file(self, path: str) -> None:
        with open(path, mode="r", encoding="utf-8") as file:
            self.__text_parts: list[str] = [part.replace("\n", " ")[:] for part in file.read().split("\n\n")]
            self.__length = len(self.__text_parts)

    def __load_parquet(self, path: str) -> None:
        print("reading parquet")
        self.__text_parts: pd.Series = pd.read_parquet(path, engine="fastparquet").loc[:10000, "text"]
        self.__length = len(self.__text_parts.index)

    def __next__(self):

        print("Iterating", self.__current_index, id(self))
        if self.__current_index != self.__length:
            self.__current_index += 1
            result = self.__text_parts[self.__current_index - 1][:]
            del self.__text_parts[self.__current_index - 1]
            return result
        else:
            raise StopIteration

    def __del__(self):
        print("Destructor called")


class AbstractTransformerTextDataset(T, U, ABC, Dataset):

    def __init__(self):
        self._len_dataset: int = property()
        self._data_list: list[U] = property()
        self._tokenization_methods: dict[str, Callable[[list[list[str]]], list[list[T]]]] = property()
        self._segmentation_methods: dict[str, Callable[[list[list[T]], int], None]] = property()

    def __len__(self) -> int:
        return self._len_dataset

    def __getitem__(self, index: int) -> list[U]:
        return self._data_list[index]

    def __setitem__(self, index: int, value: list[U]) -> None:
        self._data_list[index] = value

    @abstractmethod
    def _split_by_tokens(self, chunks: list[list[T]], sequence_length: int) -> None:
        pass

    @abstractmethod
    def _split_by_shifted_tokens(self, chunks: list[list[T]], sequence_length: int) -> None:
        pass

    def _make_dataset(self) -> None:
        assert len(self._data_list) != 0
        if len(self._data_list) % 2 != 0:
            self._data_list = self._data_list[:-1]

        self._len_dataset = len(self._data_list) // 2
        self._data_list = [[self._data_list[i], self._data_list[i + 1]] for i in range(0, len(self._data_list), 2)]

    @abstractmethod
    def __str__(self):
        return "class AbstractTransformerTextDataset"
