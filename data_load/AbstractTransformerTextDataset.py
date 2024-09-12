from abc import ABC, abstractmethod
from typing import Callable
from typing import TypeVar, Generic

from torch.utils.data import Dataset

_T = TypeVar('_T')

__all__ = ["AbstractTransformerTextDataset", "Callable", "TypeVar"]


class T(Generic[_T]):
    pass


class U(Generic[_T]):
    pass


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
