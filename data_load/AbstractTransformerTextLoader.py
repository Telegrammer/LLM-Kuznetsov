from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

__all__ = ["AbstractTransformerTextLoader", "DataLoader"]


class AbstractTransformerTextLoader:

    def __init__(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int,
                 batch_size: int):
        self._bag_size: int = property()
        self._batch_size: int = batch_size
        self._loaders: dict[str, DataLoader] = self._load_data(path, tokenization_method, segmentation_method,
                                                               sequence_length)

    def __getitem__(self, loader_type: str):
        return self._loaders[loader_type]

    @abstractmethod
    def _load_data(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int) -> \
            dict[str, DataLoader]:
        pass

    @abstractmethod
    def convert_to_words(self, indexes: torch.LongTensor) -> list[str]:
        pass

    def get_classes_count(self) -> int:
        return self._bag_size

    def get_batch_size(self) -> int:
        return self._batch_size
