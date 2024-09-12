import tiktoken
import torch
from torch.utils.data import random_split

from .AbstractTransformerTextLoader import AbstractTransformerTextLoader, DataLoader
from .GPTDataset import GPTTextDataset

__all__ = ["GPTTextLoader"]


class GPTTextLoader(AbstractTransformerTextLoader):

    def __init__(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int,
                 batch_size: int):
        AbstractTransformerTextLoader.__init__(self, path, tokenization_method, segmentation_method, sequence_length,
                                               batch_size)
        self._loaders = self._load_data(path, tokenization_method, segmentation_method, sequence_length)

    def _load_data(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int) -> \
            dict[str, DataLoader]:
        train_data = GPTTextDataset(path, tokenization_method, segmentation_method, sequence_length)

        self._bag_size = 50000 + 256 + 1

        train_data, val_data = random_split(train_data, [1, 0])
        return {'train': DataLoader(train_data, batch_size=self._batch_size, shuffle=True),
                'val': DataLoader(val_data, batch_size=self._batch_size, shuffle=False)}

    def convert_to_words(self, indexes: torch.LongTensor) -> str:
        encoding = tiktoken.get_encoding("p50k_base")
        return encoding.decode(indexes)
