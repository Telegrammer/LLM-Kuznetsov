import tiktoken
import torch
from torch.utils.data import random_split

from .AbstractTransformerTextLoader import AbstractTransformerTextLoader, DataLoader
from .GPTTextDataset import GPTTextDataset

__all__ = ["GPTTextLoader"]


class GPTTextLoader(AbstractTransformerTextLoader):

    def __init__(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int,
                 batch_size: int):
        AbstractTransformerTextLoader.__init__(self, path, tokenization_method, segmentation_method, sequence_length,
                                               batch_size)
        print()
        print("Init of Text loader")
        self._loaders = self._load_data(path, tokenization_method, segmentation_method, sequence_length)

    def _load_data(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int) -> \
            dict[str, DataLoader]:

        print("CREATING TRAIN DATA")
        train_data = GPTTextDataset(path, tokenization_method, segmentation_method, sequence_length)
        print("DONE")

        self._bag_size = 50304
        # magic number because it's divisible by many numbers that looks like 2^n,
        # which provides less work for gpu to spread data in

        train_data, val_data = random_split(train_data, [0.7, 0.3])
        return {'train': DataLoader(train_data, batch_size=self._batch_size, shuffle=True),
                'val': DataLoader(val_data, batch_size=self._batch_size, shuffle=False)}

    def convert_to_words(self, indexes: torch.LongTensor) -> str:
        encoding = tiktoken.get_encoding("p50k_base")
        return encoding.decode(list(indexes))
