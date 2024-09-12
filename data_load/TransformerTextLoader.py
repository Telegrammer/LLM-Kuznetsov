import torch
from torch.utils.data import random_split, DataLoader

from .AbstractTransformerTextLoader import AbstractTransformerTextLoader
from .TransformerTextDataset import TransformerTextDataset

__all__ = ["TransformerTextLoader"]


class TransformerTextLoader(AbstractTransformerTextLoader):

    def __init__(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int,
                 batch_size: int):
        AbstractTransformerTextLoader.__init__(self, path, tokenization_method, segmentation_method, sequence_length,
                                               batch_size)
        self._loaders = self._load_data(path, tokenization_method, segmentation_method, sequence_length)
        self.__index2word: dict[int, str] = property()
        self.__word2index: dict[str, int] = property()

    def _load_data(self, path: str, tokenization_method: str, segmentation_method: str, sequence_length: int) -> \
            dict[str, DataLoader]:
        train_data: TransformerTextDataset = TransformerTextDataset(path, tokenization_method, segmentation_method,
                                                                    sequence_length)
        bag_of_words = train_data.get_bag_of_words_once()

        self.__index2word = {index: word for index, word in enumerate(bag_of_words)}
        self.__word2index = {word: index for index, word in enumerate(bag_of_words)}
        self._bag_size = len(bag_of_words)

        for i in range(len(train_data)):
            print(train_data[i])
            train_data[i][0] = torch.LongTensor([self.__word2index[word] for word in train_data[i][0]])
            train_data[i][1] = torch.LongTensor([self.__word2index[word] for word in train_data[i][1]])

        train_data, val_data = random_split(train_data, [1, 0])
        self.__loaders = {'train': DataLoader(train_data, batch_size=1, shuffle=True),
                          'val': DataLoader(val_data, batch_size=1, shuffle=False)}

    def convert_to_words(self, indexes: torch.LongTensor) -> str:
        return "".join([self.__index2word[idx.item()] for idx in indexes])
