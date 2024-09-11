import torch
from torch.utils.data import random_split, DataLoader

from .TransformerTextDataset import TransformerTextDataset

__all__ = ["TransformerTextLoader"]


class TransformerTextLoader:

    def __init__(self,
                 path: str,
                 batch_size: int,
                 tokenization_method: str,
                 batching_method: str,
                 need_transform: bool = True):

        if tokenization_method == "tiktoken":
            need_transform = True

        train_data = TransformerTextDataset(path, tokenization_method, batching_method, batch_size)
        bag_of_words = train_data.get_bag_of_words_once()

        self.__index2word = {index: word for index, word in enumerate(bag_of_words)}
        self.__word2index = {word: index for index, word in enumerate(bag_of_words)}
        self.__data_transformed = need_transform
        self.__bag_size = len(bag_of_words)
        self.__batch_size = batch_size

        for i in range(len(train_data)):
            print(train_data[i])
            train_data[i][0] = torch.LongTensor([self.__word2index[word] for word in train_data[i][0]])
            train_data[i][1] = torch.LongTensor([self.__word2index[word] for word in train_data[i][1]])
            train_data[i][1] = torch.eye(self.__bag_size)[train_data[i][1]]

        train_data, val_data = random_split(train_data, [1, 0])
        self.__loaders = {'train': DataLoader(train_data, batch_size=1, shuffle=True),
                          'val': DataLoader(val_data, batch_size=1, shuffle=False)}

    def __getitem__(self, loader_type: str):
        return self.__loaders[loader_type]

    def convert_sample(self, words, device: str) -> torch.LongTensor:
        if self.__data_transformed:
            return words.to(device)
        else:
            return torch.LongTensor(words).to(device)

    def convert_target(self, words, device: str) -> torch.LongTensor:
        if self.__data_transformed:
            return words.to(device)
        else:
            target = torch.LongTensor(words)
            target = torch.eye(self.__bag_size)[target].to(device)
            return target

    def convert_to_words(self, indexes: torch.LongTensor) -> list[str]:
        return [self.__index2word[idx.item()] for idx in indexes]

    def get_classes_count(self):
        return self.__bag_size

    def get_batch_size(self):
        return self.__batch_size

    def get_bag_size(self):
        return self.__bag_size
