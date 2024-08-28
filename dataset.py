import numpy as np
import enum
#from tensorflow.keras.datasets import mnist

__all__ = ["load_data", "data_pair", "dataset", "DatasetType"]

data_pair = tuple[np.ndarray, int]
dataset = tuple[data_pair, data_pair]


class DatasetType(enum.IntEnum):
    train = 0
    test = 1


def load_data() -> dataset:
    pass
    #return mnist.load_data()
