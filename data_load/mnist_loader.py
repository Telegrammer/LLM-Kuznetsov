import torchvision

__all__ = ["write_dataset", "load_data"]

import os
import numpy as np

from PIL import Image

import struct

from array import array
from os import path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from .MNISTDataset import MNISTDataset

train_dataset = torchvision.datasets.MNIST(root='./content/sample_data/', train=True, download=False)
test_dataset = torchvision.datasets.MNIST(root='./content/sample_data/', train=False, download=False)


def read(dataset):
    if dataset == "training":
        path_img = './content/sample_data/MNIST/raw/train-images-idx3-ubyte'
        path_lbl = './content/sample_data/MNIST/raw/train-labels-idx1-ubyte'

    elif dataset == "testing":
        path_img = './content/sample_data/MNIST/raw/t10k-images-idx3-ubyte'
        path_lbl = './content/sample_data/MNIST/raw/t10k-labels-idx1-ubyte'

    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(path_lbl, 'rb') as f_lable:
        _, size = struct.unpack(">II", f_lable.read(8))
        lbl = array('b', f_lable.read())

    with open(path_img, 'rb') as f_img:
        _, size, rows, cols = struct.unpack(">IIII", f_img.read(16))
        img = array("B", f_img.read())

    return lbl, img, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_dir):
    classes = {i: f"class_{i}" for i in range(10)}
    output_dirs = [
        path.join(output_dir, classes[i])
        for i in range(10)
    ]
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    # write data
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".jpg")
        print("writting " + output_filename)

        with open(output_filename, "wb") as h:
            data_i = [
                data[(i * rows * cols + j * cols): (i * rows * cols + (j + 1) * cols)]
                for j in range(rows)
            ]
            data_array = np.array(data_i)

            im = Image.fromarray(data_array)
            im.save(output_filename)


def make_dataset():
    output_path = "./content/mnist"
    for dataset in ["training", "testing"]:
        write_dataset(*read(dataset), path.join(output_path, dataset))


def load_data():
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.ToTensor()
            #v2.Normalize(mean=(0.5,), std=(0.5,))
        ]
    )

    # создание датасетов
    train_data = MNISTDataset('./content/mnist/training', transform=transform)
    test_data = MNISTDataset('./content/mnist/testing', transform=transform)

    train_data, val_data = random_split(train_data, [0.7, 0.3])

    # создание загрузчиков
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}
