import numpy as np

__all__ = ["sigmoid", "sigmoid_deriv", "softmax", "ReLU", "sparse_cross_entropy", "relu_deriv", "normalize"]


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    out = np.exp(x)
    return out / np.sum(out)


def ReLU(x: np.ndarray):
    return np.maximum(x, 0)


def sparse_cross_entropy(network_output: np.ndarray, expected_result: int) -> float:
    return -np.log(network_output[expected_result, 0])


def relu_deriv(x):
    return (x >= 0).astype(float)


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def normalize(data):
    """Min-Max normalization: rescale to [0,1]"""
    data_min = data.min(axis=1).reshape((-1, 1))
    data_max = data.max(axis=1).reshape((-1, 1))
    data_range = (data_max - data_min) + (data_max == data_min)
    return (data - data_min) / data_range
