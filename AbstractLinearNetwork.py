from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch
import torch.nn as nn

__all__ = ["AbstractLinearNetwork", "np", "deque", "torch", "nn"]


class AbstractLinearNetwork(ABC, nn.Module):
    def __init__(self, topology: tuple[int]):
        nn.Module.__init__(self)
        self._topology = topology
        self._layers = nn.ModuleDict()

    @abstractmethod
    def forward(self, **kwargs) -> [torch.Tensor, np.ndarray]:
        pass
