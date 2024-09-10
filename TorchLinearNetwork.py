from AbstractLinearNetwork import *

__all__ = ["TorchLinearNetwork"]


class TorchLinearNetwork(AbstractLinearNetwork, nn.Module):

    def __init__(self, topology: tuple[int], activation_function='gelu'):
        AbstractLinearNetwork.__init__(self, topology)

        self.__activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(approximate="tanh"),
            'softmax': nn.Softmax()
        }

        self.__layers = nn.ModuleDict()
        self.__layers.add_module(name='input_layer', module=nn.Linear(topology[0], topology[1]))
        self.__layers.add_module(name='input_layer_act', module=self.__activations[activation_function])

        for i in range(1, len(topology) - 2):
            self.__layers.add_module(name=f'hidden_layer_{i}_dropout', module=nn.Dropout())
            self.__layers.add_module(name=f'hidden_layer_{i}', module=nn.Linear(topology[i], topology[i + 1]))
            self.__layers.add_module(name=f'hidden_layer_{i}_act', module=self.__activations[activation_function])
            self.__layers.add_module(name=f'hidden_layer_{i}_dropout', module=nn.Dropout(0.5))

        self.__layers.add_module(name='output_layer', module=nn.Linear(topology[-2], topology[-1]))

    def forward(self, input_tensor: torch.Tensor) -> [torch.Tensor, np.ndarray]:
        out = input_tensor.detach().clone()
        assert out is not input_tensor

        out = self.__layers['input_layer'](out)
        out = self.__layers['input_layer_act'](out)
        for i in range(1, len(self._topology) - 2):
            out = self.__layers[f'hidden_layer_{i}'](out)
            out = self.__layers[f'hidden_layer_{i}_act'](out)
            out = self.__layers[f'hidden_layer_{i}_dropout'](out)
        out = self.__layers['output_layer'](out)

        return out
