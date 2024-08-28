import enum

from AbstractLinearNetwork import *
from TorchLinearNetwork import TorchLinearNetwork


class PositionalEmbedding(nn.Module):
    def __init__(self, device: str, topology: tuple[int]):
        super(PositionalEmbedding, self).__init__()
        self.__values = torch.arange(0, topology[EncoderTopology.token_count]).unsqueeze(0).T
        self.__values = self.__values.expand(topology[EncoderTopology.token_count],
                                             topology[EncoderTopology.token_dims])

        idx = torch.arange(0, topology[EncoderTopology.token_dims])
        idx = torch.full(idx.shape, 10000) ** (2 * idx / topology[EncoderTopology.token_dims])

        self.__values = torch.div(self.__values, idx)
        sin_pos = torch.sin(self.__values[:, ::2])
        cos_pos = torch.cos(self.__values[:, 1::2])
        self.__values = torch.flatten(torch.stack((sin_pos, cos_pos), dim=2), start_dim=1).to(device)
        self.__values = nn.Parameter(self.__values, requires_grad=True)

    def forward(self, token_count=None) -> [torch.Tensor, np.ndarray]:
        if token_count is None:
            return self.__values
        else:
            return self.__values[:token_count]

    def __repr__(self):
        return list(self.parameters())


class EncoderTopology(enum.IntEnum):
    token_count = 0
    token_dims = 1
    heads_count = 2


class EncoderAttentionWrapper(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    @staticmethod
    def forward(attention_layer: nn.MultiheadAttention, input_tensor: [torch.Tensor, torch.LongTensor]) -> \
            [torch.Tensor, np.ndarray]:
        query: torch.Tensor = input_tensor.detach().clone()
        key: torch.Tensor = input_tensor.detach().clone()
        value: torch.Tensor = input_tensor.detach().clone()
        out: torch.Tensor = attention_layer(query, key, value)
        return out


class Encoder(AbstractLinearNetwork):

    def __init__(self, topology: tuple[int], embedding_layers: tuple[nn.Module]):
        AbstractLinearNetwork.__init__(self, topology)
        self.__layers = nn.ModuleDict()
        self.__layers.add_module("embedding_layer", embedding_layers[0])
        self.__layers.add_module("positional_embedding_layer", embedding_layers[1])

        self.__layers.add_module("encoder_attention_wrapper", EncoderAttentionWrapper())
        self.__layers.add_module("attention_layer",
                                 nn.MultiheadAttention(self._topology[EncoderTopology.token_dims],
                                                       self._topology[EncoderTopology.heads_count] - self._topology[
                                                           EncoderTopology.token_dims] % self._topology[
                                                           EncoderTopology.heads_count]))

        self.__layers.add_module("attention_layernorm_layer", nn.LayerNorm(self._topology[EncoderTopology.token_dims]))

        self.__layers.add_module("feed_forward_layer",
                                 TorchLinearNetwork((self._topology[EncoderTopology.token_dims],
                                                     self._topology[EncoderTopology.token_dims] * 4,
                                                     self._topology[EncoderTopology.token_dims])))

        self.__layers.add_module("forward_layernorm_layer", nn.LayerNorm(self._topology[EncoderTopology.token_dims]))

    def forward(self, input_tensor: [torch.Tensor, torch.LongTensor]) -> [torch.Tensor, np.ndarray]:
        out: torch.Tensor = input_tensor.detach().clone()
        out = self.__layers["embedding_layer"](out)
        out += self.__layers["positional_embedding_layer"](input_tensor.size(dim=0))

        # multi head attention

        attention_out: torch.Tensor = self.__layers["encoder_attention_wrapper"](self.__layers["attention_layer"], out)

        # add & norm
        out += attention_out[0]
        out = self.__layers["attention_layernorm_layer"](out)

        feed_forward_out: torch.Tensor = out.detach().clone()

        # forward
        feed_forward_out = self.__layers["feed_forward_layer"](feed_forward_out)

        # add & norm
        out += feed_forward_out
        out = self.__layers["forward_layernorm_layer"](out)

        return out

    def __repr__(self):
        return [matrix.__str__() for matrix in list(self.__layers["attention_layer"].parameters())]


class DecoderTopology(enum.IntEnum):
    token_count = 0
    token_dims = 1
    heads_count = 2


class DecoderAttentionWrapper(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    @staticmethod
    def forward(attention_layer: nn.MultiheadAttention, encoder_tensor: torch.Tensor,
                input_tensor: [torch.Tensor, torch.LongTensor], device: str) -> \
            [torch.Tensor, np.ndarray]:
        query: torch.Tensor = input_tensor.detach().clone()
        key: torch.Tensor = encoder_tensor.detach().clone()
        value: torch.Tensor = encoder_tensor.detach().clone()
        if encoder_tensor is input_tensor:
            attn_mask = (torch.triu(torch.ones(input_tensor.size(dim=0), encoder_tensor.size(dim=0))) == 1).to(device)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1,
                                                                                                 float(0.0))
            out: torch.Tensor = attention_layer(query, key, value, attn_mask=attn_mask)
        else:
            out: torch.Tensor = attention_layer(query, key, value)

        return out


class Decoder(nn.Module):
    def __init__(self, topology: tuple[int], bag_size: int, embedding_layers: tuple[nn.Module]):
        nn.Module.__init__(self)
        self._topology = topology
        self.__layers = nn.ModuleDict()
        self.__layers.add_module("embedding_layer", embedding_layers[0])
        self.__layers.add_module("positional_embedding_layer", embedding_layers[1])

        self.__layers.add_module("decoder_masked_attention_wrapper", DecoderAttentionWrapper())
        self.__layers.add_module("masked_attention_layer",
                                 nn.MultiheadAttention(self._topology[DecoderTopology.token_dims],
                                                       self._topology[DecoderTopology.heads_count]
                                                       - self._topology[DecoderTopology.token_dims]
                                                       % self._topology[DecoderTopology.heads_count])
                                 )

        self.__layers.add_module("attention_layernorm_layer", nn.LayerNorm(self._topology[DecoderTopology.token_dims]))

        self.__layers.add_module("decoder_attention_wrapper", DecoderAttentionWrapper())
        self.__layers.add_module("attention_layer",
                                 nn.MultiheadAttention(self._topology[DecoderTopology.token_dims],
                                                       self._topology[DecoderTopology.heads_count]
                                                       - self._topology[DecoderTopology.token_dims]
                                                       % self._topology[DecoderTopology.heads_count])
                                 )

        self.__layers.add_module("feed_forward_layer",
                                 TorchLinearNetwork((self._topology[EncoderTopology.token_dims],
                                                     self._topology[EncoderTopology.token_dims] * 4,
                                                     self._topology[EncoderTopology.token_dims]))
                                 )

        self.__layers.add_module("forward_layernorm_layer", nn.LayerNorm(self._topology[DecoderTopology.token_dims]))
        self.__layers.add_module("linear_layer",
                                 nn.Linear(self._topology[DecoderTopology.token_dims], bag_size))

    def forward(self, encoder_tensor: torch.tensor, input_tensor: [torch.tensor, torch.LongTensor], device: str) -> \
            tuple[list, torch.Tensor]:
        out: torch.Tensor = input_tensor.detach().clone()
        out = self.__layers["embedding_layer"](out)
        lol = self.__layers["positional_embedding_layer"](self._topology[DecoderTopology.token_count])
        out += lol

        # masked multi head attention

        attention_out: torch.Tensor = self.__layers["decoder_masked_attention_wrapper"](
            self.__layers["masked_attention_layer"], out, out, device)

        # add & norm

        out += attention_out[0]
        out = self.__layers["attention_layernorm_layer"](out)

        # multi head attention

        attention_out: torch.Tensor = self.__layers["decoder_attention_wrapper"](
            self.__layers["attention_layer"], encoder_tensor, out, device)
        # print(attention_out[0][-1])
        # add & norm
        out += attention_out[0]
        out = self.__layers["attention_layernorm_layer"](out)

        # forward
        feed_forward_out = self.__layers["feed_forward_layer"](out)

        # add & norm
        out += feed_forward_out
        out = self.__layers["forward_layernorm_layer"](out)

        # print(torch.sum(torch.abs(out)))
        # linear
        predict = out[-1].detach().clone()
        predict = self.__layers["linear_layer"](predict)
        predict = nn.functional.softmax(predict)
        out_word = torch.argmax(predict)

        self._topology = tuple(
            [self._topology[DecoderTopology.token_count] + 1]
            + list(self._topology[DecoderTopology.token_dims:])
        )

        return predict, out_word

    def reset(self):
        self._topology = tuple([1] + list(self._topology[1:]))

    def __repr__(self):
        return list(self.__layers["linear_layer"].parameters())


class TransformerTopology(enum.IntEnum):
    encoder_count = 0
    decoder_count = 1
    bag_size = 2
    response_length = 3
    input_token_count = 4


class Transformer(nn.Module):

    def __init__(self, device: str, topology: dict):
        nn.Module.__init__(self)
        self.__layers = nn.ModuleDict()
        self.__device = device
        self.__response_length = topology["self"][TransformerTopology.response_length]
        self.__input_token_count = topology["self"][TransformerTopology.input_token_count]
        self._topology = topology["self"][:TransformerTopology.response_length]

        self._encoder_embedding_layer = nn.Embedding(topology["encoder"][EncoderTopology.token_count],
                                                     topology["encoder"][EncoderTopology.token_dims])
        self._decoder_embedding_layer = nn.Embedding(topology["encoder"][EncoderTopology.token_count],
                                                     topology["encoder"][EncoderTopology.token_dims])

        self._encoder_position_embedding_layer = PositionalEmbedding(self.__device, (
            self.__input_token_count,
            topology["encoder"][EncoderTopology.token_dims]))

        self._decoder_position_embedding_layer = PositionalEmbedding(self.__device, (
            self.__response_length,
            topology["encoder"][EncoderTopology.token_dims]))

        for i in range(topology["self"][TransformerTopology.encoder_count]):
            self.__layers.add_module(f"encoder", Encoder(topology["encoder"],
                                                         (self._encoder_embedding_layer,
                                                          self._encoder_position_embedding_layer)))
            # for i in range(topology["self"][TransformerTopology.decoder_count]):
            self.__layers.add_module(f"decoder", Decoder(topology["decoder"],
                                                         topology["self"][TransformerTopology.bag_size],
                                                         (self._decoder_embedding_layer,
                                                          self._decoder_position_embedding_layer)))

    def forward(self, input_tensor: [torch.Tensor, torch.LongTensor]) -> [torch.Tensor, np.ndarray]:
        encoder_out = input_tensor.detach().clone()
        # for i in range(self._topology[TransformerTopology.encoder_count]):
        encoder_out = self.__layers[f"encoder"](encoder_out)

        result_string = torch.LongTensor([self._topology[TransformerTopology.bag_size] - 2]).to(
            self.__device)

        model_logits = torch.Tensor().to(self.__device)

        while result_string[-1].item() != self._topology[TransformerTopology.bag_size] - 1 and \
                result_string.size(0) != self.__response_length + 1:
            logits, out_idx = self.__layers[f"decoder"](encoder_out, result_string, self.__device)

            logits = logits[None, :]  # add dimension for cat
            model_logits = torch.cat((model_logits, logits))
            result_string = torch.cat((result_string, torch.Tensor([out_idx]).to(self.__device)), dim=0).long()

        self.__layers["decoder"].reset()

        return model_logits, result_string

    def set_response_length(self, value: int):
        self.__response_length = value

    def __repr__(self):
        return self.__layers["decoder"].__repr__()
