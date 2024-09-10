import enum
import typing

from AbstractLinearNetwork import *
from TorchLinearNetwork import TorchLinearNetwork


class PositionalEmbeddingTopology(enum.IntEnum):
    max_length = 0
    token_dims = 1


class PositionalEmbedding(nn.Module):
    def __init__(self, device: str, topology: tuple[int]):
        super(PositionalEmbedding, self).__init__()
        self.__values = torch.arange(0, topology[PositionalEmbeddingTopology.max_length]).unsqueeze(0).T
        self.__values = self.__values.expand(topology[PositionalEmbeddingTopology.max_length],
                                             topology[PositionalEmbeddingTopology.token_dims])

        idx = torch.arange(0, topology[PositionalEmbeddingTopology.token_dims])
        idx = torch.full(idx.shape, 10000) ** (2 * idx / topology[PositionalEmbeddingTopology.token_dims])

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
    token_dims = 0
    heads_count = 1


class Encoder(AbstractLinearNetwork):

    def __init__(self, topology: tuple[int]):
        AbstractLinearNetwork.__init__(self, topology)

        self._layers.add_module("attention_layer",
                                nn.MultiheadAttention(self._topology[EncoderTopology.token_dims],
                                                      self._topology[EncoderTopology.heads_count] - self._topology[
                                                          EncoderTopology.token_dims] % self._topology[
                                                          EncoderTopology.heads_count]))

        self._layers.add_module("attention_layernorm_layer", nn.LayerNorm(self._topology[EncoderTopology.token_dims]))

        self._layers.add_module("feed_forward_layer",
                                TorchLinearNetwork((self._topology[EncoderTopology.token_dims],
                                                    self._topology[EncoderTopology.token_dims] * 4,
                                                    self._topology[EncoderTopology.token_dims])))

        self._layers.add_module("forward_layernorm_layer", nn.LayerNorm(self._topology[EncoderTopology.token_dims]))

    def forward(self, input_tensor: [torch.Tensor, torch.LongTensor]) -> [torch.Tensor, np.ndarray]:
        out: torch.Tensor = input_tensor.clone()
        # multi head attention

        attention_in: torch.Tensor = out.detach().clone()
        attention_out: torch.Tensor = self._layers["attention_layer"](attention_in, attention_in, attention_in)
        # add & norm
        out += attention_out[0]
        out = self._layers["attention_layernorm_layer"](out)

        feed_forward_in: torch.Tensor = out.detach().clone()

        # forward
        feed_forward_out = self._layers["feed_forward_layer"](feed_forward_in)

        # add & norm
        out += feed_forward_out
        out = self._layers["forward_layernorm_layer"](out)
        return out

    def __repr__(self):
        return list(self._layers["attention_layer"].parameters())


class DecoderTopology(enum.IntEnum):
    token_dims = 0
    heads_count = 1


class Decoder(AbstractLinearNetwork):
    def __init__(self, topology: tuple[int]):
        AbstractLinearNetwork.__init__(self, topology)

        self._layers.add_module("masked_attention_layer",
                                nn.MultiheadAttention(self._topology[DecoderTopology.token_dims],
                                                      self._topology[DecoderTopology.heads_count]
                                                      - self._topology[DecoderTopology.token_dims]
                                                      % self._topology[DecoderTopology.heads_count])
                                )

        self._layers.add_module("masked_attention_layernorm_layer",
                                nn.LayerNorm(self._topology[DecoderTopology.token_dims]))

        self._layers.add_module("attention_layer",
                                nn.MultiheadAttention(self._topology[DecoderTopology.token_dims],
                                                      self._topology[DecoderTopology.heads_count]
                                                      - self._topology[DecoderTopology.token_dims]
                                                      % self._topology[DecoderTopology.heads_count])
                                )

        self._layers.add_module("attention_layernorm_layer",
                                nn.LayerNorm(self._topology[DecoderTopology.token_dims]))

        self._layers.add_module("feed_forward_layer",
                                TorchLinearNetwork((self._topology[EncoderTopology.token_dims],
                                                    self._topology[EncoderTopology.token_dims] * 4,
                                                    self._topology[EncoderTopology.token_dims]))
                                )

        self._layers.add_module("forward_layernorm_layer", nn.LayerNorm(self._topology[DecoderTopology.token_dims]))

    def forward(self, encoder_tensor: torch.tensor, input_tensor: [torch.tensor, torch.LongTensor], device: str) -> \
            tuple[list, torch.Tensor]:
        out: torch.Tensor = input_tensor.clone()

        # masked multi head attention

        attn_mask = (torch.triu(torch.ones(out.size(dim=0), out.size(dim=0))) == 1).to(device)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1,
                                                                                             float(0.0))
        attention_in: torch.Tensor() = input_tensor.detach().clone()
        attention_out: torch.Tensor = self._layers["masked_attention_layer"](attention_in, attention_in, attention_in,
                                                                             attn_mask=attn_mask)

        # add & norm

        out += attention_out[0]
        out = self._layers["masked_attention_layernorm_layer"](out)

        # multi head attention
        if encoder_tensor is not None:
            attention_in: torch.Tensor() = out.detach().clone()
            attention_out: torch.Tensor = self._layers["attention_layer"](attention_in, encoder_tensor, encoder_tensor)

            # add & norm
            out += attention_out[0]
            out = self._layers["attention_layernorm_layer"](out)

        # forward
        feed_forward_in: torch.Tensor = out.detach().clone()
        feed_forward_out = self._layers["feed_forward_layer"](feed_forward_in)

        # add & norm
        out += feed_forward_out
        out = self._layers["forward_layernorm_layer"](out)
        return out

    def __repr__(self):
        return list(self._layers["forward_layernorm_layer"].parameters())


class TransformerTopology(enum.IntEnum):
    encoder_count = 0
    decoder_count = 1
    bag_size = 2
    token_dims = 3
    response_length = 4
    input_token_count = 5


class Transformer(AbstractLinearNetwork):

    def __init__(self, device: str, topology: tuple[int], encoder_heads_count: int, decoder_heads_count: int):
        AbstractLinearNetwork.__init__(self, topology)
        self._layers = nn.ModuleDict()
        self.__device = device
        self.__response_length = topology[TransformerTopology.response_length]
        self.__input_token_count = topology[TransformerTopology.input_token_count]
        self._topology = topology[:TransformerTopology.token_dims]

        self._layers.add_module("encoder_embedding_layer", nn.Embedding(topology[TransformerTopology.bag_size],
                                                                        topology[TransformerTopology.token_dims]))
        self._layers.add_module("encoder_position_embedding_layer",
                                PositionalEmbedding(device, (self.__input_token_count,
                                                             topology[TransformerTopology.token_dims])))
        for i in range(topology[TransformerTopology.encoder_count]):
            self._layers.add_module(f"encoder_{i + 1}",
                                    Encoder((topology[TransformerTopology.token_dims], encoder_heads_count)))

        self._layers.add_module("decoder_embedding_layer", nn.Embedding(topology[TransformerTopology.bag_size],
                                                                        topology[TransformerTopology.token_dims]))

        if self._topology[TransformerTopology.encoder_count] == 0:
            embedding_length = self.__input_token_count
        else:
            embedding_length = self.__response_length
        self._layers.add_module("decoder_position_embedding_layer",
                                PositionalEmbedding(device, (embedding_length,
                                                             topology[TransformerTopology.token_dims])))
        for i in range(topology[TransformerTopology.decoder_count]):
            self._layers.add_module(f"decoder_{i + 1}",
                                    Decoder((topology[TransformerTopology.token_dims], decoder_heads_count)))
        self._layers.add_module("linear_layer", nn.Linear(topology[TransformerTopology.token_dims],
                                                          topology[TransformerTopology.bag_size]))

    def forward(self, input_tensor: torch.LongTensor, target: torch.Tensor = None,
                loss_function: typing.Callable = None) -> torch.Tensor:
        if self._topology[TransformerTopology.encoder_count] == 0:
            encoders_out = None
            result_string = input_tensor.detach().clone()
        else:
            encoders_out: torch.Tensor = self._layers["encoder_embedding_layer"](input_tensor.detach().clone()[:-1])
            encoders_out += self._layers["encoder_position_embedding_layer"](encoders_out.size(dim=0))
            for i in range(self._topology[TransformerTopology.encoder_count]):
                encoders_out = self._layers[f"encoder_{i + 1}"](encoders_out)
            result_string = torch.LongTensor([input_tensor[-1].item()]).to(self.__device)

        model_logits = torch.Tensor().to(self.__device)
        model_logits = model_logits.view(input_tensor.size(0), 0, self._topology[TransformerTopology.bag_size])

        generated_token_count = 0
        # result_string[-1].item() != self._topology[TransformerTopology.bag_size] - 1
        while generated_token_count != self.__response_length:
            decoders_out: torch.Tensor = self._layers["decoder_embedding_layer"](result_string.clone())
            decoders_out += self._layers["decoder_position_embedding_layer"](decoders_out.size(dim=1))
            for i in range(self._topology[TransformerTopology.decoder_count]):
                decoders_out = self._layers[f"decoder_{i + 1}"](encoders_out, decoders_out, self.__device)

            logits = self._layers["linear_layer"](decoders_out[:, -1]) / 2.0
            probabilities = nn.functional.softmax(logits)
            out_idx = torch.argmax(probabilities, dim=1).unsqueeze(1)

            logits = logits[:, None]
            model_logits = torch.cat((model_logits, logits), dim=1)
            result_string = torch.cat((result_string, torch.Tensor(out_idx).to(self.__device)), dim=1).long()
            generated_token_count += 1

        loss = None
        if target is not None:
            loss = loss_function(model_logits.view(-1, model_logits.size(-1)), target.view(-1, target.size(-1)))

        return model_logits, loss

    def set_response_length(self, value: int):
        self.__response_length = value

    def get_count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return self._layers["encoder_embedding_layer"].parameters()
