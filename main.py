import torch

from data_load import TransformerTextLoader
from train import TorchTeacher
from transformer import Transformer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_sequence_length = 32
    loader = TransformerTextLoader("content/texts/input.txt", input_sequence_length, "tiktoken",
                                   "shifted_tokens", False)
    bag_size = loader.get_bag_size()
    token_dim = 768
    enc_head_count = 0
    dec_head_count = 12
    batch_size = input_sequence_length * 2 + 1

    print("creating model...")
    model = Transformer(device, (0, 12, bag_size, token_dim, input_sequence_length, batch_size),
                        enc_head_count, dec_head_count).to(device)
    print(model.get_count_parameters())

    teacher = TorchTeacher(loader, education_speed=0.0001, device=device)
    teacher.teach_model(model)
