import torch
import os

from data_load.GPTTextLoader import GPTTextLoader
from train import TorchTeacher
from transformer import Transformer


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequence_length = 16
    batch_size = 4
    loader = GPTTextLoader("content/texts/input.txt", "tiktoken", "shifted_tokens", sequence_length, batch_size)
    bag_size = loader.get_classes_count()
    token_dim = 768
    enc_head_count = 0
    dec_head_count = 12
    max_buffer = sequence_length * 2

    print("creating model...")
    model = Transformer(device, (0, 12, bag_size, token_dim, sequence_length, max_buffer),
                        enc_head_count, dec_head_count).to(device)
    # model = torch.compile(model)

    teacher = TorchTeacher(loader, education_speed=0.0001, device=device)
    teacher.teach_model(model)
