import torch

from data_load.GPTTextLoader import GPTTextLoader
from train import TorchTeacher, LRScheduler
from transformer import Transformer

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 10
    total_batch_size = 2**16
    sequence_length = 512
    batch_size = 8
    accumulation_steps = total_batch_size // (sequence_length * batch_size)

    print("loading dataset...")
    loader = GPTTextLoader("content/texts/000_00000.parquet", "tiktoken", "shifted_tokens", sequence_length, batch_size)
    print("loaded.")

    max_steps = epochs*len(loader["train"])
    warmup_steps = int(max_steps * 0.07)
    lr_scheduler = LRScheduler(warmup_steps=warmup_steps, max_steps=epochs*len(loader["train"]))
    bag_size = loader.get_classes_count()
    token_dim = 768
    enc_head_count = 0
    dec_head_count = 12
    max_buffer = sequence_length * 2

    print("creating model...")
    model: Transformer = Transformer(device, (0, 12, bag_size, token_dim, sequence_length, max_buffer),
                                     enc_head_count, dec_head_count).to(device)
    print(model.get_count_parameters())
    teacher = TorchTeacher(loader, epochs=epochs, accumulation_steps=accumulation_steps, device=device)
    teacher.teach_model(model)
