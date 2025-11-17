import data_loader
import torch
from attention import  MultiHeadAttention
from config import config
from gpt import GPT

chunk_size = 4

if __name__ == "__main__":
    with open("verdict.txt", "r") as file:
        text = file.read()

    dataloader = data_loader.create_data_loader(
        text, batch_size=2, stride=chunk_size, shuffle=False, chunk_size=chunk_size)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    torch.manual_seed(0)
    model = GPT(config)
    logits = model(inputs)
    print(logits.shape)  

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    total_bytes = total_params * 4
    print(f"Total bytes: {total_bytes:,}")
    print(f"Total MB: {total_bytes / 1024 / 1024}")

    # attention = MultiHeadAttention(output_dim, output_dim, context_length, dropout=0.0, num_heads=8)
    # context_vectors = attention.forward(input_embeddings)
    # print(context_vectors.shape)
    
