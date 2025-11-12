import data_loader
import torch
from attention import SelfAttention, CausalAttention, MultiHeadAttention

vocab_size = 50257
output_dim = 256
chunk_size = 4

if __name__ == "__main__":
    with open("verdict.txt", "r") as file:
        text = file.read()

    dataloader = data_loader.create_data_loader(
        text, batch_size=8, stride=chunk_size, shuffle=False, chunk_size=chunk_size)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    torch.manual_seed(0)
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = chunk_size
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    attention = MultiHeadAttention(output_dim, output_dim, context_length, dropout=0.0, num_heads=8)
    context_vectors = attention.forward(input_embeddings)
    print(context_vectors.shape)
    
