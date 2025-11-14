import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, inputs):
        pass

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        pass

    def forward(self, inputs):
        pass

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embedding_layer = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
        self.layernorm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, inputs):
        batch_size, seq_length = inputs.shape
        token_embeddings = self.token_embedding_layer(inputs)
        position_embeddings = self.position_embedding_layer(torch.arange(seq_length),device=inputs.device)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.layernorm(x)
        logits = self.out_head(x)
        return logits
    
