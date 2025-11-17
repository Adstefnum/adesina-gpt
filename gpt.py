import torch
import torch.nn as nn
from attention import MultiHeadAttention

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4*config["emb_dim"]),
            GELU(),
            nn.Linear(4*config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, inputs):
        return self.layers(inputs)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim=config["emb_dim"], output_dim=config["emb_dim"], context_length=config["context_length"], dropout=config["dropout_rate"], num_heads=config["n_heads"], qkv_bias=config["qkv_bias"])
        self.feed_forward = FeedForward(config)
        self.layernorm1 = LayerNorm(config["emb_dim"])
        self.layernorm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, inputs):
        shortcut = inputs
        inputs = self.layernorm1(inputs)
        attention_output = self.attention(inputs)
        inputs = self.dropout(attention_output)
        inputs = inputs + shortcut

        shortcut = inputs
        inputs = self.layernorm2(inputs)
        feed_forward_output = self.feed_forward(inputs)
        inputs = self.dropout(feed_forward_output)
        inputs = inputs + shortcut
        return inputs
    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        var = inputs.var(dim=-1, keepdim=True,unbiased=False)
        norm = (inputs - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


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
        position_embeddings = self.position_embedding_layer(torch.arange(seq_length,device=inputs.device))
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.layernorm(x)
        logits = self.out_head(x)
        return logits
    
