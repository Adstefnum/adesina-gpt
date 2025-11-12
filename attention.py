import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.context_length = context_length
        self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, inputs):
        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)
        
        attention_scores = queries @ keys.transpose(-2,-1)
        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1)
        context_vector = attention_weights @ values
        
        return context_vector

class CausalAttention(SelfAttention):
    def forward(self, inputs):
        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)
        
        attention_scores = queries @ keys.transpose(-2,-1)

        masked_attention_scores = attention_scores.masked_fill(self.causal_mask.bool(), -torch.inf)
        attention_weights = torch.softmax(masked_attention_scores / torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1)

        context_vector = self.dropout(attention_weights) @ values
        
        return context_vector 

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList([CausalAttention(input_dim, output_dim, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, inputs):
        return torch.cat([head(inputs) for head in self.heads], dim=-1)