import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        
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

        context_length = attention_scores.shape[1]
        causal_mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked_attention_scores = attention_scores.masked_fill(causal_mask.bool(), -torch.inf)
        attention_weights = torch.softmax(masked_attention_scores / torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1)
        context_vector = attention_weights @ values
        
        return context_vector 