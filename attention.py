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
        assert (output_dim % num_heads == 0), "Output dimension must be divisible by number of heads"
        self.head_dim = output_dim // num_heads
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.W_query = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.context_length = context_length
        self.out_proj = torch.nn.Linear(output_dim, output_dim)
        self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, inputs):
        batch_size, num_tokens, input_dim = inputs.shape

        keys = self.W_key(inputs)
        values = self.W_value(inputs)
        queries = self.W_query(inputs)
        
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        attention_scores = queries @ keys.transpose(-2,-1)
        masked_attention_scores = attention_scores.masked_fill(self.causal_mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        attention_weights = torch.softmax(masked_attention_scores / torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1)
        context_vector = (self.dropout(attention_weights) @ values).transpose(1,2)
        
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.output_dim)
        context_vector = self.out_proj(context_vector)
        
        return context_vector
        

