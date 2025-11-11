import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=False)
        self.W_key = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=False)
        self.W_value = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=False)
        
    def forward(self, inputs):
        queries = inputs @ self.W_query
        keys = inputs @ self.W_key
        values = inputs @ self.W_value
        
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1)
        context_vector = attention_weights @ values
        
        return context_vector
        