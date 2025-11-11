import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=False)
        self.W_key = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=False)
        self.W_value = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=False)
        
    def forward(self, inputs):
        query = inputs @ self.W_query
        key = inputs @ self.W_key
        value = inputs @ self.W_value
        
        attention_scores = query @ key.transpose(-2, -1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = attention_weights @ value
        
        return context_vector
        