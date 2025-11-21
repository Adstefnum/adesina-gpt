from utils import text_to_tokens, tokens_to_text, generate_text
import tiktoken
from config import config
from gpt import GPT
import torch

torch.manual_seed(123)

model = GPT(config)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text(
    model=model,
    input_ids=text_to_tokens(tokenizer, start_context),
    max_new_tokens=10,
    context_length=config["context_length"]
)

print("Output text:\n", tokens_to_text(token_ids, tokenizer))



