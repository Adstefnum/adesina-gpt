from utils import text_to_tokens, tokens_to_text, generate_text, generate_text_random, calc_loss_loader
import tiktoken
from config import config
from gpt import GPT
import torch
import data_loader

torch.manual_seed(123)

model = GPT(config)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_random(
    model=model,
    input_ids=text_to_tokens(tokenizer, start_context),
    max_tokens=15,
    context_length=config["context_length"],
    top_k=25,
    temp=1.4
)

print("Output text:\n", tokens_to_text(token_ids, tokenizer))




