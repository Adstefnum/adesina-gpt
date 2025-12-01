from gpt_download import download_and_load_gpt2
from config import config as GPT_CONFIG_124M
from gpt import GPT as GPTModel
from utils import load_weights_into_gpt, tokens_to_text, text_to_tokens, generate_text_random
import tiktoken
import torch

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

NEW_CONFIG.update({"context_length": 1024})

NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt,params)

torch.manual_seed(123)

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_random(
    model=gpt,
    input_ids=text_to_tokens(tokenizer, start_context),
    max_tokens=25,
    context_length=NEW_CONFIG["context_length"],
    top_k=50,
    temp=1.5
)

print("Output text:\n", tokens_to_text(token_ids, tokenizer))