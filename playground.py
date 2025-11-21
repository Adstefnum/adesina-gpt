from utils import text_to_tokens, tokens_to_text, generate_text, calc_loss_loader
import tiktoken
from config import config
from gpt import GPT
import torch
import data_loader

torch.manual_seed(123)

model = GPT(config)
model.eval()

# start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

# token_ids = generate_text(
#     model=model,
#     input_ids=text_to_tokens(tokenizer, start_context),
#     max_new_tokens=10,
#     context_length=config["context_length"]
# )

# print("Output text:\n", tokens_to_text(token_ids, tokenizer))

with open("verdict.txt", "r") as file:
    text = file.read()

train_ratio = 0.9
train_size = int(len(text) * train_ratio)

train_text = text[:train_size]
test_text = text[train_size:]

train_dataloader = data_loader.create_data_loader(
    train_text, batch_size=2, stride=config["context_length"], shuffle=False,drop_last=False, chunk_size=config["context_length"])
print("Train loader:")
for x,y in train_dataloader:
    print(x.shape, y.shape)

test_dataloader = data_loader.create_data_loader(
    test_text, batch_size=2, stride=config["context_length"], shuffle=False,drop_last=False, chunk_size=config["context_length"])
print("Test loader:")
for x,y in test_dataloader:
    print(x.shape, y.shape)

device = torch.device('cpu')
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_dataloader, model, device)
    test_loss = calc_loss_loader(test_dataloader, model, device)

print(f"Training loss:{train_loss}")
print(f"Test loss {test_loss}")




