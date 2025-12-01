import torch
from gpt import GPT
from config import config
from utils import plot_losses, train_model
import data_loader
import tiktoken

torch.manual_seed(123)
model = GPT(config)
model.to(config["device"])

tokenizer = tiktoken.get_encoding("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

with open("verdict.txt", "r") as file:
    text = file.read()

train_ratio = 0.9
train_size = int(len(text) * train_ratio)

train_text = text[:train_size]
test_text = text[train_size:]

train_loss_loader = data_loader.create_data_loader(
    train_text,
    batch_size=2,
    stride=config["context_length"],
    shuffle=False,
    drop_last=False,
    chunk_size=config["context_length"],
)

val_loss_loader = data_loader.create_data_loader(
    test_text,
    batch_size=2,
    stride=config["context_length"],
    shuffle=False,
    drop_last=False,
    chunk_size=config["context_length"],
)

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model(
    model,
    optimizer,
    val_loss_loader,
    train_loss_loader,
    num_epochs,
    config["device"],
    eval_freq=5,
    start_context="Every effort moves you forward",
    eval_iter=5,
    tokenizer=tokenizer,
)
torch.save({
    "model_state_dict" : model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict()
}, "model_and_optim.pth")
epoch_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epoch_tensor, tokens_seen, train_losses, val_losses)

