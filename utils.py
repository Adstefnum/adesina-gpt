import torch

def generate_text(model, input_ids, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        cropped_input_ids = input_ids[:, -context_length:]
        with torch.no_grad():
            logits = model(cropped_input_ids)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids

def text_to_tokens(tokenizer, text):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def tokens_to_text(token_ids, tokenizer):
    tokens = token_ids.squeeze(0)
    return tokenizer.decode(tokens.tolist())

def calc_batch_loss(input_ids, target_ids, model, device):
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_ids.flatten())
    return loss

def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float('nan')

    elif num_batches is None:
        num_batches = len(dataloader)

    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_ids, target_ids) in enumerate(dataloader):
        loss = calc_batch_loss(input_ids, target_ids, model, device)
        if i < num_batches:
            total_loss += loss.item()
        else:
            break
    return total_loss/ num_batches

    
    
            