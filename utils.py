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

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_tokens(tokenizer, start_context).to(device)
    with torch.no_grad():
        token_ids = generate_text(model,encoded, 50, context_size)
    decoded = tokens_to_text(token_ids, tokenizer)
    print(decoded)
    model.train()

def train_model(model, optimizer, val_loss_loader, train_loss_loader, num_epochs, device, eval_freq, start_context, eval_iter, tokenizer):
    train_losses, val_losses, tokens_seen_list = [], [], []
    tokens_seen, global_step = 0,-1

    for i in range(num_epochs):
        model.train()
        for inputs, targets in train_loss_loader:
            optimizer.zero_grad()
            loss = calc_batch_loss(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inputs.numel()
            global_step += 1

        if global_step % eval_freq == 0:
            val_loss, train_loss = evaluate_model(model, train_loss_loader, val_loss_loader, device, eval_iter)
            val_losses.append(val_loss)
            train_losses.append(train_loss)
            tokens_seen_list.apend(tokens_seen)
            print(f"""
            Epoch {i+1}\n
            Step {global_step}
            Train loss {train_loss}\n
            Val_loss {val_loss}
            """)

    generate_and_print_sample(model,tokenizer,device,start_context)
    return train_losses, val_losses, tokens_seen_list

        


    
    
            