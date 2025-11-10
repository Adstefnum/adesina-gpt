import re
from tokenizer import SimpleTokenizer

with open("verdict.txt", "r") as file:
    data = file.read()
    print(f"no of characters: {len(data)}")
    print(f"no of words: {len(data.split())}")
    # print(data[:99])


    # tokenize with regex, remove space
    tokens = re.split(r'(\s+)|([,.;:?_()!"\']|--)', data)
    tokens = [token.strip() for token in tokens if token is not None and token.strip()]

    # convert into token ids
    all_words = sorted(set(tokens))
    vocab_size = len(all_words)
    print(f"vocab size: {vocab_size}")
    vocab = {token: idx for idx, token in enumerate(all_words)}

    tokenizer = SimpleTokenizer(vocab)
    tokens = tokenizer.encode(data)
    print(f"no of tokens: {len(tokens)}")
    text = """"It's the last he painted, you know, " Mrs. Gisburn said with pardonable pride."""
    token_ids = tokenizer.encode(text)
    print(token_ids)
    print(tokenizer.decode(token_ids))