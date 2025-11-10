import dis
import re
from tokenizer import SimpleTokenizer
import tiktoken

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
    all_words.extend([ "<|endoftext|>","<|unk|>"])
    vocab_size = len(all_words)
    print(f"vocab size: {vocab_size}")
    vocab = {token: idx for idx, token in enumerate(all_words)}

    # tokenizer = SimpleTokenizer(vocab)
    tokenizer = tiktoken.get_encoding("gpt2")
    # text = """"It's the last he painted, you know, " Mrs. Gisburn said with pardonable pride."""
    # text1 = "Hello, do you like tea?"
    # text2 = "In the sunlit terraces of the palace."
    # text = " <|endoftext|> ".join([text1, text2])
    text = "Akwirw ier"
    token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    print(text)
    print(token_ids)
    for token_id in token_ids:
        print(tokenizer.decode([token_id]))