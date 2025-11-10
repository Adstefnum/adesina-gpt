import re

with open("verdict.txt", "r") as file:
    data = file.read()
    print(f"no of characters: {len(data)}")
    print(f"no of words: {len(data.split())}")
    # print(data[:99])


    # tokenize with regex, remove space
    tokens = re.split(r'(\s+)|([,.;:?_()!"\']|--)', data)
    tokens = [token.strip() for token in tokens if token is not None and token.strip()]
    print(f"no of tokens: {len(tokens)}")
    # print(tokens[:30])

# convert into token ids
    