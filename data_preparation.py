with open("verdict.txt", "r") as file:
    data = file.read()
    print(f"no of characters: {len(data)}")
    print(f"no of words: {len(data.split())}")
    print(data[:99])
    