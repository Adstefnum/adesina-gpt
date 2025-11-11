import data_loader


if __name__ == "__main__":
    with open("verdict.txt", "r") as file:
        text = file.read()

    dataloader = data_loader.create_data_loader(
        text, batch_size=8, stride=4, shuffle=False, chunk_size=4)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
