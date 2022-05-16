import os
from config import get_config

if __name__ == "__main__":
    config = get_config()
    out = []
    with open(f"{config.dataset_dir}/train.csv") as f:
        for line in f.readlines():
            file_name, index = line.split(',')
            if index == "10-1\n":
                index = "0\n"
            elif index == "10-2\n":
                index = "10\n"

            out += [f"{file_name},{index}"]

    with open(f"{config.dataset_dir}/train.csv", "w") as f:
        f.write("".join(out))


