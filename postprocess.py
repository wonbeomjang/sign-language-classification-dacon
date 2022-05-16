import os
from config import get_config

if __name__ == "__main__":
    config = get_config()
    out = []
    with open(f"submission.csv") as f:
        for line in f.readlines():
            file_name, index = line.split(',')
            if index == "0\n":
                index = "10-1\n"
            elif index == "10\n":
                index = "10-2\n"

            out += [f"{file_name},{index}"]

    with open(f"submission.csv", "w") as f:
        f.write("".join(out))


