import os

import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

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

    train_metadata = pd.read_csv("dataset/train.csv")

    oversample = RandomOverSampler(random_state=0)
    temp_train_metadata, y = oversample.fit_resample(train_metadata.drop(columns=["label"]), train_metadata["label"])
    temp_train_metadata["label"] = y
    train_metadata = temp_train_metadata

    train_metadata.to_csv("dataset/train.csv")
