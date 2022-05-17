import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from config import get_config
from data import get_testloader


def test(net: nn.Module, data_loader: DataLoader, config):
    device = config.device
    net = net.eval()

    pbar = tqdm(data_loader, total=len(data_loader))
    pbar.set_description(f"Test...")
    result = {"file_name": [], "label": []}
    for image, file_name in pbar:
        image: torch.Tensor = image.to(device)

        predict: torch.Tensor = net(image)

        predict = predict.argmax(dim=1)

        result["file_name"] += [*file_name]
        result["label"] += predict.cpu().tolist()

    return result


if __name__ == "__main__":
    config = get_config()

    run_dir = os.path.join(os.getcwd(), config.checkpoint_dir, "run")

    data_loader = get_testloader(config)

    run_dir = os.path.join(config.checkpoint_dir, "run", "exp" + f"{len(os.listdir(run_dir))}")
    save_info = torch.load(os.path.join(run_dir, "best.pth"), map_location=config.device)
    config.run_dir = run_dir

    net: nn.Module = save_info["net"]
    net.load_state_dict(save_info["state_dict"])

    result = test(net, data_loader, config)
    df = pd.DataFrame.from_dict(result)
    df.to_csv("submission.csv", index=False)

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



