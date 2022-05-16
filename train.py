from typing import Optional

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from data import get_dataloader
from utils import *
from config import get_config


def val(net: nn.Module, data_loader: DataLoader):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    device = wandb.config.device
    criterion = nn.CrossEntropyLoss().to(device)

    net = net.eval()

    for image, target in data_loader:
        image: torch.Tensor = image.to(device)
        target: torch.Tensor = target.to(device)

        predict: torch.Tensor = net(image)
        loss: torch.Tensor = criterion(predict, target)

        predict = predict.argmax(dim=1)

        loss_meter.update(loss.mean().item())
        acc_meter.update((predict == target).sum().item() / image.shape[0])

    result = {"val/acc": acc_meter.avg, "val/loss": loss_meter.avg}

    net = net.train()
    return result


def train(net: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer, lr_scheduler, wandb, run_id,
          val_loader: Optional[DataLoader]):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    device = wandb.config.device
    criterion = nn.CrossEntropyLoss().to(device)

    net = net.to(device)
    net = net.train()

    best = wandb.config.best
    pbar = tqdm(range(wandb.config.start_epoch, wandb.config.epoch))
    for epoch in pbar:
        loss_meter.reset()
        acc_meter.reset()
        for image, target in data_loader:
            image: torch.Tensor = image.to(device)
            target: torch.Tensor = target.to(device)
            predict: torch.Tensor = net(image)
            loss: torch.Tensor = criterion(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            predict = predict.argmax(dim=1)

            loss_meter.update(loss.mean().item())
            acc_meter.update((predict == target).sum().item() / image.shape[0])

        result = {"train/acc": acc_meter.avg, "train/loss": loss_meter.avg}
        acc = acc_meter.avg
        loss = loss_meter.avg
        if val_loader:
            with torch.no_grad():
                val_result = val(net, val_loader)
                result.update(val_result)
            acc = result["val/acc"]
            loss = result["val/loss"]

        save_info = {"run_id": run_id, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(), "epoch": epoch, "best": best, "net": net}
        if acc > best:
            best = acc
            torch.save(save_info, os.path.join(wandb.config.run_dir, "best.pth"))
        torch.save(save_info, os.path.join(wandb.config.run_dir, "last.pth"))

        wandb.log(result)
        pbar.set_description(f"[{epoch + 1}/{wandb.config.epoch}] Loss: {loss:.4f}, Acc: {acc:.4f}")


if __name__ == "__main__":
    config = get_config()

    run_dir = os.path.join(os.getcwd(), config.checkpoint_dir, "run")
    attempt_make_dir(run_dir)

    train_loader, val_loader = get_dataloader(config, config.val)

    net: nn.Module = model.Model(True, config.backbone, config.num_classes).to(config.device)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, config.learning_rate, config.epoch * len(train_loader))

    if not config.resume:
        run_dir = os.path.join(config.checkpoint_dir, "run", "exp" + f"{len(os.listdir(run_dir)) + 1}")
        attempt_make_dir(run_dir)
        config.run_dir = run_dir
        run = wandb.init(project='sign_language', dir=run_dir, config=vars(config))
    else:
        run_dir = os.path.join(config.checkpoint_dir, "run", "exp" + f"{len(os.listdir(run_dir))}")
        save_info = torch.load(os.path.join(run_dir, "best.pth"), map_location=config.device)
        net.load_state_dict(save_info["state_dict"])
        optimizer.load_state_dict(save_info["optimizer"])
        lr_scheduler.load_state_dict(save_info["lr_scheduler"])
        config.start_epoch = save_info["epoch"] + 1
        config.best = save_info["best"]
        config.run_dir = run_dir
        run = wandb.init(id=save_info["run_id"], project='sign_language', resume="allow", dir=run_dir,
                         config=vars(config))

    train(net, train_loader, optimizer, lr_scheduler, wandb, run.id, val_loader)
