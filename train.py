import psutil
import argparse
from typing import Optional

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--backbone',
                    choices=dict(resnet18=model.ResNet18),
                    default=model.ResNet18)
parser.add_argument('--dataset',
                    choices=dict(cifar100=dataset.CIFAR100),
                    default=dataset.CIFAR100)

parser.add_argument("--image_size", type=int, default=224, help="size of train image")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=40, help="the number of epochs")
parser.add_argument('--lr_decay_epochs', type=int, default=[25, 30, 35], nargs='+', help="decay epoch")
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help="decay ratio")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--checkpoint_dir", default="checkpoint", help="check point directory")
parser.add_argument("--num_classes", type=int, default=100, help="the number of classes")
parser.add_argument("--resume", nargs='?', const=True, default=False, help="resume most recent training")
parser.add_argument("--cpu", nargs='?', default="cuda:0", const="cpu",  help="whether use gpu or net")
parser.add_argument("--data_dir", default="dataset", help="data directory")
parser.add_argument("--num_workers", type=int, default=psutil.cpu_count())
parser.add_argument("--best", type=int, default=0)

config = parser.parse_args()
config.device = torch.device(config.cpu)


def test(net: nn.Module, data_loader: DataLoader):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    device = wandb.config.device
    criterion = nn.CrossEntropyLoss().to(device)

    net = net.eval()

    pbar = tqdm(data_loader, total=len(data_loader))
    for image, target in pbar:
        image: torch.Tensor = image.to(device)
        target: torch.Tensor = target.to(device)

        predict: torch.Tensor = net(image)
        loss: torch.Tensor = criterion(predict, target)

        predict = predict.argmax(dim=1)

        loss_meter.update(loss.mean().item())
        acc_meter.update((predict == target).sum().item() / image.shape[0])
        pbar.set_description(f"Validate... Loss: {loss_meter.avg: .4f}, Acc: {acc_meter.avg: .4f}")

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

    for epoch in range(wandb.config.start_epoch, wandb.config.epoch):
        pbar = tqdm(data_loader, total=len(data_loader))
        loss_meter.reset()
        acc_meter.reset()
        for image, target in pbar:
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

            pbar.set_description(f"[{epoch + 1}/{wandb.config.epoch}] Loss: {loss_meter.avg: .4f}, "
                                 f"Acc: {acc_meter.avg: .4f}")

        result = {"train/acc": acc_meter.avg, "train/loss": loss_meter.avg}
        acc = acc_meter.avg
        if val_loader:
            with torch.no_grad():
                val_result = test(net, val_loader)
                result.update(val_result)
            acc = result["val/acc"]

        save_info = {"run_id": run_id, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(), "epoch": epoch, "best": best}
        if acc > best:
            best = acc
            torch.save(save_info, os.path.join(wandb.config.run_dir, "best.pth"))
        torch.save(save_info, os.path.join(wandb.config.run_dir, "last.pth"))

        wandb.log(result)


if __name__ == "__main__":
    net = config.backbone(pretrained=True)
    run_dir = os.path.join(os.getcwd(), config.checkpoint_dir, "run")
    attempt_make_dir(run_dir)

    net = model.LinearEmbedding(net, net.output_size, config.num_classes).to(config.device)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_epochs, gamma=config.lr_decay_gamma)

    if not config.resume:
        run_dir = os.path.join(config.checkpoint_dir, "run", "exp" + f"{len(os.listdir(run_dir)) + 1}")
        attempt_make_dir(run_dir)
        config.run_dir = run_dir
        run = wandb.init(project='knowledge_distillation', dir=run_dir, config=vars(config))
    else:
        run_dir = os.path.join(config.checkpoint_dir, "run", "exp" + f"{len(os.listdir(run_dir))}")
        save_info = torch.load(os.path.join(run_dir, "best.pth"), map_location=config.device)
        net.load_state_dict(save_info["state_dict"])
        optimizer.load_state_dict(save_info["optimizer"])
        lr_scheduler.load_state_dict(save_info["lr_scheduler"])
        config.start_epoch = save_info["epoch"] + 1
        config.best = save_info["best"]
        config.run_dir = run_dir
        run = wandb.init(id=save_info["run_id"], project='hair_segmentation', resume="allow", dir=run_dir,
                         config=vars(config))

    if False and (isinstance(net, model.InceptionV1BN) or isinstance(net, model.GoogleNet)):
        normalize = transforms.Compose([
            transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
            transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
        ])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(wandb.config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(wandb.config.image_size),
        transforms.ToTensor(),
        normalize
    ])

    dataset_train = config.dataset(wandb.config.data_dir, train=True, transform=train_transform, download=True)
    dataset_test = config.dataset(wandb.config.data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(dataset_train, batch_size=wandb.config.batch_size, shuffle=True,
                              num_workers=wandb.config.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=wandb.config.batch_size, shuffle=False,
                             num_workers=wandb.config.num_workers)

    train(net, train_loader, optimizer, lr_scheduler, wandb, run.id, test_loader)
