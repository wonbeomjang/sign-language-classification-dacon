import argparse
import torch
import psutil

parser = argparse.ArgumentParser()

parser.add_argument('--backbone', default="regnet")
parser.add_argument("--dataset_dir", default="dataset")
parser.add_argument("--val", nargs='?', const=False, default=True, help="validate")
parser.add_argument("--image_size", type=int, default=224, help="size of train image")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=40, help="the number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--checkpoint_dir", default="checkpoint", help="check point directory")
parser.add_argument("--num_classes", type=int, default=11, help="the number of classes")
parser.add_argument("--resume", nargs='?', const=True, default=False, help="resume most recent training")
parser.add_argument("--mixup", nargs='?', const=True, default=False, help="run with mixup")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--data_dir", default="dataset", help="data directory")
parser.add_argument("--num_workers", type=int, default=psutil.cpu_count())
parser.add_argument("--best", type=int, default=0)


def get_config():
    config = parser.parse_args()
    config.device = torch.device(config.device)
    return config


def get_sweep_config():
    sweep_config = {
        "name": "sign_language_sweep",
        "method": "random",
        "metric": {
            "name": "test/acc",
            "goal": "maximize"
        },
        "parameters": {
            "epoch": {
                "values": [50, 70, 90, 110]
            },
            "learning_rate": {
                "min": 5e-4,
                "max": 5e-3
            },
            "backbone": {
                "values": ["vgg16", "resnet18", "resnet34", "resnet50", "resnet101", "regnet", "resnext50", "resnext101"]
            }
        }
    }

    return sweep_config
