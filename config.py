import argparse
import torch
import psutil

parser = argparse.ArgumentParser()

parser.add_argument('--backbone', default="regnet")
parser.add_argument("--dataset_dir")
parser.add_argument("--val", nargs='?', const=False, default=True, help="validate")
parser.add_argument("--image_size", type=int, default=156, help="size of train image")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=40, help="the number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--checkpoint_dir", default="checkpoint", help="check point directory")
parser.add_argument("--num_classes", type=int, default=100, help="the number of classes")
parser.add_argument("--resume", nargs='?', const=True, default=False, help="resume most recent training")
parser.add_argument("--cpu", nargs='?', default="cuda:0", const="cpu",  help="whether use gpu or net")
parser.add_argument("--data_dir", default="dataset", help="data directory")
parser.add_argument("--num_workers", type=int, default=psutil.cpu_count())
parser.add_argument("--best", type=int, default=0)


def get_config():
    config = parser.parse_args()
    config.device = torch.device(config.cpu)
    return config