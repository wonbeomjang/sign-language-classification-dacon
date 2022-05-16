import pandas as pd
from torch.utils import data
from torchvision import transforms
from torch.utils.data.dataset import random_split
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, base_dir: str):
        self.df = df
        self.base_dir = base_dir
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        file_name = self.df.iloc[index]['file_name']
        label = int(self.df.iloc[index]['label'])

        image = Image.open(f'{self.base_dir}/train/{file_name}')
        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.df)


class TestDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, base_dir: str):
        self.df = df
        self.base_dir = base_dir
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        file_name = self.df.iloc[index]['file_name']

        image = Image.open(f'{self.base_dir}/test/{file_name}')
        image = self.transforms(image)

        return image, file_name

    def __len__(self):
        return len(self.df)


def get_dataloader(config, val=True, split_ratio=.9):
    meta_data = pd.read_csv(f'{config.dataset_dir}/train.csv')

    train_dataset = Dataset(meta_data, config.image_size, config.dataset_dir)
    val_loader = None
    if val:
        train_len = int(len(train_dataset) * split_ratio)
        val_len = len(train_dataset) - train_len
        train_dataset, val_dataset = random_split(train_dataset, (train_len, val_len))
        val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                     num_workers=config.num_workers)

    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                   num_workers=config.num_workers)

    return train_loader, val_loader


def get_testloader(config):
    meta_data = pd.read_csv(f'{config.dataset_dir}/test.csv')

    test_dataset = TestDataset(meta_data, config.image_size, config.dataset_dir)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                   num_workers=config.num_workers)
    return test_loader

