import torch
import torch.nn as nn
import torchvision.models as models

model_map = dict(
    vgg16=models.vgg16(pretrained=True),
    resnet=models.resnet18(pretrained=True),
    resnet=models.resnet18(pretrained=True),
    resnet=models.resnet18(pretrained=True),
    resnet=models.resnet18(pretrained=True),
)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        backbone