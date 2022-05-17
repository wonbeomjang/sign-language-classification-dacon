import torch
import torch.nn as nn
import torchvision.models as models

model_map = dict(
    vgg16=models.vgg16,
    resnet18=models.resnet18,
    resnet34=models.resnet34,
    resnet50=models.resnet50,
    resnet101=models.resnet101,
    regnet=models.regnet_y_3_2gf,
    resnext50=models.resnext50_32x4d,
    resnext101=models.resnext101_32x8d,
)


class Model(nn.Module):
    def __init__(self, pretrained=True, backbone="regnet", num_classes=10):
        super(Model, self).__init__()
        self.net = model_map[backbone](pretrained=pretrained)

        if isinstance(backbone, models.ResNet) or isinstance(backbone, models.RegNet):
            self.net.fc = nn.Linear(512, num_classes)
        elif isinstance(backbone, models.VGG):
            self.net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        return self.net(x)

