import torch
import torch.nn as nn
import torchvision.models as models

model_map = dict(
    vgg16=models.vgg16,
    resnet18=models.resnet18,
    resnet34=models.resnet34,
    resnet50=models.resnet50,
    resnet101=models.resnet101,
    regnet=models.regnet_y_3_2gf
)


class Model(nn.Module):
    def __init__(self, backbone="regnet", num_classes=10):
        super(Model, self).__init__()
        self.backbone = model_map[backbone]

        if isinstance(backbone, models.ResNet) or isinstance(backbone, models.RegNet):
            self.backbone.fc = nn.Linear(512, num_classes)
        elif isinstance(backbone, models.VGG):
            self.backbone.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        return self.backbone(x)
