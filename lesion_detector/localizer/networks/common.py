import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18, resnet50


class ResNet18Extractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        resnet = resnet18(pretrained=True)

        # Grab the original weights (shape [64, 3, 7, 7]) and average them
        w = resnet.conv1.weight.data

        # Create a new conv1 layer for 1 channel
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Initialize its weights by averaging the RGB weights
        resnet.conv1.weight.data = w.mean(dim=1, keepdim=True)

        # Drop the final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.cnn(x)  # [batch, 512, 1, 1]
        return x.view(x.size(0), -1)  # [batch, 512]


class ResNet50Extractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=2048):
        super().__init__(obs_space, features_dim)
        resnet = resnet50(pretrained=True)

        # Replace the first conv layer and adapt to single-channel input
        w = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Initialize its weights by averaging the RGB weights
        resnet.conv1.weight.data = w.mean(dim=1, keepdim=True)

        # Drop the final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.cnn(x)  # [batch, 2048, 1, 1]
        return x.view(x.size(0), -1)  # [batch, 2048]
