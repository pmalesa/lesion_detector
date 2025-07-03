import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights

from localizer.networks.common import resnet50_backbone


class BoxRegressor(nn.Module):
    """
    Class representing the network for bounding box regression.
    """

    def __init__(self, weights: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()

        # Initialize ResNet50 backbone
        self._backbone_cnn = resnet50_backbone(
            weights=weights, unfreeze_final_layer=True
        )

        # Remove final FC and pooling, keeping 2048-dim features
        self._feature_extractor = nn.Sequential(
            *list(self._backbone_cnn.children())[:-1]
        )

        # Initialize regression head
        self._regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        features = self._feature_extractor(x)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        return self._regressor(features)  # [B, 4]

    def save_backbone(self, path: str):
        """
        Saves only the backbone weights.
        """

        torch.save(self._backbone_cnn.state_dict(), path)

    def load_backbone(self, path: str):
        """
        Loads backbone weights into this model.
        """

        state = torch.load(path, map_location="cuda:0")
        self._backbone_cnn.load_state_dict(state, strict=False)
