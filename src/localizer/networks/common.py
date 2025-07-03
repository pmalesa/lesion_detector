import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import ResNet50_Weights, ResNet18_Weights, resnet18
from gymnasium.spaces import Dict, Box
from localizer.networks.resnet import resnet50_backbone
from pathlib import Path
import logging

logger = logging.getLogger("LESION-DETECTOR")

class ResNet18Extractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

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
    def __init__(self, obs_space, features_dim=2048, weights_path: str = None, device: str = "cuda:0"):
        super().__init__(obs_space, features_dim)

        resnet = resnet50_backbone(
            weights=ResNet50_Weights.IMAGENET1K_V2,
            unfreeze_final_layer=False,
            device=device
        )

        # Load fine-tuned weights
        if weights_path is not None:
            weights_path = Path(weights_path)
            if weights_path.is_file():
                state = torch.load(weights_path, map_location=device, weights_only=True)
                resnet.load_state_dict(state, strict=False)
                logger.info(f"ResNet50CoordsExtractor: Using weights found at '{str(weights_path)}'.")
            else:
                logger.info(f"ResNet50CoordsExtractor: No weights found at '{str(weights_path)}', using ImageNet2 wegihts.")
        else:
            logger.info(f"ResNet50CoordsExtractor: Using ImageNet2 weights.")

        # Drop the final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.cnn(x)  # [batch, 2048, 1, 1]
        return x.view(x.size(0), -1)  # [batch, 2048]

class ResNet50CoordsExtractor(BaseFeaturesExtractor):
    """
    A ResNet50 backbone extractor that also includes the 4-dim
    normalized bounding box coordinates into the feature vector.
    Expects:
        obs_space = Dict({"image": Box(1, H, W), "coords": Box(4,)}).
    Returns:
        Tensor of dim (2048 + 4).
    """

    def __init__(self, obs_space: Dict, features_dim: int = 2048 + 4, weights_path: str = None, device: str = "cuda:0"):
        # Validate whether obs_space has the needed keys
        assert isinstance(obs_space, Dict)
        assert "image" in obs_space.spaces and "coords" in obs_space.spaces        
        img_space = obs_space.spaces["image"]
        coord_space = obs_space.spaces["coords"]
        assert isinstance(img_space, Box)
        assert isinstance(coord_space, Box) and coord_space.shape == (4,)

        super().__init__(obs_space, features_dim)

        resnet = resnet50_backbone(
            weights=ResNet50_Weights.IMAGENET1K_V2,
            unfreeze_final_layer=False,
            device=device
        )

        # Load fine-tuned weights
        if weights_path is not None:
            weights_path = Path(weights_path)
            if weights_path.is_file():
                state = torch.load(weights_path, map_location=device, weights_only=True)
                resnet.load_state_dict(state, strict=False)
                logger.info(f"ResNet50CoordsExtractor: Using weights found at '{str(weights_path)}'.")
            else:
                logger.info(f"ResNet50CoordsExtractor: No weights found at '{str(weights_path)}', using ImageNet2 wegihts.")
        else:
            logger.info(f"ResNet50CoordsExtractor: Using ImageNet2 weights.")
        
        # Drop the final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = 2048 # ResNet50's last conv has 2048 channels

    def forward(self, observation: dict) -> torch.Tensor:
        """
        observation: {
            "image": Tensor[B, 1, H, W],
            "coords": Tensor[B, 4]
        }
        returns Tensor[B, 2048 + 4]
        """

        img = observation["image"]
        coords = observation["coords"]

        # 1) ResNet50 forward pass -> [B, 2048, 1, 1]
        x = self.cnn(img)
        x = x.view(x.size(0), -1) # [B, 2048]

        # 2) Concatenate bbox coordinates -> [B, 2048 + 4]
        return torch.cat([x, coords], dim=1)
