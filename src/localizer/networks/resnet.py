import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def resnet50_backbone(
    weights: ResNet50_Weights, unfreeze_final_layer: bool = True, device: str = "cuda:0"
):
    """
    Returns a ResNet50 pretrained on ImageNet (or with custom weights),
    adapted to single-channel input with every layer frozen and optionally
    unfreezing the final conv block (layer4).
    """

    model = resnet50(weights=weights)

    # Replace the first conv layer and adapt to single-channel input
    w = model.conv1.weight.data
    model.conv1 = nn.Conv2d(
        in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.conv1.weight.data = w.mean(dim=1, keepdim=True)

    # Freeze all the layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer (layer4)
    if unfreeze_final_layer:
        for name, param in model.named_parameters():
            if name.startswith("layer4"):
                param.requires_grad = True

    return model.to(device)
