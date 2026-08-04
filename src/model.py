import torch
import torch.nn as nn
from torchvision import models


class LivenessNet(nn.Module):
    """MobileNetV3-Small backbone with a binary classification head for PAD.

    Architecture:
        - Pretrained MobileNetV3-Small feature extractor (frozen or fine-tuned)
        - Global Average Pooling
        - Dropout + Linear head → 2 classes (LIVE=0, SPOOF=1)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()

        backbone = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)

        self.features = nn.Sequential(*list(backbone.features.children()))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(576, 2),
        )

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_activation_hook(self):
        """Register a hook on the last feature map for Grad-CAM."""
        activation = {}

        def hook(module, input, output):
            activation["features"] = output.detach()

        self.features[-1].register_forward_hook(hook)
        return activation