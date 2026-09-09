"""
CNN 공간 특징 추출기 — ResNet 백본.

ResNet 기반 특징 추출기
- torchvision ResNet18/34 (ImageNet 사전학습) 를 백본으로 사용
- Global Average Pooling -> 512-d 특징 벡터 -> 1-logit
- Grad-CAM 대상 레이어: layer4 (마지막 conv 블록)
- forward() 는 로짓을 반환 (sigmoid 는 호출 측에서)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm


_BACKBONES = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class CNNResNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"지원하지 않는 백본: {backbone} (가능: {list(_BACKBONES)})")
        ctor, weights, feat_dim = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        # 분류 헤드 제거, 특징 추출부만 사용
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.backbone_name = backbone
        self.feat_dim = feat_dim

    # Grad-CAM 이 후크를 걸 대상
    @property
    def gradcam_layer(self) -> nn.Module:
        return self.layer4

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,feat_dim,h,w) 공간 특징맵."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,feat_dim) 풀링된 특징 벡터 (LSTM 등에서 재사용 가능)."""
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(1) # (B,) 로짓
