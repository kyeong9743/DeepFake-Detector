"""
LSTM 시간 분석기 — 공간 스트림 + 주파수 스트림 -> LSTM.

입력: spatial (B,T,3,H,W) [정규화된 이미지], frequency (B,T,3,H,W) [fft_spectrum 출력]
출력: (B,) 로짓
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class StreamEncoder(nn.Module):
    """프레임 1장 -> 특징 벡터. 얕은 conv 스택 + GAP (시퀀스 전체를 배치로 처리하므로 가볍게)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 96),
            _conv_block(96, out_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class ResNetStreamEncoder(nn.Module):
    """
    프레임 1장 -> 특징 벡터, ImageNet 사전학습 ResNet 트렁크 사용.
    처음부터 학습하는 소형 conv(StreamEncoder)는 545개 영상 검증에서 AUC 0.58 에 그쳤음 -> 공간 스트림은 사전학습 백본으로.
    """

    def __init__(self, backbone: str = "resnet18", out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        from torchvision import models as tvm
        ctor, weights, feat = {
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, 512),
            "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1, 512),
        }[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4,
                                   nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(feat, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(x).flatten(1))


class LSTMAnalyzer(nn.Module):
    def __init__(self, hidden: int = 128, bidirectional: bool = True, num_layers: int = 2,
                 dropout: float = 0.3, stream_dim: int = 128, spatial_backbone: str | None = "resnet18",
                 pretrained: bool = True):
        super().__init__()
        # 공간 스트림: 사전학습 ResNet (None 이면 소형 conv). 주파수 스트림: FFT 스펙트럼은 자연 이미지가 아니라 소형 conv 유지
        self.spatial_enc = ResNetStreamEncoder(spatial_backbone, stream_dim, pretrained) if spatial_backbone else StreamEncoder(stream_dim)
        self.freq_enc = StreamEncoder(stream_dim) # 별도 인스턴스 — 가중치 공유 없음
        self.fuse = nn.Sequential(
            nn.Linear(stream_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0, # num_layers=1 이면 dropout 무효 -> 0 으로 명시
        )
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out),
            nn.Dropout(dropout),
            nn.Linear(out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.bidirectional = bidirectional

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        b, t = spatial.shape[:2]
        s = self.spatial_enc(spatial.flatten(0, 1)).view(b, t, -1)
        f = self.freq_enc(frequency.flatten(0, 1)).view(b, t, -1)
        x = self.fuse(torch.cat([s, f], dim=-1)) # (B,T,256)
        out, _ = self.lstm(x) # (B,T,out)
        # 양방향이면 시퀀스 전체 평균이 마지막 스텝보다 안정적
        pooled = out.mean(dim=1) if self.bidirectional else out[:, -1]
        return self.head(pooled).squeeze(1)
