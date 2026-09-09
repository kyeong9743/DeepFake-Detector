"""
주파수 분석기 — FFT 로그 진폭 스펙트럼 -> CNN.

입력은 common.preprocess.fft_spectrum() 의 출력 (B,3,H,W)
(학습·추론이 같은 함수를 사용해야 분포가 일치한다)
Flatten+Linear 대신 GAP 을 사용해 입력 해상도에 무관하고 파라미터가 작다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class FrequencyAnalyzer(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 160),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(160, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(spectrum).flatten(1)).squeeze(1)
