"""
학습·추론 전처리
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from config import NORMALIZE_MEAN, NORMALIZE_STD

_MEAN = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1)


def bgr_to_tensor(frame_bgr: np.ndarray, size: int) -> torch.Tensor:
    """BGR uint8 (H,W,3) -> float32 (3,size,size), [0,1] 스케일. 정규화는 normalize()에서."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) 또는 (3,H,W) [0,1] -> ImageNet 정규화."""
    if x.dim() == 3:
        return (x - _MEAN[0].to(x.device)) / _STD[0].to(x.device)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def fft_spectrum(x: torch.Tensor) -> torch.Tensor:
    """
    공간 이미지 -> 로그 진폭 스펙트럼 (채널별, 샘플별 정규화).
    입력·출력 모두 (B,3,H,W). 학습의 compute_frequency_domain 과 추론이 반드시 이 함수를 공유해야 함.

    - 채널별 표준화 -> DC 성분이 스펙트럼을 지배하는 것을 완화
    - fftshift -> 저주파를 중앙에 배치 (Conv 커널이 주파수 구조를 학습하기 쉬움)
    - log1p -> 동적 범위 압축
    - 샘플별 재표준화 -> 배치 구성에 무관한 스케일
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.float()
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + 1e-6
    x = (x - mean) / std
    spec = torch.fft.fft2(x, norm="ortho")
    spec = torch.fft.fftshift(spec, dim=(-2, -1))
    mag = torch.log1p(spec.abs())
    mean = mag.mean(dim=(2, 3), keepdim=True)
    std = mag.std(dim=(2, 3), keepdim=True) + 1e-6
    return (mag - mean) / std


def make_windows(n: int, window: int, stride: int) -> list[tuple[int, int]]:
    """
    길이 n 시퀀스를 (window, stride) 오버랩 윈도우로 분할한 [start, end) 목록.
    n < window 이면 [0, n) 하나만 반환(모델 쪽에서 패딩).
    마지막 윈도우가 끝에 닿지 않으면 끝에 맞춘 윈도우를 하나 추가.
    """
    if n <= window:
        return [(0, n)]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    return [(s, s + window) for s in starts]


def pad_sequence(x: torch.Tensor, length: int) -> torch.Tensor:
    """(T,...) -> (length,...) 뒤쪽을 마지막 프레임으로 패딩. T ≥ length 면 앞 length 개."""
    t = x.shape[0]
    if t >= length:
        return x[:length]
    pad = x[-1:].expand(length - t, *x.shape[1:])
    return torch.cat([x, pad], dim=0)
