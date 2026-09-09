"""
데이터 증강 (발표자료 DataAugmentor: RandomCrop · ColorJitter).

딥페이크 탐지에 특히 중요한 증강을 추가한다.
- JPEG 재압축 / 다운스케일->업스케일 : SNS 등 수차례 재업로드된 압축 영상에 대응
- 가우시안 블러 / 노이즈 : 화질 열화 대응
- 수평 반전
- (2026-09-07 추가, SNS 일반화) 선명화(unsharp) · 피부 보정풍 스무딩 · 감마 : 소셜 앱 필터 대응.
  실제 H.264 저비트레이트 재인코딩은 프레임 단위로 흉내 낼 수 없으므로 training.make_sns_variants 가
  캐시 클립을 오프라인으로 재인코딩한 변형본(sns{k}/)을 만들고 dataset 이 확률적으로 그것을 읽는다.
모든 변환은 BGR uint8 numpy 이미지에 적용되며, 시퀀스 증강은 시퀀스 내 모든 프레임에
**같은 파라미터**를 적용해 시간적 일관성을 유지한다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugParams:
    """시퀀스 전체에 동일하게 적용할 증강 파라미터 한 세트."""
    crop: tuple[float, float, float, float] | None   # (x0,y0,x1,y1) 비율
    flip: bool
    brightness: float
    contrast: float
    saturation: float
    jpeg_q: int | None
    downscale: float | None
    blur_k: int | None
    noise_sigma: float | None
    sharpen: float | None = None # unsharp mask 강도 (0.3~1.2) — 소셜 앱 '선명하게' 필터
    smooth: float | None = None  # 피부 보정풍 스무딩 혼합 비율 (0.3~0.7)
    gamma: float | None = None   # 감마 (0.7~1.4) — 밝기 보정 필터


def sample_params(strength: float = 1.0) -> AugParams:
    """strength 0~1: 증강 강도."""
    s = strength
    crop = None
    if random.random() < 0.8:
        scale = random.uniform(1 - 0.25 * s, 1.0)
        ar = random.uniform(0.9, 1.1)
        w = min(1.0, scale * ar)
        h = min(1.0, scale / ar)
        x0 = random.uniform(0, 1 - w)
        y0 = random.uniform(0, 1 - h)
        crop = (x0, y0, x0 + w, y0 + h)
    # 소셜 필터: 선명화와 스무딩은 상호배타적으로 하나만
    r = random.random()
    sharpen = random.uniform(0.3, 1.2) if r < 0.2 * s else None
    smooth = random.uniform(0.3, 0.7) if (sharpen is None and r < 0.35 * s) else None
    return AugParams(
        crop=crop,
        flip=random.random() < 0.5,
        brightness=random.uniform(1 - 0.3 * s, 1 + 0.3 * s),
        contrast=random.uniform(1 - 0.3 * s, 1 + 0.3 * s),
        saturation=random.uniform(1 - 0.3 * s, 1 + 0.3 * s),
        jpeg_q=random.randint(30, 90) if random.random() < 0.5 * s else None,
        downscale=random.uniform(0.4, 0.9) if random.random() < 0.3 * s else None,
        blur_k=random.choice([3, 5]) if random.random() < 0.2 * s else None,
        noise_sigma=random.uniform(2, 8) if random.random() < 0.2 * s else None,
        sharpen=sharpen,
        smooth=smooth,
        gamma=random.uniform(0.7, 1.4) if random.random() < 0.25 * s else None,
    )


def apply(img: np.ndarray, p: AugParams) -> np.ndarray:
    h, w = img.shape[:2]
    out = img
    if p.crop is not None:
        x0, y0, x1, y1 = p.crop
        out = out[int(y0 * h):max(int(y1 * h), int(y0 * h) + 8), int(x0 * w):max(int(x1 * w), int(x0 * w) + 8)]
    if p.flip:
        out = cv2.flip(out, 1)
    # ColorJitter (HSV 채도 + 밝기/대비)
    if abs(p.saturation - 1) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * p.saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if abs(p.brightness - 1) > 1e-3 or abs(p.contrast - 1) > 1e-3:
        f = out.astype(np.float32)
        mean = f.mean()
        f = (f - mean) * p.contrast + mean
        f = f * p.brightness
        out = np.clip(f, 0, 255).astype(np.uint8)
    if p.gamma is not None and abs(p.gamma - 1) > 1e-3:
        lut = (np.linspace(0, 1, 256) ** p.gamma * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    # 소셜 필터 (압축 전에 적용 — 실제 앱도 필터 -> 업로드 인코딩 순서)
    if p.smooth is not None:
        base = cv2.bilateralFilter(out, d=7, sigmaColor=40, sigmaSpace=7)
        out = cv2.addWeighted(out, 1 - p.smooth, base, p.smooth, 0)
    if p.sharpen is not None:
        blur = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = cv2.addWeighted(out, 1 + p.sharpen, blur, -p.sharpen, 0)
    if p.downscale is not None:
        hh, ww = out.shape[:2]
        small = cv2.resize(out, (max(8, int(ww * p.downscale)), max(8, int(hh * p.downscale))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
    if p.blur_k is not None:
        out = cv2.GaussianBlur(out, (p.blur_k, p.blur_k), 0)
    if p.noise_sigma is not None:
        noise = np.random.normal(0, p.noise_sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if p.jpeg_q is not None:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out
