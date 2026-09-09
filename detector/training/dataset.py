"""
학습 데이터셋 — prepare_data 가 만든 프레임 캐시를 읽는다.

- FrameDataset    : 단일 프레임 샘플 (CNN / Frequency 학습). 영상당 여러 프레임을 균등 샘플.
- SequenceDataset : (window, stride) 오버랩 시퀀스 샘플 (LSTM 학습). 발표자료의 SeqGenerator.
- 증강은 시퀀스 내 모든 프레임에 같은 파라미터를 적용.
- 라벨은 index.json 의 label (0=real, 1=fake). 영상 단위로 split 하므로 같은 영상의 프레임이
  train/val 양쪽에 들어가지 않는다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.preprocess import bgr_to_tensor, fft_spectrum, make_windows, normalize
from training import augment


def load_index(data_dir: Path, cache_dir: Path | None = None) -> list[dict]:
    p = (cache_dir or data_dir / "cache") / "index.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없습니다. 먼저  python -m training.prepare_data  를 실행하십시오.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _frame_paths(cache_dir: Path, meta: dict, variant: int | None = None) -> list[Path]:
    """variant=None -> 원본 캐시, k -> make_sns_variants 가 만든 H.264 재인코딩 변형본 sns{k}/"""
    d = cache_dir / meta["id"]
    if variant is not None:
        d = d / f"sns{variant}"
    return [d / f"f_{i:04d}.jpg" for i in range(meta["n_frames"])]


def pick_variant(meta: dict, sns_prob: float, force: int | None = None) -> int | None:
    """학습 시 확률 sns_prob 로 SNS 변형본 중 하나를 고른다. force 가 주어지면(평가용) 그 변형을 강제."""
    n = int(meta.get("sns_variants") or 0)
    if n <= 0:
        return None
    if force is not None:
        return min(force, n - 1)
    if sns_prob > 0 and random.random() < sns_prob:
        return random.randrange(n)
    return None


def _read(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"프레임을 읽을 수 없음: {p}")
    return img


class FrameDataset(Dataset):
    """영상 목록 -> 프레임 샘플. 영상당 frames_per_video 개를 균등 샘플(에폭마다 오프셋 랜덤)."""

    def __init__(self, videos: list[dict], cache_dir: Path, size: int, frames_per_video: int = 8,
                 train: bool = True, aug_strength: float = 1.0, sns_prob: float = 0.0, variant: int | None = None):
        self.videos, self.cache_dir, self.size = videos, cache_dir, size
        self.fpv, self.train, self.aug_strength = frames_per_video, train, aug_strength
        self.sns_prob, self.variant = sns_prob, variant # variant: 평가 시 SNS 변형본 강제 (None=원본)
        self.items: list[tuple[int, int]] = [] # (video_idx, frame_idx)
        for vi, m in enumerate(videos):
            n = m["n_frames"]
            k = min(self.fpv, n)
            for fi in np.linspace(0, n - 1, num=k, dtype=int):
                self.items.append((vi, int(fi)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        vi, fi = self.items[i]
        m = self.videos[vi]
        if self.train:
            # 같은 슬롯에서 인접 프레임을 랜덤 선택 -> 에폭마다 다른 프레임
            fi = min(m["n_frames"] - 1, max(0, fi + random.randint(-2, 2)))
        v = pick_variant(m, self.sns_prob if self.train else 0.0, self.variant)
        img = _read(_frame_paths(self.cache_dir, m, v)[fi])
        if self.train:
            # 이미 H.264 열화된 변형본에는 프레임 증강을 약하게 (이중 열화로 학습 신호가 사라지는 것 방지)
            img = augment.apply(img, augment.sample_params(self.aug_strength * (0.5 if v is not None else 1.0)))
        x = normalize(bgr_to_tensor(img, self.size))
        return x, torch.tensor(float(m["label"]))


class SequenceDataset(Dataset):
    """영상 목록 -> (window, stride) 오버랩 시퀀스 샘플. LSTM 학습용."""

    def __init__(self, videos: list[dict], cache_dir: Path, size: int, window: int, stride: int,
                 train: bool = True, aug_strength: float = 1.0, max_windows_per_video: int = 6,
                 sns_prob: float = 0.0, variant: int | None = None):
        self.videos, self.cache_dir, self.size = videos, cache_dir, size
        self.window, self.train, self.aug_strength = window, train, aug_strength
        self.sns_prob, self.variant = sns_prob, variant
        self.items: list[tuple[int, int, int]] = [] # (video_idx, start, end)
        for vi, m in enumerate(videos):
            cl = int(m.get("clip_len") or 0)
            if cl and cl == window:
                # prepare_data --clips 로 만든 캐시: 구간(연속 프레임) 경계에 윈도우를 정렬 -> 구간 넘나드는 불연속 없음
                wins = [(s, s + cl) for s in range(0, m["n_frames"] - cl + 1, cl)] or [(0, m["n_frames"])]
            else:
                wins = make_windows(m["n_frames"], window, stride)
            if len(wins) > max_windows_per_video:
                sel = np.linspace(0, len(wins) - 1, num=max_windows_per_video, dtype=int)
                wins = [wins[j] for j in sel]
            for s, e in wins:
                self.items.append((vi, s, e))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        vi, s, e = self.items[i]
        m = self.videos[vi]
        v = pick_variant(m, self.sns_prob if self.train else 0.0, self.variant)
        paths = _frame_paths(self.cache_dir, m, v)[s:e]
        params = augment.sample_params(self.aug_strength * (0.5 if v is not None else 1.0)) if self.train else None
        frames = []
        for p in paths:
            img = _read(p)
            if params is not None:
                img = augment.apply(img, params) # 시퀀스 전체 동일 파라미터
            frames.append(bgr_to_tensor(img, self.size))
        x = torch.stack(frames) # (T,3,H,W) [0,1]
        if x.shape[0] < self.window: # 짧은 영상 -> 마지막 프레임 패딩
            pad = x[-1:].expand(self.window - x.shape[0], *x.shape[1:])
            x = torch.cat([x, pad], 0)
        # uint8 로 반환 — 정규화·FFT 는 GPU 에서 (train.forward). float32 spatial+freq 를 워커가 넘기면
        # 배치당 ~450 MB 로 DataLoader 공유메모리(shm)가 고갈되어 Bus error 발생 (2026-09-06 실측)
        return (x * 255).round().to(torch.uint8), torch.tensor(float(m["label"]))


def split_videos(index: list[dict], n_splits: int, fold: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    영상 단위 K-fold 분할 — 같은 인물(group)의 영상은 반드시 같은 fold

    - meta 에 'group' 이 있으면 StratifiedGroupKFold (AI Hub: real=UUID, fake=타겟UUID)
    - 없으면 StratifiedKFold
    - manifest 의 split='val' 로 표시된 영상은 K-fold 에 넣지 않고 모든 fold 의 검증셋에 추가시킴 (AI Hub 공식 Validation 셋을 별도 홀드아웃으로 활용)
    """
    holdout = [m for m in index if m.get("split") == "val"]
    pool = [m for m in index if m.get("split") != "val"]
    if len(pool) < n_splits * 2:
        raise ValueError(f"K-fold 대상 영상이 너무 적습니다: {len(pool)}")

    y = np.array([m["label"] for m in pool])
    groups = [m.get("group") or m["id"] for m in pool]
    if len(set(groups)) < len(groups): # 그룹 정보가 실제로 있음
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = splitter.split(np.zeros(len(y)), y, groups=np.array(groups))
    else:
        from sklearn.model_selection import StratifiedKFold
        it = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    for k, (tr, va) in enumerate(it):
        if k == fold:
            return [pool[i] for i in tr], [pool[i] for i in va] + holdout
    raise ValueError(f"fold {fold} 가 범위를 벗어남 (n_splits={n_splits})")
