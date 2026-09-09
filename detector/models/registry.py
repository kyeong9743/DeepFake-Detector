"""
모델 생성 · 체크포인트 저장/로드.

체크포인트는 state_dict 만이 아니라 **재현에 필요한 메타데이터**를 함께 저장
[ mode, 백본, frame_size, window, 정규화 통계, 학습 시각, 검증 지표 ]
로드 시 메타데이터로 모델을 재구성하고 strict=True 로 검증
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from config import NORMALIZE_MEAN, NORMALIZE_STD, PROFILES, checkpoint_path
from .cnn_resnet import CNNResNet
from .frequency_analyzer import FrequencyAnalyzer
from .lstm_analyzer import LSTMAnalyzer

logger = logging.getLogger(__name__)

KINDS = ("cnn", "lstm", "frequency")


def build_model(kind: str, mode: str, pretrained: bool = True) -> nn.Module:
    p = PROFILES[mode]
    if kind == "cnn":
        return CNNResNet(backbone=p.backbone, pretrained=pretrained)
    if kind == "lstm":
        return LSTMAnalyzer(bidirectional=p.bidirectional, spatial_backbone=p.lstm_backbone, pretrained=pretrained)
    if kind == "frequency":
        return FrequencyAnalyzer()
    raise ValueError(f"unknown model kind: {kind}")


def save_checkpoint(model: nn.Module, kind: str, mode: str, metrics: dict[str, Any] | None = None,
                    path: Path | None = None, extra: dict[str, Any] | None = None) -> Path:
    p = PROFILES[mode]
    path = path or checkpoint_path(kind, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": {
            "kind": kind,
            "mode": mode,
            "backbone": p.backbone,
            "frame_size": p.frame_size,
            "window": p.window,
            "stride": p.stride,
            "bidirectional": p.bidirectional,
            "normalize_mean": NORMALIZE_MEAN,
            "normalize_std": NORMALIZE_STD,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            **(extra or {}),
        },
    }
    torch.save(payload, path)
    return path


def load_checkpoint(kind: str, mode: str, device: torch.device, path: Path | None = None) -> tuple[nn.Module, dict]:
    """체크포인트가 없거나 구조가 맞지 않으면 예외처리"""
    path = path or checkpoint_path(kind, mode)
    if not path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {path}\n"
            f"먼저 학습하십시오:  python -m training.train --mode {mode} --model {kind}"
        )
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload.get("meta", {})
    if meta.get("kind", kind) != kind or meta.get("mode", mode) != mode:
        raise ValueError(f"체크포인트 메타 불일치: 파일={meta.get('kind')}/{meta.get('mode')} 요청={kind}/{mode}")
    model = build_model(kind, mode, pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    m = meta.get("metrics", {})
    auc = m.get("auc") or m.get("oof", {}).get("auc")
    logger.info("모델 로드: %s (%s, OOF AUC=%s)", path.name, meta.get("saved_at", "?")[:19],
                f"{auc:.4f}" if isinstance(auc, (int, float)) else "?")
    return model, meta


def load_all_models(mode: str, device: torch.device) -> dict[str, nn.Module]:
    return {k: load_checkpoint(k, mode, device)[0] for k in KINDS}


def load_ensemble_config(mode: str) -> dict[str, Any]:
    """
    앙상블 가중치·임계값. 학습(training.ensemble_opt) 결과가 없으면 균등 가중치 + 0.5.
    형식: {"weights": {"cnn":..,"lstm":..,"frequency":..}, "threshold": 0.5, "method": "...", ...}
    """
    path = checkpoint_path("ensemble", mode)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    logger.warning("앙상블 설정이 없어 균등 가중치를 사용합니다: %s", path)
    return {"weights": {k: 1 / 3 for k in KINDS}, "threshold": 0.5, "method": "uniform(default)"}
