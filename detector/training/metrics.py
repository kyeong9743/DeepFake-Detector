"""
평가 지표 및 시각화

- compute_metrics: AUC · Accuracy · Precision · Recall · F1 · Confusion Matrix · 최적 임계값
- MetricLogger: 에폭별 지표를 JSONL 로 기록하고 손실/AUC/정확도 그래프 PNG 생성
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt # noqa: E402
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["ap"] = float(average_precision_score(y_true, y_prob))
        # Youden's J 로 최적 임계값
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        out["best_threshold"] = float(thr[int(np.argmax(j))])
    else:
        # 단일 클래스 — AUC 정의 불가. 학습 데이터 구성 오류를 즉시 드러내기 위해 명시
        out["auc"] = float("nan")
        out["ap"] = float("nan")
        out["best_threshold"] = 0.5
        out["warning"] = "검증셋에 클래스가 하나뿐입니다. real/fake 가 모두 포함되어야 합니다."
    return out


@dataclass
class MetricLogger:
    run_dir: Path
    name: str
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.run_dir / f"{self.name}_metrics.jsonl"

    def log(self, epoch: int, train_loss: float, val_loss: float, val_metrics: dict[str, Any], lr: float) -> None:
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, **val_metrics}
        self.history.append(row)
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def plot(self) -> Path:
        if not self.history:
            return self.run_dir
        ep = [h["epoch"] for h in self.history]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(ep, [h["train_loss"] for h in self.history], label="train")
        axes[0].plot(ep, [h["val_loss"] for h in self.history], label="val")
        axes[0].set_title(f"{self.name} — Loss"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(ep, [h.get("auc", np.nan) for h in self.history], color="tab:green")
        axes[1].set_title("Val AUC"); axes[1].set_ylim(0.4, 1.0); axes[1].grid(alpha=.3)
        axes[2].plot(ep, [h.get("accuracy", np.nan) for h in self.history], label="acc")
        axes[2].plot(ep, [h.get("f1", np.nan) for h in self.history], label="f1")
        axes[2].set_title("Val Accuracy / F1"); axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(alpha=.3)
        for a in axes:
            a.set_xlabel("epoch")
        fig.tight_layout()
        out = self.run_dir / f"{self.name}_curves.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str = "") -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    a1.plot([0, 1], [0, 1], "--", color="gray"); a1.set_xlabel("FPR"); a1.set_ylabel("TPR")
    a1.set_title(f"ROC {title}"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(rec, prec, label=f"AP={average_precision_score(y_true, y_prob):.3f}")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title(f"PR {title}"); a2.legend(); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_confusion(cm: list[list[int]], out_path: Path, title: str = "") -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion {title}")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
