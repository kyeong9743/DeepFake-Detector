"""
앙상블 가중치 최적화

입력: training.train 이 K-fold 로 만든 영상 단위 out-of-fold 예측
      runs/{mode}/oof_cnn.csv, oof_lstm.csv, oof_frequency.csv  (video_id,label,prob)
방법:
  1. Grid Search  — 심플렉스 위 가중치 격자 전수 탐색
  2. Bayesian Optimization — GP(Matern) + Expected Improvement 로 격자 사이 미세 탐색
  목적함수: 가중 평균 확률의 ROC-AUC (임계값 무관). 최종 임계값은 Youden's J.
  투표 시스템: 모델별 이진 투표 다수결도 함께 계산해 리포트에 병기.
출력: checkpoints/{mode}_ensemble.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

# AUC 가 1.0 으로 포화되면(합성 데이터 등) GP 커널 최적화가 수렴 경고를 내지만 결과에는 영향이 없음
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import roc_auc_score

from config import checkpoint_path, settings
from models.registry import KINDS
from training.metrics import compute_metrics

logger = logging.getLogger("ensemble_opt")


def load_oof(run_dir: Path) -> pd.DataFrame:
    dfs = []
    for k in KINDS:
        p = run_dir / f"oof_{k}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음 — 먼저 세 모델을 모두 학습하십시오 (--model all).")
        d = pd.read_csv(p).rename(columns={"prob": k})
        # 홀드아웃(split=val) 영상은 모든 fold 검증셋에 포함되어 fold 수만큼 중복 -> 영상별 평균으로 1행화
        d = d.groupby("video_id", as_index=False).agg({"label": "first", k: "mean"})
        dfs.append(d[["video_id", "label", k]] if not dfs else d[["video_id", k]])
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="video_id", how="inner")
    if df["label"].nunique() < 2:
        raise ValueError("OOF 예측에 클래스가 하나뿐입니다. 데이터셋에 real/fake 가 모두 필요합니다.")
    return df


MIN_W = 0.0 # --min-weight: 모델별 최소 가중치 (0 = 데이터가 정하는 대로. 0.1 이면 세 모델 모두 최소 10% 반영)


def _effective(w: np.ndarray) -> np.ndarray:
    """탐색 가중치(심플렉스) -> 최소 가중치 제약을 적용한 실제 가중치."""
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
    return MIN_W + (1 - len(w) * MIN_W) * w


def _auc(w: np.ndarray, P: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, P @ _effective(w)))


def grid_search(P: np.ndarray, y: np.ndarray, step: float = 0.05) -> tuple[np.ndarray, float, list[dict]]:
    n = P.shape[1]
    steps = int(round(1 / step))
    best_w, best_auc, trace = None, -1.0, []
    for combo in itertools.product(range(steps + 1), repeat=n - 1):
        if sum(combo) > steps:
            continue
        w = np.array(list(combo) + [steps - sum(combo)], dtype=float) / steps
        a = _auc(w, P, y)
        trace.append({"w": w.round(3).tolist(), "auc": round(a, 5)})
        if a > best_auc:
            best_auc, best_w = a, w
    return best_w, best_auc, trace


def _to_simplex(u: np.ndarray) -> np.ndarray:
    """(n-1) 차원 단위 큐브 -> n 차원 심플렉스 (stick-breaking)."""
    u = np.clip(u, 1e-6, 1 - 1e-6)
    w, rest = [], 1.0
    for ui in u:
        w.append(rest * ui)
        rest *= (1 - ui)
    w.append(rest)
    return np.array(w)


def bayesian_opt(P: np.ndarray, y: np.ndarray, init_w: np.ndarray, n_iter: int = 40, seed: int = 0) -> tuple[np.ndarray, float, list[dict]]:
    """GP + Expected Improvement. Grid Search 최적점 주변을 초기 관측으로 포함."""
    rng = np.random.default_rng(seed)
    n = P.shape[1]
    d = n - 1
    X, Y, trace = [], [], []

    def observe(u):
        w = _to_simplex(u)
        a = _auc(w, P, y)
        X.append(u); Y.append(a)
        trace.append({"w": w.round(3).tolist(), "auc": round(a, 5)})
        return a

    # 초기 관측: 랜덤 8개 + 그리드 최적점의 역변환
    for _ in range(8):
        observe(rng.random(d))
    u0 = []
    rest = 1.0
    for wi in init_w[:-1]:
        u0.append(0.5 if rest <= 1e-9 else wi / rest)
        rest -= wi
    observe(np.clip(np.array(u0), 1e-3, 1 - 1e-3))

    kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    for _ in range(n_iter):
        gp.fit(np.array(X), np.array(Y))
        cand = rng.random((2000, d))
        mu, sd = gp.predict(cand, return_std=True)
        best = max(Y)
        z = (mu - best) / (sd + 1e-9)
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        observe(cand[int(np.argmax(ei))])
    i = int(np.argmax(Y))
    return _to_simplex(X[i]), float(Y[i]), trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=settings.default_mode, choices=["fast", "high"])
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--bo-iter", type=int, default=40)
    ap.add_argument("--min-weight", type=float, default=0.0,
                    help="모델별 최소 가중치 (예: 0.1). 0 이면 순수 데이터 기반 — 약한 모델은 가중치 0 이 될 수 있음")
    args = ap.parse_args()
    global MIN_W
    MIN_W = max(0.0, min(args.min_weight, 1 / len(KINDS) - 1e-6))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = settings.runs_dir / args.mode
    df = load_oof(run_dir)
    y = df["label"].to_numpy().astype(int)
    P = df[list(KINDS)].to_numpy().astype(float)
    logger.info("OOF 영상 %d개 (real=%d, fake=%d)", len(y), int((y == 0).sum()), int((y == 1).sum()))
    for i, k in enumerate(KINDS):
        logger.info("  단일 모델 %-10s AUC=%.4f", k, roc_auc_score(y, P[:, i]))

    gw, gauc, gtrace = grid_search(P, y, args.grid_step)
    bw, bauc, btrace = bayesian_opt(P, y, gw, args.bo_iter)
    gw, bw = _effective(gw), _effective(bw) # 최소 가중치 제약 반영된 실제 가중치
    logger.info("Grid Search  최적 w=%s AUC=%.4f (%d 조합)  min_weight=%.2f", gw.round(3), gauc, len(gtrace), MIN_W)
    logger.info("Bayesian Opt 최적 w=%s AUC=%.4f (%d 관측)", bw.round(3), bauc, len(btrace))

    if bauc >= gauc:
        w, method = bw, "bayesian_optimization"
    else:
        w, method = gw, "grid_search"
    ens_prob = P @ w
    m = compute_metrics(y, ens_prob, threshold=0.5)
    thr = m["best_threshold"]
    m_thr = compute_metrics(y, ens_prob, threshold=thr)

    # 투표 시스템: 각 모델을 자기 최적 임계값으로 이진화 -> 다수결
    model_thr = {k: compute_metrics(y, P[:, i])["best_threshold"] for i, k in enumerate(KINDS)}
    votes = np.stack([(P[:, i] >= model_thr[k]).astype(int) for i, k in enumerate(KINDS)], 1)
    vote_pred = (votes.sum(1) >= 2).astype(int)
    vote_acc = float((vote_pred == y).mean())

    out = {
        "mode": args.mode,
        "method": method,
        "weights": {k: round(float(w[i]), 4) for i, k in enumerate(KINDS)},
        "threshold": round(float(thr), 4),
        "model_thresholds": {k: round(float(v), 4) for k, v in model_thr.items()},
        "metrics_at_threshold": m_thr,
        "metrics_at_0.5": m,
        "voting_accuracy": round(vote_acc, 4),
        "grid_search": {"best_auc": round(gauc, 5), "best_w": gw.round(4).tolist(), "n_combos": len(gtrace)},
        "bayesian_opt": {"best_auc": round(bauc, 5), "best_w": bw.round(4).tolist(), "n_obs": len(btrace)},
        "single_model_auc": {k: round(float(roc_auc_score(y, P[:, i])), 5) for i, k in enumerate(KINDS)},
        "n_videos": int(len(y)),
    }
    path = checkpoint_path("ensemble", args.mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(run_dir / "ensemble_search_trace.json", "w", encoding="utf-8") as f:
        json.dump({"grid": gtrace, "bayesian": btrace}, f)
    logger.info("저장: %s", path)
    logger.info("앙상블 AUC=%.4f  ACC@thr=%.4f  F1=%.4f  thr=%.3f  voting_acc=%.4f",
                m_thr.get("auc", float("nan")), m_thr["accuracy"], m_thr["f1"], thr, vote_acc)


if __name__ == "__main__":
    main()
