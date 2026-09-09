"""
홀드아웃(AI Hub Validation, split=val) 재평가 — 원본 캐시와 SNS 압축 변형본(sns0) 양쪽.
학습 없이 기존 체크포인트의 압축 강건성을 잰다 (증강 재학습 전/후 비교의 기준선).

    python -m training.eval_holdout --mode high --tag v1_noaug
    python -m training.eval_holdout --mode high --tag v1_noaug --ckpt-dir /checkpoints/archive/high_v1_aihub_noaug
-> runs/{mode}/holdout_eval_{tag}.json  (모델별 orig / sns 지표 + 앙상블)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from config import checkpoint_path, settings
from models.registry import KINDS, load_checkpoint, load_ensemble_config
from training.dataset import load_index
from training.metrics import compute_metrics
from training.train import predict_videos

logger = logging.getLogger("eval_holdout")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=settings.default_mode, choices=["fast", "high"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="체크포인트 디렉터리 (기본: /checkpoints)")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="디버그: 홀드아웃 앞 N개만")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache_dir = args.cache_dir or (settings.data_dir / "cache")
    index = load_index(settings.data_dir, cache_dir)
    holdout = [m for m in index if m.get("split") == "val"]
    if args.limit:
        holdout = holdout[:args.limit]
    sns_hold = [m for m in holdout if int(m.get("sns_variants") or 0) > 0]
    logger.info("홀드아웃 %d (SNS 변형본 보유 %d)", len(holdout), len(sns_hold))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out: dict = {"tag": args.tag, "mode": args.mode, "n_holdout": len(holdout), "n_sns": len(sns_hold), "models": {}}
    probs: dict[str, dict[str, np.ndarray]] = {}
    labels = np.array([m["label"] for m in holdout]); labels_sns = np.array([m["label"] for m in sns_hold])
    for k in KINDS:
        path = (args.ckpt_dir / checkpoint_path(k, args.mode).name) if args.ckpt_dir else None
        model, _ = load_checkpoint(k, args.mode, device, path=path)
        df = predict_videos(k, args.mode, model, holdout, cache_dir, device)
        m_orig = compute_metrics(labels, df["prob"].to_numpy())
        probs[k] = {"orig": df["prob"].to_numpy()}
        rec = {"orig": m_orig}
        if sns_hold:
            sdf = predict_videos(k, args.mode, model, sns_hold, cache_dir, device, variant=0)
            rec["sns"] = compute_metrics(labels_sns, sdf["prob"].to_numpy())
            probs[k]["sns"] = sdf["prob"].to_numpy()
        out["models"][k] = rec
        logger.info("[%s] orig AUC=%.4f ACC=%.4f | sns AUC=%s ACC=%s", k, m_orig["auc"], m_orig["accuracy"],
                    f"{rec['sns']['auc']:.4f}" if "sns" in rec else "-", f"{rec['sns']['accuracy']:.4f}" if "sns" in rec else "-")
        del model; torch.cuda.empty_cache()

    ens_path = (args.ckpt_dir / checkpoint_path("ensemble", args.mode).name) if args.ckpt_dir else None
    if ens_path and ens_path.exists():
        with open(ens_path, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = load_ensemble_config(args.mode)
    w, thr = cfg["weights"], float(cfg.get("threshold", 0.5))
    out["ensemble"] = {"weights": w, "threshold": thr}
    for part, y in (("orig", labels), ("sns", labels_sns)):
        if all(part in probs[k] for k in KINDS) and len(y):
            ens = sum(w[k] * probs[k][part] for k in KINDS)
            out["ensemble"][part] = compute_metrics(y, ens, threshold=thr)
            logger.info("[ensemble] %s AUC=%.4f ACC@%.3f=%.4f", part, out["ensemble"][part]["auc"], thr, out["ensemble"][part]["accuracy"])

    run_dir = settings.runs_dir / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"holdout_eval_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", run_dir / f"holdout_eval_{args.tag}.json")


if __name__ == "__main__":
    main()
