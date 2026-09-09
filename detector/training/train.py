"""
모델 학습 — K-fold 교차검증, AMP, 조기 종료, 지표 로깅, OOF 예측 저장.

    python -m training.train --mode high --model all --folds 5 --epochs 20

동작:
  - data/cache/index.json 의 영상 목록을 Stratified K-fold 로 분할 (영상 단위 -> 프레임 누출 없음)
  - fold 마다 모델을 새로 학습, 검증 AUC 최고 에폭을 fold 체크포인트로 저장
  - fold 검증 영상에 대한 예측을 모아 out-of-fold(OOF) 예측 CSV 저장 -> ensemble_opt 입력
  - 모든 fold 완료 후 **전체 데이터**로 최종 모델을 한 번 더 학습(에폭 = fold 평균 best epoch)
    -> checkpoints/{mode}_{kind}.pt  (추론이 사용)
  - runs/{mode}/ 에 지표 JSONL, 곡선 PNG, ROC/PR, confusion matrix 저장
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PROFILES, settings
from models.registry import KINDS, build_model, save_checkpoint
from training.dataset import FrameDataset, SequenceDataset, load_index, split_videos
from training.metrics import MetricLogger, compute_metrics, plot_confusion, plot_roc_pr

logger = logging.getLogger("train")


# --------------------------------------------------------------------------- 유틸
def make_loaders(kind: str, mode: str, train_v: list[dict], val_v: list[dict], cache_dir: Path,
                 batch_size: int, workers: int, aug: float, sns_prob: float = 0.0):
    p = PROFILES[mode]
    if kind == "lstm":
        tr = SequenceDataset(train_v, cache_dir, p.frame_size, p.window, p.stride, train=True, aug_strength=aug,
                             sns_prob=sns_prob)
        va = SequenceDataset(val_v, cache_dir, p.frame_size, p.window, p.stride, train=False)
        bs = max(1, batch_size // 4) # 시퀀스는 프레임 16장 -> 배치 축소
    else:
        tr = FrameDataset(train_v, cache_dir, p.frame_size, frames_per_video=8, train=True, aug_strength=aug,
                          sns_prob=sns_prob)
        va = FrameDataset(val_v, cache_dir, p.frame_size, frames_per_video=8, train=False)
        bs = batch_size
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, **common),
            DataLoader(va, batch_size=bs, shuffle=False, **common))


def forward(kind: str, model: nn.Module, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "lstm":
        from common.preprocess import fft_spectrum, normalize
        xu8, y = batch  # (B,T,3,H,W) uint8 — 워커는 바이트만 전달
        x = xu8.to(device, non_blocking=True).float().div_(255.0)
        b, t = x.shape[:2]
        spatial = normalize(x)
        freq = fft_spectrum(x.flatten(0, 1)).view(b, t, *x.shape[2:])
        return model(spatial, freq), y.to(device)
    x, y = batch
    x = x.to(device, non_blocking=True)
    if kind == "frequency":
        from common.preprocess import fft_spectrum
        # FrameDataset 은 정규화된 이미지를 주므로 역정규화 없이 스펙트럼 계산 (fft_spectrum 이 자체 표준화)
        x = fft_spectrum(x)
    # channels_last: 텐서코어 conv 처리량 향상 (벤치마크 447->568 img/s)
    return model(x.contiguous(memory_format=torch.channels_last)), y.to(device)


def pos_weight(videos: list[dict]) -> float:
    n_pos = sum(v["label"] == 1 for v in videos)
    n_neg = len(videos) - n_pos
    return float(n_neg / max(1, n_pos))


@torch.no_grad()
def evaluate(kind, model, loader, device, criterion) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, probs, labels = [], [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logit, y = forward(kind, model, batch, device)
            losses.append(criterion(logit.float(), y.float()).item())
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        labels.append(y.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def predict_videos(kind: str, mode: str, model: nn.Module, videos: list[dict], cache_dir: Path, device,
                   variant: int | None = None) -> pd.DataFrame:
    """영상 단위 확률 (프레임/윈도우 확률 평균). variant=k -> SNS 압축 변형본 sns{k} 로 평가(강건성 측정)."""
    p = PROFILES[mode]
    rows = []
    model.eval()
    for m in videos:
        if kind == "lstm":
            ds = SequenceDataset([m], cache_dir, p.frame_size, p.window, p.stride, train=False, max_windows_per_video=12,
                                 variant=variant)
        else:
            ds = FrameDataset([m], cache_dir, p.frame_size, frames_per_video=16, train=False, variant=variant)
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        ps = []
        for batch in dl:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logit, _ = forward(kind, model, batch, device)
            ps.append(torch.sigmoid(logit.float()).cpu().numpy())
        rows.append({"video_id": m["id"], "label": m["label"], "prob": float(np.concatenate(ps).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 학습 루프
def train_one(kind: str, mode: str, train_v: list[dict], val_v: list[dict] | None, cache_dir: Path,
              run_dir: Path, tag: str, epochs: int, batch_size: int, lr: float, workers: int,
              aug: float, patience: int, device, sns_prob: float = 0.0) -> tuple[nn.Module, dict, int]:
    p = PROFILES[mode]
    model = build_model(kind, mode, pretrained=True).to(device)
    if kind != "lstm":
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight(train_v), device=device))
    # 백본(사전학습)은 작은 lr, 헤드는 큰 lr
    head_params = [q for n, q in model.named_parameters() if n.startswith("head")]
    body_params = [q for n, q in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body_params, "lr": lr}, {"params": head_params, "lr": lr * 5}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    if val_v is None:
        # 최종(전체 데이터) 학습: 모니터링용 의사 검증셋 — 클래스별로 10% 씩 뽑아 AUC 계산이 가능하게
        k = max(1, len(train_v) // 20)
        val_v = [v for v in train_v if v["label"] == 0][:k] + [v for v in train_v if v["label"] == 1][:k]
    tr_loader, va_loader = make_loaders(kind, mode, train_v, val_v, cache_dir, batch_size, workers, aug, sns_prob)
    mlog = MetricLogger(run_dir, f"{kind}_{tag}")
    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0

    # 최종(final) 학습은 에폭마다 상태를 저장하고, 파일이 있으면 자동으로 이어서 학습 (CUDA 오류 등으로 죽어도 에폭 단위로만 손실)
    state_path = run_dir / f"{kind}_{tag}_state.pt" if tag == "final" else None
    start_ep = 1
    if state_path and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_ep = int(st["epoch"]) + 1
        logger.info("[%s/%s] 상태 파일에서 이어하기: epoch %d 부터 (%s)", kind, tag, start_ep, state_path.name)

    for ep in range(start_ep, epochs + 1):
        model.train()
        tl, t0 = [], time.time()
        pbar = tqdm(tr_loader, desc=f"[{kind}/{tag}] ep{ep}/{epochs}", leave=False)
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logit, y = forward(kind, model, batch, device)
                loss = criterion(logit.float(), y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(tl[-20:]):.4f}")
        sched.step()

        vl, probs, labels = evaluate(kind, model, va_loader, device, criterion)
        vm = compute_metrics(labels, probs)
        mlog.log(ep, float(np.mean(tl)), vl, vm, opt.param_groups[0]["lr"])
        auc = vm.get("auc", float("nan"))
        logger.info("[%s/%s] ep%d train=%.4f val=%.4f auc=%.4f acc=%.4f f1=%.4f (%.0fs)",
                    kind, tag, ep, np.mean(tl), vl, auc, vm["accuracy"], vm["f1"], time.time() - t0)
        score = auc if not np.isnan(auc) else -vl
        # final 은 의사검증(5%)이 첫 에폭부터 AUC 1.0 으로 포화되므로 best 선택을 하지 않고 마지막 에폭 모델을 사용
        # (best 선택 시 ep1 모델이 최종본이 되는 문제 — 2026-09-07 발견)
        if tag != "final" and score > best_auc:
            best_auc, best_epoch, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                logger.info("조기 종료 (patience=%d)", patience)
                break
        if state_path:
            torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict()}, state_path)

    mlog.plot()
    if state_path and state_path.exists():
        state_path.unlink() # 정상 완료 -> 이어하기 상태 파일 제거
    if best_state is not None:
        model.load_state_dict(best_state)
    _, probs, labels = evaluate(kind, model, va_loader, device, criterion)
    final_m = compute_metrics(labels, probs)
    plot_roc_pr(labels, probs, run_dir / f"{kind}_{tag}_roc_pr.png", f"{kind}/{tag}")
    plot_confusion(final_m["confusion_matrix"], run_dir / f"{kind}_{tag}_cm.png", f"{kind}/{tag}")
    return model, final_m, best_epoch


def run(kind: str, mode: str, args, device) -> None:
    cache_dir = args.cache_dir or (settings.data_dir / "cache")
    index = load_index(settings.data_dir, cache_dir)
    run_dir = settings.runs_dir / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    n_real, n_fake = sum(m["label"] == 0 for m in index), sum(m["label"] == 1 for m in index)
    logger.info("=== %s / %s  영상 %d (real=%d, fake=%d)  folds=%d ===", kind, mode, len(index), n_real, n_fake, args.folds)
    if min(n_real, n_fake) < args.folds:
        raise SystemExit(f"클래스별 영상 수({min(n_real, n_fake)})가 fold 수({args.folds})보다 적습니다.")

    oof, fold_metrics, best_epochs = [], [], []
    oof_path = run_dir / f"oof_{kind}.csv"
    if args.skip_folds and oof_path.exists():
        # K-fold 가 이미 끝났고 최종 학습 중 실패한 경우(예: CUDA 오류) — fold 재학습 없이 최종 단계만 재개
        logger.info("[%s] --skip-folds: 기존 %s 재사용, fold 학습 생략", kind, oof_path.name)
        oof_df = pd.read_csv(oof_path)
        for fold in range(args.folds):
            mp = run_dir / f"{kind}_fold{fold}_metrics.jsonl"
            if mp.exists():
                rows = [json.loads(l) for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()]
                best = max(rows, key=lambda r: (r.get("auc") if r.get("auc") == r.get("auc") else -1) or -1)
                fold_metrics.append({k: best.get(k) for k in ("auc", "accuracy", "f1", "threshold", "confusion_matrix")})
                best_epochs.append(int(best["epoch"]))
        if not best_epochs:
            best_epochs = [args.epochs]
    else:
        for fold in range(args.folds):
            tr_v, va_v = split_videos(index, args.folds, fold, seed=args.seed)
            model, m, be = train_one(kind, mode, tr_v, va_v, cache_dir, run_dir, f"fold{fold}", args.epochs,
                                     args.batch_size, args.lr, args.workers, args.aug, args.patience, device,
                                     sns_prob=args.sns_prob)
            fold_metrics.append(m); best_epochs.append(be)
            # OOF 는 fold 검증 영상만 (홀드아웃 split=val 은 매 fold 에 붙으므로 제외 -> 중복 방지)
            oof.append(predict_videos(kind, mode, model, [v for v in va_v if v.get("split") != "val"], cache_dir, device))
            save_checkpoint(model, kind, mode, m, path=run_dir / f"{kind}_fold{fold}.pt", extra={"fold": fold})
            del model; torch.cuda.empty_cache()
        oof_df = pd.concat(oof, ignore_index=True)
        oof_df.to_csv(oof_path, index=False)
    oof_m = compute_metrics(oof_df["label"].to_numpy(), oof_df["prob"].to_numpy())
    plot_roc_pr(oof_df["label"].to_numpy(), oof_df["prob"].to_numpy(), run_dir / f"{kind}_oof_roc_pr.png", f"{kind} OOF")
    logger.info("[%s] K-fold OOF(영상 단위): AUC=%.4f ACC=%.4f F1=%.4f  fold AUC=%s",
                kind, oof_m.get("auc", float("nan")), oof_m["accuracy"], oof_m["f1"],
                [round(f.get("auc", float("nan")), 3) for f in fold_metrics])

    # 변조 방법별 OOF 지표 (AI Hub: fo / dfl / dffs / fsgan / audio_driven / anti_*)
    by_method: dict[str, dict] = {}
    meta_by_id = {m["id"]: m for m in index}
    oof_df["method"] = oof_df["video_id"].map(lambda i: meta_by_id.get(i, {}).get("method", "") or "unknown")
    reals = oof_df[oof_df["label"] == 0]
    for meth, g in oof_df[oof_df["label"] == 1].groupby("method"):
        sub = pd.concat([g, reals])
        by_method[meth] = {"n_fake": int(len(g)), "recall@0.5": float((g["prob"] >= 0.5).mean()),
                           "auc_vs_real": compute_metrics(sub["label"].to_numpy(), sub["prob"].to_numpy()).get("auc")}
    if by_method:
        logger.info("[%s] 변조 방법별: %s", kind, {k: f"recall={v['recall@0.5']:.2f} auc={v['auc_vs_real']:.3f}" for k, v in by_method.items()})

    # 최종 모델: K-fold 대상(pool) 전체로 학습, 에폭 = fold best epoch 평균 (최소 3).
    # manifest split='val' 홀드아웃은 최종 학습에서도 제외하고 별도 평가 -> 정직한 최종 지표
    pool = [m for m in index if m.get("split") != "val"]
    holdout = [m for m in index if m.get("split") == "val"]
    final_epochs = max(3, int(round(np.mean(best_epochs))))
    logger.info("[%s] 최종 학습 (%d 영상, %d 에폭, 홀드아웃 %d)", kind, len(pool), final_epochs, len(holdout))
    model, _, _ = train_one(kind, mode, pool, None, cache_dir, run_dir, "final", final_epochs,
                            args.batch_size, args.lr, args.workers, args.aug, patience=10**9, device=device,
                            sns_prob=args.sns_prob)
    holdout_m, holdout_sns_m = None, None
    if holdout:
        hdf = predict_videos(kind, mode, model, holdout, cache_dir, device)
        holdout_m = compute_metrics(hdf["label"].to_numpy(), hdf["prob"].to_numpy())
        hdf.to_csv(run_dir / f"holdout_{kind}.csv", index=False)
        plot_roc_pr(hdf["label"].to_numpy(), hdf["prob"].to_numpy(), run_dir / f"{kind}_holdout_roc_pr.png", f"{kind} holdout")
        logger.info("[%s] 홀드아웃(AI Hub Validation) 최종 모델: AUC=%.4f ACC=%.4f F1=%.4f",
                    kind, holdout_m.get("auc", float("nan")), holdout_m["accuracy"], holdout_m["f1"])
        # SNS 압축 강건성: 같은 홀드아웃을 H.264 저비트레이트 변형본(sns0)으로 평가 (make_sns_variants 가 만든 경우만)
        sns_hold = [m for m in holdout if int(m.get("sns_variants") or 0) > 0]
        if sns_hold:
            sdf = predict_videos(kind, mode, model, sns_hold, cache_dir, device, variant=0)
            holdout_sns_m = compute_metrics(sdf["label"].to_numpy(), sdf["prob"].to_numpy())
            sdf.to_csv(run_dir / f"holdout_sns_{kind}.csv", index=False)
            logger.info("[%s] 홀드아웃 SNS 압축 변형본(%d): AUC=%.4f ACC=%.4f F1=%.4f",
                        kind, len(sns_hold), holdout_sns_m.get("auc", float("nan")), holdout_sns_m["accuracy"], holdout_sns_m["f1"])
    path = save_checkpoint(model, kind, mode, {"oof": oof_m, "holdout": holdout_m, "holdout_sns": holdout_sns_m,
                                               "folds": fold_metrics, "final_epochs": final_epochs, "by_method": by_method},
                           extra={"n_videos": len(pool), "n_real": n_real, "n_fake": n_fake, "folds": args.folds,
                                  "sns_prob": args.sns_prob, "aug": args.aug})
    with open(run_dir / f"{kind}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"oof": oof_m, "holdout": holdout_m, "holdout_sns": holdout_sns_m, "by_method": by_method,
                   "folds": fold_metrics, "best_epochs": best_epochs, "final_epochs": final_epochs,
                   "sns_prob": args.sns_prob, "aug": args.aug, "checkpoint": str(path)},
                  f, ensure_ascii=False, indent=2)
    logger.info("[%s] 저장: %s", kind, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="CAKE 모델 학습 (K-fold)")
    ap.add_argument("--mode", default=settings.default_mode, choices=["fast", "high"])
    ap.add_argument("--model", default="all", choices=["all", *KINDS])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aug", type=float, default=1.0, help="증강 강도 0~1")
    ap.add_argument("--sns-prob", type=float, default=0.0,
                    help="학습 샘플을 SNS 압축 변형본(make_sns_variants)으로 읽을 확률 (예: 0.5). 0 이면 원본만")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=Path, default=None, help="prepare_data 캐시 디렉터리 (기본: /data/cache)")
    ap.add_argument("--skip-folds", action="store_true",
                    help="runs/{mode}/oof_{kind}.csv 가 있으면 K-fold 를 건너뛰고 최종 학습만 수행 (최종 단계 실패 후 재개용)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(settings.runs_dir / f"train_{args.mode}.log", encoding="utf-8")])
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True          # 고정 입력 크기 -> 최적 conv 알고리즘 캐시 (벤치마크 +25%)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s  %s", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    kinds = list(KINDS) if args.model == "all" else [args.model]
    for k in kinds:
        run(k, args.mode, args, device)
    if args.model == "all":
        logger.info("세 모델 학습 완료. 앙상블 가중치 최적화:  python -m training.ensemble_opt --mode %s", args.mode)


if __name__ == "__main__":
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    main()
