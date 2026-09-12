"""
data/samples/ 의 실제 SNS 영상을 현재 체크포인트로 일괄 분석 -> 일반화(도메인 밖) 점검.
라벨은 --labels CSV(file,label; 1=fake 0=real -1=범위 밖) 로 파일별 지정. 없으면 --label 로 전체 동일 라벨.
범위 밖(-1: VFX·완전 생성 영상 등 얼굴 변조가 아닌 것)은 점수만 기록하고 정답 집계에서 뺀다.

    docker compose exec -T detector python -m training.eval_samples --tag v1_labeled --labels /data/samples/labels.csv
    -> runs/samples_eval_{tag}.json + 표 출력.  Grad-CAM 생략(점수만).

같은 파일(중복)은 md5 로 묶어 한 번만 센다. 임계값은 ensemble.json 의 threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from config import settings                       # noqa: E402
from inference.engine import DetectorEngine       # noqa: E402


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=settings.default_mode)
    ap.add_argument("--dir", type=Path, default=settings.data_dir / "samples")
    ap.add_argument("--tag", required=True, help="결과 파일 이름 (예: v1, v2_sns)")
    ap.add_argument("--label", type=int, default=1, help="샘플 실제 라벨 (기본 1=fake). --labels 가 없을 때만 사용")
    ap.add_argument("--labels", type=Path, default=None, help="파일별 라벨 CSV (file,label). 1=fake, 0=real, -1=범위 밖(집계 제외)")
    args = ap.parse_args()

    labels: dict[str, int] = {}
    if args.labels:
        with open(args.labels, encoding="utf-8-sig", newline="") as f:
            labels = {r["file"].strip(): int(r["label"]) for r in csv.DictReader(f)}

    engine = DetectorEngine(args.mode)
    thr = engine.ens_cfg["threshold"]
    rows, seen = [], {}
    for p in sorted(args.dir.glob("*.mp4")):
        h = md5(p)
        if h in seen:
            rows.append({"file": p.name, "dup_of": seen[h]}); continue
        seen[h] = p.name
        t0 = time.time()
        r = engine.analyze_video(str(p), f"eval_{args.tag}_{h}", out_dir=Path("/tmp") / f"eval_{h}", gradcam=False)
        s = r["scores"]
        lab = labels.get(p.name, args.label)
        rows.append({"file": p.name, "label": lab, "score": round(r["deepfake_score"], 4), "risk": r["risk_level"],
                     "pred_fake": bool(r["deepfake_score"] >= thr), "correct": (bool(r["deepfake_score"] >= thr) == (lab == 1)) if lab >= 0 else None,
                     "vote": f'{r["vote"]["fake"]}/{r["vote"]["total"]}',
                     "cnn": round(s["cnn"], 3), "lstm": round(s["lstm"], 3), "fft": round(s["frequency"], 3),
                     "face_ratio": r["face_ratio"], "face_px": r.get("face_px"), "frames_no_face": r.get("frames_no_face"),
                     "quality_warnings": r.get("quality_warnings", []), "sec": round(time.time() - t0, 1)})
    uniq = [r for r in rows if "dup_of" not in r]
    fakes = [r for r in uniq if r["label"] == 1]; reals = [r for r in uniq if r["label"] == 0]
    oos = [r for r in uniq if r["label"] < 0]
    scored = fakes + reals
    n_hit = sum(bool(r["correct"]) for r in scored)
    tp = sum(r["pred_fake"] for r in fakes); tn = sum(not r["pred_fake"] for r in reals)
    out = {"tag": args.tag, "mode": args.mode, "weights": engine.ens_cfg["weights"], "threshold": thr,
           "n_unique": len(uniq), "n_fake": len(fakes), "n_real": len(reals), "n_out_of_scope": len(oos), "n_correct": n_hit,
           "accuracy": round(n_hit / max(1, len(scored)), 3),
           "fake_detected": tp, "recall": round(tp / len(fakes), 3) if fakes else None,
           "real_correct": tn, "specificity": round(tn / len(reals), 3) if reals else None,
           "rows": rows}
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.runs_dir / f"samples_eval_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n[{args.tag}] mode={args.mode} weights={out['weights']} thr={thr:.3f}")
    print(f"{'file':<36}{'label':<6}{'score':>7}  {'risk':<8}{'pred':<6}{'vote':<5}{'cnn':>6}{'lstm':>6}{'fft':>6}{'face':>6}{'px':>5}{'drop':>5}  warnings")
    for r in sorted(uniq, key=lambda r: -r["score"]):
        print(f"{r['file'][:35]:<36}{'FAKE' if r['label'] == 1 else ('real' if r['label'] == 0 else 'OOS'):<6}{r['score']*100:>7.1f}  {r['risk']:<8}{'FAKE' if r['pred_fake'] else 'real':<6}"
              f"{r['vote']:<5}{r['cnn']:>6.2f}{r['lstm']:>6.2f}{r['fft']:>6.2f}{r['face_ratio']:>6.2f}"
              f"{str(r.get('face_px') or '-'):>5}{str(r.get('frames_no_face') or 0):>5}  {','.join(r.get('quality_warnings') or [])}")
    for r in rows:
        if "dup_of" in r:
            print(f"{r['file'][:35]:<36}   (= {r['dup_of']})")
    print(f"정답: {n_hit}/{len(scored)} = {out['accuracy']*100:.0f}%  |  가짜 탐지 {tp}/{len(fakes)}  |  진짜 정답 {tn}/{len(reals)}  |  범위 밖 {len(oos)}개(집계 제외)")


if __name__ == "__main__":
    main()
