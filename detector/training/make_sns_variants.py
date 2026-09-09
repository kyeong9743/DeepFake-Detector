"""
SNS 압축 변형본 생성 — 얼굴 크롭 캐시의 각 클립(연속 프레임)을 실제 H.264 저비트레이트로 재인코딩해
{cache}/{id}/sns{k}/f_XXXX.jpg 로 저장한다.

    python -m training.make_sns_variants --variants 2 --workers 12
    python -m training.make_sns_variants --variants 1 --only-holdout      # 평가용(split=val)만

왜 오프라인인가: 프레임 단위 JPEG 증강(augment.py)은 H.264 의 블록/디블로킹/시간축(P-frame) 아티팩트를
재현하지 못한다. 클립을 ffmpeg libx264 로 인코딩->디코딩하면 SNS 재업로드와 같은 종류의 열화가 생긴다.
학습 중 매번 인코딩하면 DataLoader 가 병목이 되므로 미리 만들어 둔다 (변형본 1개 ≈ 원본 캐시 크기).

변형 파라미터(클립마다 랜덤, 재현 가능 seed): 해상도 축소 0.5~0.85 (-> 업스케일 복원) · CRF 23~34 ·
preset medium · 선택적 선명화/스무딩 필터(업로드 전 앱 필터 흉내). 파라미터는 sns{k}/params.json 에 기록.
  (처음 시도한 0.4~0.8 · CRF 28~42 · veryfast 는 정지 얼굴에서 P-frame 이 전부 skip 되어 16프레임이 완전히
   동일해지는 등 시간축 정보가 소실됨 — 2026-09-07 실측. medium 은 같은 CRF 에서 미세 움직임을 유지하고 속도 동일)
index.json 의 각 항목에 "sns_variants": k 가 기록되며 dataset 이 이를 읽는다.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from config import settings

logger = logging.getLogger("make_sns_variants")
PARAMS_VERSION = 2 # 파라미터 범위를 바꾸면 올릴 것 -> 구버전 변형본 자동 재생성


def _filter(img: np.ndarray, kind: str, amount: float) -> np.ndarray:
    if kind == "sharpen":
        blur = cv2.GaussianBlur(img, (0, 0), 1.5)
        return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    if kind == "smooth":
        base = cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)
        return cv2.addWeighted(img, 1 - amount, base, amount, 0)
    return img


def h264_roundtrip(frames: list[np.ndarray], scale: float, crf: int, fps: int = 15) -> list[np.ndarray]:
    """프레임 목록 -> (축소) -> libx264 인코딩 -> 디코딩 -> 원래 크기로 복원."""
    h, w = frames[0].shape[:2]
    sw, sh = max(16, int(w * scale) // 2 * 2), max(16, int(h * scale) // 2 * 2)
    with tempfile.TemporaryDirectory() as td:
        mp4 = Path(td) / "c.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "pipe:0", "-vf", f"scale={sw}:{sh}:flags=area", "-c:v", "libx264", "-preset", "medium",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-g", "30", "-an", str(mp4)]
        proc = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                              capture_output=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="ignore")[:300])
        cap = cv2.VideoCapture(str(mp4))
        out = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            out.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR))
        cap.release()
    if len(out) != len(frames): # 디코드 프레임 수가 다르면(드묾) 마지막 프레임으로 채움/자름
        out = (out + [out[-1]] * len(frames))[:len(frames)] if out else frames
    return out


def process_video(meta: dict, cache_dir: str, variants: int, seed: int, force: bool = False) -> tuple[str, int, str | None]:
    vdir = Path(cache_dir) / meta["id"]
    n, cl = int(meta["n_frames"]), int(meta.get("clip_len") or 0) or int(meta["n_frames"])
    done = 0
    for k in range(variants):
        out = vdir / f"sns{k}"
        pj = out / "params.json"
        if pj.exists() and not force:
            try:
                with open(pj, encoding="utf-8") as f:
                    if json.load(f).get("version") == PARAMS_VERSION:
                        done += 1; continue
            except Exception: # noqa: BLE001
                pass
        if out.exists():
            shutil.rmtree(out) # 구버전/불완전 변형본 제거 후 재생성
        rng = random.Random(f"{seed}:{meta['id']}:{k}")
        out.mkdir(parents=True, exist_ok=True)
        params = []
        try:
            for s in range(0, n, cl):
                idx = list(range(s, min(n, s + cl)))
                frames = [cv2.imread(str(vdir / f"f_{i:04d}.jpg"), cv2.IMREAD_COLOR) for i in idx]
                if any(f is None for f in frames):
                    raise IOError(f"프레임 누락: {vdir}")
                r = rng.random()
                fkind, famt = ("sharpen", rng.uniform(0.3, 1.0)) if r < 0.25 else (("smooth", rng.uniform(0.3, 0.7)) if r < 0.5 else ("none", 0.0))
                scale, crf = rng.uniform(0.5, 0.85), rng.randint(23, 34)
                frames = [_filter(f, fkind, famt) for f in frames]
                frames = h264_roundtrip(frames, scale, crf)
                for i, f in zip(idx, frames):
                    cv2.imwrite(str(out / f"f_{i:04d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
                params.append({"start": s, "n": len(idx), "filter": fkind, "amount": round(famt, 3),
                               "scale": round(scale, 3), "crf": crf})
            with open(out / "params.json", "w", encoding="utf-8") as f:
                json.dump({"version": PARAMS_VERSION, "clips": params}, f)
            done += 1
        except Exception as e: # noqa: BLE001
            return meta["id"], done, f"{type(e).__name__}: {e}"
    return meta["id"], done, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=settings.data_dir / "cache")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only-holdout", action="store_true", help="split=val 영상만 (평가용)")
    ap.add_argument("--limit", type=int, default=0, help="디버그: 앞 N개만")
    ap.add_argument("--force", action="store_true", help="기존 변형본을 지우고 다시 생성")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    idx_path = args.cache_dir / "index.json"
    with open(idx_path, encoding="utf-8") as f:
        index = json.load(f)
    targets = [m for m in index if (not args.only_holdout or m.get("split") == "val")]
    if args.limit:
        targets = targets[:args.limit]
    logger.info("대상 %d 영상 × 변형 %d (cache=%s)", len(targets), args.variants, args.cache_dir)

    results: dict[str, int] = {}
    errors = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_video, m, str(args.cache_dir), args.variants, args.seed, args.force) for m in targets]
        for i, fu in enumerate(as_completed(futs), 1):
            vid, done, err = fu.result()
            results[vid] = done
            if err:
                errors += 1
                if errors <= 20:
                    logger.warning("%s: %s", vid, err)
            if i % 500 == 0 or i == len(futs):
                logger.info("진행 %d/%d (오류 %d)", i, len(futs), errors)

    for m in index:
        if m["id"] in results:
            m["sns_variants"] = results[m["id"]]
            meta_p = args.cache_dir / m["id"] / "meta.json"
            if meta_p.exists():
                try:
                    with open(meta_p, encoding="utf-8") as f:
                        mm = json.load(f)
                    mm["sns_variants"] = results[m["id"]]
                    with open(meta_p, "w", encoding="utf-8") as f:
                        json.dump(mm, f, ensure_ascii=False)
                except Exception: # noqa: BLE001
                    pass
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    n_ok = sum(1 for m in index if m.get("sns_variants", 0) >= args.variants)
    logger.info("완료: 변형 %d개 보유 영상 %d/%d, 오류 %d. index.json 갱신", args.variants, n_ok, len(index), errors)


if __name__ == "__main__":
    main()
