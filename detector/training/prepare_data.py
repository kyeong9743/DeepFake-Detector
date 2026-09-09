"""
데이터셋 전처리 — 영상 -> 얼굴 크롭 프레임 캐시.

입력 (컨테이너 기준 /data):
    data/real/*.mp4, data/fake/*.mp4           폴더명으로 라벨 유도
    data/manifest.csv (선택)                    path,label[,source][,split]

출력:
    data/cache/{video_id}/f_0000.jpg ...        얼굴 크롭 프레임 (정사각)
    data/cache/index.json                        영상별 메타(라벨, 프레임 수, 얼굴 검출 비율, 원본 경로)

매 에폭 영상 디코딩을 반복하지 않도록 한 번만 디코딩해 캐시한다. 학습은 캐시만 읽는다.

사용:
    python -m training.prepare_data --frames 64 --size 384
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common.face import FaceDetector, FaceTracker
from config import settings

logger = logging.getLogger("prepare_data")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".m4v"}


def read_manifest(data_dir: Path, manifest: Path | None = None) -> list[dict]:
    """manifest.csv 가 있으면 사용, 없으면 real/ fake/ 폴더에서 생성."""
    m = manifest or (data_dir / "manifest.csv")
    rows: list[dict] = []
    if m.exists():
        with open(m, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                p = (data_dir / r["path"]).resolve() if not Path(r["path"]).is_absolute() else Path(r["path"])
                rows.append({"path": str(p), "label": int(r["label"]),
                             "source": r.get("source", ""), "split": r.get("split", ""),
                             "method": r.get("method", ""), "group": r.get("group", ""), "gender": r.get("gender", "")})
        logger.info("manifest.csv 에서 %d 건 로드", len(rows))
        return rows
    for label, sub in ((0, "real"), (1, "fake")):
        d = data_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in VIDEO_EXT:
                rows.append({"path": str(p), "label": label, "source": sub, "split": ""})
    logger.info("폴더에서 %d 건 수집 (real=%d, fake=%d)", len(rows),
                sum(r["label"] == 0 for r in rows), sum(r["label"] == 1 for r in rows))
    return rows


def video_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def _frame_positions(total: int, n_frames: int, clips: int, clip_len: int, clip_stride: int) -> list[int]:
    """
    clips == 0 : 영상 전체에서 n_frames 개 균등 샘플 (seek n_frames 회 — HEVC 장편에서 느림)
    clips  > 0 : clips 개 구간을 균등 배치, 각 구간에서 clip_stride 간격으로 clip_len 프레임 연속 추출
                 (seek clips 회 + 순차 디코드 — 빠르고, LSTM 이 실제 연속 움직임을 학습)
    """
    if clips <= 0:
        return np.linspace(0, total - 1, num=min(n_frames, total), dtype=int).tolist()
    span = (clip_len - 1) * clip_stride + 1
    if total <= span: # 짧은 영상: 처음부터 가능한 만큼
        return list(range(0, total, clip_stride))[:clip_len]
    starts = np.linspace(0, total - span, num=clips, dtype=int)
    pos: list[int] = []
    for s in starts:
        pos += [int(s + k * clip_stride) for k in range(clip_len)]
    return pos


def process_video(row: dict, cache_dir: Path, n_frames: int, size: int, det: FaceDetector,
                  clips: int = 0, clip_len: int = 16, clip_stride: int = 2) -> dict | None:
    vid = video_id(row["path"])
    out_dir = cache_dir / vid
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    cap = cv2.VideoCapture(row["path"])
    if not cap.isOpened():
        logger.warning("열 수 없음: %s", row["path"])
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total < 8:
        cap.release()
        logger.warning("프레임 부족(%d): %s", total, row["path"])
        return None

    idxs = _frame_positions(total, n_frames, clips, clip_len, clip_stride) # 시간 순서 유지
    # 구간(연속 프레임) 모드에서는 4프레임마다 검출, 사이는 박스 재사용 (얼굴이 0.1~0.2초 안에 크게 움직이지 않음)
    tracker = FaceTracker(det, detect_every=4 if clips > 0 else 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    cur = -1 # 현재 디코더 위치 (다음에 read() 될 프레임 번호)
    for fi in idxs:
        # 연속/근접 위치는 seek 대신 grab() 으로 순차 진행 (HEVC 키프레임 재디코드 비용 회피)
        if cur < 0 or fi < cur or fi - cur > 30:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); cur = fi
        while cur < fi:
            if not cap.grab():
                break
            cur += 1
        ok, img = cap.read(); cur += 1
        if not ok:
            continue
        crop, _ = tracker.crop(img)
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"f_{saved:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    if saved < 4:
        return None
    meta = {
        "id": vid, "path": row["path"], "label": int(row["label"]), "source": row.get("source", ""),
        "split": row.get("split", ""), "method": row.get("method", ""), "group": row.get("group", ""),
        "gender": row.get("gender", ""), "n_frames": saved, "face_ratio": round(tracker.face_ratio, 3),
        "clips": clips, "clip_len": clip_len if clips > 0 else 0, "clip_stride": clip_stride,
        "fps": fps, "total_frames": total, "size": size,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="영상 -> 얼굴 크롭 프레임 캐시")
    ap.add_argument("--data-dir", type=Path, default=settings.data_dir)
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.csv 경로 (기본: data-dir/manifest.csv)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="캐시 디렉터리 (기본: data-dir/cache)")
    ap.add_argument("--frames", type=int, default=64, help="영상당 추출 프레임 수 (--clips 0 일 때 균등 샘플)")
    ap.add_argument("--clips", type=int, default=0, help="구간 수. >0 이면 구간별 연속 프레임 추출 (예: --clips 3 --clip-len 16)")
    ap.add_argument("--clip-len", type=int, default=16, help="구간당 프레임 수 (LSTM window 와 같게)")
    ap.add_argument("--clip-stride", type=int, default=2, help="구간 내 프레임 간격 (2 -> 30fps 영상에서 15fps)")
    ap.add_argument("--size", type=int, default=384, help="캐시 해상도 (학습 시 다시 리사이즈 가능)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-face-ratio", type=float, default=0.2,
                    help="얼굴 검출 비율이 이보다 낮은 영상은 index 에서 제외")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = read_manifest(args.data_dir, args.manifest)
    if not rows:
        raise SystemExit(f"영상이 없습니다. {args.data_dir}/real, {args.data_dir}/fake 에 영상을 넣거나 --manifest 를 지정하십시오.")
    labels = {r["label"] for r in rows}
    if labels != {0, 1}:
        raise SystemExit(f"real(0)과 fake(1)이 모두 있어야 합니다. 현재 라벨: {labels}")
    n_synth = sum(Path(r["path"]).name.startswith("synth_") for r in rows)
    if 0 < n_synth < len(rows):
        logger.warning("합성 스모크 테스트 영상 %d개가 실데이터와 섞여 있습니다. "
                       "실학습 전에 data/real, data/fake 의 synth_*.mp4 와 data/cache 를 삭제하십시오.", n_synth)

    cache_dir = args.cache_dir or (args.data_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    det = FaceDetector()
    index: list[dict] = []
    skipped_face = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_video, r, cache_dir, args.frames, args.size, det,
                          args.clips, args.clip_len, args.clip_stride): r for r in rows}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="전처리"):
            meta = fut.result()
            if meta is None:
                continue
            if meta["face_ratio"] < args.min_face_ratio:
                skipped_face += 1
                continue
            index.append(meta)

    index.sort(key=lambda m: m["path"])
    with open(cache_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    n_real = sum(m["label"] == 0 for m in index)
    n_fake = sum(m["label"] == 1 for m in index)
    logger.info("완료: %d 영상 캐시 (real=%d, fake=%d), 얼굴 비율 미달 제외=%d", len(index), n_real, n_fake, skipped_face)
    if min(n_real, n_fake) < 10:
        logger.warning("클래스별 영상이 10개 미만입니다. 학습 결과의 신뢰성이 매우 낮습니다.")


if __name__ == "__main__":
    main()
