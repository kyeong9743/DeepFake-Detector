"""
합성 스모크 테스트 데이터셋 생성.

실데이터 확보 전에 prepare_data -> train -> ensemble_opt -> 추론 파이프라인이
끝까지 동작하는지 검증하기 위한 것. 모델 성능 평가용이 아님

real : 부드럽게 움직이는 얼굴 모양 패턴 + 자연스러운 조명 변화
fake : 같은 패턴에 얼굴 경계, 고주파 노이즈, 프레임 간 깜빡임 추가
-> CNN(패턴), Frequency(격자), LSTM(시간적 흐름)

    python -m training.make_synthetic --n 40 --out /data
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def _face(canvas: np.ndarray, cx: int, cy: int, r: int, tone: tuple[int, int, int]) -> None:
    cv2.ellipse(canvas, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, tone, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx - r // 2, cy - r // 4), r // 7, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx + r // 2, cy - r // 4), r // 7, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy + r // 2), (r // 2, r // 5), 0, 0, 180, (60, 40, 90), 3, cv2.LINE_AA)


def make_video(path: Path, fake: bool, seed: int, n_frames: int = 90, size: int = 480, fps: int = 30) -> None:
    rng = random.Random(seed)
    bg = np.array([rng.randint(60, 200) for _ in range(3)], dtype=np.uint8)
    tone = (rng.randint(150, 220), rng.randint(140, 200), rng.randint(180, 240))
    cx0, cy0, r = size // 2 + rng.randint(-40, 40), size // 2 + rng.randint(-30, 30), rng.randint(70, 100)
    vx, vy = rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    grid = None
    if fake:
        yy, xx = np.mgrid[0:size, 0:size]
        freq = rng.choice([6, 8, 12])
        grid = (((np.sin(xx / freq * np.pi) * np.sin(yy / freq * np.pi)) > 0.6) * 18).astype(np.int16)
    for t in range(n_frames):
        frame = np.empty((size, size, 3), np.uint8); frame[:] = bg
        light = 1.0 + 0.08 * np.sin(t / 15.0) # 자연스러운 조명 변화
        cx, cy = int(cx0 + vx * t), int(cy0 + vy * t)
        _face(frame, cx, cy, r, tuple(int(min(255, c * light)) for c in tone))
        # 배경 텍스처
        cv2.putText(frame, f"clip {seed}", (20, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if fake:
            # CNN(패턴)
            ring = np.zeros_like(frame); cv2.ellipse(ring, (cx, cy), (r + 6, int(r * 1.25) + 6), 0, 0, 360, (200, 200, 255), 8, cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 1.0, ring, 0.25, 0)
            # Frequency(격자)
            mask = np.zeros((size, size), np.uint8); cv2.ellipse(mask, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, 255, -1)
            f16 = frame.astype(np.int16); f16[mask > 0] += grid[..., None][mask > 0]
            frame = np.clip(f16, 0, 255).astype(np.uint8)
            # LSTM(시간적 흐름)
            if t % 7 == 0:
                frame = np.clip(frame.astype(np.int16) + rng.randint(-25, 25), 0, 255).astype(np.uint8)
        noise = np.random.default_rng(seed * 1000 + t).normal(0, 3, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="클래스별 영상 수")
    ap.add_argument("--out", type=Path, default=Path("/data"))
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()
    for label, sub in ((0, "real"), (1, "fake")):
        d = args.out / sub; d.mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            make_video(d / f"synth_{sub}_{i:03d}.mp4", fake=bool(label), seed=label * 10000 + i, n_frames=args.frames)
    print(f"생성 완료: {args.out}/real ({args.n}), {args.out}/fake ({args.n})")
    print("주의: 합성 데이터는 파이프라인 검증용입니다. 얼굴 검출이 되지 않으므로 prepare_data 에 --min-face-ratio 0 설정하세요")


if __name__ == "__main__":
    main()
