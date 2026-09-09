"""
CLI — 서버 없이 영상 1개(파일 또는 URL)를 분석한다.

    # 컨테이너 안에서
    python -m inference.cli /data/samples/test.mp4 --mode fast
    python -m inference.cli "https://www.youtube.com/watch?v=..." --mode high --out /reports/cli_test

    # 호스트에서 (compose)
    docker compose run --rm --no-deps detector python -m inference.cli /data/samples/test.mp4

옵션:
    --mode fast|high    (기본: .env DEFAULT_MODE)
    --out DIR           산출물 디렉터리 (기본: /reports/cli_{timestamp})
    --no-gradcam        히트맵 생략 (빠른 점수만)
    --json              결과 JSON 만 stdout 에 출력 (스크립트 연동용)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from common.video import download_video
from config import settings
from inference.engine import DetectorEngine


def _progress(stage: str, cur: int, tot: int, msg: str) -> None:
    pct = int(cur / tot * 100) if tot else 0
    print(f"\r  [{stage:<10}] {pct:3d}%  {msg:<40}", end="", file=sys.stderr, flush=True)
    if stage == "done":
        print(file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="CAKE 딥페이크 분석 CLI")
    ap.add_argument("source", help="영상 파일 경로 또는 http(s) URL (YouTube 지원)")
    ap.add_argument("--mode", choices=["fast", "high"], default=settings.default_mode)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-gradcam", action="store_true")
    ap.add_argument("--json", action="store_true", help="결과 JSON 만 출력")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.json else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    analysis_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out or (settings.reports_dir / analysis_id)
    prog = None if args.json else _progress

    t0 = time.time()
    src = args.source
    if src.startswith(("http://", "https://")):
        src = download_video(src, settings.media_dir / "downloads", progress=prog)
    elif not Path(src).exists():
        print(f"파일이 없습니다: {src}", file=sys.stderr)
        return 2

    try:
        engine = DetectorEngine(args.mode)
    except FileNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        return 3

    result = engine.analyze_video(src, analysis_id, out_dir=out_dir, progress=prog, gradcam=not args.no_gradcam)

    if args.json:
        slim = {k: v for k, v in result.items() if k not in ("frame_scores", "lstm_windows")}
        print(json.dumps(slim, ensure_ascii=False, indent=1))
        return 0

    s = result["scores"]
    print("\n=== CAKE 분석 결과 ===")
    print(f"입력        : {args.source}")
    print(f"모드        : {result['mode']}  ({result['frame_count']} 프레임, 얼굴 검출 {result['face_ratio']*100:.0f}%)")
    print(f"위험도      : {result['risk_level']}   (딥페이크 점수 {result['deepfake_score']*100:.1f} / 임계값 {result['threshold']*100:.0f})")
    print(f"신뢰도      : {result['confidence']*100:.1f}%")
    print(f"모델별 점수 : CNN {s['cnn']*100:.1f}  LSTM {s['lstm']*100:.1f}  FFT {s['frequency']*100:.1f}"
          f"   가중치 {result['weights']}")
    print(f"투표        : fake {result['vote']['fake']} / {result['vote']['total']}")
    print(f"입력 품질   : 얼굴 {result.get('face_px', '?')}px · 얼굴 없는 프레임 {result.get('frames_no_face', 0)}/{result.get('frames_extracted', '?')} 제외")
    for m in result.get("quality_messages", []):
        print(f"경고        : {m}")
    print(f"산출물      : {out_dir}  (report.html, scores.csv, heatmap.mp4 ...)")
    print(f"소요 시간   : {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
