"""
영상 프레임 추출 — 시간 순서를 보존하는 적응형 샘플링.

- 분석 구간: 영상 중앙 target_duration 초 (짧으면 전체)
- 기본 샘플링 간격 + 장면 전환/큰 모션 프레임 우선 포함
- max_frames 초과 시 중요도 상위 N개를 고르되 **시간 순서로 재정렬** (LSTM 입력 정합성)
- 프레임마다 원본 프레임 번호와 타임스탬프를 함께 반환 (리포트에서 "몇 초 지점" 표기 가능)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from config import ModeProfile, settings

ProgressCb = Callable[[str, int, int, str], None] | None


@dataclass
class Frame:
    index: int          # 원본 프레임 번호
    time_sec: float     # 원본 타임스탬프
    image: np.ndarray   # BGR uint8
    importance: float   # 샘플링 우선순위 (장면전환=1.0)


@dataclass
class VideoInfo:
    path: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int


def probe(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if total <= 0 or fps <= 0:
        raise ValueError("영상 메타데이터를 읽을 수 없습니다 (프레임 수 또는 FPS가 0).")
    return VideoInfo(video_path, fps, total, total / fps, w, h)


def _gray_small(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA)


def extract_frames(video_path: str, profile: ModeProfile, progress: ProgressCb = None) -> tuple[list[Frame], VideoInfo]:
    info = probe(video_path)
    cap = cv2.VideoCapture(video_path)

    # 분석 구간 결정 (중앙)
    if info.duration <= profile.target_duration:
        start, end = 0, info.total_frames
    else:
        span = int(profile.target_duration * info.fps)
        start = (info.total_frames - span) // 2
        end = start + span

    interval = max(1, int(round(info.fps / profile.target_fps)))
    scene_thr = 22.0 # 그레이 평균 차이 (0~255)
    motion_thr = 8.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[Frame] = []
    prev_small: np.ndarray | None = None
    idx = start
    total_span = max(1, end - start)

    while idx < end:
        ok, img = cap.read()
        if not ok:
            break
        small = _gray_small(img)
        importance = 0.0
        if prev_small is not None:
            # float32 로 캐스팅해 uint8 언더플로 방지
            diff = float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32))))
            if diff > scene_thr:
                importance = 1.0
            elif diff > motion_thr:
                importance = min(0.99, diff / scene_thr)
        prev_small = small

        if (idx - start) % interval == 0 or importance > 0.0:
            frames.append(Frame(idx, idx / info.fps, img, importance if importance > 0 else 0.5))

        if progress and (idx - start) % 30 == 0:
            progress("extract", idx - start, total_span, "프레임 추출 중")
        idx += 1
    cap.release()

    if not frames:
        raise ValueError("분석 구간에서 프레임을 추출하지 못했습니다.")

    # 상한 초과 시 중요도 상위 N개 -> 시간 순서 복원
    if len(frames) > profile.max_frames:
        order = np.argsort([-f.importance for f in frames], kind="stable")[: profile.max_frames]
        keep = sorted(order.tolist())
        frames = [frames[i] for i in keep]

    if progress:
        progress("extract", total_span, total_span, f"프레임 {len(frames)}개 추출 완료")
    return frames, info


def download_video(url: str, dest_dir: Path | None = None, progress: ProgressCb = None) -> str:
    """URL(YouTube 포함) -> 로컬 mp4 경로. yt-dlp 사용."""
    import yt_dlp  # 지연 import

    dest_dir = dest_dir or Path(tempfile.gettempdir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest_dir / "%(id)s.%(ext)s")

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            progress("download", int(d.get("downloaded_bytes", 0)), int(total), "영상 다운로드 중")

    opts = {
        "outtmpl": out_tmpl,
        # H.264(avc1) 우선 — OpenCV 내장 FFmpeg 가 AV1/VP9 를 디코드하지 못하는 경우가 있음.
        # 그래도 다른 코덱이 오면 ensure_decodable() 이 ffmpeg 로 변환.
        "format": ("bestvideo[ext=mp4][vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]"
                   "/best[ext=mp4][vcodec^=avc1][height<=1080]/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]"
                   "/best[ext=mp4]/best"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "max_filesize": settings_max_download_bytes(),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    if not path.endswith(".mp4"):
        path = os.path.splitext(path)[0] + ".mp4"
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        raise ValueError("영상 다운로드에 실패했습니다.")
    return path


def settings_max_download_bytes() -> int:
    mb = int(os.environ.get("MAX_UPLOAD_MB", "300"))
    return mb * 1024 * 1024


def _opencv_can_decode(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened()
    if ok:
        ok = bool(cap.read()[0])
    cap.release()
    return ok


def ensure_decodable(video_path: str, progress: ProgressCb = None) -> str:
    """
    OpenCV 가 첫 프레임을 읽지 못하면(AV1, HEVC 등) ffmpeg 로 H.264 로 변환한 경로를 반환.
    읽을 수 있으면 원본 경로 그대로. ffmpeg 도 실패하면 원본 경로를 돌려주고 이후 단계에서 명확한 오류가 발생.
    """
    if _opencv_can_decode(video_path):
        return video_path
    if progress:
        progress("download", 0, 1, "코덱 변환 중 (H.264)")
    out = os.path.splitext(video_path)[0] + ".h264.mp4"
    try:
        subprocess.run(
            [settings.ffmpeg_path, "-y", "-loglevel", "error", "-i", video_path,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
             "-movflags", "+faststart", out],
            check=True, capture_output=True, text=True, timeout=900,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return video_path
    if _opencv_can_decode(out):
        return out
    return video_path


def images_to_mp4(image_paths: list[str], out_path: str, fps: int = 10) -> bool:
    """
    이미지 시퀀스 -> 브라우저 호환 H.264 mp4.
    ffmpeg 가 없거나 실패하면 OpenCV mp4v 로 대체하고 False 반환(분석 결과는 영향 없음).
    """
    if not image_paths:
        return False
    first = cv2.imread(image_paths[0])
    h, w = first.shape[:2]
    h, w = h - h % 2, w - w % 2  # yuv420p 는 짝수 크기 필요

    tmp = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        writer.write(cv2.resize(img, (w, h)))
    writer.release()

    ffmpeg = settings.ffmpeg_path
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", tmp,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
            check=True, capture_output=True, text=True, timeout=300,
        )
        os.remove(tmp)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        os.replace(tmp, out_path)
        return False
