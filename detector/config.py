"""
Detect Server
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["fast", "high"]


class ModeProfile(BaseModel):
    """분석 모드별 하이퍼파라미터. fast/high 두 프로필이 존재합니다. (환경변수 영향 없음)"""

    frame_size: int
    backbone: str                  # torchvision resnet 이름
    window: int                    # LSTM 시퀀스 길이
    stride: int                    # 시퀀스 오버랩 간격 (window - stride = 오버랩)
    target_duration: float         # 분석 구간(초)
    target_fps: float              # 추출 목표 FPS
    max_frames: int                # 분석 최대 프레임
    bidirectional: bool            # LSTM 양방향 여부
    gradcam_max_frames: int        # Grad-CAM 히트맵 생성 프레임 상한
    lstm_backbone: str = "resnet18"   # LSTM 공간 스트림 인코더 (사전학습). 소형 conv 는 545개 검증에서 AUC 0.58 에 그침


FAST_PROFILE = ModeProfile(
    frame_size=224, backbone="resnet18", window=16, stride=8,
    target_duration=20, target_fps=8, max_frames=160,
    bidirectional=False, gradcam_max_frames=48, lstm_backbone="resnet18",
)
HIGH_PROFILE = ModeProfile(
    frame_size=384, backbone="resnet34", window=16, stride=8,
    target_duration=30, target_fps=10, max_frames=300,
    bidirectional=True, gradcam_max_frames=96, lstm_backbone="resnet18",
)
PROFILES: dict[str, ModeProfile] = {"fast": FAST_PROFILE, "high": HIGH_PROFILE}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- 경로 (컨테이너 기준) ----
    checkpoints_dir: Path = Path("/checkpoints")
    reports_dir: Path = Path("/reports")
    media_dir: Path = Path("/media")
    data_dir: Path = Path("/data")
    runs_dir: Path = Path("/runs")
    # 이미지 빌드 시 /opt/cake/assets 에 다운로드 (코드 바인드 마운트에 가려지지 않는 위치)
    assets_dir: Path = Path("/opt/cake/assets")

    # ---- 외부 연동 ----
    backend_url: str = "http://backend:8000"
    internal_api_token: str = Field(default="", description="backend ↔ detector 내부 인증 토큰")
    ffmpeg_path: str = "ffmpeg"
    cors_origins: str = "http://localhost:8000"

    # ---- 분석 기본값 ----
    default_mode: Mode = "high"
    # 얼굴 검출 실패 시 프레임 전체를 사용하기 전, 최근 검출 박스를 재사용할 프레임 수
    face_box_ttl: int = 15
    # 얼굴 박스 확장 비율 (1.0 = 박스 그대로)
    face_margin: float = 1.3
    # 얼굴이 검출된 프레임 비율이 이 값 미만이면 결과에 경고 플래그
    min_face_ratio: float = 0.3
    # (2026-09-07 입력 품질 검사) 검출된 얼굴 박스(긴 변, 원본 픽셀)의 중앙값이 이 값 미만이면 '얼굴이 작음' 경고
    # — 크롭이 frame_size 까지 2배 이상 업스케일되어 생성 흔적이 흐려짐 (sample 영상: 얼굴 126 px -> 14점)
    min_face_px: int = 128
    # 얼굴이 실제 검출된 프레임만 점수에 사용. 그 수가 이 값 미만이면 전체 프레임을 쓰되 'no_face' 경고
    min_face_frames: int = 8

    # ---- 판정 등급 경계 (ensemble.json 의 threshold 가 있으면 그것을 이진 판정에 사용) ----
    risk_bounds: tuple[float, float, float, float] = (0.30, 0.50, 0.70, 0.85)

    # ---- 동시성 ----
    max_concurrent_analyses: int = 1   # GPU 1장 기준

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

# 학습/추론이 공유하는 정규화 통계. ImageNet 사전학습 백본을 쓰므로 ImageNet 값을 양쪽 모두에 적용.
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# 체크포인트 파일명 — 학습·추론이 같은 상수를 참조
CHECKPOINT_NAMES = {
    "cnn": "{mode}_cnn.pt",
    "lstm": "{mode}_lstm.pt",
    "frequency": "{mode}_frequency.pt",
    "ensemble": "{mode}_ensemble.json",
}


def checkpoint_path(kind: str, mode: str) -> Path:
    return settings.checkpoints_dir / CHECKPOINT_NAMES[kind].format(mode=mode)


def risk_level(score: float) -> str:
    """0~1 점수를 5단계 등급 문자열로 변환."""
    b = settings.risk_bounds
    if score < b[0]:
        return "안전"
    if score < b[1]:
        return "낮음"
    if score < b[2]:
        return "보통"
    if score < b[3]:
        return "위험"
    return "매우위험"
