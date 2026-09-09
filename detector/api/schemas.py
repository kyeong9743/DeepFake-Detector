from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AnalyzeRequest(BaseModel):
    """backend -> detector 분석 요청."""
    analysis_id: str = Field(..., description="backend 가 발급한 ID (영숫자/-/_ 64자 이내)")
    source_type: Literal["upload", "url"]
    # upload: /media 볼륨 내 상대 경로 (예: 2024/ab/xyz.mp4) — 파일을 두 번 전송하지 않음
    # url   : 외부 URL (YouTube 포함), detector 가 직접 다운로드
    source: str
    mode: Literal["fast", "high"] | None = None
    gradcam: bool = True
    callback_url: str | None = Field(None, description="진행률/결과를 POST 할 backend URL (없으면 폴링만)")

    @field_validator("analysis_id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        if not _ID_RE.fullmatch(v):
            raise ValueError("analysis_id 는 영숫자, -, _ 만 허용되며 64자 이내여야 합니다.")
        return v

    @field_validator("source")
    @classmethod
    def _safe_source(cls, v: str) -> str:
        if ".." in v.replace("\\", "/").split("/"):
            raise ValueError("source 에 상위 디렉토리 참조를 포함할 수 없습니다.")
        return v


class AnalyzeAccepted(BaseModel):
    analysis_id: str
    status: Literal["queued"]
    queue_position: int


class StatusResponse(BaseModel):
    analysis_id: str
    status: Literal["queued", "processing", "done", "failed"]
    stage: str = ""
    progress: int = 0          # 0~100
    message: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    device: str
    models_loaded: bool
    queue_size: int
    version: str = "2.0.0"
