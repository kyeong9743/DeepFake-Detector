"""
Data Server — SQLAlchemy 모델 (MariaDB / SQLite 공통).

analyses 테이블 하나에 요청·진행·결과를 모두 기록한다. 
상세 프레임 점수 등 큰 데이터는 reports 볼륨의 result.json 에 저장하며,
DB 에는 요약과 경로만 저장한다.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:20]


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    # 입력
    source_type: Mapped[str] = mapped_column(String(16))              # upload | url
    source_name: Mapped[str] = mapped_column(String(255))             # 원본 파일명 또는 URL
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_path: Mapped[str | None] = mapped_column(String(255), nullable=True)   # /media 기준 상대 경로
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[str] = mapped_column(String(8), default="high")
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 진행
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True) # queued|processing|done|failed
    stage: Mapped[str] = mapped_column(String(32), default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(String(255), default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 결과 요약
    deepfake_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    is_deepfake: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    cnn_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    lstm_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    vote_fake: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    face_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True) # 요약(프레임 점수 제외)

    def to_dict(self, full: bool = False) -> dict:
        d = {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_type": self.source_type, "source_name": self.source_name, "source_url": self.source_url,
            "file_size": self.file_size, "mode": self.mode,
            "status": self.status, "stage": self.stage, "progress": self.progress, "message": self.message,
            "error": self.error,
            "deepfake_score": self.deepfake_score, "confidence": self.confidence, "risk_level": self.risk_level,
            "is_deepfake": self.is_deepfake,
            "scores": {"cnn": self.cnn_score, "lstm": self.lstm_score, "frequency": self.frequency_score},
            "vote_fake": self.vote_fake, "frame_count": self.frame_count, "face_ratio": self.face_ratio,
            "processing_time": self.processing_time, "video_duration": self.video_duration,
            "report_url": f"/reports/{self.id}/report.html" if self.status == "done" else None,
        }
        if full and self.result_json:
            d["result"] = self.result_json
        return d


# DB 연결이 끊기면 자동 재접속
engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=1800, future=True,
                       connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    with engine.connect() as c:
        c.execute(text("SELECT 1"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
