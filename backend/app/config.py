from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:////media/cake.db"
    detector_url: str = "http://detector:3001"
    backend_url: str = "http://backend:8000" # detector 가 callback할 주소
    public_base_url: str = "http://localhost:8000"
    internal_api_token: str = ""
    default_mode: str = "high"
    max_upload_mb: int = 300
    allowed_extensions: str = "mp4,avi,mov,mkv,webm,wmv"
    media_dir: Path = Path("/media")
    reports_dir: Path = Path("/reports")
    cors_origins: str = "http://localhost:8000"

    @property
    def allowed_ext_set(self) -> set[str]:
        return {e.strip().lower().lstrip(".") for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
