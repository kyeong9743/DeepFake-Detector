"""
CAKE Backend — FastAPI.

브라우저 -> backend(:8000) -> detector(:3001)
  POST /api/analyses            영상 업로드(스트리밍) 또는 URL 접수 -> DB 기록 -> detector 에 분석 요청(202)
  GET  /api/analyses            분석 기록 목록 (페이지)
  GET  /api/analyses/{id}       상태·결과 (프론트 폴링)
  POST /internal/callback       detector -> backend 진행률/결과 콜백 (내부 토큰)
  GET  /reports/{id}/...        분석 산출물 정적 서빙 (공유 볼륨)
  WS   /ws/realtime             브라우저 ↔ detector WebSocket 프록시 (실시간, 웹 UI 에서는 비활성)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiofiles
import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import Analysis, get_db, init_db, new_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend")

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
_URL_RE = re.compile(r"^https?://", re.I)
_MP4_MAGIC = (b"ftyp",)          # ISO BMFF
_MAGIC = {                         # 확장자 -> 매직 바이트 검사 함수
    "mp4": lambda h: h[4:8] == b"ftyp", "mov": lambda h: h[4:8] == b"ftyp", "m4v": lambda h: h[4:8] == b"ftyp",
    "avi": lambda h: h[:4] == b"RIFF" and h[8:12] == b"AVI ",
    "mkv": lambda h: h[:4] == b"\x1aE\xdf\xa3", "webm": lambda h: h[:4] == b"\x1aE\xdf\xa3",
    "wmv": lambda h: h[:4] == b"\x30\x26\xb2\x75",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(10):
        try:
            init_db(); logger.info("DB 연결 성공: %s", settings.database_url.split("@")[-1]); break
        except Exception as e:
            logger.warning("DB 연결 재시도 %d/10: %s", attempt + 1, e); await asyncio.sleep(3)
    else:
        raise RuntimeError("DB 에 연결할 수 없습니다. DATABASE_URL 을 확인하십시오.")
    yield


app = FastAPI(title="CAKE Backend", version="2.0.0", lifespan=lifespan)
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
settings.reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(settings.reports_dir)), name="reports")


def _ctx(request: Request, **kw):
    return {"request": request, "max_mb": settings.max_upload_mb, "exts": sorted(settings.allowed_ext_set),
            "default_mode": settings.default_mode, **kw}


# ------------------------------------------------------------------ 페이지
@app.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    return templates.TemplateResponse(request, "index.html", _ctx(request, page="upload"))


@app.get("/analyses/{analysis_id}", response_class=HTMLResponse)
async def page_analysis(request: Request, analysis_id: str, db: Session = Depends(get_db)):
    a = db.get(Analysis, analysis_id)
    if a is None:
        raise HTTPException(404, "분석을 찾을 수 없습니다.")
    return templates.TemplateResponse(request, "analysis.html", _ctx(request, page="analysis", analysis=a.to_dict()))


@app.get("/history", response_class=HTMLResponse)
async def page_history(request: Request):
    return templates.TemplateResponse(request, "history.html", _ctx(request, page="history"))


@app.get("/realtime", response_class=HTMLResponse)
async def page_realtime(request: Request):
    return templates.TemplateResponse(request, "realtime.html", _ctx(request, page="realtime"))


@app.get("/monitor", response_class=HTMLResponse)
async def page_monitor(request: Request):
    return templates.TemplateResponse(request, "monitor.html", _ctx(request, page="monitor"))


@app.get("/api/metrics")
async def api_metrics():
    """detector /metrics 프록시 + backend 자체 지표. 대시보드가 2초 간격으로 폴링."""
    import psutil
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            det = (await c.get(f"{settings.detector_url}/metrics")).json()
    except Exception as e:
        det = {"error": str(e)}
    vm = psutil.virtual_memory()
    return {"detector": det,
            "backend": {"cpu_percent": psutil.cpu_percent(interval=None), "mem_percent": vm.percent,
                        "mem_used_mb": round(vm.used / 2**20)}}


# ------------------------------------------------------------------ 헬퍼
def _detector_headers() -> dict[str, str]:
    return {"X-Internal-Token": settings.internal_api_token}


def require_internal(x_internal_token: str | None = Header(default=None)) -> None:
    if settings.internal_api_token and x_internal_token != settings.internal_api_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal token")


async def _submit_to_detector(a: Analysis) -> None:
    payload = {
        "analysis_id": a.id,
        "source_type": "url" if a.source_type == "url" else "upload",
        "source": a.source_url if a.source_type == "url" else a.media_path,
        "mode": a.mode,
        "gradcam": True,
        "callback_url": f"{settings.backend_url}/internal/callback",
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{settings.detector_url}/analyze", json=payload, headers=_detector_headers())
    if r.status_code == 503:
        raise HTTPException(503, "분석 서버에 학습된 모델이 없습니다. 관리자에게 문의하십시오.")
    r.raise_for_status()


def _apply_result(a: Analysis, res: dict) -> None:
    a.deepfake_score = res.get("deepfake_score"); a.confidence = res.get("confidence")
    a.risk_level = res.get("risk_level"); a.is_deepfake = res.get("is_deepfake")
    sc = res.get("scores", {})
    a.cnn_score, a.lstm_score, a.frequency_score = sc.get("cnn"), sc.get("lstm"), sc.get("frequency")
    a.vote_fake = res.get("vote", {}).get("fake"); a.frame_count = res.get("frame_count")
    a.face_ratio = res.get("face_ratio"); a.processing_time = res.get("processing_time")
    a.video_duration = res.get("video", {}).get("duration")
    a.result_json = {k: v for k, v in res.items() if k not in ("frame_scores", "lstm_windows")}


# ------------------------------------------------------------------ API
@app.post("/api/analyses", status_code=202)
async def create_analysis(request: Request, db: Session = Depends(get_db),
                          file: UploadFile | None = File(default=None),
                          url: str | None = Form(default=None),
                          mode: str = Form(default=None)):
    mode = mode if mode in ("fast", "high") else settings.default_mode
    aid = new_id()
    client_ip = request.client.host if request.client else None

    if url and url.strip():
        url = url.strip()
        if not _URL_RE.match(url):
            raise HTTPException(400, "http(s) URL 만 지원합니다.")
        a = Analysis(id=aid, source_type="url", source_name=url[:255], source_url=url, mode=mode, client_ip=client_ip,
                     message="대기 중")
    elif file is not None and file.filename:
        ext = Path(file.filename).suffix.lower().lstrip(".")
        if ext not in settings.allowed_ext_set:
            raise HTTPException(400, f"지원하지 않는 형식입니다. 허용: {', '.join(sorted(settings.allowed_ext_set))}")
        sub = datetime.utcnow().strftime("%Y%m")
        rel = f"uploads/{sub}/{aid}.{ext}"
        dst = settings.media_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 스트리밍 저장 + 실제 바이트 수로 크기 제한 (클라이언트 값 신뢰 안 함)
        written, head = 0, b""
        async with aiofiles.open(dst, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                if not head:
                    head = chunk[:16]
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    await out.close(); dst.unlink(missing_ok=True)
                    raise HTTPException(413, f"파일이 {settings.max_upload_mb}MB 를 초과합니다.")
                await out.write(chunk)
        if written == 0 or not _MAGIC.get(ext, lambda h: True)(head):
            dst.unlink(missing_ok=True)
            raise HTTPException(400, "영상 파일이 아니거나 손상되었습니다.")
        a = Analysis(id=aid, source_type="upload", source_name=file.filename[:255], media_path=rel,
                     file_size=written, mode=mode, client_ip=client_ip, message="대기 중")
    else:
        raise HTTPException(400, "영상 파일 또는 URL 이 필요합니다.")

    db.add(a); db.commit()
    try:
        await _submit_to_detector(a)
    except HTTPException:
        a.status, a.error = "failed", "분석 서버 사용 불가"; db.commit(); raise
    except Exception as e:
        a.status, a.error = "failed", f"분석 서버 오류: {e}"; db.commit()
        raise HTTPException(502, f"분석 서버에 연결할 수 없습니다: {e}")
    return {"id": aid, "status": "queued", "url": f"/analyses/{aid}"}


@app.get("/api/analyses")
async def list_analyses(db: Session = Depends(get_db), page: int = 1, size: int = 20, status_f: str | None = None):
    size = max(1, min(size, 100)); page = max(1, page)
    q = select(Analysis)
    if status_f:
        q = q.where(Analysis.status == status_f)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(desc(Analysis.created_at)).offset((page - 1) * size).limit(size)).all()
    return {"total": total, "page": page, "size": size, "items": [r.to_dict() for r in rows]}


@app.get("/api/analyses/{analysis_id}")
async def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    a = db.get(Analysis, analysis_id)
    if a is None:
        raise HTTPException(404, "분석을 찾을 수 없습니다.")
    d = a.to_dict(full=True)
    if a.status == "done":
        # 프레임 점수 등 상세는 reports 볼륨의 result.json
        rp = settings.reports_dir / a.id / "result.json"
        if rp.exists():
            import json
            d["result"] = json.loads(rp.read_text(encoding="utf-8"))
    return d


@app.get("/api/analyses/{analysis_id}/download/{name}")
async def download_artifact(analysis_id: str, name: str, db: Session = Depends(get_db)):
    if db.get(Analysis, analysis_id) is None:
        raise HTTPException(404)
    if name not in ("scores.csv", "scores.json", "result.json", "report.html", "heatmap.mp4"):
        raise HTTPException(404)
    p = settings.reports_dir / analysis_id / name
    if not p.exists():
        raise HTTPException(404, "산출물이 없습니다.")
    return FileResponse(p, filename=f"cake_{analysis_id}_{name}")


@app.get("/api/stats")
async def stats(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Analysis.id))) or 0
    done = db.scalar(select(func.count(Analysis.id)).where(Analysis.status == "done")) or 0
    fake = db.scalar(select(func.count(Analysis.id)).where(Analysis.is_deepfake.is_(True))) or 0
    avg_t = db.scalar(select(func.avg(Analysis.processing_time)).where(Analysis.status == "done"))
    return {"total": total, "done": done, "fake": fake, "avg_processing_time": round(avg_t or 0, 1)}


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{settings.detector_url}/health")
            det = r.json()
    except Exception as e:
        det = {"status": "unreachable", "error": str(e)}
    return {"backend": "ok", "detector": det}


# ------------------------------------------------------------------ detector -> backend 콜백
@app.post("/internal/callback", dependencies=[Depends(require_internal)])
async def callback(payload: dict, db: Session = Depends(get_db)):
    a = db.get(Analysis, payload.get("analysis_id", ""))
    if a is None:
        raise HTTPException(404)
    a.status = payload.get("status", a.status)
    a.stage = payload.get("stage") or a.stage
    a.progress = int(payload.get("progress") or a.progress)
    a.message = (payload.get("message") or a.message)[:255]
    if payload.get("event") == "result" and payload.get("result"):
        _apply_result(a, payload["result"])
    if payload.get("event") == "failed":
        a.error = payload.get("error")
    db.commit()
    return {"ok": True}


# ------------------------------------------------------------------ 실시간 프록시 (웹 UI 에서는 비활성화)
@app.websocket("/ws/realtime")
async def ws_realtime(ws: WebSocket):
    """브라우저 프레임을 detector /ws/analyze 로 중계. 내부 토큰은 서버가 붙이므로 브라우저에 노출되지 않음."""
    await ws.accept()
    import websockets
    det_ws_url = settings.detector_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/analyze"
    try:
        async with websockets.connect(det_ws_url, max_size=8 * 1024 * 1024) as det:
            async def up():
                while True:
                    m = await ws.receive()
                    if m.get("bytes"):
                        await det.send(m["bytes"])
                    elif m.get("text"):
                        await det.send(m["text"])
            async def down():
                async for m in det:
                    await ws.send_text(m if isinstance(m, str) else m.decode())
            await asyncio.gather(up(), down())
    except (WebSocketDisconnect, Exception) as e:
        logger.info("realtime ws 종료: %s", e)
