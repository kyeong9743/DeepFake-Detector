"""
CAKE Detect Server — FastAPI.

- 모델은 서버 기동 시 1회 로드. 체크포인트가 없으면 /health 가 models_loaded=false 를 반환하고 분석 요청은 503 을 돌려준다.
- POST /analyze 는 즉시 202 를 반환하고 워커가 GPU 세마포어 아래에서 순차 처리한다.
- 진행률·결과는 callback_url 로 POST 하며 GET /status/{id} 로 폴링도 가능.
- WS /ws/analyze 는 수신 프레임을 추론해서 결과를 반환
- backend 와의 내부 호출은 X-Internal-Token 헤더로 인증
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import AnalyzeAccepted, AnalyzeRequest, HealthResponse, StatusResponse
from common.face import FaceTracker, get_face_detector
from common.video import download_video
from config import settings
from inference.engine import DetectorEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("detector.api")

# ------------------------------------------------------------------ 상태
STATE: dict[str, Any] = {"engine": None, "load_error": None}
JOBS: dict[str, dict[str, Any]] = {} # analysis_id -> StatusResponse dict
QUEUE: asyncio.Queue[AnalyzeRequest] = asyncio.Queue()
GPU_SEM = asyncio.Semaphore(settings.max_concurrent_analyses)


def _engine() -> DetectorEngine:
    eng = STATE["engine"]
    if eng is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"모델이 로드되지 않았습니다: {STATE['load_error'] or '학습된 체크포인트가 없습니다.'}")
    return eng


def require_token(x_internal_token: str | None = Header(default=None)) -> None:
    if settings.internal_api_token and x_internal_token != settings.internal_api_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal token")


async def _callback(url: str | None, payload: dict[str, Any]) -> None:
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(url, json=payload, headers={"X-Internal-Token": settings.internal_api_token})
    except Exception as e: # 콜백 실패는 분석을 막지 않음
        logger.warning("callback 실패 %s: %s", url, e)


# ------------------------------------------------------------------ 워커
async def _run_job(req: AnalyzeRequest) -> None:
    job = JOBS[req.analysis_id]
    loop = asyncio.get_running_loop()
    eng = STATE["engine"]
    job.update(status="processing", stage="start", progress=0, message="분석 시작")
    await _callback(req.callback_url, {"analysis_id": req.analysis_id, "event": "progress", **_public(job)})

    last_sent = [0.0]

    def progress(stage: str, cur: int, tot: int, msg: str):
        pct = int(min(100, max(0, (cur / tot * 100) if tot else 0)))
        # 단계별 전체 진행률 매핑
        span = {"download": (0, 10), "extract": (10, 30), "preprocess": (30, 45), "infer": (45, 75), "report": (75, 98), "done": (98, 100)}
        lo, hi = span.get(stage, (0, 100))
        job.update(stage=stage, progress=int(lo + (hi - lo) * pct / 100), message=msg)
        now = time.time()
        if now - last_sent[0] > 1.0 or stage == "done": # 콜백은 초당 1회로 제한
            last_sent[0] = now
            asyncio.run_coroutine_threadsafe(
                _callback(req.callback_url, {"analysis_id": req.analysis_id, "event": "progress", **_public(job)}), loop)

    def work() -> dict[str, Any]:
        if req.source_type == "url":
            video_path = download_video(req.source, settings.media_dir / "downloads", progress=progress)
        else:
            video_path = str((settings.media_dir / req.source).resolve())
            if not Path(video_path).is_relative_to(settings.media_dir.resolve()) or not Path(video_path).exists():
                raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {req.source}")
        engine = eng if (req.mode is None or req.mode == eng.mode) else DetectorEngine(req.mode)
        return engine.analyze_video(video_path, req.analysis_id, progress=progress, gradcam=req.gradcam)

    try:
        async with GPU_SEM:
            result = await asyncio.to_thread(work)
        result["report_url"] = f"/reports/{req.analysis_id}/report.html"
        job.update(status="done", stage="done", progress=100, message="분석 완료", result=result)
        await _callback(req.callback_url, {"analysis_id": req.analysis_id, "event": "result", **_public(job)})
    except Exception as e:
        logger.exception("분석 실패 %s", req.analysis_id)
        job.update(status="failed", error=str(e), message="분석 실패")
        await _callback(req.callback_url, {"analysis_id": req.analysis_id, "event": "failed", **_public(job)})


async def _worker() -> None:
    while True:
        req = await QUEUE.get()
        try:
            await _run_job(req)
        finally:
            QUEUE.task_done()


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: job.get(k) for k in ("status", "stage", "progress", "message", "error", "result")}


# ------------------------------------------------------------------ 앱
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    try:
        STATE["engine"] = await asyncio.to_thread(DetectorEngine, settings.default_mode)
    except Exception as e:
        STATE["load_error"] = str(e)
        logger.error("모델 로드 실패 — 503: %s", e)
    workers = [asyncio.create_task(_worker()) for _ in range(settings.max_concurrent_analyses)]
    yield
    for w in workers:
        w.cancel()


app = FastAPI(title="CAKE Detect Server", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Internal-Token"])
settings.reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(settings.reports_dir)), name="reports")


def _gpu_metrics() -> dict[str, Any]:
    """GPU 사용률·메모리. pynvml 우선, 실패 시 torch.cuda 메모리만."""
    out: dict[str, Any] = {"available": torch.cuda.is_available()}
    if not out["available"]:
        return out
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        out.update(name=pynvml.nvmlDeviceGetName(h), util_percent=float(util.gpu),
                   mem_used_mb=round(mem.used / 2**20), mem_total_mb=round(mem.total / 2**20),
                   mem_percent=round(mem.used / mem.total * 100, 1))
        try:
            out["temp_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            out["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
        except Exception:
            pass
        pynvml.nvmlShutdown()
    except Exception:
        free, total = torch.cuda.mem_get_info(0)
        out.update(name=torch.cuda.get_device_name(0), util_percent=None,
                   mem_used_mb=round((total - free) / 2**20), mem_total_mb=round(total / 2**20),
                   mem_percent=round((total - free) / total * 100, 1))
    out["torch_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 2**20)
    return out


@app.get("/metrics")
async def metrics():
    """리소스 사용량 스냅샷"""
    import psutil
    vm = psutil.virtual_memory()
    counts = {"queued": 0, "processing": 0, "done": 0, "failed": 0}
    for j in JOBS.values():
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    current = next(({"analysis_id": j["analysis_id"], "stage": j["stage"], "progress": j["progress"], "message": j["message"]}
                    for j in JOBS.values() if j["status"] == "processing"), None)
    return {
        "ts": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(),
        "mem_percent": vm.percent, "mem_used_mb": round(vm.used / 2**20), "mem_total_mb": round(vm.total / 2**20),
        "gpu": _gpu_metrics(),
        "queue_size": QUEUE.qsize(),
        "jobs": counts,
        "current": current,
        "models_loaded": STATE["engine"] is not None,
        "mode": STATE["engine"].mode if STATE["engine"] else settings.default_mode,
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    eng = STATE["engine"]
    return HealthResponse(status="ok" if eng else "degraded", mode=settings.default_mode,
                          device=str(eng.device) if eng else ("cuda" if torch.cuda.is_available() else "cpu"),
                          models_loaded=eng is not None, queue_size=QUEUE.qsize())


@app.post("/analyze", response_model=AnalyzeAccepted, status_code=202, dependencies=[Depends(require_token)])
async def analyze(req: AnalyzeRequest):
    _engine()
    if req.analysis_id in JOBS and JOBS[req.analysis_id]["status"] in ("queued", "processing"):
        raise HTTPException(409, "이미 처리 중인 analysis_id 입니다.")
    JOBS[req.analysis_id] = {"analysis_id": req.analysis_id, "status": "queued", "stage": "queued",
                             "progress": 0, "message": f"대기 중 (앞에 {QUEUE.qsize()}건)", "error": None, "result": None}
    await QUEUE.put(req)
    return AnalyzeAccepted(analysis_id=req.analysis_id, status="queued", queue_position=QUEUE.qsize())


@app.get("/status/{analysis_id}", response_model=StatusResponse)
async def get_status(analysis_id: str):
    job = JOBS.get(analysis_id)
    if job is None:
        # 재기동 후에는 메모리에 없어도 디스크 결과가 있을 수 있음
        rp = settings.reports_dir / analysis_id / "result.json"
        if rp.exists():
            return StatusResponse(analysis_id=analysis_id, status="done", progress=100, message="분석 완료",
                                  result=json.loads(rp.read_text(encoding="utf-8")))
        raise HTTPException(404, "analysis_id 를 찾을 수 없습니다.")
    return StatusResponse(analysis_id=analysis_id, **_public(job))


# ------------------------------------------------------------------ 실시간
@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket):
    """
    프로토콜: 클라이언트가 JPEG 바이너리 프레임을 보내면 프레임마다 결과 JSON 을 회신.
    텍스트 메시지(JSON)로 메타를 먼저 보낼 수 있음.
    """
    await ws.accept()
    eng = STATE["engine"]
    if eng is None:
        await ws.send_json({"error": "모델이 로드되지 않았습니다."}); await ws.close(code=1011); return
    tracker = FaceTracker(get_face_detector())
    meta: dict[str, Any] = {}
    n = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("text"):
                try:
                    meta.update(json.loads(msg["text"]))
                except json.JSONDecodeError:
                    await ws.send_json({"error": "잘못된 메타 JSON"})
                continue
            data = msg.get("bytes")
            if not data:
                continue
            arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"error": "이미지를 디코드할 수 없습니다."}); continue
            n += 1
            res = await asyncio.to_thread(eng.analyze_frame, frame, tracker)
            await ws.send_json({"analysis_id": meta.get("analysis_id"), "frame_number": meta.get("frame_number", n),
                                "face_detected": tracker.face_ratio > 0, **res})
    except WebSocketDisconnect:
        logger.info("ws 종료 (%d 프레임)", n)
    except Exception as e:
        logger.warning("ws 오류: %s", e)
        try:
            await ws.close(code=1011)
        except Exception:
            pass
