"""
탐지 엔진 — 모델을 1회 로드해 재사용하며 영상/프레임을 분석한다.

analyze_video():
  프레임 추출(시간 순서 보존) -> 얼굴 크롭 -> CNN·Frequency 배치 추론 ->
  LSTM 오버랩 윈도우 추론 -> 앙상블(가중 평균 + 투표 + 신뢰도) -> Grad-CAM -> 리포트
analyze_frame():
  실시간용 단일 프레임 (CNN + Frequency).
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from common.face import FaceTracker, get_face_detector
from common.preprocess import bgr_to_tensor, fft_spectrum, make_windows, normalize, pad_sequence
from common.video import Frame, ensure_decodable, extract_frames
from config import PROFILES, risk_level, settings
from inference import report
from inference.gradcam import GradCAM, overlay_heatmap
from models.registry import KINDS, load_all_models, load_ensemble_config

logger = logging.getLogger(__name__)

# 입력 품질 경고 -> 사용자 문구 (웹·리포트·CLI 공통). 데이터 필터이지 판정 조작이 아님
QUALITY_MESSAGES = {
    "no_face": "얼굴이 검출된 프레임이 거의 없어 프레임 전체로 분석했습니다. 결과 신뢰성이 크게 떨어집니다.",
    "low_face_ratio": "얼굴이 검출된 프레임 비율이 낮습니다. 가려짐·잦은 이탈이 있는 영상은 결과 신뢰성이 떨어집니다.",
    "small_face": "얼굴이 작습니다(원본 기준). 크롭이 크게 확대되어 생성 흔적이 흐려지므로 낮은 점수를 그대로 믿기 어렵습니다.",
}
ProgressCb = Callable[[str, int, int, str], None] | None

RISK_COLORS = {"안전": "#2e9e5b", "낮음": "#7bb661", "보통": "#e0a800", "위험": "#e8702a", "매우위험": "#d63333"}


@dataclass
class EnsembleResult:
    score: float
    confidence: float
    vote_fake: int
    vote_total: int
    weights: dict[str, float]
    threshold: float
    method: str
    per_model: dict[str, float] = field(default_factory=dict)


def ensemble(per_model: dict[str, float], cfg: dict[str, Any]) -> EnsembleResult:
    """
    가중 평균(PredIntegrator) + 다수결 투표 + 신뢰도(ConfCalculator).
    신뢰도 = 확신도(평균이 0.5에서 떨어진 정도) × 일치도(모델 간 표준편차가 작을수록 높음).
    세 모델이 모두 0.5 근처면 낮고, 모두 한쪽으로 확신하면 높다.
    """
    w = cfg["weights"]
    total_w = sum(w[k] for k in KINDS)
    score = sum(per_model[k] * w[k] for k in KINDS) / max(total_w, 1e-9)
    thr = float(cfg.get("threshold", 0.5))
    model_thr = cfg.get("model_thresholds", {k: 0.5 for k in KINDS})
    votes = [int(per_model[k] >= model_thr.get(k, 0.5)) for k in KINDS]
    vals = np.array([per_model[k] for k in KINDS])
    certainty = min(1.0, abs(score - 0.5) * 2)
    agreement = max(0.0, 1.0 - vals.std() * 2)
    confidence = float(round(0.6 * certainty + 0.4 * agreement, 4))
    return EnsembleResult(float(score), confidence, sum(votes), len(votes), {k: float(w[k]) for k in KINDS},
                          thr, cfg.get("method", ""), per_model)


class DetectorEngine:
    def __init__(self, mode: str | None = None, device: str | None = None):
        self.mode = mode or settings.default_mode
        self.profile = PROFILES[self.mode]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        t0 = time.time()
        self.models = load_all_models(self.mode, self.device)
        self.ens_cfg = load_ensemble_config(self.mode)
        self.face = get_face_detector()
        self._gradcam = GradCAM(self.models["cnn"], self.models["cnn"].gradcam_layer, plus_plus=True)
        logger.info("엔진 준비 (%s, %s) %.1fs — 앙상블 %s w=%s thr=%.2f", self.mode, self.device,
                    time.time() - t0, self.ens_cfg.get("method"), self.ens_cfg["weights"], self.ens_cfg.get("threshold", .5))

    # ------------------------------------------------------------------ 내부
    @torch.no_grad()
    def _frame_scores(self, x: torch.Tensor, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """x: (N,3,H,W) [0,1] CPU -> (cnn_prob[N], freq_prob[N])"""
        cnn, freq = [], []
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                cnn.append(torch.sigmoid(self.models["cnn"](normalize(xb)).float()).cpu())
                freq.append(torch.sigmoid(self.models["frequency"](fft_spectrum(xb)).float()).cpu())
        return torch.cat(cnn).numpy(), torch.cat(freq).numpy()

    @torch.no_grad()
    def _lstm_scores(self, x: torch.Tensor) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """오버랩 윈도우별 LSTM 확률. x: (N,3,H,W) [0,1] CPU."""
        p = self.profile
        wins = make_windows(x.shape[0], p.window, p.stride)
        probs = []
        for s, e in wins:
            seq = pad_sequence(x[s:e], p.window).unsqueeze(0).to(self.device)     # (1,T,3,H,W)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                logit = self.models["lstm"](normalize(seq), fft_spectrum(seq.squeeze(0)).unsqueeze(0))
            probs.append(float(torch.sigmoid(logit.float()).item()))
        return np.array(probs), wins

    def _gradcam_overlays(self, crops: list[np.ndarray], x: torch.Tensor, idx: list[int], batch: int = 16) -> list[np.ndarray]:
        overlays = []
        for i in range(0, len(idx), batch):
            sel = idx[i:i + batch]
            cams, _ = self._gradcam(normalize(x[sel].to(self.device)))
            overlays += [overlay_heatmap(crops[j], cams[k]) for k, j in enumerate(sel)]
        return overlays

    # ------------------------------------------------------------------ 공개 API
    def analyze_video(self, video_path: str, analysis_id: str, out_dir: Path | None = None,
                      progress: ProgressCb = None, gradcam: bool = True) -> dict[str, Any]:
        t_start = time.time()
        p = self.profile
        out_dir = out_dir or (settings.reports_dir / analysis_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        def prog(stage, cur, tot, msg):
            if progress:
                progress(stage, cur, tot, msg)

        # 1) 프레임 추출 (OpenCV 가 못 읽는 코덱이면 ffmpeg 로 H.264 변환 후)
        video_path = ensure_decodable(video_path, progress=progress)
        frames, info = extract_frames(video_path, p, progress=progress)
        n = len(frames)

        # 2) 얼굴 크롭 + 텐서화
        prog("preprocess", 0, n, "얼굴 검출 및 전처리")
        tracker = FaceTracker(self.face)
        crops: list[np.ndarray] = []
        tensors = []
        valid: list[bool] = []
        for i, fr in enumerate(frames):
            crop, found = tracker.crop(fr.image)
            crop = np.ascontiguousarray(crop)
            crops.append(crop)
            tensors.append(bgr_to_tensor(crop, p.frame_size))
            valid.append(found)
            if i % 20 == 0:
                prog("preprocess", i, n, "얼굴 검출 및 전처리")
        face_ratio = tracker.face_ratio
        face_px = tracker.median_face_px
        n_extracted = n

        # 2-b) 입력 품질 검사 (2026-09-07). 얼굴이 실제 검출된 프레임만 점수에 사용한다.
        #   검출 실패 프레임은 최근 박스 재사용(스티커·배경이 잘림) 또는 프레임 전체(분포 밖 입력)가 들어가
        #   영상 점수를 오염시킴 — sample 영상에서 마지막 21/184 프레임이 얼굴 스티커·전체 프레임이었음.
        quality_warnings: list[str] = []
        keep = [i for i, ok in enumerate(valid) if ok]
        if len(keep) < settings.min_face_frames:
            quality_warnings.append("no_face") # 얼굴 프레임이 거의 없음 -> 전체 프레임으로 분석하되 경고
            keep = list(range(n))
        if len(keep) < n:
            frames = [frames[i] for i in keep]
            crops = [crops[i] for i in keep]
            tensors = [tensors[i] for i in keep]
            n = len(frames)
        if face_ratio < settings.min_face_ratio:
            quality_warnings.append("low_face_ratio")
        if 0 < face_px < settings.min_face_px:
            quality_warnings.append("small_face")
        x = torch.stack(tensors) # (N,3,S,S) CPU [0,1]

        # 3) 프레임 모델
        prog("infer", 0, 3, "CNN · Frequency 추론")
        cnn_p, freq_p = self._frame_scores(x)
        prog("infer", 1, 3, "LSTM 시퀀스 추론")
        lstm_w, wins = self._lstm_scores(x)
        # 윈도우 확률을 프레임에 펼치기(겹치는 구간은 평균) -> 그래프용
        lstm_frame = np.full(n, np.nan)
        acc, cnt = np.zeros(n), np.zeros(n)
        for (s, e), pr in zip(wins, lstm_w):
            acc[s:e] += pr; cnt[s:e] += 1
        lstm_frame[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
        prog("infer", 2, 3, "앙상블 계산")

        per_model = {"cnn": float(cnn_p.mean()), "frequency": float(freq_p.mean()), "lstm": float(lstm_w.mean())}
        ens = ensemble(per_model, self.ens_cfg)
        w = ens.weights
        frame_ens = (w["cnn"] * cnn_p + w["frequency"] * freq_p + w["lstm"] * np.nan_to_num(lstm_frame, nan=per_model["lstm"])) \
            / (w["cnn"] + w["frequency"] + w["lstm"])
        prog("infer", 3, 3, "추론 완료")

        # 4) 결과 dict
        level = risk_level(ens.score)
        result: dict[str, Any] = {
            "analysis_id": analysis_id,
            "mode": self.mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deepfake_score": round(ens.score, 4),
            "is_deepfake": bool(ens.score >= ens.threshold),
            "threshold": ens.threshold,
            "confidence": ens.confidence,
            "risk_level": level,
            "risk_color": RISK_COLORS[level],
            "scores": {k: round(v, 4) for k, v in per_model.items()},
            "weights": {k: round(v, 4) for k, v in w.items()},
            "ensemble_method": ens.method,
            "vote": {"fake": ens.vote_fake, "total": ens.vote_total},
            "frame_count": n,
            "frames_extracted": n_extracted,
            "frames_no_face": n_extracted - n,
            "face_ratio": round(face_ratio, 3),
            "face_px": face_px,
            "quality_warnings": quality_warnings,
            "quality_messages": [QUALITY_MESSAGES[k] for k in quality_warnings],
            "face_warning": bool(quality_warnings),
            "video": {"duration": round(info.duration, 2), "fps": round(info.fps, 2),
                      "width": info.width, "height": info.height, "total_frames": info.total_frames},
            "frame_scores": {
                "time": [round(f.time_sec, 3) for f in frames],
                "cnn": cnn_p.round(4).tolist(),
                "frequency": freq_p.round(4).tolist(),
                "lstm": [None if math.isnan(v) else round(float(v), 4) for v in lstm_frame],
                "ensemble": frame_ens.round(4).tolist(),
            },
            "lstm_windows": [{"start": s, "end": e, "prob": round(float(pv), 4)} for (s, e), pv in zip(wins, lstm_w)],
            "report_dir": str(out_dir),
            "artifacts": {},
        }

        # 5) 시각화 (실패해도 점수는 유효)
        try:
            prog("report", 0, 2, "그래프·리포트 생성")
            report.save_score_plot(out_dir / "scores.png", result["frame_scores"]["time"],
                                   {"cnn": cnn_p.tolist(), "frequency": freq_p.tolist(),
                                    "lstm": lstm_frame.tolist(), "ensemble": frame_ens.tolist()}, ens.threshold)
            rows = [{"frame_index": f.index, "time_sec": round(f.time_sec, 3), "cnn": round(float(a), 4),
                     "frequency": round(float(b), 4), "lstm": None if math.isnan(c) else round(float(c), 4),
                     "ensemble": round(float(d), 4)} for f, a, b, c, d in zip(frames, cnn_p, freq_p, lstm_frame, frame_ens)]
            report.save_scores_table(out_dir / "scores.csv", out_dir / "scores.json", rows)
            result["artifacts"].update({"scores_png": "scores.png", "scores_csv": "scores.csv", "scores_json": "scores.json"})

            if gradcam:
                prog("report", 1, 2, "Grad-CAM 히트맵 생성")
                k = min(n, p.gradcam_max_frames)
                idx = np.linspace(0, n - 1, num=k, dtype=int).tolist()
                overlays = self._gradcam_overlays(crops, x, idx)
                heat_paths = report.save_heatmap_frames(out_dir, [crops[i] for i in idx], overlays,
                                                        [frames[i].time_sec for i in idx], [float(frame_ens[i]) for i in idx])
                sus = report.save_suspicious(out_dir, heat_paths, [float(frame_ens[i]) for i in idx], ens.threshold)
                hv = report.save_heatmap_video(out_dir, heat_paths)
                result["suspicious_frames"] = sus
                result["heatmap_video"] = hv["video"]
                result["artifacts"].update({"heatmap_dir": "heatmap", "compare_dir": "compare",
                                            "heatmap_video": hv["video"], "heatmap_h264": hv["h264"],
                                            "suspicious_dir": "suspicious"})
            report.write_html(out_dir, result)
            result["artifacts"]["report_html"] = "report.html"
        except Exception as e:  # 시각화 실패는 결과를 무효화하지 않음
            logger.exception("시각화 생성 실패 (점수는 유효): %s", e)
            result["visualization_error"] = str(e)

        result["processing_time"] = round(time.time() - t_start, 2)
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        prog("done", 1, 1, "분석 완료")
        return result

    @torch.no_grad()
    def analyze_frame(self, frame_bgr: np.ndarray, tracker: FaceTracker | None = None) -> dict[str, Any]:
        """실시간 단일 프레임: CNN + Frequency 만 사용 (LSTM 은 시퀀스가 필요)."""
        t0 = time.time()
        crop = frame_bgr
        if tracker is not None:
            crop, _ = tracker.crop(frame_bgr)
        x = bgr_to_tensor(np.ascontiguousarray(crop), self.profile.frame_size).unsqueeze(0)
        cnn_p, freq_p = self._frame_scores(x)
        w = self.ens_cfg["weights"]
        score = (cnn_p[0] * w["cnn"] + freq_p[0] * w["frequency"]) / (w["cnn"] + w["frequency"])
        level = risk_level(float(score))
        return {
            "deepfake_score": round(float(score), 4),
            "risk_level": level,
            "risk_color": RISK_COLORS[level],
            "scores": {"cnn": round(float(cnn_p[0]), 4), "frequency": round(float(freq_p[0]), 4)},
            "confidence": round(float(min(1.0, abs(score - 0.5) * 2)), 4),
            "processing_time": round(time.time() - t0, 4),
        }
