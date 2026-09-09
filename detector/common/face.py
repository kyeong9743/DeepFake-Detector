"""
얼굴 검출 및 크롭 — OpenCV YuNet (ONNX).

딥페이크 조작 흔적은 얼굴 영역에 집중되므로, 프레임 전체를 축소하는 대신 얼굴을 잘라 모델에 입력. 
검출 실패 시 최근 박스를 재사용하고, 그것도 없으면 프레임 전체를 사용. (결과에 얼굴 검출 비율을 함께 기록)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import settings

_MODEL_FILE = "face_detection_yunet_2023mar.onnx"


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float

    def expanded(self, margin: float, frame_w: int, frame_h: int) -> "FaceBox":
        """정사각형으로 맞추고 margin 배 확장한 뒤 프레임 경계로 클리핑."""
        cx, cy = self.x + self.w / 2, self.y + self.h / 2
        side = max(self.w, self.h) * margin
        x0 = int(max(0, cx - side / 2))
        y0 = int(max(0, cy - side / 2))
        x1 = int(min(frame_w, cx + side / 2))
        y1 = int(min(frame_h, cy + side / 2))
        return FaceBox(x0, y0, x1 - x0, y1 - y0, self.score)


class FaceDetector:
    """YuNet 래퍼. 스레드 안전을 위해 detect 호출을 락으로 보호"""

    def __init__(self, model_path: Path | None = None, score_threshold: float = 0.7):
        path = model_path or settings.assets_dir / _MODEL_FILE
        if not Path(path).exists():
            raise FileNotFoundError(
                f"얼굴 검출 모델이 없습니다: {path}\n"
                "Dockerfile 가 자동으로 다운로드합니다."
                f"{_MODEL_FILE} 을 받아 assets/ 에 두십시오."
            )
        self._det = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold=score_threshold, nms_threshold=0.3, top_k=50
        )
        self._lock = threading.Lock()

    # 검출 입력 최대 변 길이. 1080p 를 그대로 넣으면 검출이 프레임당 수십 ms -> 640 으로 축소 후 박스를 원본 좌표로 환산
    DETECT_MAX_SIDE = 640

    def detect(self, frame_bgr: np.ndarray) -> FaceBox | None:
        """가장 큰 얼굴 1개를 반환(원본 좌표). 없으면 None."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.DETECT_MAX_SIDE / max(h, w))
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        sh, sw = small.shape[:2]
        with self._lock:
            self._det.setInputSize((sw, sh))
            _, faces = self._det.detect(small)
        if faces is None or len(faces) == 0:
            return None
        # faces: (N, 15) — x, y, w, h, 5 landmarks(10), score
        best = max(faces, key=lambda f: f[2] * f[3])
        inv = 1.0 / scale
        return FaceBox(int(best[0] * inv), int(best[1] * inv), int(best[2] * inv), int(best[3] * inv), float(best[14]))


class FaceTracker:
    """
    영상 시퀀스용. 검출 실패 프레임에서 최근 박스를 TTL 동안 재사용해 크롭이 튀는 것을
    막고, 검출 성공 비율을 집계.
    """

    def __init__(self, detector: FaceDetector, ttl: int | None = None, margin: float | None = None,
                 detect_every: int = 1):
        self.det = detector
        self.ttl = ttl if ttl is not None else settings.face_box_ttl
        self.margin = margin if margin is not None else settings.face_margin
        # 연속 프레임에서는 매 프레임 검출이 낭비 -> detect_every 프레임마다 검출, 사이는 최근 박스 재사용
        self.detect_every = max(1, detect_every)
        self._last: FaceBox | None = None
        self._age = 0
        self.detected = 0
        self.total = 0
        self.sizes: list[int] = [] # 실제 검출된 박스의 긴 변(원본 픽셀) — 얼굴 크기 품질 검사용

    def crop(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """(크롭 이미지, 이 프레임에서 실제 검출 여부). 검출을 건너뛴 프레임은 '검출됨'으로 집계(박스 유효)."""
        h, w = frame_bgr.shape[:2]
        self.total += 1
        skip = self._last is not None and self._age < self.ttl and (self.total - 1) % self.detect_every != 0
        box = None if skip else self.det.detect(frame_bgr)
        if box is not None:
            self.detected += 1
            self._last, self._age = box, 0
            self.sizes.append(max(box.w, box.h))
            found = True
        elif skip:
            self.detected += 1
            box, found = self._last, True
        else:
            self._age += 1
            if self._last is None or self._age > self.ttl:
                return frame_bgr, False
            box, found = self._last, False
        b = box.expanded(self.margin, w, h)
        if b.w < 16 or b.h < 16:
            return frame_bgr, found
        return frame_bgr[b.y:b.y + b.h, b.x:b.x + b.w], found

    @property
    def face_ratio(self) -> float:
        return self.detected / self.total if self.total else 0.0

    @property
    def median_face_px(self) -> int:
        """실제 검출된 얼굴 박스 긴 변의 중앙값(원본 픽셀). 검출이 없으면 0."""
        return int(np.median(self.sizes)) if self.sizes else 0


_shared: FaceDetector | None = None
_shared_lock = threading.Lock()


def get_face_detector() -> FaceDetector:
    """프로세스 전역 싱글턴."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = FaceDetector()
        return _shared
