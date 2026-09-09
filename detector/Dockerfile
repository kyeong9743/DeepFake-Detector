# CAKE Detect Server
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/checkpoints/torch_cache

# ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 얼굴 검출 모델
RUN mkdir -p /opt/cake/assets && curl -fsSL -o /opt/cake/assets/face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

COPY . .

EXPOSE 3001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3001", "--workers", "1"]
