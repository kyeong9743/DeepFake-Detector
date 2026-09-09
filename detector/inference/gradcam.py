"""
Grad-CAM / Grad-CAM++ — 배치 처리, register_full_backward_hook 사용.

CNNResNet.gradcam_layer(layer4) 의 활성화와 그래디언트로 히트맵을 만든다.
한 번의 forward/backward 로 배치 전체의 CAM 을 계산한다.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, plus_plus: bool = True):
        self.model = model
        self.plus_plus = plus_plus
        self._act: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._fwd)
        self._h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self._act = out.detach()

    def _bwd(self, _m, _gi, gout):
        self._grad = gout[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """
        x: (B,3,H,W) 정규화된 입력.
        반환: cams (B,h,w) float32 [0,1],  logits (B,)
        """
        was_training = self.model.training
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        # 각 샘플의 자기 로짓에 대해 backward -> 배치 단위 gradient
        logits.sum().backward()
        act, grad = self._act, self._grad # (B,C,h,w)
        if self.plus_plus:
            g2, g3 = grad ** 2, grad ** 3
            sum_act = act.sum(dim=(2, 3), keepdim=True)
            alpha = g2 / (2 * g2 + sum_act * g3 + 1e-8)
            weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)
        else:
            weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1)) # (B,h,w)
        cam = cam - cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam = cam / (cam.flatten(1).max(dim=1)[0].view(-1, 1, 1) + 1e-8)
        if was_training:
            self.model.train()
        return cam.cpu().numpy().astype(np.float32), logits.detach()


def overlay_heatmap(frame_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """CAM (h,w) [0,1] 을 프레임 크기로 키워 JET 컬러맵으로 합성."""
    h, w = frame_bgr.shape[:2]
    heat = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat, alpha, 0)
