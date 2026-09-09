"""
분석 산출물 생성 — 점수 그래프 · CSV/JSON · Grad-CAM 프레임/영상 · HTML 리포트.

모든 파일은 reports/{analysis_id}/ 아래에 생성되며 backend 가 정적으로 서빙.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt # noqa: E402
import numpy as np

from common.video import images_to_mp4


def save_score_plot(out: Path, frame_times: list[float], series: dict[str, list[float]], threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {"cnn": "tab:blue", "frequency": "tab:orange", "lstm": "tab:green", "ensemble": "tab:red"}
    for name, ys in series.items():
        ys_arr = np.array(ys, dtype=float)
        if np.all(np.isnan(ys_arr)):
            continue
        ax.plot(frame_times[: len(ys_arr)], ys_arr, marker=".", ms=3, lw=1.4,
                label=name.upper(), color=colors.get(name), alpha=0.9 if name == "ensemble" else 0.7)
    ax.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold {threshold:.2f}")
    ax.set_ylim(0, 1); ax.set_xlabel("time (s)"); ax.set_ylabel("deepfake probability")
    ax.set_title("Frame-wise scores"); ax.grid(alpha=.3); ax.legend(loc="upper right", ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def save_scores_table(out_csv: Path, out_json: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def save_heatmap_frames(out_dir: Path, frames_bgr: list[np.ndarray], overlays: list[np.ndarray],
                        times: list[float], scores: list[float]) -> list[str]:
    """원본|히트맵 비교 이미지 + 히트맵 단독 이미지 저장. 반환: 히트맵 단독 파일 경로 목록."""
    heat_dir, cmp_dir = out_dir / "heatmap", out_dir / "compare"
    heat_dir.mkdir(exist_ok=True); cmp_dir.mkdir(exist_ok=True)
    paths = []
    for i, (orig, ov, t, s) in enumerate(zip(frames_bgr, overlays, times, scores)):
        ov = cv2.resize(ov, (orig.shape[1], orig.shape[0]))
        label = f"t={t:.1f}s  p={s:.2f}"
        cv2.putText(ov, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        hp = heat_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(hp), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(cmp_dir / f"frame_{i:04d}.jpg"), cv2.hconcat([orig, ov]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(hp))
    return paths


def save_suspicious(out_dir: Path, heat_paths: list[str], scores: list[float], threshold: float, top_k: int = 12) -> list[str]:
    """임계값 이상 프레임 중 점수 상위 top_k 를 suspicious/ 로 복사."""
    sus = out_dir / "suspicious"
    sus.mkdir(exist_ok=True)
    idx = [i for i, s in enumerate(scores) if s >= threshold]
    idx.sort(key=lambda i: -scores[i])
    out = []
    for i in idx[:top_k]:
        dst = sus / Path(heat_paths[i]).name
        img = cv2.imread(heat_paths[i])
        if img is not None:
            cv2.imwrite(str(dst), img)
            out.append(dst.name)
    return out


def save_heatmap_video(out_dir: Path, heat_paths: list[str], fps: int = 8) -> dict[str, Any]:
    if not heat_paths:
        return {"video": None, "h264": False}
    out = out_dir / "heatmap.mp4"
    ok = images_to_mp4(heat_paths, str(out), fps=fps)
    return {"video": out.name, "h264": ok}


def write_html(out_dir: Path, result: dict[str, Any]) -> Path:
    """단독으로 열 수 있는 요약 리포트."""
    r = result
    def pct(v): return f"{v*100:.1f}%" if isinstance(v, (int, float)) and not np.isnan(v) else "—"
    sus_imgs = "".join(f'<img src="suspicious/{n}" loading="lazy">' for n in r.get("suspicious_frames", []))
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CAKE 분석 리포트 {r['analysis_id']}</title>
<style>
body{{font-family:system-ui,-apple-system,'Noto Sans KR',sans-serif;margin:0;background:#f5f6fa;color:#1c1f2b}}
main{{max-width:960px;margin:0 auto;padding:24px}} h1{{font-size:1.4rem;margin:0 0 4px}} .sub{{color:#666;font-size:.9rem}}
.card{{background:#fff;border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.kv b{{display:block;font-size:1.5rem}} .kv span{{color:#666;font-size:.85rem}}
.risk{{display:inline-block;padding:4px 14px;border-radius:999px;color:#fff;font-weight:700;font-size:1rem;line-height:1.6;vertical-align:middle;background:{r.get('risk_color','#888')}}}
img,video{{max-width:100%;border-radius:8px}} .sus img{{width:180px;margin:4px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} td,th{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}}
</style></head><body><main>
<h1>CAKE 딥페이크 분석 리포트</h1><div class="sub">분석 ID {r['analysis_id']} · 모드 {r['mode']} · {r.get('created_at','')}</div>
<div class="card"><div class="grid">
<div class="kv"><b><span class="risk">{r['risk_level']}</span></b><span>위험도</span></div>
<div class="kv"><b>{r['deepfake_score']*100:.0f}점</b><span>딥페이크 위험 점수</span></div>
<div class="kv"><b>{pct(r['confidence'])}</b><span>신뢰도</span></div>
<div class="kv"><b>{r['vote']['fake']}/{r['vote']['total']}</b><span>모델 투표 (fake)</span></div>
</div></div>
<div class="card"><h3>모델별 점수</h3><table>
<tr><th>모델</th><th>점수</th><th>가중치</th></tr>
<tr><td>CNN (ResNet)</td><td>{pct(r['scores']['cnn'])}</td><td>{r['weights']['cnn']:.2f}</td></tr>
<tr><td>LSTM (시간)</td><td>{pct(r['scores']['lstm'])}</td><td>{r['weights']['lstm']:.2f}</td></tr>
<tr><td>Frequency (FFT)</td><td>{pct(r['scores']['frequency'])}</td><td>{r['weights']['frequency']:.2f}</td></tr>
</table><p class="sub">가중 방식: {r.get('ensemble_method','')} · 판정 임계값 {r['threshold']:.2f} · 얼굴 검출 비율 {pct(r['face_ratio'])} · 얼굴 크기 {r.get('face_px', '—')}px · 얼굴 없는 프레임 제외 {r.get('frames_no_face', 0)}</p>{''.join(f'<p class="sub" style="color:#b54708">⚠ {m}</p>' for m in r.get('quality_messages', []))}</div>
<div class="card"><h3>프레임별 점수</h3><img src="scores.png" alt="scores"></div>
<div class="card"><h3>Grad-CAM 히트맵</h3>{'<video src="heatmap.mp4" controls loop muted playsinline></video>' if r.get('heatmap_video') else '<p>히트맵 영상 없음</p>'}</div>
<div class="card sus"><h3>의심 프레임 (임계값 이상)</h3>{sus_imgs or '<p>임계값 이상 프레임 없음</p>'}</div>
<div class="card"><h3>다운로드</h3><a href="scores.csv">scores.csv</a> · <a href="scores.json">scores.json</a> · <a href="result.json">result.json</a></div>
<p class="sub">본 결과는 딥페이크 탐지 <b>보조</b> 정보이며 최종 판단은 이용자에게 있습니다.</p>
</main></body></html>"""
    p = out_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p
