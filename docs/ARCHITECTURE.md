# CAKE v2 — 하이브리드 탐지 시스템 동작 구조

---

## 0. 한 문장 요약

영상에서 **시간 순서가 보존된 프레임**을 뽑아 **얼굴을 크롭**하고, **공간(CNN·ResNet) · 시간(LSTM) · 주파수(FFT)** 세 관점의 모델이 각각 낸 확률을 **검증 데이터로 최적화된 가중치와 투표**로 합쳐 딥페이크 위험 점수를 내고, **Grad-CAM** 으로 근거 영역을 시각화한다.

---

## 1. 시스템 토폴로지

```
          브라우저 (PC · 모바일 웹)
                │  ① multipart 업로드 / YouTube URL
                ▼
   ┌─────────────────────────────────┐
   │ backend  (FastAPI :8000)        │  backend/app/main.py
   │  · 웹 UI (Jinja2 + vanilla JS)  │
   │  · 업로드 스트리밍 저장 -> /media │
   │  · MariaDB analyses 테이블      │  backend/app/db.py
   │  · /reports 정적 서빙           │
   └──────────────┬──────────────────┘
                  │ ② POST /analyze {analysis_id, source, mode, callback_url}   (X-Internal-Token)
                  ▼
   ┌─────────────────────────────────┐
   │ detector (FastAPI :3001, GPU)   │  detector/api/main.py
   │  · asyncio.Queue -> 워커         │   GPU 세마포어(1) 아래 to_thread 로 추론 -> 이벤트 루프 비블로킹
   │  · DetectorEngine (모델 1회 로드)│  detector/inference/engine.py
   │  · /ws/analyze 실시간 추론      │
   │  · /metrics 리소스 지표         │
   └──────────────┬──────────────────┘
                  │ ③ POST callback_url {event: progress|result|failed, ...}
                  ▼
            backend DB 갱신 -> 브라우저는 GET /api/analyses/{id} 폴링(1.5->4s)

 공유 볼륨:  media/   (업로드 영상, yt-dlp 다운로드)     reports/  (분석 산출물)
 외부:       MariaDB  (.env DATABASE_URL, SQLAlchemy pool_pre_ping)
```

**왜 파일을 두 번 보내지 않는가** — backend 가 `/media/uploads/YYYYMM/{id}.mp4` 에 저장하고 detector 에는 **상대 경로만** 전달한다(`AnalyzeRequest.source_type="upload"`). 두 컨테이너가 같은 볼륨을 마운트하므로 detector 가 직접 읽다. v1 은 URL 을 넘겨 detector 가 다시 다운로드했다.

---

## 2. 추론 파이프라인 (`inference/engine.py: DetectorEngine.analyze_video`)

```
영상 파일
  │
  ├─ ensure_decodable()            common/video.py   OpenCV 가 못 읽는 코덱(AV1 등) -> ffmpeg 로 H.264 변환
  │
  ├─ ① extract_frames()           common/video.py
  │     · 영상 중앙 target_duration 초 구간 (fast 20s / high 30s)
  │     · 기본 샘플링 간격 + 장면전환(gray diff>22) · 모션(>8) 프레임 우선
  │     · max_frames 초과 시 importance 상위 N -> **np.sort 로 시간순 복원**
  │     · Frame(index, time_sec, image, importance) 반환 -> 리포트에 실제 초 표기
  │
  ├─ ② FaceTracker.crop()          common/face.py     YuNet(ONNX) 최대 얼굴, 1.3배 정사각 확장
  │     · 검출은 최대 변 640 으로 축소한 프레임에서 수행 후 원본 좌표 환산 (1080p 직접 검출 대비 ~10배 빠름)
  │     · 미검출 시 최근 박스 TTL 15프레임 재사용 -> 그래도 없으면 전체 프레임
  │     · face_ratio 집계 -> 0.3 미만이면 low_face_ratio 경고
  │     · 검출 실패 프레임(최근 박스 재사용·전체 프레임)은 **점수에서 제외** (frames_no_face).
  │       얼굴 프레임 < min_face_frames(8) 면 전체 프레임으로 분석 + no_face 경고. 검출 박스 중앙값 < min_face_px(128) 면 small_face 경고.
  │       quality_warnings[] · quality_messages[] · face_px 를 결과에 기록, face_warning = 경고 존재 여부. 판정 조작 아님 — 입력 필터
  │       (근거: sample 영상 A 는 마지막 21/184 프레임이 얼굴 스티커·전체 프레임 -> 제외 후 55->63점, sample 영상 B 는 얼굴 125px -> 경고)
  │
  ├─ ③ bgr_to_tensor()             common/preprocess.py   (N,3,S,S) [0,1]  ← 학습과 동일 함수
  │
  ├─ ④ 프레임 모델  _frame_scores()                     32장 배치, autocast fp16
  │     · CNN        : normalize(x) -> CNNResNet -> sigmoid            models/cnn_resnet.py
  │     · Frequency  : fft_spectrum(x) -> FrequencyAnalyzer -> sigmoid  models/frequency_analyzer.py
  │
  ├─ ⑤ 시퀀스 모델  _lstm_scores()
  │     · make_windows(N, window=16, stride=8) 오버랩 윈도우          common/preprocess.py
  │     · 윈도우마다 LSTMAnalyzer(normalize(seq), fft_spectrum(seq)) -> 확률
  │     · 윈도우 확률을 프레임에 펼쳐(겹침 평균) 그래프용 lstm_frame 생성
  │
  ├─ ⑥ ensemble()                  inference/engine.py
  │     · per_model = {cnn: mean, frequency: mean, lstm: mean(windows)}
  │     · score = Σ w_k·p_k  (w 는 checkpoints/{mode}_ensemble.json — training/ensemble_opt.py 산출)
  │     · vote  = Σ [p_k ≥ model_threshold_k]   (모델별 Youden 임계값)
  │     · confidence = 0.6·certainty + 0.4·agreement
  │           certainty = |score-0.5|·2 ,  agreement = 1 - 2·std(p)
  │     · is_deepfake = score ≥ threshold (ensemble.json)
  │     · risk_level  = 5단계 (config.risk_bounds: 0.30/0.50/0.70/0.85)
  │
  ├─ ⑦ Grad-CAM++                 inference/gradcam.py
  │     · 대상 층 CNNResNet.layer4, register_full_backward_hook, 16장 배치
  │     · 최대 gradcam_max_frames (fast 48 / high 96) 균등 선택
  │     · overlay -> heatmap/frame_XXXX.jpg, compare/ (원본|히트맵), suspicious/ (임계값 이상 top-12)
  │     · images_to_mp4() -> heatmap.mp4 (ffmpeg libx264, 실패 시 mp4v 대체)
  │
  └─ ⑧ 리포트                      inference/report.py
        scores.png · scores.csv/json (frame_index, time_sec, cnn, frequency, lstm, ensemble)
        report.html (단독 열람) · result.json (전체 결과)
```

**시각화(⑦⑧) 실패는 점수를 무효화하지 않다** — `try/except` 로 감싸 `visualization_error` 필드에 기록하고 결과는 반환한다. v1 은 ffmpeg 경로 하나로 전체가 실패했다.

### 실시간 (`analyze_frame`)
`/ws/analyze` 로 들어온 JPEG 1장 -> 얼굴 크롭 -> CNN + Frequency 만 (LSTM 은 시퀀스 필요) -> `{deepfake_score, risk_level, scores, confidence}` 즉시 회신. 측정 14 ms/프레임(RTX 5070). 웹 UI 의 버튼은 비활성(브라우저 화면 캡처 제약), 서버 API 는 완전 동작.

---

## 3. 세 모델의 실제 구조

### 3.1 CNN — `models/cnn_resnet.py: CNNResNet`
```
입력 (B,3,S,S) ImageNet 정규화
  -> torchvision resnet34(high) / resnet18(fast), ImageNet 사전학습 (stem, layer1~4)
  -> AdaptiveAvgPool2d(1) -> 512-d
  -> Dropout(0.3) -> Linear(512,256) -> ReLU -> Dropout -> Linear(256,1)   -> 로짓
Grad-CAM 대상: layer4    파라미터: ResNet34 ≈ 21.5M (v1 75.6M)
```
잔차 연결(residual)로 깊은 네트워크에서도 기울기 소실 없이 학습 — 발표자료가 말한 "ResNet 기반 특징 추출기"가 이제 실제이다. 사전학습 가중치는 `TORCH_HOME=/checkpoints/torch_cache` 에 캐시.

### 3.2 LSTM — `models/lstm_analyzer.py: LSTMAnalyzer`
```
spatial   (B,T,3,S,S) ─ ResNetStreamEncoder ─┐  ImageNet 사전학습 ResNet18 트렁크 -> GAP 512 -> Linear 128
frequency (B,T,3,S,S) ─ StreamEncoder_f ─────┤  소형 conv(3->32->64->96->128, stride2)+BN+ReLU+GAP (FFT 스펙트럼용)
                                              ▼   ← 두 스트림은 별도 인스턴스 (v1 은 가중치 공유)
                         concat(256) -> Linear(256,256)+ReLU+Dropout -> (B,T,256)
                                          ▼
                         LSTM(256->128, 2층, bidirectional=high) dropout 0.3 (2층이라 실제 적용)
                                          ▼
                         양방향: 시퀀스 평균 / 단방향: 마지막 스텝 -> LayerNorm -> Linear(64) -> Linear(1)
```
LSTM 게이트 (legacy technical_details 수식 그대로 적용됨):
`f_t=σ(W_f·[h_{t-1},x_t]+b_f)`, `i_t=σ(…)`, `c̃_t=tanh(…)`, `c_t=f_t⊙c_{t-1}+i_t⊙c̃_t`, `o_t=σ(…)`, `h_t=o_t⊙tanh(c_t)`; 양방향은 `[h_t^->; h_t^←]` 결합.

### 3.3 Frequency — `models/frequency_analyzer.py` + `common/preprocess.py: fft_spectrum`
```
x [0,1] -> 채널별 표준화 -> fft2(norm="ortho") -> fftshift -> log1p(|X|) -> 샘플별 표준화   (B,3,S,S)
  -> conv(3->32->64->128->160, stride2)+BN+ReLU -> GAP -> Dropout -> Linear(160,128) -> Linear(128,1)
```
`X_k = Σ x_n·e^{-2πikn/N}`; 진폭 스펙트럼 `|X(f)|` 의 로그를 사용. fftshift 로 저주파를 중앙에 두어 Conv 커널이 주파수 구조를 학습. **학습(`training/dataset.py`)과 추론(`engine.py`)이 같은 `fft_spectrum` 을 호출** — v1 은 학습에서 다른(잘못된) 변환을 썼고 추론에서는 변환을 하지 않았다.

### 3.4 모드 프로필 — `config.py: PROFILES`
| | fast | high |
|---|---|---|
| frame_size | 224 | 384 |
| backbone | resnet18 | resnet34 |
| window / stride | 16 / 8 | 16 / 8 |
| target_duration / fps | 20s / 8 | 30s / 10 |
| max_frames | 160 | 300 |
| LSTM | 단방향 | **양방향** |
| gradcam_max_frames | 48 | 96 |

---

## 4. 앙상블 — 왜 이렇게 합치는가

| 요소 | 구현 | 발표자료 용어 |
|---|---|---|
| 가중 평균 | `Σ w_k p_k`, w 는 OOF 예측에서 AUC 최대화 | PredIntegrator · 가중 평균 |
| 투표 | 모델별 Youden 임계값으로 이진화 -> 다수결 | PredIntegrator · 투표 시스템 |
| 임계값 | OOF 앙상블 확률의 Youden's J | ConfCalculator · 임계값 적용 |
| 신뢰도 | 확신도×일치도 | ConfCalculator · 확률 추정 |
| 등급 | 5단계 risk_level | ResultAggregator · 신뢰도 등급 분류 |

가중치 학습 (`training/ensemble_opt.py`):
1. **Grid Search** — 심플렉스 위 0.05 간격 전수 탐색 (3모델 -> 231 조합)
2. **Bayesian Optimization** — GP(Matern ν=2.5) + Expected Improvement, stick-breaking 으로 심플렉스 파라미터화, 40 반복
3. 둘 중 AUC 높은 쪽 채택, `checkpoints/{mode}_ensemble.json` 에 방법·가중치·임계값·모델별 임계값·지표 기록

이 파일이 없으면 `load_ensemble_config()` 가 균등 가중치 + 0.5 로 동작하며 로그에 경고한다.

---

## 5. 학습 파이프라인 (`training/`)

```
AI Hub 4TB 디스크 (/aihub, ro)  또는  data/real, data/fake
  │
  ├─ build_aihub_manifest.py   메타 CSV(라벨·변조모델·UUID·타겟) -> manifest.csv
  │                             방법별 균등, 타겟(원본)당 ≤2 변조본, group = 인물 UUID, AI Hub Validation -> split=val
  │
  ├─ prepare_data.py     영상 -> 얼굴 크롭 JPEG 캐시 (/data/cache = Docker 네임드 볼륨 cake_cache)
  │    · --clips 3 --clip-len 16 --clip-stride 2 : 3 구간 × 연속 16프레임 (seek 3회 + grab 순차; HEVC 장편에서 균등 48 seek 대비 13배 빠름)
  │    · FaceDetector.detect: 최대 변 640 으로 축소 검출 후 원본 좌표 환산 / FaceTracker(detect_every=4): 구간 내 4프레임마다 검출
  │    · meta.json 에 method/group/gender/clips/clip_len 기록, index.json 집계. 얼굴 비율 < min_face_ratio 제외
  │
  ├─ make_sns_variants.py  캐시의 각 클립(연속 16프레임)을 ffmpeg libx264 로 **실제 H.264 재인코딩**
  │    · 클립마다 랜덤: 축소 0.5~0.85 -> CRF 23~34 (preset medium) -> 원 크기 복원, 25% 선명화 / 25% 스무딩 필터 선적용
  │    · {id}/sns{k}/f_XXXX.jpg + params.json(version) 저장, index.json·meta.json 에 sns_variants=k 기록. 기본 2 변형(캐시 +2배)
  │    · 프레임 단위 JPEG 증강으로는 못 만드는 블록/디블로킹/P-frame 열화를 오프라인으로 생성 (학습 중 인코딩은 DataLoader 병목)
  │
  ├─ train.py            StratifiedGroupKFold (group=인물 -> 같은 사람이 train/val 에 갈라지지 않음)
  │    · split=val 영상은 fold 밖 홀드아웃 -> 모든 fold 검증셋에 추가, 최종 모델 평가에 사용
  │    fold 마다:  build_model(pretrained) -> AdamW(백본 lr, 헤드 5×lr) -> Cosine -> AMP fp16 -> grad clip 5
  │                cudnn.benchmark · channels_last(CNN/FFT) · BCEWithLogitsLoss(pos_weight) · 조기 종료 · best-AUC 스냅샷
  │                dataset.py: FrameDataset(CNN/FFT, 정규화 float) · SequenceDataset(LSTM) — **uint8 프레임만 반환**,
  │                            정규화·FFT 는 GPU(forward)에서 -> DataLoader shm 사용 16배↓ (shm 4 GB Bus error 의 원인)
  │                            clip_len == window 이면 구간 경계에 윈도우 정렬
  │                augment.py: RandomResizedCrop · flip · ColorJitter(HSV) · JPEG(30~90) · downscale · blur · noise (시퀀스 동일 파라미터)
  │                            + gamma 0.7~1.4 · unsharp 선명화 · bilateral 스무딩 (소셜 앱 필터 흉내, 압축 전에 적용)
  │                --sns-prob p : 학습 샘플을 확률 p 로 sns{k}/ 변형본에서 읽음 (dataset.pick_variant). 변형본에는 프레임 증강 강도 ½
  │                검증·OOF·홀드아웃은 항상 원본 -> 이전 결과와 직접 비교 가능
  │                metrics.py: AUC/AP/ACC/P/R/F1/CM/Youden -> *_metrics.jsonl, *_curves.png, roc_pr, cm
  │    fold 검증 영상(홀드아웃 제외)의 영상 단위 확률 -> oof_{kind}.csv ; 변조 방법별 recall/AUC -> by_method
  │    최종: pool(홀드아웃 제외) 전체로 재학습(에폭 = fold best 평균) -> **마지막 에폭 모델** 사용(의사검증 AUC 포화로 best 선택 금지)
  │          에폭마다 {kind}_final_state.pt 저장 -> 죽으면 자동 이어하기 (CUDA unknown error 2회 경험)
  │          홀드아웃 예측 -> holdout_{kind}.csv, 지표는 체크포인트 메타·{kind}_summary.json 에 기록
  │          홀드아웃에 sns0 변형본이 있으면 **압축 강건성** 지표(holdout_sns)도 계산 -> holdout_sns_{kind}.csv
  │    --skip-folds : oof 가 있으면 fold 생략, 최종만 (최종 단계 실패 후 재개)   --cache-dir : 캐시 위치
  │
  └─ ensemble_opt.py     oof_*.csv 를 영상별 평균으로 1행화(홀드아웃 fold 중복 제거) -> merge -> Grid(0.05) + BO(GP-EI 40) -> {mode}_ensemble.json
                         --min-weight : 모델별 최소 가중치. 0 이면 cnn .35 / lstm .65 / frequency .00 (AUC .9977).
                         **팀 결정: 0.1 로 세 모델 모두 반영** -> cnn .52 / lstm .38 / fft .10, AUC .9965, ACC 98.1%, thr .514
  ├─ eval_holdout.py     학습 없이 체크포인트(--ckpt-dir 로 보관본 지정 가능)를 홀드아웃 원본 + sns0 변형본으로 재평가 -> holdout_eval_{tag}.json
  └─ eval_samples.py     data/samples/ (TikTok 실제 딥페이크 8개, md5 중복 제거) 일괄 분석 -> runs/samples_eval_{tag}.json  (도메인 밖 일반화 점검)
```

실행 래퍼: `scripts/aihub_pipeline.sh` (STEP/END_STEP/MODELS/TRAIN_ARGS/CLIPS/…), `scripts/aihub_run_detached.sh` (setsid 분리 + `### PIPELINE DONE|FAILED` 마커). **WSL 또는 Linux 에서** `docker-compose.aihub.yml` 오버라이드와 함께. 데이터 위치는 `.env` 의 `AIHUB_DATA_DIR` (기본값 없음).

체크포인트 메타(`models/registry.py: save_checkpoint`): kind, mode, backbone, frame_size, window, stride, bidirectional, normalize_mean/std, saved_at, metrics(oof·holdout·folds·by_method), n_videos. 로드는 `strict=True`, 메타 불일치 시 예외 — **랜덤 가중치로 조용히 추론하는 경로가 없다.**

**실측**: AI Hub 홀드아웃 2,640 — CNN AUC 0.9987/ACC 98.8%, LSTM 0.9991/98.6%, FFT 0.873/78.9%, 앙상블 0.9977/98.1%. SNS 실제 영상 8개(가짜 5·진짜 3, 라벨 정정·샘플 추가): 가짜 3/5 탐지, 진짜 3/3 정답 -> 도메인 밖 일반화 약함. `runs/samples_eval_v1_labeled.json`.

---

## 6. API 계약

### backend -> detector (`api/schemas.py: AnalyzeRequest`)
```json
{"analysis_id":"<영숫자-_ ≤64>", "source_type":"upload|url", "source":"uploads/202609/abc.mp4 | https://…",
 "mode":"fast|high|null", "gradcam":true, "callback_url":"http://backend:8000/internal/callback"}
```
`analysis_id` 정규식 검증, `source` 의 `..` 거부, `/media` 밖 경로 거부 (`Path.is_relative_to`).

### detector -> backend 콜백
```json
{"analysis_id":"…", "event":"progress|result|failed", "status":"…", "stage":"…", "progress":0-100,
 "message":"…", "error":null, "result":{…result.json 전체…}}
```
진행률 콜백은 초당 1회로 제한. 단계->전체% 매핑: download 0-10, extract 10-30, preprocess 30-45, infer 45-75, report 75-98, done 98-100.

### result.json 주요 필드
`deepfake_score, is_deepfake, threshold, confidence, risk_level, risk_color, scores{cnn,lstm,frequency}, weights, ensemble_method, vote{fake,total}, frame_count, frames_extracted, frames_no_face, face_ratio, face_px, quality_warnings[], quality_messages[], face_warning, video{duration,fps,width,height,total_frames}, frame_scores{time[],cnn[],frequency[],lstm[],ensemble[]}, lstm_windows[], suspicious_frames[], heatmap_video, artifacts{}, processing_time`

---

## 7. 데이터 모델 (`backend/app/db.py: Analysis`)

`analyses` 테이블 1개. 입력(source_type/name/url, media_path, file_size, mode, client_ip) · 진행(status, stage, progress, message, error) · 결과 요약(deepfake_score, confidence, risk_level, is_deepfake, cnn/lstm/frequency_score, vote_fake, frame_count, face_ratio, processing_time, video_duration, result_json). 프레임별 점수 등 큰 데이터는 DB 에 넣지 않고 `reports/{id}/result.json` 에서 읽다.

---

## 8. 보안·운영 설계

| 항목 | 구현 |
|---|---|
| 내부 인증 | `X-Internal-Token` (backend↔detector, .env `INTERNAL_API_TOKEN`) |
| 업로드 검증 | 스트리밍 저장하며 **실제 바이트 수**로 제한, 확장자 + **매직 바이트** 검사 |
| 경로 탐색 | analysis_id 정규식, `..` 거부, `/media` 상대경로 검증, StaticFiles 로 산출물 서빙 |
| 동시성 | `asyncio.Queue` + `Semaphore(max_concurrent_analyses=1)`; 추론은 `to_thread` |
| 모델 미로드 | `/health models_loaded=false`, `/analyze` -> 503, backend 는 사용자에게 안내 |
| 비밀정보 | 소스에 없음. `.env`(gitignore) 만. DB 계정은 `cake.*` 권한만 |
| 코덱 | AV1/HEVC -> ffmpeg H.264 변환 후 분석 |
| 모니터링 | detector `/metrics`(psutil + pynvml), backend `/monitor` 대시보드 2s 폴링 |

---

## 9. 파일 지도

```
detector/
  config.py                 모드 프로필, 경로, 체크포인트 이름, risk_level()
  common/face.py            YuNet FaceDetector / FaceTracker (sizes · median_face_px)
  common/preprocess.py      bgr_to_tensor · normalize · fft_spectrum · make_windows · pad_sequence  ← 학습·추론 공용
  common/video.py           probe · extract_frames · download_video(yt-dlp) · ensure_decodable · images_to_mp4
  models/cnn_resnet.py      CNNResNet
  models/lstm_analyzer.py   StreamEncoder · LSTMAnalyzer
  models/frequency_analyzer.py
  models/registry.py        build_model · save/load_checkpoint(메타) · load_ensemble_config
  inference/engine.py       DetectorEngine.analyze_video / analyze_frame · ensemble()
  inference/gradcam.py      GradCAM(++) · overlay_heatmap
  inference/report.py       그래프 · CSV/JSON · 히트맵 · 의심 프레임 · HTML
  inference/cli.py          서버 없는 단일 영상 분석
  training/prepare_data.py  프레임 캐시
  training/dataset.py       FrameDataset · SequenceDataset · split_videos(K-fold)
  training/augment.py       증강 (+ 소셜 필터)
  training/make_sns_variants.py  H.264 재인코딩 변형본(sns{k}/) 오프라인 생성
  training/eval_holdout.py  홀드아웃 원본·SNS 변형본 재평가 (강건성)
  training/eval_samples.py  data/samples/ TikTok 일괄 분석 (일반화)
  training/train.py         K-fold 학습 · OOF · 최종 모델
  training/ensemble_opt.py  Grid Search + Bayesian Opt
  training/metrics.py       지표 · 곡선 · ROC/PR · CM
  training/make_synthetic.py 스모크 테스트 데이터
  api/main.py               FastAPI: /analyze /status /ws/analyze /metrics /health
  api/schemas.py
backend/
  app/main.py               페이지 · /api/analyses · /internal/callback · /api/metrics · /ws/realtime 프록시
  app/db.py                 SQLAlchemy Analysis
  app/config.py
  web/templates/            base · index · analysis · history · realtime · monitor
  web/static/               app.css · app.js (업로드 XHR 진행률, 폴링, 캔버스 차트, 모니터링)
```
