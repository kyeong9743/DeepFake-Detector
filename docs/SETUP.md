# CAKE — 설치 · 실행 · 학습 가이드

> 프로젝트 소개와 실험 결과는 [README](../README.md) 를 참고한다.
> 이 문서는 직접 돌려 보려는 사람을 위한 운영 문서이다.

---

## 1. 빠른 시작

### 요구 사항
- Docker Desktop (WSL2) + NVIDIA 드라이버 (RTX 50 시리즈 포함, CUDA 12.8)
- `.env` — `.env.example` 을 복사해 `DATABASE_URL`, `INTERNAL_API_TOKEN` 설정
- MariaDB(또는 SQLAlchemy 지원 DB) — DB 와 계정만 만들어 두면 backend 가 시작 시 테이블을 자동 생성한다. 참고 스키마·계정 생성문: [backend/schema.sql](../backend/schema.sql)

```bash
cp .env.example .env          # 값 채우기
docker compose up -d --build  # detector(GPU) + backend
```

- 웹 UI: http://localhost:8000
- detector 상태: http://localhost:3001/health -> `models_loaded: true` 여야 분석 가능

> 처음 실행하면 **학습된 체크포인트가 없어** `models_loaded: false` 이고 분석 요청은 503 을 반환한다.
> 랜덤 가중치로 조용히 추론하지 않는다. 아래 [3. 학습 파이프라인] 절차로 학습하거나, 학습된 `high_cnn.pt · high_lstm.pt · high_frequency.pt · high_ensemble.json` 을 `checkpoints/` 에 넣는다 (가중치는 저장소에 포함되지 않는다).

### 처음 받은 뒤 동작 확인 (5분)

```bash
# 1) 컨테이너 상태 — detector 가 healthy, models_loaded 가 true 여야 함
docker compose ps
curl localhost:3001/health          # {"status":"ok","mode":"high","models_loaded":true,...}

# 2) CLI 로 영상 1개 분석 (서버 API 를 거치지 않음). 영상은 data/samples/ 에 두면 컨테이너의 /data/samples 에서 보임
docker compose exec detector python -m inference.cli /data/samples/내영상.mp4 --no-gradcam      # Git Bash 는 앞에 MSYS_NO_PATHCONV=1

# 3) 웹 — http://localhost:8000 에서 같은 영상 업로드 -> 결과 페이지(점수·모델별 점수·그래프·히트맵), /history 에 기록, /monitor 에 GPU 지표
#    API 로 하려면:
curl -F "file=@data/samples/내영상.mp4" -F "mode=high" localhost:8000/api/analyses     # -> {"id":...}
curl localhost:8000/api/analyses/<id>                                                  # status 가 done 이 될 때까지 폴링
```

실데이터 없이 파이프라인(전처리 -> 학습 -> 앙상블 -> 분석)만 검증하려면 `.\scripts\smoke_test.ps1` (합성 데이터, 성능 수치 무의미).

---

## 2. 데이터셋 규격

```
data/
├─ real/   *.mp4 | *.avi | *.mov | *.mkv | *.webm      (label 0)
├─ fake/   *.mp4 ...                                    (label 1)
└─ manifest.csv   (선택)  path,label[,source][,split]
                          real/a.mp4,0,aihub,train
                          fake/b.mp4,1,aihub,val
```

- `manifest.csv` 가 없으면 폴더명으로 라벨을 유도한다.
- 영상은 **얼굴이 보이는 3초 이상**이면 된다. 해상도·길이 제한 없음.
- 클래스별 최소 수십 개, 권장 수백 개 이상. 두 클래스의 **영상 길이 분포가 비슷**해야 한다(길이로 클래스를 맞추는 누출 방지).

---

## 2-1. AI Hub 「딥페이크 변조 영상」 데이터셋 (실학습용)

데이터는 WSL(Ubuntu)에 붙은 4 TB ext4 디스크에 있다 — **복사하지 않고** 컨테이너에 읽기 전용으로 바인드한다.

```
<AIHUB_DATA_DIR>/003.딥페이크/                (2.8 TB, 총 231,000 mp4)
├─ 1.Training/
│  ├─ 라벨링데이터/train_meta_data/{원본,변조}영상_training_메타데이터.csv
│  └─ 원천데이터/
│     ├─ train_원본/원본N/{UUID}/{UUID}_NNN.mp4       49,419개  label 0   HEVC 1080p ~90s
│     ├─ train_변조/{fo|dffs|dfl|fsgan|audio_driven}N/{UUID}/*.mp4   139,950개  label 1   H.264 1080p ~20s
│     └─ train_탐지방해/{원본영상|변조영상}/NN_*/*.mp4   16,760개  탐지 방해 처리본(고비트레이트)
└─ 2.Validation/  동일 구조 (원본 6,093 · 변조 16,894 · 탐지방해 2,044)
```

| 변조 방법 | 타겟(원본) 영상 수 | 의미 |
|---|---:|---|
| fo | 39,492 | FaceSwap 계열 |
| dffs | 23,451 | DeepFaceLab-FaceSwap 계열 |
| dfl | 21,836 | DeepFaceLab |
| audio-driven | 12,480 | 음성 구동 립싱크 |
| fsgan | 3,728 | FSGAN |

**절차 (WSL 또는 Linux 에서)**

먼저 `.env` 에 데이터 위치를 적는다 (기본값 없음 — 없으면 스크립트와 compose 가 즉시 오류를 낸다):
```dotenv
AIHUB_DATA_DIR=/mnt/<디스크>/<AI Hub 폴더>     # 003.딥페이크/ 가 바로 아래에 있는 디렉터리 (필수)
AIHUB_MOUNT=/mnt/<디스크>                      # (선택) 마운트 지점 — 지정하면 마운트 여부를 확인
AIHUB_DISK_UUID=                               # (선택) 마운트가 풀려 있을 때 sudo mount UUID=… 로 자동 마운트. blkid 로 확인
```
```bash
cd <repo>/cake
bash scripts/aihub_pipeline.sh   # manifest -> 전처리 -> K-fold -> 앙상블 -> 재시작
```
`build_aihub_manifest.py` 가 메타 CSV로 라벨·변조방법·인물(UUID)을 붙여 클래스 균형(방법별 균등, 타겟당 ≤2개)으로 `data/manifest.csv` 를 만들고, `split_videos` 는 **인물 단위 StratifiedGroupKFold** 로 나눠 같은 사람이 train/val 에 갈라지는 누출을 막다. AI Hub Validation 셋은 K-fold 밖의 **홀드아웃**으로 최종 지표에 쓰이다 (`runs/{mode}/{kind}_summary.json` 의 `holdout`, 변조 방법별 `by_method`).

> ⚠️ **클래스 간 인코딩 편향** — 원본은 HEVC·90초·저비트레이트, 변조는 H.264·20초이다. 모델이 "압축 아티팩트 = real" 식의 지름길을 배울 위험이 있어 (a) 프레임 캐시는 JPEG q95 로 통일, (b) 학습 시 JPEG 재압축·다운스케일 증강을 적용하며, (c) 홀드아웃과 `data/samples/`(TikTok) 로 일반화를 확인해야 한다. 방법별 recall 이 `by_method` 에 기록되므로 특정 변조 방법만 못 잡는지도 볼 수 있다.

## 3. 학습 파이프라인

**AI Hub 실데이터는 §2-1 의 `scripts/aihub_pipeline.sh` (WSL) 를 사용한다.** 아래는 단계별 수동 실행 (소규모 자체 데이터 · 디버깅용):

```bash
# ① 전처리: 영상 -> 얼굴 크롭 프레임 캐시 (/data/cache).  --clips 3 --clip-len 16 = 3구간 × 연속 16프레임 (권장)
docker compose run --rm detector python -m training.prepare_data --frames 48 --size 384 --clips 3 --clip-len 16 --clip-stride 2

# ①-b SNS 압축 변형본 (선택, 권장): 캐시 클립을 H.264 로 재인코딩한 sns0/ sns1/ 생성 (캐시 용량 ×3, 15.8k 영상 ≈ 15 분)
docker compose run --rm detector python -m training.make_sns_variants --variants 2 --workers 12

# ② K-fold 학습 (CNN · LSTM · Frequency) -> checkpoints/high_*.pt, runs/high/   (실데이터 실측 기준: 3-fold × 8 epoch ≈ 12.5 h)
#    --sns-prob 0.5 : 학습 샘플의 절반을 SNS 변형본에서 읽음 (①-b 를 만든 경우). 검증·홀드아웃은 항상 원본
docker compose run --rm detector python -m training.train --mode high --model all --folds 3 --epochs 8 --batch-size 24 --workers 12 --sns-prob 0.5
#    --skip-folds : oof_{kind}.csv 가 있으면 fold 생략, 최종 학습만 (최종 단계 실패 후 재개 — 에폭별 상태 자동 이어하기 포함)

# ③ 앙상블 가중치 최적화 (Grid Search + Bayesian Opt) -> checkpoints/high_ensemble.json
docker compose run --rm detector python -m training.ensemble_opt --mode high            # --min-weight 0.1 : 세 모델 모두 최소 10% 반영

# ④ detector 재시작 -> 새 체크포인트 로드
docker compose restart detector
```

`scripts/aihub_pipeline.sh` 환경변수: `MODE FOLDS EPOCHS BATCH WORKERS PER_CLASS VAL_PER_CLASS CLIPS CLIP_LEN CLIP_STRIDE STEP END_STEP MODELS TRAIN_ARGS INCLUDE_ANTI SNS_VARIANTS(2) SNS_PROB(0.5) MIN_WEIGHT(0.1)`. 

데이터 위치는 `.env` 의 `AIHUB_DATA_DIR`(필수, 기본값 없음)로 지정하고, `AIHUB_MOUNT`·`AIHUB_DISK_UUID`(선택)를 주면 마운트되어 있지 않을 때 자동 마운트한다. 

`docker-compose.aihub.yml` 도 `AIHUB_DATA_DIR` 를 읽는다. 

단계: `manifest -> prepare -> sns -> train -> ensemble -> restart`.

평가 전용: `python -m training.eval_holdout --tag …` (홀드아웃 원본·SNS 변형본), `python -m training.eval_samples --tag …` (data/samples TikTok).

분리 실행(터미널을 닫아도 계속)은 `scripts/aihub_run_detached.sh` — 로그 `runs/aihub_pipeline.log`, 종료 시 마지막 줄에 `### PIPELINE DONE` 또는 `### PIPELINE FAILED (exit N)`.

AI Hub 없이 `data/real|fake` 에 직접 넣은 소규모 영상으로 학습할 때는 위 §3 의 `docker compose run …` 명령을 순서대로 실행한다 (PowerShell·bash 동일).

fast 모드도 필요하면 `--mode fast` 로 반복한다.

### 학습 산출물 (`runs/{mode}/`)
| 파일 | 내용 |
|---|---|
| `{kind}_fold{k}_metrics.jsonl` | 에폭별 loss · AUC · Accuracy · F1 · Confusion Matrix |
| `{kind}_fold{k}_curves.png` | 손실 / AUC / 정확도 곡선 (MetricVisualizer) |
| `{kind}_fold{k}_roc_pr.png`, `_cm.png` | ROC·PR 곡선, 혼동 행렬 |
| `oof_{kind}.csv` | 영상 단위 out-of-fold 예측 (앙상블 최적화 입력) |
| `{kind}_summary.json` | fold 별 지표 요약 (+ `holdout`, `holdout_sns` 압축 강건성) |
| `holdout_eval_{tag}.json`, `samples_eval_{tag}.json` | 재평가 결과 (eval_holdout / eval_samples) |
| `ensemble_search_trace.json` | Grid / Bayesian 탐색 기록 |

체크포인트(`checkpoints/{mode}_{kind}.pt`)에는 state_dict 와 함께 **mode · 백본 · frame_size · 정규화 통계 · 검증 지표 · 학습 시각**이 저장되며 로드 시 `strict=True` 로 검증한다.

### 파이프라인만 검증하려면 (실데이터 없이)
```bash
docker compose run --rm detector python -m training.make_synthetic --n 40
docker compose run --rm detector python -m training.prepare_data --frames 32 --size 224 --min-face-ratio 0
docker compose run --rm detector python -m training.train --mode fast --folds 3 --epochs 3
docker compose run --rm detector python -m training.ensemble_opt --mode fast
```
합성 데이터는 **동작 검증용**이며 성능 수치는 의미가 없다.

---

## 4. 분석 흐름 (detector)

1. **프레임 추출** — 영상 중앙 구간에서 적응형 샘플링(장면 전환·모션 우선), **시간 순서 보존**
2. **얼굴 검출·크롭** — OpenCV YuNet. 검출 실패 프레임은 크롭은 만들되 **점수에서 제외**(`frames_no_face`); 얼굴 프레임이 8개 미만이면 전체 프레임으로 분석하고 `no_face` 경고. 얼굴 박스 중앙값 < 128 px 면 `small_face` 경고 (`quality_warnings`)
3. **CNN** — ResNet34(high) / ResNet18(fast) 프레임별 확률
4. **Frequency** — 채널별 표준화 -> FFT -> fftshift -> log -> 표준화 -> CNN
5. **LSTM** — 16프레임 윈도우, 8프레임 stride 오버랩. 공간·주파수 **독립 인코더** -> 융합 -> BiLSTM
6. **앙상블** — 검증 데이터로 최적화된 가중 평균 + 모델별 임계값 다수결 투표 + 신뢰도(확신도×일치도)
7. **Grad-CAM++** — `layer4` 기준 배치 처리, 히트맵 프레임/비교 이미지/mp4
8. **리포트** — `reports/{id}/` : `result.json`, `scores.csv/json/png`, `heatmap/`, `compare/`, `suspicious/`, `report.html`

학습과 추론은 `common/preprocess.py` 의 **같은 함수**로 전처리하므로 입력 분포가 일치한다.

---

## 5. API

### backend (:8000)
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/analyses` | `file`(multipart) 또는 `url`, `mode` -> `{id, url}` (202) |
| GET | `/api/analyses` | 기록 목록 `?page=&size=&status_f=` |
| GET | `/api/analyses/{id}` | 상태·진행률·결과 (완료 시 프레임 점수 포함) |
| GET | `/api/analyses/{id}/download/{name}` | `scores.csv` · `result.json` · `report.html` · `heatmap.mp4` |
| GET | `/api/stats`, `/api/health` | 통계, 서버 상태 |
| POST | `/internal/callback` | detector -> backend (내부 토큰) |
| WS | `/ws/realtime` | detector `/ws/analyze` 프록시 |

### detector (:3001)
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/analyze` | 내부 토큰 필요. 202 + 큐 등록, GPU 세마포어로 순차 처리 |
| GET | `/status/{id}` | 폴링 |
| WS | `/ws/analyze` | JPEG 프레임 -> 프레임별 실제 추론 결과 |
| GET | `/health` | `models_loaded`, `queue_size` |

---

## 5-1. CLI (서버 없이 단일 영상 분석)

```bash
# 컨테이너 안에서 (Git Bash 는 경로 변환 방지를 위해 MSYS_NO_PATHCONV=1)
docker compose exec detector python -m inference.cli /data/samples/test.mp4 --mode fast
docker compose exec detector python -m inference.cli "https://www.youtube.com/watch?v=..." --json
```
산출물은 `reports/cli_{timestamp}/` 에 생성된다. `--no-gradcam` 으로 점수만 빠르게 확인할 수 있다.

## 5-2. 모니터링 대시보드

- 웹: http://localhost:8000/monitor — GPU 사용률·메모리·온도·전력, 분석 서버 CPU/RAM, 큐·작업 수, 현재 작업 진행률 (2초 갱신, 최근 2분 그래프)
- API: detector `GET /metrics`, backend `GET /api/metrics`

## 5-3. 샘플 영상

`data/samples/` 에 SNS 에서 수집한 **실제 영상 9개**(가짜 5, 진짜 3, VFX 1)와 라벨 파일 labels.csv 가 있다 (학습 데이터 아님, 일반화 점검용. -1 은 집계 제외). 제3자 콘텐츠이므로 영상은 git 에 포함되지 않으며(data/ 제외) 공개 자료에 쓰지 않는다. 라벨과 평가 결과는 파일명을 샘플 A~I 로 바꿔 docs/metrics/ 에 두었다.

## 5-4. 문서

| 문서 | 내용 |
|---|---|
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | **하이브리드 탐지 시스템 동작 구조** — 파이프라인, 세 모델, 앙상블, 학습, API 계약, 보안 |


## 6. 실시간 탐지에 대하여

detector 의 `/ws/analyze` 는 수신 프레임을 **실제로 CNN + Frequency 모델에 통과**시켜 결과를 회신한다.
웹 UI 의 「실시간 분석」 버튼은 **의도적으로 비활성**이다 — 브라우저는 다른 앱의 화면을 지속 캡처할 수 없어
모바일 앱 전용 기능이기 때문이다. `/realtime` 페이지 하단의 개발자 도구로 서버 API 동작은 확인할 수 있다.

---

## 7. 환경 변수 (`.env`)

| 변수 | 설명 |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL. MariaDB: `mysql+pymysql://user:pw@host:port/cake?charset=utf8mb4` |
| `DETECTOR_URL` / `BACKEND_URL` | compose 내부 주소 (기본값 유지) |
| `PUBLIC_BASE_URL` | 브라우저가 접근하는 backend 주소 |
| `INTERNAL_API_TOKEN` | backend ↔ detector 내부 인증 (임의 32바이트) |
| `DEFAULT_MODE` | `high` / `fast` |
| `MAX_UPLOAD_MB`, `ALLOWED_EXTENSIONS` | 업로드 제한 |
| `CORS_ORIGINS` | 허용 오리진 (쉼표 구분) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` — 장시간 학습 중 CUDA 메모리 단편화 완화 (CUDA unknown error 2회 후 추가) |

---

## 8. 디렉터리

```
cake/
├─ docker-compose.yml           detector(GPU) + backend
├─ docker-compose.aihub.yml     WSL 전용 오버라이드: /aihub(4TB, ro) + 네임드 볼륨 cake_cache -> /data/cache
├─ detector/            분석 서버
│  ├─ api/              FastAPI (analyze · status · ws · metrics · health)
│  ├─ common/           face(YuNet, 640 축소 검출) · preprocess(공용) · video(추출/다운로드/코덱 변환/ffmpeg)
│  ├─ inference/        engine · gradcam · report · cli
│  ├─ models/           cnn_resnet · lstm_analyzer(ResNet18 인코더) · frequency_analyzer · registry
│  └─ training/         build_aihub_manifest · prepare_data(구간 샘플링) · dataset(GroupKFold) · augment · train(이어하기) · ensemble_opt(Grid+BO) · metrics · make_synthetic
│                       · make_sns_variants(H.264 변형본) · eval_holdout(압축 강건성) · eval_samples(SNS 일반화)
├─ backend/             웹·중계 서버
│  ├─ app/              main · db(SQLAlchemy) · config
│  └─ web/              templates(index·analysis·history·realtime·monitor) · static(css/js)
├─ docs/                SETUP · ARCHITECTURE   (발표 pdf 는 저장소 밖 — .gitignore 로 제외, 요청 시 제공)
├─ LICENSE              MIT
├─ backend/schema.sql   MariaDB 참고 스키마 (테이블은 자동 생성)
├─ checkpoints/         학습 가중치 high_*.pt, high_ensemble.json (git 제외)
├─ data/                samples/(SNS 실제 영상 9개 + labels.csv) · manifest.csv · cache_test/ (git 제외)
├─ runs/                학습 로그·지표·OOF·fold 체크포인트 (git 제외)
└─ scripts/             aihub_pipeline.sh(실데이터 학습 전체) · aihub_run_detached.sh(분리 실행) · smoke_test.ps1(합성 데이터 검증) · bench_gpu.py(GPU 처리량 진단)
```

---

## 8-1. 실측 성능 (high 모드, AI Hub 「딥페이크 변조 영상」)

학습 15,834 영상(train 13,194 + AI Hub Validation 홀드아웃 2,640), 인물(UUID) 단위 3-fold, 8 epoch.

| 모델 | K-fold OOF AUC / ACC | 홀드아웃 AUC / ACC / F1 |
|---|---|---|
| CNN (ResNet34) | 0.9975 / 97.8% | **0.9987 / 98.8% / 0.988** |
| LSTM (ResNet18 인코더 + BiLSTM) | 0.9971 / 97.9% | **0.9991 / 98.6% / 0.986** |
| Frequency (FFT + conv) | 0.8533 / 77.0% | 0.8732 / 78.9% / 0.769 |
| 앙상블 (cnn 0.35 · lstm 0.65 · fft 0.00, thr 0.41) | AUC 0.9977 / ACC 98.1% / F1 0.981 | — |

- 변조 방법별로는 **fsgan** 이 가장 어렵고(CNN recall 0.89, LSTM 0.83), Frequency 는 고비트레이트 재인코딩본(anti_fake)에 취약한다.
- Frequency 가중치 0 은 데이터 기반 최적화 결과이다. **팀 결정으로 `--min-weight 0.1` 을 적용해 세 모델을 모두 반영 중**: cnn 0.52 · lstm 0.38 · fft 0.10, thr 0.514 -> OOF AUC 0.9965 / ACC 98.1% / F1 0.981 (AUC −0.0012, ACC 동일). SNS 샘플 8개(가짜 5·진짜 3) 결과는 가짜 4/5 탐지·진짜 3/3 정답으로 변화 없음(`runs/samples_eval_v1_labeled.json`, 라벨 `data/samples/labels.csv`).
- **압축 강건성 (실측, `training.eval_holdout`)**: 같은 홀드아웃을 SNS 급 H.264 재압축(축소 0.5 ~ 0.85, CRF 23~34)하면 앙상블 정확도 98.8% -> **92.8%**, 그중 **real->fake 오탐이 0.2% -> 12.4%** 로 늘고 fake 놓침은 그대로이다 (CNN 단독은 88.7%, 오탐 20.7%). 원본(HEVC 고화질) vs 변조(H.264) 인코딩 편향을 일부 학습한 결과이며, 압축 증강 재학습(`make_sns_variants` + `--sns-prob`)이 그 대책이다. 근거: `runs/high/holdout_eval_v1_minw0.1.json`.
- 이 수치는 **AI Hub 데이터셋 분포 기준**이다. 분포가 다른 실제 SNS 영상(`data/samples/` 8개: fake 5, real 3. 라벨은 labels.csv, VFX 1개는 -1 로 집계 제외)에서는 **fake 5개 중 4개 탐지, real 3개는 정답** 이었다 — 생성기·필터·재압축이 달라 일반화가 약하다. 표본이 작고 구성(한국인 3·외국인 4·다수 인물 1)이 고르지 않아 비율이 아닌 사례로 본다. 개선 선택지(FF++/Celeb-DF 추가, SNS 압축 증강, 임계값 조정, Frequency 재설계)가 있다. **공개 문서에는 두 수치를 함께 적는 것이 정직한다.**

## 9. 알려진 한계

- 정확도 등 성능 수치는 **학습 데이터에 따라** 결정된다. `runs/{mode}/*_summary.json` 의 OOF·홀드아웃 지표를 문서에 인용한다 (§8-1).
- 얼굴이 매우 작거나(`small_face`) 가려지거나 자주 벗어나는(`low_face_ratio`/`no_face`) 영상은 `quality_warnings` 와 함께 `face_warning` 이 표시되며 신뢰성이 낮다. 얼굴 없는 프레임은 점수에서 제외된다.
- 학습 데이터(AI Hub 6개 변조 방법, 스튜디오 촬영)와 다른 생성기·VFX·완전 생성 영상은 탐지 범위 밖이다. 실제 SNS 딥페이크 5개 중 4개 탐지(§8-1).
- GPU 1장 기준 동시 분석 1건(`max_concurrent_analyses`). 추가 요청은 큐에서 대기한다.
- 본 결과는 **보조 정보**이며 법적 판단의 근거가 될 수 없다.
