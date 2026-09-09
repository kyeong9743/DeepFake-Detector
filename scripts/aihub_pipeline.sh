#!/usr/bin/env bash
# AI Hub 「딥페이크 변조 영상」 -> CAKE 학습 전체 파이프라인.  **WSL(Ubuntu) 에서 실행**
#
#   cd <repo>/cake            (WSL 안에서 저장소 경로로 이동)
#   bash scripts/aihub_pipeline.sh                 # 기본: high, 클래스당 3000 + 검증 600
#   PER_CLASS=6000 MODE=high bash scripts/aihub_pipeline.sh
#   STEP=train bash scripts/aihub_pipeline.sh     # 특정 단계부터 (manifest|prepare|sns|train|ensemble|restart)
#   SNS_VARIANTS=2 SNS_PROB=0.5 MIN_WEIGHT=0.1 …    # SNS 압축 변형본 수 / 학습 시 변형본 사용 확률 / 앙상블 최소 가중치
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${MODE:-high}"
PER_CLASS="${PER_CLASS:-3000}"
VAL_PER_CLASS="${VAL_PER_CLASS:-600}"
FRAMES="${FRAMES:-48}"            # CLIPS=0 일 때 균등 샘플 수
CLIPS="${CLIPS:-3}"               # >0: 구간별 연속 프레임 (기본 3구간 × 16프레임 = 48). HEVC 장편에서 수 배 빠름
CLIP_LEN="${CLIP_LEN:-16}"        # LSTM window 와 동일
CLIP_STRIDE="${CLIP_STRIDE:-2}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-6}"
STEP="${STEP:-manifest}"          # 시작 단계
END_STEP="${END_STEP:-restart}"   # 종료 단계 (예: END_STEP=prepare -> 전처리까지만)
INCLUDE_ANTI="${INCLUDE_ANTI:-1}"
SNS_VARIANTS="${SNS_VARIANTS:-2}"   # 0 이면 sns 단계 생략
SNS_PROB="${SNS_PROB:-0.5}"         # train --sns-prob
MIN_WEIGHT="${MIN_WEIGHT:-0.1}"     # ensemble_opt --min-weight (팀 결정 2026-09-07: 세 모델 모두 반영)
SIZE=$([ "$MODE" = high ] && echo 384 || echo 224)
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.aihub.yml"
RUN="$COMPOSE run --rm --no-deps -T detector"

step_ge() { # 단계 $1 이 [STEP, END_STEP] 범위 안인가 (순서: manifest prepare train ensemble restart)
  local order="manifest prepare sns train ensemble restart" a b e
  a=$(echo "$order" | tr ' ' '\n' | grep -nx "$STEP"     | cut -d: -f1)
  e=$(echo "$order" | tr ' ' '\n' | grep -nx "$END_STEP" | cut -d: -f1)
  b=$(echo "$order" | tr ' ' '\n' | grep -nx "$1"        | cut -d: -f1)
  [ "$b" -ge "$a" ] && [ "$b" -le "$e" ]
}

# ---- AI Hub 데이터 위치: 기본값 없음. .env 또는 환경변수로 반드시 지정 ----
#   AIHUB_DATA_DIR   003.딥페이크/ 가 바로 아래에 있는 디렉터리 (필수)
#   AIHUB_MOUNT      (선택) 그 디렉터리가 있는 마운트 지점. 지정하면 마운트 여부를 확인
#   AIHUB_DISK_UUID  (선택) AIHUB_MOUNT 가 안 되어 있을 때 sudo mount UUID=… 로 마운트 (blkid 로 확인)
if [ -f .env ]; then set -a; . ./.env; set +a; fi        # .env 의 AIHUB_* 를 읽음 (환경변수가 있으면 그것이 우선)
: "${AIHUB_DATA_DIR:?AIHUB_DATA_DIR 가 설정되지 않았습니다. .env 에 AIHUB_DATA_DIR=<003.딥페이크 가 있는 디렉터리> 를 추가하십시오 (.env.example 참고)}"

echo "== 0. AI Hub 데이터 확인 ($AIHUB_DATA_DIR) =="
if [ -n "${AIHUB_MOUNT:-}" ] && ! mountpoint -q "$AIHUB_MOUNT"; then
  if [ -n "${AIHUB_DISK_UUID:-}" ]; then
    sudo mount UUID="$AIHUB_DISK_UUID" "$AIHUB_MOUNT"
  else
    echo "오류: $AIHUB_MOUNT 가 마운트되어 있지 않습니다. 디스크를 마운트하거나 AIHUB_DISK_UUID 를 지정하십시오." >&2; exit 1
  fi
fi
if [ ! -d "$AIHUB_DATA_DIR/003.딥페이크" ]; then
  echo "오류: $AIHUB_DATA_DIR/003.딥페이크 가 없습니다. AIHUB_DATA_DIR 를 확인하십시오." >&2; exit 1
fi
df -h "$AIHUB_DATA_DIR" | tail -1
export AIHUB_DATA_DIR

if step_ge manifest; then
  echo "== 1. manifest 생성 (클래스당 $PER_CLASS + 검증 $VAL_PER_CLASS) =="
  ANTI=$([ "$INCLUDE_ANTI" = 1 ] && echo --include-anti || true)
  $RUN python -m training.build_aihub_manifest --aihub /aihub --per-class "$PER_CLASS" \
       --val-per-class "$VAL_PER_CLASS" $ANTI --out /data/manifest.csv
fi

if step_ge prepare; then
  echo "== 2. 전처리: 얼굴 크롭 프레임 캐시 (frames=$FRAMES size=$SIZE) =="
  if ls data/real/synth_*.mp4 >/dev/null 2>&1; then
    echo "   합성 스모크 데이터 제거"; rm -f data/real/synth_*.mp4 data/fake/synth_*.mp4
  fi
  # 캐시는 Docker 네임드 볼륨 cake_cache (컨테이너 /data/cache). 컨테이너(root)가 만든 파일이므로 컨테이너에서 정리
  $RUN bash -c 'mkdir -p /data/cache && find /data/cache -mindepth 1 -maxdepth 1 -exec rm -rf {} +'
  $RUN python -m training.prepare_data --manifest /data/manifest.csv --frames "$FRAMES" --size "$SIZE" --workers "$WORKERS" --min-face-ratio 0.3 \
       --clips "$CLIPS" --clip-len "$CLIP_LEN" --clip-stride "$CLIP_STRIDE"
fi

if step_ge sns && [ "$SNS_VARIANTS" -gt 0 ]; then
  echo "== 2b. SNS 압축 변형본 생성 (H.264 재인코딩, 변형 $SNS_VARIANTS개) =="
  $RUN python -m training.make_sns_variants --variants "$SNS_VARIANTS" --workers "$WORKERS"
fi

MODELS="${MODELS:-all}" # 학습할 모델: all 또는 공백 구분 (예: "lstm frequency" — CNN 이 이미 끝났을 때)
if step_ge train; then
  echo "== 3. K-fold 학습 (mode=$MODE folds=$FOLDS epochs=$EPOCHS models=$MODELS sns_prob=$SNS_PROB) =="
  TRAIN_ARGS="${TRAIN_ARGS:-}" # 추가 인자 (예: TRAIN_ARGS="--skip-folds" — fold 완료 후 최종 단계 실패 시 재개)
  for m in $MODELS; do
    $RUN python -m training.train --mode "$MODE" --model "$m" --folds "$FOLDS" --epochs "$EPOCHS" --batch-size "$BATCH" --workers "$WORKERS" --sns-prob "$SNS_PROB" $TRAIN_ARGS
  done
fi

if step_ge ensemble; then
  echo "== 4. 앙상블 가중치 최적화 =="
  $RUN python -m training.ensemble_opt --mode "$MODE" --min-weight "$MIN_WEIGHT"
fi

if step_ge restart; then
  echo "== 5. DEFAULT_MODE=$MODE 로 detector 재시작 =="
  sed -i "s/^DEFAULT_MODE=.*/DEFAULT_MODE=$MODE/" .env
  docker compose up -d detector
  sleep 10; curl -s http://localhost:3001/health; echo
fi
echo "완료. 지표: runs/$MODE/*_summary.json  (oof / holdout / by_method)"
