#!/usr/bin/env bash
# aihub_pipeline.sh 를 세션과 분리(nohup)해 실행하고 runs/aihub_pipeline.log 에 기록.
# 종료 시 마지막 줄에  ### PIPELINE DONE  또는  ### PIPELINE FAILED (exit N)  을 남김 -> 모니터가 감지.
#   cd <repo>/cake && bash scripts/aihub_run_detached.sh          (환경변수는 aihub_pipeline.sh 와 동일)
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
LOG=runs/aihub_pipeline.log
{
  echo "### PIPELINE START $(date -Is)  MODE=${MODE:-high} PER_CLASS=${PER_CLASS:-3000} VAL_PER_CLASS=${VAL_PER_CLASS:-600} STEP=${STEP:-manifest} WORKERS=${WORKERS:-6}"
} > "$LOG"
# setsid -f: 새 세션으로 완전히 분리 -> wsl.exe / 터미널이 종료되어도 계속 실행 (nohup 만으로는 wsl -e 종료 시 함께 정리됨)
setsid -f bash -c "bash scripts/aihub_pipeline.sh; rc=\$?; if [ \$rc -eq 0 ]; then echo \"### PIPELINE DONE \$(date -Is)\"; else echo \"### PIPELINE FAILED (exit \$rc) \$(date -Is)\"; fi" >> "$LOG" 2>&1 < /dev/null
sleep 2
echo "실행 중 PID: $(pgrep -f 'scripts/aihub_pipeline.sh' | head -1)   로그: $LOG"
