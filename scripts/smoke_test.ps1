# 합성 데이터로 전체 파이프라인 동작 검증 (성능 평가 아님)
#   .\scripts\smoke_test.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== 합성 데이터 생성 ==" -ForegroundColor Cyan
docker compose run --rm detector python -m training.make_synthetic --n 30 --frames 60
docker compose run --rm detector python -m training.prepare_data --frames 32 --size 224 --min-face-ratio 0 --workers 4
docker compose run --rm detector python -m training.train --mode fast --model all --folds 3 --epochs 3 --batch-size 32 --workers 2
docker compose run --rm detector python -m training.ensemble_opt --mode fast
Write-Host "== DEFAULT_MODE=fast 로 detector 기동 후 분석 1건 ==" -ForegroundColor Cyan
$env:DEFAULT_MODE = "fast"
docker compose up -d
Start-Sleep -Seconds 15
Invoke-RestMethod http://localhost:3001/health | ConvertTo-Json
Write-Host "웹 UI: http://localhost:8000  — data\fake\synth_fake_000.mp4 를 업로드해 확인하십시오." -ForegroundColor Green
