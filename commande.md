```powershell
$RunName = "p3_p4_locked_c2_w6_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$HostRunDir = Join-Path $PWD "ai_trading\info_retour\$RunName"
$ContainerRunDir = "/app/ai_trading/info_retour/$RunName"
$P2ContainerFile = "/app/ai_trading/info_retour/data/sentiment/observations.parquet"

New-Item -ItemType Directory -Force $HostRunDir | Out-Null

docker compose -f docker\docker-compose.yml exec -T ai_api python3 -m ai_trading.scripts.export_fear_greed_p2_observations `
  --output $P2ContainerFile --assets BTC/USDT ETH/USDT --timeframe 1d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose -f docker\docker-compose.yml exec -T ai_api timeout --signal=INT --kill-after=30s 10800s python3 -m ai_trading.scripts.run_real_multi_asset_walk_forward `
  --assets BTC/USDT ETH/USDT --timeframe 1d --days 1800 --train-candles 400 --validation-candles 100 --test-candles 100 --step-candles 100 `
  --window-start 6 --max-windows 1 --validation-seeds 73 211 997 `
  --episodes 20 --max-training-steps 14000 --max-optimization-steps 1000 `
  --batch-size 64 --buffer-size 10000 --sequence-length 8 --window-size 50 `
  --min-closed-trades 20 --allow-short --evaluation-mode validation `
  --candidate-config /app/ai_trading/configs/p3_locked_candidate.json `
  --candidate-id p3-bf72563778749e59 `
  --enable-p4-features --p4-sentiment-observations $P2ContainerFile `
  --p4-horizons 1d --p4-min-train-candles 120 --p4-retrain-interval 100 `
  --output-dir $ContainerRunDir 2>&1 | Tee-Object (Join-Path $HostRunDir "command.log")
```
