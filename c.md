Oui. Les 6 commandes ci-dessous sont propres : sorties isolées, GPU Docker, aucun test hors-échantillon consommé pendant la sélection.

Le filtre rapide de 15 min est un diagnostic des 10 candidats. Les cinq runs 3 h sont des validations multi-seed sur cinq périodes différentes, chacune sur les 10 candidats. Ne les lance pas en parallèle.

Le filtre rapide ne permet pas de sélectionner ou verrouiller P3 : une seed et trois épisodes ne sont pas une preuve de performance. Les cinq validations restent toutes en `--evaluation-mode validation`, donc aucun test hors-échantillon n'est consommé.

### 1. Filtre rapide 15 min — 10 candidats, une seed

```powershell
$RunName = "p3_screen_15m_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$HostRunDir = Join-Path $PWD "ai_trading\info_retour\$RunName"
$ContainerRunDir = "/app/ai_trading/info_retour/$RunName"

New-Item -ItemType Directory -Force $HostRunDir | Out-Null

docker compose -f docker\docker-compose.yml exec -T ai_api timeout --signal=INT --kill-after=30s 900s python3 -m ai_trading.scripts.run_real_multi_asset_walk_forward `
  --days 1800 --train-candles 400 --validation-candles 100 --test-candles 100 --step-candles 100 `
  --window-start 0 --max-windows 1 --validation-seeds 42 `
  --episodes 3 --max-training-steps 1200 --max-optimization-steps 120 `
  --batch-size 64 --buffer-size 5000 --sequence-length 8 --window-size 50 `
  --min-closed-trades 20 --allow-short --evaluation-mode validation `
  --output-dir $ContainerRunDir 2>&1 | Tee-Object (Join-Path $HostRunDir "command.log")
```

### 2. Validation longue — fenêtre 1

```powershell
$RunName = "p3_validation_w01_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$HostRunDir = Join-Path $PWD "ai_trading\info_retour\$RunName"
$ContainerRunDir = "/app/ai_trading/info_retour/$RunName"

New-Item -ItemType Directory -Force $HostRunDir | Out-Null

docker compose -f docker\docker-compose.yml exec -T ai_api timeout --signal=INT --kill-after=30s 10800s python3 -m ai_trading.scripts.run_real_multi_asset_walk_forward `
  --days 1800 --train-candles 400 --validation-candles 100 --test-candles 100 --step-candles 100 `
  --window-start 0 --max-windows 1 --validation-seeds 42 314 2024 `
  --episodes 20 --max-training-steps 14000 --max-optimization-steps 1000 `
  --batch-size 64 --buffer-size 10000 --sequence-length 8 --window-size 50 `
  --min-closed-trades 20 --allow-short --evaluation-mode validation `
  --output-dir $ContainerRunDir 2>&1 | Tee-Object (Join-Path $HostRunDir "command.log")
```

### 3. Validation longue — fenêtre 2

Même commande, seul `--window-start` change :

```powershell
$RunName = "p3_validation_w02_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$HostRunDir = Join-Path $PWD "ai_trading\info_retour\$RunName"
$ContainerRunDir = "/app/ai_trading/info_retour/$RunName"

New-Item -ItemType Directory -Force $HostRunDir | Out-Null

docker compose -f docker\docker-compose.yml exec -T ai_api timeout --signal=INT --kill-after=30s 10800s python3 -m ai_trading.scripts.run_real_multi_asset_walk_forward `
  --days 1800 --train-candles 400 --validation-candles 100 --test-candles 100 --step-candles 100 `
  --window-start 1 --max-windows 1 --validation-seeds 42 314 2024 `
  --episodes 20 --max-training-steps 14000 --max-optimization-steps 1000 `
  --batch-size 64 --buffer-size 10000 --sequence-length 8 --window-size 50 `
  --min-closed-trades 20 --allow-short --evaluation-mode validation `
  --output-dir $ContainerRunDir 2>&1 | Tee-Object (Join-Path $HostRunDir "command.log")
```

### 4. Validation longue — fenêtre 3

```powershell
$RunName = "p3_validation_w03_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$HostRunDir = Join-Path $PWD "ai_trading\info_retour\$RunName"
$ContainerRunDir = "/app/ai_trading/info_retour/$RunName"

New-Item -ItemType Directory -Force $HostRunDir | Out-Null

docker compose -f docker\docker-compose.yml exec -T ai_api timeout --signal=INT --kill-after=30s 10800s python3 -m ai_trading.scripts.run_real_multi_asset_walk_forward `
  --days 1800 --train-candles 400 --validation-candles 100 --test-candles 100 --step-candles 100 `
  --window-start 2 --max-windows 1 --validation-seeds 42 314 2024 `
  --episodes 20 --max-training-steps 14000 --max-optimization-steps 1000 `
  --batch-size 64 --buffer-size 10000 --sequence-length 8 --window-size 50 `
  --min-closed-trades 20 --allow-short --evaluation-mode validation `
  --output-dir $ContainerRunDir 2>&1 | Tee-Object (Join-Path $HostRunDir "command.log")
```

### 5. Validation longue — fenêtre 4

```powershell
$RunName = "p3_validation_w04_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$HostRunDir = Join-Path $PWD "ai_trading\info_retour\$RunName"
$ContainerRunDir = "/app/ai_trading/info_retour/$RunName"

New-Item -ItemType Directory -Force $HostRunDir | Out-Null

docker compose -f docker\docker-compose.yml exec -T ai_api timeout --signal=INT --kill-after=30s 10800s python3 -m ai_trading.scripts.run_real_multi_asset_walk_forward `
  --days 1800 --train-candles 400 --validation-candles 100 --test-candles 100 --step-candles 100 `
  --window-start 3 --max-windows 1 --validation-seeds 42 314 2024 `
  --episodes 20 --max-training-steps 14000 --max-optimization-steps 1000 `
  --batch-size 64 --buffer-size 10000 --sequence-length 8 --window-size 50 `
  --min-closed-trades 20 --allow-short --evaluation-mode validation `
  --output-dir $ContainerRunDir 2>&1 | Tee-Object (Join-Path $HostRunDir "command.log")
```

### 6. Validation longue — fenêtre 5

```powershell
$RunName = "p3_validation_w05_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$HostRunDir = Join-Path $PWD "ai_trading\info_retour\$RunName"
$ContainerRunDir = "/app/ai_trading/info_retour/$RunName"

New-Item -ItemType Directory -Force $HostRunDir | Out-Null

docker compose -f docker\docker-compose.yml exec -T ai_api timeout --signal=INT --kill-after=30s 10800s python3 -m ai_trading.scripts.run_real_multi_asset_walk_forward `
  --days 1800 --train-candles 400 --validation-candles 100 --test-candles 100 --step-candles 100 `
  --window-start 4 --max-windows 1 --validation-seeds 42 314 2024 `
  --episodes 20 --max-training-steps 14000 --max-optimization-steps 1000 `
  --batch-size 64 --buffer-size 10000 --sequence-length 8 --window-size 50 `
  --min-closed-trades 20 --allow-short --evaluation-mode validation `
  --output-dir $ContainerRunDir 2>&1 | Tee-Object (Join-Path $HostRunDir "command.log")
```

Chaque dossier créé contient `walk_forward_summary.json`, les graphes, `metrics.json`, `equity_curve.csv`, `actions.csv`, `trades.csv`, `closed_trades.csv` et `command.log`.

Après les cinq validations, envoie-moi les cinq dossiers `p3_validation_w*`. Compare les 10 candidats sur les cinq fenêtres et retiens uniquement celui qui passe toutes les gardes : rendement net, benchmark, profit factor, drawdown, turnover, diversité et stabilité des trois seeds. Le vrai test hors-échantillon final ne sera lancé qu’après cette sélection.
