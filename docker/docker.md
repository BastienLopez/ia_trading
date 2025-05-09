# Guide d'utilisation Docker pour AI Trading

Ce document explique comment utiliser Docker pour exécuter les différents composants du projet AI Trading.

## Table des matières

1. [Prérequis](#prérequis)
2. [Structure des fichiers Docker](#structure-des-fichiers-docker)
3. [Services disponibles](#services-disponibles)
4. [Commandes principales](#commandes-principales)
5. [URLs et ports](#urls-et-ports)
6. [Configuration](#configuration)
7. [Résolution des problèmes courants](#résolution-des-problèmes-courants)

## Prérequis

- [Docker](https://www.docker.com/get-started) (version 20.10 ou supérieure)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 2.0 ou supérieure)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (pour le support GPU)

## Structure des fichiers Docker

Le projet utilise les fichiers Docker suivants, situés dans le dossier `docker/` :

- `Dockerfile` : Image principale pour AI Trading, compatible avec le dashboard et l'API web
- `Dockerfile.test` : Image spécialisée pour exécuter les tests
- `docker-compose.yml` : Configuration des services et de leur orchestration
- `.dockerignore` : Fichiers exclus du contexte Docker

## Services disponibles

Le projet comprend plusieurs services configurés dans le fichier `docker-compose.yml` :

| Service | Description |
|---------|-------------|
| dashboard | Interface graphique pour l'analyse de données et la visualisation (port 8050) |
| web_app | Interface web principale pour les utilisateurs finaux (port 5000) |
| postgres | Base de données PostgreSQL pour le stockage persistant |
| redis | Cache et file d'attente pour les opérations |

## Commandes principales

### Démarrer tous les services

```bash
cd docker
docker compose up
```

### Démarrer un service spécifique

```bash
cd docker
docker compose up dashboard  # Pour démarrer uniquement le dashboard
docker compose up web_app    # Pour démarrer uniquement l'interface web
```

### Reconstruire les images et démarrer

```bash
cd docker
docker compose up --build
```

### Exécuter en arrière-plan

```bash
cd docker
docker compose up -d
```

### Arrêter les services

```bash
cd docker
docker compose down
```

### Exécuter des tests

```bash
# Utiliser le Dockerfile.test spécifique
cd ..  # À la racine du projet
docker build -t ai-trading-test:v1 -f docker/Dockerfile.test .
docker run --rm --gpus all ai-trading-test:v1

# Pour exécuter tous les tests (incluant les tests lents)
docker run --rm --gpus all -e RUN_SLOW_TESTS=1 ai-trading-test:v1
```

### Exécuter une commande dans un conteneur en cours d'exécution

```bash
docker exec -it ai_trading_dashboard python -c "import torch; print(torch.cuda.is_available())"
```

## URLs et ports

Le projet expose plusieurs ports pour différents services :

| Port | URL | Description |
|------|-----|-------------|
| 5000 | http://localhost:5000 | Interface web principale (Flask) |
| 8050 | http://localhost:8050 | Dashboard d'analyse et visualisations (Dash) |
| 8000 | http://localhost:8000 | API web (actuellement redirigée vers le dashboard) |
| 8501 | http://localhost:8501 | Interface Streamlit pour visualisations alternatives (non active par défaut) |
| 5432 | localhost:5432 | Base de données PostgreSQL |
| 6379 | localhost:6379 | Serveur Redis |

### Endpoints principaux de l'interface web (port 5000)

- `http://localhost:5000/` - Page d'accueil de l'application
- `http://localhost:5000/dashboard` - Dashboard principal
- `http://localhost:5000/backtest` - Page pour exécuter des backtests
- `http://localhost:5000/tradingview` - Visualisation des graphiques
- `http://localhost:5000/technology` - Informations sur la technologie utilisée
- `http://localhost:5000/health` - Endpoint pour vérifier l'état de santé du service (utilisé par Docker)
- `http://localhost:5000/status` - Page web montrant l'état global du système et de ses dépendances

### Endpoints du Dashboard (port 8050)

- `http://localhost:8050/` - Interface principale du dashboard
- `http://localhost:8050/health` - Endpoint de vérification d'état (healthcheck)

Onglets disponibles dans le dashboard:
- Vue d'ensemble - Vue générale du portefeuille et performances
- Performances - Analyse détaillée des performances du portefeuille 
- Transactions - Historique et détails des transactions
- Allocation - Analyse de l'allocation d'actifs dans le portefeuille
- Risque - Métriques et visualisations des risques (VaR, drawdowns)
- Facteurs - Analyse des facteurs de risque et leur contribution
- Visualisations 3D - Visualisations 3D des données et indicateurs techniques
- Analyse Post-Mortem - Analyse détaillée des trades passés (gains/pertes)

> **Note**: La navigation entre les onglets se fait directement dans l'interface du dashboard. Les URL spécifiques sont gérées par le routing interne de Dash.

## Configuration

### Variables d'environnement

Les services Docker peuvent être configurés en modifiant le fichier `.env` à la racine du projet ou en passant des variables d'environnement via la ligne de commande.

Principales variables :

```bash
# Base de données
POSTGRES_DB=ai_trading
POSTGRES_USER=ai_trading
POSTGRES_PASSWORD=changeme

# Dashboard et API
HOST=0.0.0.0
PORT=8050
DASH_DEBUG=true

# Interface Web Flask
FLASK_ENV=development
FLASK_DEBUG=1
API_URL=http://dashboard:8000  # URL interne pour la communication entre services

# GPU et performances
NVIDIA_VISIBLE_DEVICES=all  # ou spécifier des GPU spécifiques (ex: "0,1")
OMP_NUM_THREADS=8  # Nombre de threads OpenMP
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Configuration allocation CUDA
```

### Volumes persistants

Les données sont stockées dans les volumes Docker suivants :

- `postgres_data` : Données PostgreSQL
- `redis_data` : Données Redis

Les données spécifiques à l'application sont montées depuis les dossiers du projet :

- `../data:/app/data` : Données de l'application
- `../logs:/app/logs` : Logs de l'application
- `../models:/app/models` : Modèles entraînés
- `../web_app/data:/app/web_app/data` : Données spécifiques à l'interface web

## Résolution des problèmes courants

### Support GPU non disponible

Si vous rencontrez des problèmes avec le GPU :

1. Vérifiez que le NVIDIA Container Toolkit est installé :
   ```bash
   nvidia-smi
   ```

2. Vérifiez que Docker peut voir les GPU :
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi
   ```

3. Dans le conteneur, vérifiez que PyTorch voit les GPU :
   ```bash
   docker exec -it ai_trading_dashboard python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
   ```

### Problèmes de port

Si un service ne répond pas sur le port attendu :

1. Vérifiez qu'aucun autre service n'utilise déjà le port :
   ```bash
   netstat -tulpn | grep <PORT>  # Sur Linux
   netstat -ano | findstr "<PORT>"  # Sur Windows
   ```

2. Modifiez le port dans `docker-compose.yml` si nécessaire :
   ```yaml
   ports:
     - "8051:8050"  # Exposition du port 8050 du conteneur sur le port 8051 local
   ```

3. Vérifiez que l'application écoute sur toutes les interfaces (`0.0.0.0`) et pas seulement localhost (`127.0.0.1`)

### Problèmes de mémoire

Si Docker manque de mémoire lors de la construction ou de l'exécution :

1. Augmentez les ressources allouées à Docker dans les paramètres Docker Desktop
2. Ajoutez des limites de mémoire dans le `docker-compose.yml` :
   ```yaml
   deploy:
     resources:
       limits:
         memory: 8G
   ```

### Vérifier l'état des services 

Pour vérifier que tous les services fonctionnent correctement, vous pouvez :

1. Utiliser la commande Docker :
   ```bash
   docker ps  # Liste tous les conteneurs en cours d'exécution
   docker logs ai_trading_dashboard  # Affiche les logs du dashboard
   docker logs ai_trading_web_app  # Affiche les logs de l'interface web
   ```

2. Vérifier les endpoints de santé :
   ```bash
   curl http://localhost:5000/health  # Vérification de l'interface web
   curl http://localhost:8050/health  # Vérification du dashboard
   ```

3. Accéder à la page d'état du système :
   ```
   http://localhost:5000/status
   ```
   Cette page affiche l'état global du système, y compris la connexion à l'API et à la base de données. 