# Pipeline CI/CD & Intégration Automatisée

## 🎯 Objectif

L'objectif de cette phase est de garantir que **tous** les modules (data, RL, LLM, API…) sont construits, testés et validés en continu avant tout déploiement. Cette pipeline assure l'intégration harmonieuse de tous les composants du système AI Trading.

## 🏗️ Architecture de la pipeline

### 1. Orchestration des flux de données

#### Solution choisie: Airflow
```python
# dags/ai_trading_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ai_trading',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ai_trading_pipeline',
    default_args=default_args,
    description='Pipeline complète AI Trading',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Tâches principales
data_collection = PythonOperator(
    task_id='data_collection',
    python_callable=collect_data,
    dag=dag,
)

sentiment_analysis = PythonOperator(
    task_id='sentiment_analysis',
    python_callable=analyze_sentiment,
    dag=dag,
)

market_prediction = PythonOperator(
    task_id='market_prediction',
    python_callable=predict_market,
    dag=dag,
)

signal_generation = PythonOperator(
    task_id='signal_generation',
    python_callable=generate_signals,
    dag=dag,
)

rl_training = PythonOperator(
    task_id='rl_training',
    python_callable=train_rl_agent,
    dag=dag,
)

# Définition des dépendances
data_collection >> [sentiment_analysis, market_prediction]
sentiment_analysis >> market_prediction
market_prediction >> signal_generation
signal_generation >> rl_training
```

#### Alternative légère: Script Bash
```bash
#!/bin/bash
# run_pipeline.sh

set -e  # Arrêt en cas d'erreur

echo "🔄 Démarrage de la pipeline AI Trading"

# 1. Collecte des données
echo "📊 Collecte des données..."
python -m ai_trading.utils.enhanced_data_collector || { echo "❌ Échec de la collecte"; exit 1; }

# 2. Analyse de sentiment
echo "🔍 Analyse de sentiment..."
python -m ai_trading.llm.sentiment_analysis.run || { echo "❌ Échec de l'analyse"; exit 1; }

# 3. Prédictions de marché
echo "🔮 Prédictions de marché..."
python -m ai_trading.llm.predictions.run || { echo "❌ Échec des prédictions"; exit 1; }

# 4. Génération de signaux
echo "📈 Génération de signaux..."
python -m ai_trading.ml.trading_signals.run || { echo "❌ Échec des signaux"; exit 1; }

# 5. Entraînement RL
echo "🤖 Entraînement RL..."
python -m ai_trading.rl.train || { echo "❌ Échec de l'entraînement"; exit 1; }

echo "✅ Pipeline terminée avec succès"
```

### 2. Système de file d'attente robuste

#### Configuration Docker Compose pour RabbitMQ
```yaml
# docker-compose.yml
version: '3'

services:
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"   # AMQP protocol
      - "15672:15672" # Management UI
    environment:
      - RABBITMQ_DEFAULT_USER=ai_trading
      - RABBITMQ_DEFAULT_PASS=secure_password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

volumes:
  rabbitmq_data:
```

#### Intégration Python avec Pika
```python
# ai_trading/utils/message_queue.py
import pika
import json
import logging
from typing import Dict, Any, Callable

class MessageQueue:
    """Gestionnaire de file d'attente pour la communication entre modules"""
    
    def __init__(self, host='localhost', port=5672, user='ai_trading', password='secure_password'):
        """Initialise la connexion à RabbitMQ"""
        credentials = pika.PlainCredentials(user, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port, credentials=credentials)
        )
        self.channel = self.connection.channel()
        
        # Déclarer les queues principales
        self.queues = [
            'data_ready', 'sentiment_ready', 'predictions_ready', 
            'signals_ready', 'rl_ready', 'api_ready'
        ]
        
        for queue in self.queues:
            self.channel.queue_declare(queue=queue, durable=True)
            
        logging.info("MessageQueue initialisé avec succès")
    
    def publish(self, queue: str, message: Dict[str, Any]) -> None:
        """Publie un message dans une queue spécifique"""
        if queue not in self.queues:
            logging.warning(f"Queue inconnue: {queue}, création automatique")
            self.channel.queue_declare(queue=queue, durable=True)
            self.queues.append(queue)
            
        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # message persistant
            )
        )
        logging.info(f"Message publié dans {queue}")
    
    def consume(self, queue: str, callback: Callable) -> None:
        """Configure un consommateur pour une queue spécifique"""
        if queue not in self.queues:
            logging.warning(f"Queue inconnue: {queue}, création automatique")
            self.channel.queue_declare(queue=queue, durable=True)
            self.queues.append(queue)
            
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=callback,
            auto_ack=False
        )
        logging.info(f"Consommateur configuré pour {queue}")
        
    def start_consuming(self) -> None:
        """Démarre la consommation des messages"""
        logging.info("Démarrage de la consommation des messages")
        self.channel.start_consuming()
        
    def close(self) -> None:
        """Ferme la connexion"""
        self.connection.close()
        logging.info("Connexion fermée")
```

### 3. Gestion des dépendances & retries intelligents

```python
# ai_trading/utils/retry_manager.py
import time
import logging
import functools
from typing import Callable, Any, Optional

class RetryManager:
    """Gestionnaire de retries avec backoff exponentiel"""
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        """
        Initialise le gestionnaire de retries
        
        Args:
            max_retries: Nombre maximum de tentatives
            initial_delay: Délai initial en secondes
            max_delay: Délai maximum en secondes
            backoff_factor: Facteur de multiplication pour le backoff exponentiel
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
    def with_retry(self, func: Callable) -> Callable:
        """
        Décorateur pour ajouter des retries à une fonction
        
        Args:
            func: Fonction à décorer
            
        Returns:
            Fonction décorée avec retries
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            delay = self.initial_delay
            
            while retries <= self.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > self.max_retries:
                        logging.error(f"Échec après {self.max_retries} tentatives: {str(e)}")
                        raise
                    
                    logging.warning(f"Tentative {retries}/{self.max_retries} échouée: {str(e)}")
                    logging.info(f"Nouvelle tentative dans {delay:.2f} secondes")
                    
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
        
        return wrapper
```

### 4. Mécanismes de reprise sur erreur & points de contrôle

```python
# ai_trading/utils/checkpoint_manager.py
import os
import json
import pickle
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class CheckpointManager:
    """Gestionnaire de points de contrôle pour la reprise sur erreur"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialise le gestionnaire de points de contrôle
        
        Args:
            checkpoint_dir: Répertoire de stockage des points de contrôle
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"CheckpointManager initialisé dans {checkpoint_dir}")
        
    def save_checkpoint(self, module_name: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Sauvegarde un point de contrôle
        
        Args:
            module_name: Nom du module (ex: 'data_collector', 'sentiment_analysis')
            data: Données à sauvegarder (doit être sérialisable)
            metadata: Métadonnées additionnelles
            
        Returns:
            Chemin du fichier de point de contrôle
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{module_name}_{timestamp}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.pkl")
        
        # Métadonnées par défaut
        meta = {
            "module": module_name,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat()
        }
        
        # Ajouter les métadonnées personnalisées
        if metadata:
            meta.update(metadata)
            
        # Sauvegarder les métadonnées
        meta_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
            
        # Sauvegarder les données
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
            
        logging.info(f"Point de contrôle sauvegardé: {checkpoint_path}")
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_id: str) -> Any:
        """
        Charge un point de contrôle
        
        Args:
            checkpoint_id: Identifiant du point de contrôle (ex: 'data_collector_20230101_120000')
            
        Returns:
            Données du point de contrôle
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.pkl")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Point de contrôle non trouvé: {checkpoint_path}")
            
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            
        logging.info(f"Point de contrôle chargé: {checkpoint_path}")
        return data
        
    def get_latest_checkpoint(self, module_name: str) -> Optional[str]:
        """
        Récupère le dernier point de contrôle d'un module
        
        Args:
            module_name: Nom du module
            
        Returns:
            Identifiant du dernier point de contrôle ou None si aucun
        """
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith(".json") and filename.startswith(f"{module_name}_"):
                checkpoint_id = filename[:-5]  # Enlever l'extension .json
                meta_path = os.path.join(self.checkpoint_dir, filename)
                
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    
                checkpoints.append((checkpoint_id, metadata["created_at"]))
                
        if not checkpoints:
            return None
            
        # Trier par date de création (plus récent d'abord)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
```

### 5. Architecture modulaire & rolling updates

#### Structure Docker Compose pour déploiement modulaire
```yaml
# docker-compose.yml
version: '3'

services:
  rabbitmq:
    image: rabbitmq:3-management
    # Configuration comme précédemment...
    
  data_collector:
    build:
      context: .
      dockerfile: dockerfiles/data_collector.Dockerfile
    depends_on:
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
    volumes:
      - data_volume:/app/data
    restart: unless-stopped
    
  sentiment_analyzer:
    build:
      context: .
      dockerfile: dockerfiles/sentiment_analyzer.Dockerfile
    depends_on:
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
    volumes:
      - models_volume:/app/models
    restart: unless-stopped
    
  market_predictor:
    build:
      context: .
      dockerfile: dockerfiles/market_predictor.Dockerfile
    depends_on:
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
    volumes:
      - models_volume:/app/models
    restart: unless-stopped
    
  signal_generator:
    build:
      context: .
      dockerfile: dockerfiles/signal_generator.Dockerfile
    depends_on:
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
    restart: unless-stopped
    
  rl_agent:
    build:
      context: .
      dockerfile: dockerfiles/rl_agent.Dockerfile
    depends_on:
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
    volumes:
      - models_volume:/app/models
    restart: unless-stopped
    
  api:
    build:
      context: .
      dockerfile: dockerfiles/api.Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
    restart: unless-stopped

volumes:
  data_volume:
  models_volume:
  rabbitmq_data:
```

#### Exemple de Dockerfile pour un module
```dockerfile
# dockerfiles/data_collector.Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_trading/ /app/ai_trading/

CMD ["python", "-m", "ai_trading.utils.enhanced_data_collector"]
```

### 6. Observabilité complète

#### Configuration Prometheus pour les métriques
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai_trading'
    static_configs:
      - targets: ['api:8000', 'data_collector:8001', 'sentiment_analyzer:8002', 
                  'market_predictor:8003', 'signal_generator:8004', 'rl_agent:8005']
```

#### Intégration des métriques dans le code
```python
# ai_trading/utils/metrics.py
import time
import functools
from typing import Callable, Any
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Métriques globales
REQUESTS_TOTAL = Counter('requests_total', 'Total des requêtes traitées', ['module', 'endpoint'])
PROCESSING_TIME = Histogram('processing_time_seconds', 'Temps de traitement', ['module', 'function'])
ERRORS_TOTAL = Counter('errors_total', 'Total des erreurs', ['module', 'error_type'])
ACTIVE_TASKS = Gauge('active_tasks', 'Tâches actives', ['module'])
DATA_PROCESSED = Counter('data_processed_total', 'Volume de données traitées', ['module', 'data_type'])

def init_metrics(port: int = 8000) -> None:
    """Initialise le serveur de métriques Prometheus"""
    start_http_server(port)

def monitor(module: str) -> Callable:
    """Décorateur pour monitorer une fonction"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            function_name = func.__name__
            
            # Incrémenter le compteur de requêtes
            REQUESTS_TOTAL.labels(module=module, endpoint=function_name).inc()
            
            # Incrémenter le compteur de tâches actives
            ACTIVE_TASKS.labels(module=module).inc()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Incrémenter le compteur d'erreurs
                error_type = type(e).__name__
                ERRORS_TOTAL.labels(module=module, error_type=error_type).inc()
                raise
            finally:
                # Mesurer le temps de traitement
                duration = time.time() - start_time
                PROCESSING_TIME.labels(module=module, function=function_name).observe(duration)
                
                # Décrémenter le compteur de tâches actives
                ACTIVE_TASKS.labels(module=module).dec()
        
        return wrapper
    
    return decorator
```

### 7. Optimisation des performances de bout en bout

#### Script de profiling automatisé
```python
# ai_trading/utils/profiling.py
import cProfile
import pstats
import io
import functools
import logging
from typing import Callable, Any, Optional
from pathlib import Path

def profile(output_dir: Optional[str] = None) -> Callable:
    """
    Décorateur pour profiler une fonction
    
    Args:
        output_dir: Répertoire de sortie pour les résultats de profiling
        
    Returns:
        Fonction décorée avec profiling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Créer le répertoire de sortie si nécessaire
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
            # Créer le profiler
            pr = cProfile.Profile()
            pr.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                pr.disable()
                
                # Traiter les résultats
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 fonctions les plus coûteuses
                
                # Afficher les résultats
                logging.info(f"Profiling de {func.__name__}:\n{s.getvalue()}")
                
                # Sauvegarder les résultats
                if output_dir:
                    stats_path = Path(output_dir) / f"{func.__name__}.prof"
                    pr.dump_stats(str(stats_path))
                    logging.info(f"Résultats de profiling sauvegardés dans {stats_path}")
        
        return wrapper
    
    return decorator
```

### 8. Exécution & cleanup

#### Script principal d'exécution de la pipeline
```bash
#!/bin/bash
# run_ci.sh

set -e  # Arrêt en cas d'erreur

# Configuration
SLACK_WEBHOOK="https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ"
EMAIL_RECIPIENT="team@ai-trading.com"
PROJECT_ROOT=$(pwd)
REPORT_DIR="$PROJECT_ROOT/reports"

# Créer le répertoire de rapports
mkdir -p "$REPORT_DIR"

# Fonction pour envoyer une notification
notify() {
    local status=$1
    local message=$2
    
    # Notification Slack
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"[$status] Pipeline CI/CD: $message\"}" \
            "$SLACK_WEBHOOK"
    fi
    
    # Notification email
    if [ -n "$EMAIL_RECIPIENT" ]; then
        echo "[$status] Pipeline CI/CD: $message" | \
        mail -s "[$status] Pipeline AI Trading" "$EMAIL_RECIPIENT"
    fi
}

# Démarrer les services
echo "🚀 Démarrage des services..."
docker-compose up -d rabbitmq
sleep 5  # Attendre que RabbitMQ soit prêt

# Exécuter les tests unitaires
echo "🧪 Exécution des tests unitaires..."
python -m pytest tests/ --junitxml="$REPORT_DIR/unit_tests.xml" || {
    notify "ERREUR" "Tests unitaires échoués"
    docker-compose down
    exit 1
}

# Exécuter les tests d'intégration
echo "🔄 Exécution des tests d'intégration..."
python -m pytest tests/integration/ --junitxml="$REPORT_DIR/integration_tests.xml" || {
    notify "ERREUR" "Tests d'intégration échoués"
    docker-compose down
    exit 1
}

# Exécuter les exemples en mode non interactif
echo "📚 Exécution des exemples..."
python -m ai_trading.examples.run_all_examples --non-interactive || {
    notify "ERREUR" "Exemples échoués"
    docker-compose down
    exit 1
}

# Générer la documentation
echo "📝 Génération de la documentation..."
cd docs && make html && cd ..

# Exécuter la pipeline complète
echo "🔄 Exécution de la pipeline complète..."
python -m ai_trading.run_pipeline || {
    notify "ERREUR" "Pipeline échouée"
    docker-compose down
    exit 1
}

# Arrêter les services
echo "🛑 Arrêt des services..."
docker-compose down

# Notification de succès
notify "SUCCÈS" "Pipeline terminée avec succès"

echo "✅ CI/CD terminée avec succès"
```

### 9. Badges & documentation

#### Configuration pour les badges dans README.md
```markdown
# AI Trading

![Build Status](https://img.shields.io/github/workflow/status/username/ai-trading/CI)
![Test Coverage](https://img.shields.io/codecov/c/github/username/ai-trading)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/github/license/username/ai-trading)

## Pipeline d'intégration continue

Notre pipeline CI/CD garantit la qualité et la fiabilité du code à chaque étape du développement.

### Étapes automatisées

- ✅ Tests unitaires
- ✅ Tests d'intégration
- ✅ Vérification de la couverture de code
- ✅ Analyse statique du code
- ✅ Benchmarks de performance
- ✅ Génération de la documentation
```

## 🚀 Mise en œuvre

### Étapes d'implémentation

1. **Configuration de l'environnement**
   - Installer Docker et Docker Compose
   - Configurer les variables d'environnement

2. **Mise en place des services de base**
   - Déployer RabbitMQ
   - Configurer Prometheus et Grafana

3. **Développement des utilitaires**
   - Implémenter le gestionnaire de messages
   - Implémenter le gestionnaire de points de contrôle
   - Implémenter le gestionnaire de retries

4. **Configuration de la CI/CD**
   - Configurer GitHub Actions ou GitLab CI
   - Mettre en place les workflows automatisés

5. **Intégration des métriques**
   - Instrumenter le code avec Prometheus
   - Créer des dashboards Grafana

6. **Documentation**
   - Configurer Sphinx ou MkDocs
   - Mettre en place la génération automatique de documentation

## 📊 Métriques et KPIs

- Temps d'exécution de la pipeline complète
- Taux de réussite des builds
- Couverture de code
- Temps de réponse des API
- Utilisation des ressources (CPU, RAM, GPU)
- Nombre d'erreurs par module

## 🔄 Maintenance et évolution

- Révision mensuelle des performances de la pipeline
- Optimisation continue des goulots d'étranglement
- Mise à jour des dépendances et des images Docker
- Ajout de nouveaux tests et métriques selon les besoins 