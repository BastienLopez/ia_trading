# Guide des Tests - Crypto Trading AI

Guide détaillé pour l'exécution et la maintenance des tests du projet.

## 📋 Vue d'ensemble

Le projet utilise pytest comme framework de test principal, avec:
- Tests unitaires pour chaque composant
- Tests d'intégration pour les interactions entre composants
- Mesure de la couverture de code avec pytest-cov

## 🔧 Installation des outils de test

```bash
# Installation des dépendances de test
pip install pytest pytest-cov
```

## 🚀 Exécution des tests

### Tests complets

Pour exécuter tous les tests:

```bash
python -m pytest
```

### Tests par module

Pour tester un module spécifique:

```bash
# Tests du module d'IA
python -m pytest tests/ai_trading/

# Tests du bot Discord
python -m pytest tests/discord_bot/

# Tests du module de configuration
python -m pytest tests/test_config.py
```

### Tests individuels

Pour exécuter un test spécifique:

```bash
# Test d'une classe spécifique
python -m pytest tests/ai_trading/test_data_processor.py::TestDataProcessor

# Test d'une méthode spécifique
python -m pytest tests/ai_trading/test_data_processor.py::TestDataProcessor::test_add_indicators
```

### Couverture de code

Pour mesurer la couverture de code:

```bash
python -m pytest --cov=ai_trading --cov=discord_bot --cov-report=term-missing
```

## 📝 Utilisation des logs et prints pour le débogage

Pour faciliter le débogage des tests, vous pouvez ajouter des logs ou des prints qui vous aideront à comprendre l'exécution:

### Utilisation des prints simples

```python
def test_add_indicators(self):
    """Test de l'ajout d'indicateurs techniques"""
    # Affichage des informations de débogage
    print(f"\nTest démarré: {self._testMethodName}")
    print(f"Forme du DataFrame initial: {self.test_df.shape}")
    
    # Exécution du test
    result = self.data_processor.add_indicators(self.test_df.copy())
    
    # Affichage des résultats intermédiaires
    print(f"Colonnes du DataFrame résultant: {result.columns.tolist()}")
    print(f"Forme du DataFrame résultant: {result.shape}")
    
    # Vérification des indicateurs...
```

Pour activer l'affichage des prints pendant les tests:

```bash
python -m pytest -v tests/ai_trading/test_data_processor.py::TestDataProcessor::test_add_indicators -s
```

Le paramètre `-s` est essentiel car il désactive la capture de la sortie standard, permettant aux messages de s'afficher.

### Utilisation du logging

Le logging est plus flexible que les prints simples:

```python
import logging

# Configuration du logger au début du fichier de test
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestDataProcessor(unittest.TestCase):
    def test_add_indicators(self):
        logger.debug("Test démarré: %s", self._testMethodName)
        logger.info("Forme du DataFrame initial: %s", self.test_df.shape)
        
        # Exécution du test
        result = self.data_processor.add_indicators(self.test_df.copy())
        
        logger.debug("Colonnes résultantes: %s", result.columns.tolist())
        # ...
```

Pour activer les logs pendant les tests:

```bash
# Définir le niveau de logging pour les tests
LOGLEVEL=DEBUG python -m pytest tests/ai_trading/test_data_processor.py -v
```

### Débogage avancé avec des assertions détaillées

Pour des tests plus lisibles et informatifs:

```python
def test_crypto_trading_env_step_buy(self):
    """Test de l'action 'acheter' dans l'environnement"""
    self.env.reset()
    initial_balance = self.env.balance
    print(f"\n[DÉBOGAGE] Solde initial: {initial_balance}")
    
    # Action 1 = Acheter
    observation, reward, done, truncated, info = self.env.step(1)
    print(f"[DÉBOGAGE] Après achat - Solde: {self.env.balance}, Crypto: {self.env.crypto_held}")
    print(f"[DÉBOGAGE] Récompense: {reward}, Position: {self.env.current_position}")
    print(f"[DÉBOGAGE] Valeur du portefeuille: {self.env.portfolio_value}")
    
    # Vérifications avec messages explicites
    self.assertEqual(self.env.balance, 0, 
                    f"Le solde devrait être 0 après achat mais est {self.env.balance}")
    self.assertGreater(self.env.crypto_held, 0, 
                      f"Cryptos détenus devraient être > 0 mais sont {self.env.crypto_held}")
```

### Utilisation de fichiers de log

Pour garder une trace permanente des exécutions de test:

```python
import logging
from datetime import datetime

# Configuration du logger avec fichier
log_file = f"logs/test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Affiche aussi dans la console
    ]
)

def setUp(self):
    """Configuration avant chaque test"""
    logging.info("====== Début du test: %s ======", self._testMethodName)
    # ...

def tearDown(self):
    """Nettoyage après chaque test"""
    logging.info("====== Fin du test: %s ======\n", self._testMethodName)
    # ...
```

## 🧩 Structure des tests

```
tests/
├── conftest.py              # Configuration et fixtures communes
├── test_config.py           # Tests de configuration
├── __init__.py              # Initialisation du module de tests
├── ai_trading/              # Tests du module d'IA
│   ├── test_api.py          # Tests de l'API
│   ├── test_rl_agent.py     # Tests de l'agent RL
│   ├── test_data_processor.py # Tests du traitement des données
│   └── __init__.py
└── discord_bot/             # Tests du bot Discord
    ├── test_bot.py          # Tests du bot
    └── __init__.py
```

## 📊 Tests de l'API

Les tests de l'API utilisent TestClient de FastAPI pour simuler des requêtes HTTP:

```python
def test_predict_endpoint_success(self, mock_get_agent, mock_get_data_processor):
    """Test de l'endpoint /predict avec succès"""
    # Configuration...
    print(f"\n[DÉBOGAGE] Test de l'endpoint /predict - Envoi d'une requête")
    response = self.client.post("/predict", json={"symbol": "BTC/USDT"})
    print(f"[DÉBOGAGE] Code de statut: {response.status_code}")
    print(f"[DÉBOGAGE] Réponse: {response.json()}")
    
    self.assertEqual(response.status_code, 200)
    # Assertions...
```

## 🤖 Tests de l'Agent RL

Les tests de l'agent RL vérifient son comportement avec différentes actions de trading:

```python
def test_crypto_trading_env_step_buy(self):
    """Test de l'action 'acheter' dans l'environnement"""
    # Configuration...
    logging.info("Test de l'action Acheter - Étape initiale")
    observation, reward, done, truncated, info = self.env.step(1)  # Action 1 = Acheter
    logging.info("Après action Acheter - Solde: %.2f, Crypto: %.8f, Position: %d", 
                self.env.balance, self.env.crypto_held, self.env.current_position)
    # Assertions...
```

## 📉 Tests du Traitement des Données

Les tests du processeur de données vérifient les calculs d'indicateurs techniques:

```python
def test_add_indicators(self):
    """Test de l'ajout d'indicateurs techniques"""
    result = self.data_processor.add_indicators(self.test_df.copy())
    # Vérification des indicateurs...
```

## 📨 Tests du Bot Discord

Les tests du bot Discord simulent des commandes et vérifient les réponses:

```python
async def test_price_command(self, mock_bot, mock_ctx):
    """Test de la commande !prix"""
    await mock_bot.price.callback(mock_bot, mock_ctx, symbol="BTC")
    mock_ctx.send.assert_called_once()
    # Vérifications...
```

## ⚠️ Résolution des problèmes

### Dépendances manquantes

Si certains tests échouent en raison de dépendances manquantes:

```bash
# Installer les dépendances
pip install ccxt gymnasium matplotlib
```

### Erreurs d'importation

Si vous rencontrez des erreurs d'importation dans les tests:

1. Vérifiez que les modules sont correctement installés
2. Vérifiez la variable PYTHONPATH
3. Assurez-vous que les `__init__.py` sont présents dans chaque répertoire

### Mocks pour les API externes

Les tests utilisent des mocks pour les services externes (API de change, Discord).
Assurez-vous que les mocks sont correctement configurés.

### Débuggez un test spécifique avec pdb

Pour débugger interactivement un test:

```bash
python -m pytest tests/ai_trading/test_data_processor.py::TestDataProcessor::test_add_indicators --pdb
```

Vous pouvez aussi ajouter des points d'arrêt dans votre code de test:

```python
def test_add_indicators(self):
    """Test de l'ajout d'indicateurs techniques"""
    print("Test démarré")
    import pdb; pdb.set_trace()  # Le test s'arrête ici
    result = self.data_processor.add_indicators(self.test_df.copy())
    # ...
``` 