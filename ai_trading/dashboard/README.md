# Dashboard Interactif avec Analyse Post-Trade

Ce module fournit un dashboard interactif permettant de visualiser et d'analyser les performances du système de trading crypto basé sur l'IA.

## Fonctionnalités

Le dashboard offre les fonctionnalités suivantes:

- **Vue d'ensemble**: Aperçu global de la performance du portefeuille
- **Analyse de performance**: Graphiques détaillés de performance et comparaison avec des benchmarks
- **Transactions**: Visualisation et analyse des transactions effectuées
- **Allocation d'actifs**: Suivi de l'allocation du portefeuille et suggestions de rééquilibrage
- **Analyse de risque**: Métriques de risque avancées (VaR, ES, Drawdown)
- **Analyse factorielle**: Exposition et contribution des facteurs

## Installation

### Prérequis

- Python 3.8+
- Packages requis: `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `numpy`

### Installation des dépendances

```bash
pip install dash dash-bootstrap-components plotly pandas numpy
```

## Utilisation

### Lancer le dashboard

```bash
# Depuis le répertoire racine du projet
python -m ai_trading.dashboard.run

# Ou directement avec le script
python ai_trading/dashboard/run.py
```

Par défaut, le dashboard est accessible à l'adresse http://127.0.0.1:8050/

### Chargement des données

Le dashboard utilise par défaut des données simulées pour la démonstration. Pour utiliser vos propres données:

1. Créez un dossier `data/dashboard` à la racine du projet
2. Placez-y vos fichiers CSV avec les formats suivants:
   - `{portfolio_id}_history.csv`: Historique de la valeur du portefeuille
   - `{portfolio_id}_transactions.csv`: Détail des transactions
   - `{portfolio_id}_allocations.csv`: Allocation d'actifs
   - `{portfolio_id}_factor_exposures.csv`: Expositions aux facteurs

## Structure du code

```
dashboard/
├── app.py                # Application principale
├── callbacks.py          # Callbacks pour l'interactivité
├── data_loader.py        # Chargement et préparation des données
├── layouts.py            # Définition des layouts de l'interface
├── run.py                # Script d'exécution
└── assets/               # Ressources statiques (CSS, images)
    ├── custom.css        # CSS personnalisé
    └── images/           # Images et icônes
```

## Personnalisation

Vous pouvez personnaliser l'apparence du dashboard en modifiant les fichiers suivants:

- `assets/custom.css`: Styles CSS personnalisés
- `layouts.py`: Structure et composants de l'interface
- `callbacks.py`: Comportement interactif

## Intégration

Le dashboard s'intègre avec les autres modules du système de trading AI:

- Utilise les métriques de risque du module `ai_trading.risk`
- S'appuie sur l'optimisation de portefeuille du module `ai_trading.rl.models`
- Visualise les données des transactions et du portefeuille

## Licence

Ce module est distribué sous la même licence que le reste du projet. 