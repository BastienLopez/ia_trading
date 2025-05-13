# Optimisation Bayésienne des Hyperparamètres

Ce document explique comment utiliser l'optimisation bayésienne pour trouver efficacement les hyperparamètres optimaux des agents d'apprentissage par renforcement dans le projet AI Trading.

## Introduction

L'optimisation bayésienne est une technique puissante pour l'optimisation d'hyperparamètres qui présente plusieurs avantages par rapport à la recherche par grille ou aléatoire traditionnelle :

- **Efficacité** : Trouve de meilleures solutions avec moins d'évaluations
- **Adaptation dynamique** : Concentre les évaluations dans les régions prometteuses
- **Prise en compte des incertitudes** : Balance exploration et exploitation
- **Support pour l'optimisation multi-objectifs** : Optimise plusieurs métriques simultanément

## Architecture

L'implémentation de l'optimisation bayésienne dans ce projet comprend :

1. **BayesianOptimizer** : Classe principale qui étend `HyperparameterOptimizer` avec des méthodes d'optimisation bayésienne
2. **Fonctions d'acquisition** : Implémentation de l'amélioration espérée (Expected Improvement)
3. **Processus gaussien** : Modèle probabiliste pour estimer la fonction objectif
4. **Optimisation multi-objectifs** : Support pour l'optimisation de plusieurs métriques avec pondération

## Utilisation

### Exemple simple

```python
from ai_trading.rl.bayesian_optimizer import optimize_sac_agent_bayesian

# Définir l'espace de recherche
param_space = {
    "actor_learning_rate": (1e-5, 1e-3),  # Intervalle continu
    "critic_learning_rate": (1e-5, 1e-3),
    "batch_size": (32, 256),              # Intervalle discret
    "hidden_size": (64, 512),
    "activation": ["relu", "tanh", "elu"]  # Paramètre catégoriel
}

# Exécuter l'optimisation bayésienne
best_params = optimize_sac_agent_bayesian(
    train_data=data,
    param_space=param_space,
    n_episodes=50,
    eval_episodes=10,
    n_initial_points=5,   # Points initiaux aléatoires
    n_iterations=15,      # Itérations bayésiennes
    exploration_weight=0.01
)

print(f"Meilleurs paramètres: {best_params}")
```

### Optimisation multi-objectifs

```python
# Définir les poids pour chaque métrique
multi_objective_weights = {
    "total_reward": 0.4,
    "sharpe_ratio": 0.3,
    "max_drawdown": -0.2,  # Négatif car on veut minimiser
    "win_rate": 0.1
}

best_params = optimize_sac_agent_bayesian(
    train_data=data,
    param_space=param_space,
    multi_objective_weights=multi_objective_weights
)
```

### Ligne de commande

Vous pouvez également utiliser l'exemple fourni via la ligne de commande :

```bash
# Optimisation bayésienne pour l'agent SAC
python -m ai_trading.examples.bayesian_optimization_example --agent sac --episodes 50 --iterations 20 --save

# Optimisation bayésienne pour l'agent GRU-SAC avec multi-objectifs
python -m ai_trading.examples.bayesian_optimization_example --agent gru_sac --multi-objective --exploration 0.05
```

## Paramètres clés

- **n_initial_points** : Nombre de points initiaux échantillonnés aléatoirement
- **n_iterations** : Nombre d'itérations d'optimisation bayésienne
- **exploration_weight** : Poids d'exploration dans la fonction d'acquisition (plus élevé = plus d'exploration)
- **gp_noise** : Niveau de bruit pour le processus gaussien
- **multi_objective_weights** : Dictionnaire des poids pour l'optimisation multi-objectifs

## Visualisation des résultats

L'optimiseur bayésien génère plusieurs visualisations pour analyser les résultats :

1. **Graphique de convergence** : Montre l'évolution du meilleur score au fil des itérations
2. **Importance des paramètres** : Analyse de la sensibilité de chaque hyperparamètre
3. **Surface de réponse** : Visualisation de la fonction objectif estimée

Ces graphiques sont sauvegardés dans le répertoire spécifié par `save_dir`.

## Comparaison avec la recherche par grille

L'optimisation bayésienne est particulièrement avantageuse lorsque :
- L'évaluation de chaque configuration est coûteuse en temps
- L'espace de recherche est vaste
- Les hyperparamètres ont des interactions complexes

Dans nos tests, l'optimisation bayésienne a permis de trouver de meilleures configurations avec 3 à 5 fois moins d'évaluations qu'une recherche par grille exhaustive.

## Limitations et considérations

- **Dimensionnalité** : L'optimisation bayésienne devient moins efficace avec un très grand nombre de paramètres (>20)
- **Paramètres catégoriels** : Bien que supportés, ils augmentent la dimensionnalité de l'espace
- **Bruit d'évaluation** : Les environnements stochastiques peuvent nécessiter plusieurs évaluations par configuration

## Références

- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms.
- Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2015). Taking the human out of the loop: A review of Bayesian optimization. 