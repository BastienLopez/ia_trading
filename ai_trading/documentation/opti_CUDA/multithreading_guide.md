# Guide d'optimisation multithreading/multiprocessing

Ce guide explique comment optimiser les performances CPU de votre système de trading en utilisant efficacement le multithreading et le multiprocessing.

## Table des matières

1. [Introduction](#introduction)
2. [Concepts clés](#concepts-clés)
3. [Classes et fonctions d'optimisation](#classes-et-fonctions-doptimisation)
4. [Exemples d'utilisation](#exemples-dutilisation)
5. [Bonnes pratiques](#bonnes-pratiques)
6. [Benchmarks et résultats](#benchmarks-et-résultats)

## Introduction

Le multithreading et le multiprocessing sont des techniques essentielles pour tirer parti de la puissance des processeurs multi-cœurs modernes. Dans le contexte d'un système de trading, ces techniques peuvent significativement améliorer :

- Le chargement et le prétraitement des données financières
- L'exécution de calculs parallèles pour les indicateurs et stratégies
- Le backtesting sur de grands ensembles de données
- La génération de features en temps réel

Notre système de trading intègre une optimisation automatique du multithreading pour maximiser les performances sur différentes configurations matérielles.

## Concepts clés

### Multithreading vs Multiprocessing

- **Multithreading** : Plusieurs threads s'exécutent simultanément au sein d'un même processus, partageant la mémoire. Idéal pour les tâches I/O-bound (entrées/sorties, comme le chargement de fichiers).
- **Multiprocessing** : Plusieurs processus s'exécutent en parallèle avec des espaces mémoire séparés. Idéal pour les tâches CPU-bound (calculs intensifs).

### Le Global Interpreter Lock (GIL) de Python

Le GIL est une limitation de l'interpréteur Python standard qui empêche plusieurs threads d'exécuter du code Python simultanément dans un même processus. Implications :

- Pour les opérations I/O-bound : le multithreading reste efficace
- Pour les opérations CPU-bound : privilégier le multiprocessing
- Les bibliothèques numériques comme NumPy et Pandas libèrent souvent le GIL pendant les calculs

### Hyperthreading

L'hyperthreading permet à un cœur physique de se comporter comme deux cœurs logiques. Notre optimiseur en tient compte pour déterminer le nombre optimal de workers.

## Classes et fonctions d'optimisation

### ThreadingOptimizer

La classe principale pour optimiser l'utilisation des threads et processus :

```python
from ai_trading.utils.threading_optimizer import ThreadingOptimizer

# Initialiser l'optimiseur
optimizer = ThreadingOptimizer()

# Obtenir le nombre optimal de workers
optimal_workers = optimizer.calculate_optimal_workers()
print(f"DataLoader: {optimal_workers['dataloader']} workers")
print(f"Prétraitement: {optimal_workers['preprocessing']} workers")
```

### Configuration automatique du DataLoader

Le DataLoader financier peut désormais s'auto-configurer :

```python
from ai_trading.data.financial_dataset import get_financial_dataloader

# Utiliser l'auto-configuration
dataloader = get_financial_dataloader(
    dataset=my_dataset,
    batch_size=32,
    num_workers=-1,  # -1 active l'auto-détection
    auto_threading=True
)
```

### Parallélisation des opérations

Pour paralléliser des opérations sur une liste d'éléments :

```python
from ai_trading.utils.threading_optimizer import parallel_map

# Pour des tâches I/O-bound (lectures de fichiers, etc.)
results = parallel_map(my_function, items, use_processes=False)

# Pour des tâches CPU-bound (calculs complexes)
results = parallel_map(my_function, items, use_processes=True)
```

### Optimisation système pour l'entraînement

Configure automatiquement le système pour des performances optimales :

```python
from ai_trading.utils.threading_optimizer import optimize_system_for_training

# Appeler avant l'entraînement
optimize_system_for_training()
```

## Exemples d'utilisation

### 1. Prétraitement parallèle de données financières

```python
from ai_trading.utils.threading_optimizer import ThreadingOptimizer, parallel_map

# Fonction de traitement pour une ligne de données
def process_row(row):
    # Calculs d'indicateurs, normalisation, etc.
    return processed_data

# Obtenir le nombre optimal de processus
optimizer = ThreadingOptimizer()
optimal = optimizer.calculate_optimal_workers()
num_processes = optimal['preprocessing']

# Exécuter en parallèle
results = parallel_map(process_row, dataframe.iterrows(), 
                      max_workers=num_processes,
                      use_processes=True)
```

### 2. DataLoader optimisé pour l'entraînement

```python
from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader

# Créer le dataset
dataset = FinancialDataset(
    data=my_data,
    sequence_length=50,
    lazy_loading=True
)

# Créer un dataloader optimisé
dataloader = get_financial_dataloader(
    dataset=dataset,
    batch_size=32,
    num_workers=-1,  # Auto-détection
    auto_threading=True
)
```

## Bonnes pratiques

1. **Adaptez à la tâche** : Utilisez le threading pour les tâches I/O-bound et le multiprocessing pour les tâches CPU-bound

2. **Attention à la granularité** : Pour les petites tâches, le coût de la parallélisation peut dépasser les bénéfices

3. **Pensez à la mémoire** : Surveillez l'utilisation mémoire, car trop de workers peuvent saturer la RAM

4. **Laissez les resources au système** : Réservez toujours quelques cœurs pour le système d'exploitation et les autres processus

5. **Préférez l'auto-optimisation** : Utilisez num_workers=-1 et auto_threading=True quand c'est possible

## Benchmarks et résultats

Des tests sur différentes configurations matérielles ont montré que :

- Les petits datasets (<5000 points) obtiennent de meilleures performances avec 0-2 workers
- Les grands datasets bénéficient de 4-8 workers selon le nombre de cœurs disponibles
- Un dataloader avec persistent_workers=True est plus efficace pour les entraînements longs
- Sur des machines avec hyperthreading, limiter le nombre de threads NumPy/PyTorch au nombre de cœurs physiques améliore souvent les performances

### Accélérations typiques

| Opération | Accélération typique |
|-----------|----------------------|
| Chargement de données (I/O-bound) | 2-4x |
| Prétraitement (CPU-bound) | 3-7x |
| Conversion de format | 2-3x |

## Conclusion

L'optimisation multithreading/multiprocessing est essentielle pour des performances maximales sur les systèmes modernes. Notre optimiseur automatique simplifie cette tâche en déterminant les meilleures configurations pour chaque type d'opération.

Pour des performances optimales, utilisez toujours les fonctions d'auto-optimisation et adaptez les stratégies de parallélisation en fonction de la nature de vos tâches (I/O-bound vs CPU-bound). 