# Optimisations Multiprocessing/Multithreading pour le Trading IA

Ce document présente les techniques d'optimisation multithreading et multiprocessing implémentées dans notre système de trading basé sur l'IA, avec des recommandations pour maximiser les performances sur différentes configurations matérielles.

## Table des matières

1. [Introduction](#introduction)
2. [Principes fondamentaux](#principes-fondamentaux)
3. [Optimisations implémentées](#optimisations-implémentées)
4. [Paramètres clés](#paramètres-clés)
5. [Tests et benchmarks](#tests-et-benchmarks)
6. [Recommandations par type de matériel](#recommandations-par-type-de-matériel)
7. [Résolution des problèmes courants](#résolution-des-problèmes-courants)

## Introduction

Le multiprocessing et le multithreading sont des techniques essentielles pour optimiser les performances des systèmes de trading, en particulier lors du chargement et du prétraitement des données. Ces optimisations permettent d'exploiter pleinement les ressources CPU disponibles et de minimiser les temps d'attente.

Notre système de trading implémente plusieurs stratégies d'optimisation pour le traitement parallèle des données, avec un focus particulier sur le chargement efficace des données financières.

## Principes fondamentaux

### Différence entre multiprocessing et multithreading

- **Multithreading** : Exécution de plusieurs threads au sein d'un même processus, partageant la même mémoire.
- **Multiprocessing** : Exécution de plusieurs processus indépendants avec des espaces mémoire séparés.

### Application dans le système de trading

Le système de trading utilise principalement le multithreading via PyTorch DataLoader, qui permet de charger et prétraiter les données financières en parallèle tout en alimentant efficacement les modèles d'apprentissage.

## Optimisations implémentées

### 1. DataLoader optimisé

La classe `FinancialDataset` et la fonction `get_financial_dataloader()` intègrent plusieurs optimisations :

```python
dataloader = get_financial_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,              # Parallélisation
    prefetch_factor=2,          # Préchargement
    pin_memory=True,            # Optimisation mémoire pour GPU
    optimize_memory=True,       # Nettoyage automatique
    persistent_workers=True     # Réutilisation des workers
)
```

### 2. Chargement asynchrone

Le chargement asynchrone permet de préparer les données pendant que le modèle s'entraîne :

```python
dataset = FinancialDataset(
    data=data,
    lazy_loading=True,
    async_prefetch=True,          # Préchargement asynchrone
    prefetch_num_chunks=2,        # Nombre de chunks à précharger
    chunk_size=5000               # Taille des chunks
)
```

### 3. Gestion de mémoire optimisée

La classe `GarbageCollectionDataLoader` intègre une gestion automatique de la mémoire pour éviter les fuites et optimiser l'utilisation des ressources.

### 4. Variables d'environnement optimisées

Le module `install_optimizations.py` configure automatiquement les variables d'environnement pour maximiser les performances des bibliothèques numériques :

```
MKL_NUM_THREADS=8
NUMEXPR_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
OMP_NUM_THREADS=8
OMP_WAIT_POLICY=ACTIVE
```

## Paramètres clés

### num_workers

Ce paramètre définit le nombre de processus parallèles utilisés pour charger les données :

- `0` : Chargement synchrone (utile pour le débogage)
- `1-4` : Parallélisation modérée
- `-1` : Auto-détection (utilise le nombre de cœurs CPU disponibles)

### prefetch_factor

Contrôle combien de batchs sont préchargés par worker :

- `1` : Préchargement minimal
- `2` : Préchargement modéré (recommandé pour la plupart des cas)
- `4-8` : Préchargement agressif (utile pour les données complexes)

### persistent_workers

- `True` : Garde les workers en vie entre les époques, réduisant le temps de démarrage
- `False` : Recrée les workers à chaque époque

### pin_memory

- `True` : Accélère le transfert des données vers le GPU
- `False` : Économise la mémoire CPU

## Tests et benchmarks

L'exemple `multiprocessing_optimization_example.py` permet de tester différentes configurations et de déterminer les paramètres optimaux pour votre matériel.

Résultats typiques observés :

| Configuration | Performance (batches/s) | Utilisation CPU | Mémoire |
|---------------|-------------------------|-----------------|---------|
| Synchrone     | 15-20                   | 25-35%          | Basse   |
| 4 workers     | 30-40                   | 70-90%          | Moyenne |
| Auto + persist| 35-45                   | 80-95%          | Haute   |

## Recommandations par type de matériel

### Machines à 4 cœurs ou moins

- `num_workers=2`
- `prefetch_factor=2`
- `pin_memory=False` (sauf si GPU)
- `chunk_size=2000`

### Machines à 8+ cœurs

- `num_workers=4` ou `-1` (auto)
- `prefetch_factor=4`
- `pin_memory=True`
- `persistent_workers=True`
- `chunk_size=5000`

### Environnements avec RAM limitée

- Réduire `num_workers` et `prefetch_factor`
- Utiliser `memory_optimize=True`
- Augmenter `gc_frequency` dans `GarbageCollectionDataLoader`

## Résolution des problèmes courants

### Utilisation CPU faible

Si l'utilisation CPU reste faible malgré plusieurs workers :

1. Vérifier si le disque est le goulot d'étranglement (IO-bound)
2. Augmenter `batch_size` pour des calculs plus lourds par batch
3. Vérifier que `lazy_loading=True` pour forcer le chargement parallèle

### Utilisation mémoire excessive

Si la consommation mémoire est trop élevée :

1. Réduire `prefetch_factor` et `num_workers`
2. Diminuer `chunk_size` pour charger moins de données à la fois
3. Augmenter la fréquence de garbage collection avec `gc_frequency`

### Performances inférieures avec plus de workers

Ce phénomène peut survenir lorsque :

1. Le CPU est saturé par d'autres processus
2. Les workers passent plus de temps à se synchroniser qu'à travailler
3. La lecture du disque est le facteur limitant

Solution : trouver le nombre optimal de workers via benchmarking. 