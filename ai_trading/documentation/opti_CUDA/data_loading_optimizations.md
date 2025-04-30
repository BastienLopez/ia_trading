# Optimisations du chargement de données financières

Ce document décrit les optimisations disponibles pour le chargement efficace de données financières dans le système de trading basé sur l'apprentissage par renforcement.

## Table des matières

1. [DataLoader optimisé](#dataloader-optimisé)
2. [Chargement paresseux (Lazy Loading)](#chargement-paresseux-lazy-loading)
3. [Mise en cache des features](#mise-en-cache-des-features)
4. [Formats de données optimisés](#formats-de-données-optimisés)
5. [Exemples d'utilisation](#exemples-dutilisation)
6. [Benchmark et résultats](#benchmark-et-résultats)

## DataLoader optimisé

Le système utilise un `DataLoader` optimisé pour le chargement parallèle et efficace des données financières. Les optimisations incluent :

- **Parallélisation** : Utilisation de `num_workers > 0` pour charger plusieurs batchs en parallèle
- **Préchargement** : Utilisation de `prefetch_factor` pour précharger les batchs suivants pendant le traitement
- **Mémoire partagée** : Utilisation de la mémoire partagée pour éviter les copies inutiles entre processus
- **Pin Memory** : Option `pin_memory=True` pour accélérer les transferts vers GPU quand disponible

Exemple d'utilisation :

```python
dataloader = get_financial_dataloader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # Ajuster selon le nombre de cœurs CPU disponibles
    prefetch_factor=2,
    pin_memory=True
)
```

## Chargement paresseux (Lazy Loading)

Pour traiter efficacement les grands jeux de données sans saturer la mémoire RAM, nous avons implémenté une stratégie de chargement paresseux :

### Principe

- Les données sont divisées en "chunks" (blocs) de taille configurable
- Seuls les chunks nécessaires sont chargés en mémoire à la demande
- Les chunks inutilisés peuvent être déchargés pour libérer de la mémoire

### Avantages

- **Initialisation rapide** : Le dataset est prêt à l'emploi sans attendre le chargement complet des données
- **Empreinte mémoire réduite** : Seule une fraction des données est en mémoire à un moment donné
- **Accès aléatoire efficace** : Seuls les chunks pertinents sont chargés lors de l'accès aléatoire

### Configuration

```python
dataset = FinancialDataset(
    data="chemin/vers/donnees.parquet",
    sequence_length=50,
    is_train=True,
    lazy_loading=True,  # Activer le chargement paresseux
    chunk_size=10000,   # Taille des chunks (ou None pour automatique)
    precompute_features=True  # Précharger certains chunks initiaux
)
```

## Mise en cache des features

Pour les transformations coûteuses en calcul (comme les indicateurs techniques complexes), un système de mise en cache est disponible :

### Principe

- Les résultats des transformations sont mis en cache
- Les transformations identiques ne sont calculées qu'une seule fois
- Utilisation d'une stratégie LRU (Least Recently Used) pour gérer le cache

### Utilisation

```python
from ai_trading.data.financial_dataset import get_feature_transform_fn

@get_feature_transform_fn(cache_size=500)
def calcul_indicateurs_techniques(sequence):
    # Calculs coûteux...
    return sequence_transformee

# Utilisation avec le dataset
dataset = FinancialDataset(
    data=data,
    transform=calcul_indicateurs_techniques
)
```

### Performances

Dans nos tests, les gains de performance sont significatifs :
- **Réduction des temps de calcul jusqu'à 110x** pour les accès répétés aux mêmes séquences
- Particulièrement efficace pour l'entraînement avec plusieurs époques sur les mêmes données

## Formats de données optimisés

Le système prend en charge plusieurs formats de stockage optimisés pour les données financières :

### Parquet

Recommandé pour la plupart des cas d'usage grâce à :
- Compression efficace (réduction de 4-10x par rapport au CSV)
- Chargement colonne par colonne (lecture partielle)
- Partitionnement par date pour un accès rapide à des périodes spécifiques

Conversion :
```python
from ai_trading.data.data_optimizers import convert_to_parquet

convert_to_parquet(
    data=df,  # DataFrame ou chemin vers CSV
    output_path="donnees.parquet",
    compression="snappy",  # Bon équilibre vitesse/compression
    partition_cols=["year", "month"]  # Partitionnement temporel
)
```

### HDF5

Alternative intéressante pour :
- Accès aléatoire très rapide
- Support de requêtes complexes
- Stockage hiérarchique

Conversion :
```python
from ai_trading.data.data_optimizers import convert_to_hdf5

convert_to_hdf5(
    data=df,
    output_path="donnees.h5",
    key="data",
    complevel=9,  # Niveau de compression (0-9)
    complib="blosc:lz4"  # Algorithme de compression
)
```

## Exemples d'utilisation

### Exemple 1 : Chargement optimisé pour entraînement sur GPU

```python
# Créer un dataset avec chargement paresseux et transformation mise en cache
dataset = FinancialDataset(
    data="chemin/vers/donnees.parquet",
    sequence_length=50,
    target_column="close",
    feature_columns=["open", "high", "low", "close", "volume"],
    is_train=True,
    lazy_loading=True,
    transform=indicateurs_techniques_cached,
    use_shared_memory=True
)

# DataLoader optimisé
dataloader = get_financial_dataloader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True
)

# Entraînement
for epoch in range(n_epochs):
    for batch in dataloader:
        features, targets = batch
        features = features.cuda()  # Transfert rapide grâce à pin_memory
        targets = targets.cuda()
        # Reste de la logique d'entraînement...
```

### Exemple 2 : Traitement de données volumineuses

```python
from ai_trading.data.data_optimizers import optimize_dataset_storage

# Convertir tous les fichiers CSV d'un répertoire en format optimisé
optimize_dataset_storage(
    data_dir="donnees/brutes",
    output_dir="donnees/optimisees",
    format="parquet",
    recursive=True,
    pattern="*.csv",
    partition_dates=True
)

# Charger un dataset volumineux avec lazy loading
dataset = FinancialDataset(
    data="donnees/optimisees",  # Peut être un répertoire avec plusieurs fichiers
    sequence_length=100,
    is_train=True,
    lazy_loading=True,
    chunk_size=50000
)
```

## Benchmark et résultats

Nos tests montrent des gains significatifs avec ces optimisations :

| Configuration               | Temps initialisation | Mémoire utilisée | Temps/batch |
|-----------------------------|---------------------|------------------|-------------|
| Chargement standard         | 1.00x               | 100%             | 1.00x       |
| Lazy loading                | 0.08x (12x plus rapide) | 15%           | 1.20x       |
| Lazy + mise en cache        | 0.08x               | 20%              | 0.40x       |
| Lazy + cache + num_workers=4| 0.08x               | 22%              | 0.25x       |

Pour un dataset de 500,000 points avec des séquences de longueur 50, nous avons observé :
- **Réduction de mémoire** : jusqu'à 85% comparé au chargement standard
- **Initialisation** : jusqu'à 12x plus rapide
- **Accès répétés** : jusqu'à 110x plus rapides avec mise en cache

Les gains sont encore plus importants avec des datasets plus volumineux ou des transformations plus complexes.

## Dépendances optionnelles

Ces fonctionnalités peuvent nécessiter l'installation de packages supplémentaires :

```
pip install pyarrow    # Pour le format Parquet
pip install tables     # Pour HDF5 avec PyTables (plus rapide)
pip install h5py       # Alternative pour HDF5
``` 