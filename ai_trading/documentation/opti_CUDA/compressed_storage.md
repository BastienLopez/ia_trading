# Compression de fichiers et lecture rapide avec zstd

Ce module met en œuvre une solution de compression de fichiers et de lecture rapide basée sur le format zstd (Zstandard), qui est particulièrement adapté pour les fichiers de données volumineux utilisés dans l'analyse financière et le trading.

## Fonctionnalités principales

- **Compression haute performance** : Utilisation de zstd, un algorithme de compression à haute vitesse et haut ratio
- **Gestion des fichiers volumineux** : Traitement optimisé des fichiers de données financières de grande taille
- **Lecture en streaming** : Possibilité de lire des fichiers compressés par chunks sans les décompresser entièrement
- **Dictionnaire de compression** : Support des dictionnaires pour améliorer la compression des données similaires
- **Optimisation automatique** : Détermination automatique du meilleur niveau de compression
- **Intégration avec pandas/NumPy** : Support natif des DataFrames et tableaux NumPy

## Modules implémentés

1. **CompressedStorage** : Classe principale pour la gestion de la compression
2. **OptimizedFinancialDataset** : Dataset optimisé pour les données financières avec compression

## Utilisation de base

```python
from ai_trading.data.compressed_storage import CompressedStorage

# Créer une instance de CompressedStorage
storage = CompressedStorage(compression_level=9)

# Compresser un fichier
compressed_file = storage.compress_file('data.csv')

# Sauvegarder un DataFrame dans un format compressé
import pandas as pd
df = pd.read_csv('data.csv')
storage.save_dataframe(df, 'data.parquet.zst', format='parquet')

# Charger un DataFrame depuis un fichier compressé
loaded_df = storage.load_dataframe('data.parquet.zst', format='parquet')

# Traitement par chunks pour les très gros fichiers
for chunk in storage.stream_dataframe_chunks('big_data.csv.zst', format='csv', chunksize=10000):
    # Traiter chaque chunk
    process_chunk(chunk)
```

## Dataset optimisé pour les données financières

```python
from ai_trading.data.optimized_dataset import OptimizedFinancialDataset, get_optimized_dataloader

# Créer un dataset optimisé
dataset = OptimizedFinancialDataset(
    data=df,
    sequence_length=50,
    cache_dir='/path/to/cache',
    compression_level=9
)

# Créer un DataLoader optimisé
dataloader = get_optimized_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Mettre en cache et charger des caractéristiques calculées
dataset.cache_data("features", features_tensor, format="numpy")
loaded_features = dataset.load_cached_data("features", format="numpy", tensor_dtype=torch.float32)
```

## Performances

Les résultats de compression dépendent de la nature des données :

- **Données textuelles répétitives** : Ratio de compression ~746x
- **Données numériques avec motifs** : Ratio ~10-50x
- **Données aléatoires** : Ratio ~1-2x

La vitesse de décompression reste élevée même avec des niveaux de compression élevés, ce qui permet une lecture rapide des données.

## Fonctionnalités avancées

- **Dictionnaires de compression** : Possibilité d'entraîner un dictionnaire sur des échantillons de données similaires pour améliorer la compression
- **Optimisation du niveau** : Fonction `optimize_compression_level` pour trouver le meilleur compromis vitesse/taux
- **Cache intelligent** : Mise en cache automatique des données fréquemment utilisées pour accélérer l'accès

## Recommandations d'utilisation

- Utiliser le niveau de compression 3 pour un bon équilibre vitesse/ratio
- Utiliser le niveau 9 ou plus pour privilégier le ratio de compression
- Utiliser le niveau 1 pour privilégier la vitesse de compression
- Préférer le format parquet pour les DataFrames tabulaires
- Utiliser les dictionnaires de compression pour les données similaires 