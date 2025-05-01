"""
Package pour le chargement paresseux (lazy loading) des données et les optimisations associées.
Ce package contient des modules pour:
- Chargement paresseux des fichiers de données volumineux
- Mise en cache des transformations fréquemment utilisées
- Optimisation des inférences par lots
"""

from ai_trading.data.lazy_loading.lazy_storage import (
    LazyFileReader,
    LazyDataset,
    get_lazy_dataloader
)

from ai_trading.data.lazy_loading.cached_transform import (
    get_cache_transform_fn,
    CachedFeatureTransform,
    cached_transform,
    default_transform_cache
)

from ai_trading.data.lazy_loading.batch_inference import (
    BatchInferenceOptimizer,
    batch_inference
)

__all__ = [
    # Lazy loading
    'LazyFileReader',
    'LazyDataset',
    'get_lazy_dataloader',
    
    # Cached transforms
    'get_cache_transform_fn',
    'CachedFeatureTransform',
    'cached_transform',
    'default_transform_cache',
    
    # Batch inference
    'BatchInferenceOptimizer',
    'batch_inference'
] 