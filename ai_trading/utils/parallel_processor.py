"""
Processeur parallèle avancé utilisant Dask pour la parallélisation des calculs.

Ce module permet:
- Parallélisation des calculs sur de grands ensembles de données
- Traitement distribué à l'échelle d'un cluster
- Optimisation automatique de la charge de travail
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, progress

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParallelProcessor")

class DaskParallelProcessor:
    """
    Processeur parallèle utilisant Dask pour accélérer les calculs sur de grands ensembles de données.
    """
    
    def __init__(self, 
                n_workers: int = None, 
                threads_per_worker: int = 2,
                memory_limit: str = "4GB",
                scheduler_port: int = 8786,
                dashboard_address: str = ":8787",
                use_existing_cluster: bool = False,
                existing_scheduler_address: str = None):
        """
        Initialise le processeur parallèle avec Dask.
        
        Args:
            n_workers: Nombre de workers (None = utiliser tous les CPU disponibles)
            threads_per_worker: Nombre de threads par worker
            memory_limit: Limite de mémoire par worker
            scheduler_port: Port pour le scheduler Dask
            dashboard_address: Adresse du dashboard Dask
            use_existing_cluster: Utiliser un cluster Dask existant
            existing_scheduler_address: Adresse du scheduler existant
        """
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.scheduler_port = scheduler_port
        self.dashboard_address = dashboard_address
        self.client = None
        self.cluster = None
        
        # Connexion au cluster
        if use_existing_cluster and existing_scheduler_address:
            logger.info(f"Connexion à un cluster Dask existant à {existing_scheduler_address}")
            self.client = Client(existing_scheduler_address)
        else:
            self._setup_local_cluster()
    
    def _setup_local_cluster(self):
        """Configure et démarre un cluster Dask local."""
        try:
            logger.info("Démarrage d'un cluster Dask local")
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                scheduler_port=self.scheduler_port,
                dashboard_address=self.dashboard_address
            )
            self.client = Client(self.cluster)
            logger.info(f"Cluster Dask démarré avec {len(self.cluster.workers)} workers")
            logger.info(f"Dashboard disponible à {self.client.dashboard_link}")
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du cluster Dask: {e}")
            # Fallback en mode local sans cluster
            self.client = Client(processes=False)
            logger.warning("Fallback vers un client Dask local (sans cluster)")
    
    def parallelize_dataframe(self, df: pd.DataFrame, partition_size: int = None) -> dd.DataFrame:
        """
        Convertit un DataFrame pandas en DataFrame Dask pour traitement parallèle.
        
        Args:
            df: DataFrame pandas à paralléliser
            partition_size: Taille approximative des partitions (en nombre de lignes)
            
        Returns:
            dd.DataFrame: DataFrame Dask parallélisé
        """
        if df.empty:
            logger.warning("DataFrame vide, retour sans parallélisation")
            return dd.from_pandas(df, npartitions=1)
        
        # Déterminer le nombre de partitions basé sur la taille du DataFrame
        if partition_size is None:
            # Heuristique: ~10 tâches par coeur disponible
            estimated_cores = len(self.client.ncores()) * 2  # Approximation
            npartitions = max(1, min(estimated_cores * 10, len(df) // 1000))
        else:
            npartitions = max(1, len(df) // partition_size)
        
        logger.info(f"Parallélisation du DataFrame en {npartitions} partitions")
        return dd.from_pandas(df, npartitions=npartitions)
    
    def apply_parallel(self, df: pd.DataFrame, func: Callable, column_subset: List[str] = None) -> pd.DataFrame:
        """
        Applique une fonction à un DataFrame en parallèle.
        
        Args:
            df: DataFrame à traiter
            func: Fonction à appliquer (doit être sérialisable)
            column_subset: Sous-ensemble de colonnes à utiliser (None = toutes les colonnes)
            
        Returns:
            pd.DataFrame: DataFrame traité
        """
        logger.info("Application parallèle de fonction sur DataFrame")
        
        # Sélectionner le sous-ensemble de colonnes si spécifié
        if column_subset:
            df_subset = df[column_subset].copy()
        else:
            df_subset = df.copy()
        
        # Convertir en DataFrame Dask
        ddf = self.parallelize_dataframe(df_subset)
        
        # Appliquer la fonction en parallèle
        result = ddf.map_partitions(lambda partition: partition.apply(func, axis=1))
        
        # Récupérer le résultat
        return result.compute()
    
    def map_reduce(self, 
                  data: List[Any], 
                  map_func: Callable, 
                  reduce_func: Callable,
                  partition_size: int = None) -> Any:
        """
        Exécute un traitement map-reduce parallèle sur une liste de données.
        
        Args:
            data: Liste de données à traiter
            map_func: Fonction de mapping (appliquée à chaque élément)
            reduce_func: Fonction de réduction (combine les résultats)
            partition_size: Taille des partitions
            
        Returns:
            Any: Résultat final après réduction
        """
        logger.info(f"Exécution de map-reduce sur {len(data)} éléments")
        
        # Créer une collection Dask bag
        bag = dask.bag.from_sequence(data, partition_size)
        
        # Appliquer la fonction map
        mapped = bag.map(map_func)
        
        # Appliquer la fonction reduce
        reduced = mapped.fold(reduce_func)
        
        # Exécuter le calcul
        result = reduced.compute()
        
        return result
    
    def process_chunks(self, 
                      df: pd.DataFrame, 
                      chunk_func: Callable,
                      chunk_size: int = 10000,
                      args: Tuple = (),
                      kwargs: Dict = None) -> pd.DataFrame:
        """
        Traite un grand DataFrame par morceaux en parallèle.
        
        Args:
            df: DataFrame à traiter
            chunk_func: Fonction à appliquer à chaque morceau
            chunk_size: Taille des morceaux
            args: Arguments positionnels pour chunk_func
            kwargs: Arguments nommés pour chunk_func
            
        Returns:
            pd.DataFrame: DataFrame résultant de la concaténation des morceaux traités
        """
        kwargs = kwargs or {}
        
        # Diviser le DataFrame en morceaux
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        logger.info(f"Traitement parallèle de {len(chunks)} morceaux de taille {chunk_size}")
        
        # Traiter chaque morceau en parallèle
        futures = []
        for chunk in chunks:
            # Soumettre chaque tâche au client Dask
            future = self.client.submit(chunk_func, chunk, *args, **kwargs)
            futures.append(future)
        
        # Afficher la progression
        progress(futures)
        
        # Collecter les résultats
        results = self.client.gather(futures)
        
        # Concaténer les résultats
        if all(isinstance(r, pd.DataFrame) for r in results):
            return pd.concat(results, ignore_index=True)
        return results
    
    def parallelize_computation(self, 
                               func: Callable, 
                               data_list: List[Any],
                               *args, **kwargs) -> List[Any]:
        """
        Exécute une fonction sur une liste de données en parallèle.
        
        Args:
            func: Fonction à exécuter
            data_list: Liste de données à traiter
            *args, **kwargs: Arguments supplémentaires pour func
            
        Returns:
            List[Any]: Liste des résultats
        """
        logger.info(f"Exécution parallèle sur {len(data_list)} éléments")
        
        # Créer une liste de tâches
        futures = []
        for data in data_list:
            future = self.client.submit(func, data, *args, **kwargs)
            futures.append(future)
        
        # Afficher la progression
        progress(futures)
        
        # Récupérer les résultats
        return self.client.gather(futures)
    
    def close(self):
        """Ferme le client et le cluster Dask."""
        if self.client:
            logger.info("Fermeture du client Dask")
            self.client.close()
            self.client = None
        
        if self.cluster:
            logger.info("Fermeture du cluster Dask")
            self.cluster.close()
            self.cluster = None
    
    def __enter__(self):
        """Support pour le context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ferme les ressources lors de la sortie du context manager."""
        self.close()


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un grand DataFrame de test
    data = np.random.randn(100000, 5)
    columns = ["A", "B", "C", "D", "E"]
    df = pd.DataFrame(data, columns=columns)
    
    # Fonction de test pour le traitement
    def process_row(row):
        # Simuler un calcul intensif
        import time
        time.sleep(0.001)  # simuler un traitement de 1ms par ligne
        return row["A"] * row["B"] + row["C"] * row["D"] - row["E"]
    
    # Initialiser le processeur parallèle
    with DaskParallelProcessor(n_workers=4) as processor:
        # Exemple 1: Appliquer une fonction en parallèle
        print("Exemple 1: Appliquer une fonction en parallèle")
        result1 = processor.apply_parallel(df, process_row)
        print(f"Résultat: {result1.shape}")
        
        # Exemple 2: Traitement par morceaux
        print("\nExemple 2: Traitement par morceaux")
        def process_chunk(chunk):
            return chunk.assign(Result=chunk["A"] * chunk["B"])
        
        result2 = processor.process_chunks(df, process_chunk, chunk_size=20000)
        print(f"Résultat: {result2.shape}")
        
        # Exemple 3: Map-reduce
        print("\nExemple 3: Map-reduce")
        data_list = [(i, i+1) for i in range(1000)]
        map_func = lambda x: x[0] * x[1]
        reduce_func = lambda x, y: x + y
        
        result3 = processor.map_reduce(data_list, map_func, reduce_func)
        print(f"Résultat: {result3}") 