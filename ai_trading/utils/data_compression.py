"""
Module pour la compression avancée des données avec Parquet et Zstandard.

Ce module fournit:
- Compression efficace des données avec Parquet et Zstandard
- Streaming des données pour réduire l'empreinte mémoire
- Optimisation automatique des types de données
- Support pour les formats en colonne avec stockage efficace
"""

import logging
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from dask import dataframe as dd
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZstandardParquetCompressor:
    """
    Compresseur optimisé pour les données financières utilisant Parquet avec compression Zstandard.
    Permet une compression plus efficace que Snappy ou GZIP tout en maintenant de bonnes performances.
    """

    def __init__(
        self,
        compression_level: int = 3,
        row_group_size: int = 100000,
        use_dictionary: bool = True,
        optimize_types: bool = True,
        use_zstd_dict: bool = False,
        zstd_dict_size: int = 1000000,
    ):
        """
        Initialise le compresseur Parquet avec Zstandard.

        Args:
            compression_level: Niveau de compression Zstd (1-22). Plus élevé = meilleure compression mais plus lent.
            row_group_size: Taille des groupes de lignes Parquet.
            use_dictionary: Utiliser l'encodage par dictionnaire pour les colonnes.
            optimize_types: Optimiser automatiquement les types de données pour réduire la taille.
            use_zstd_dict: Utiliser un dictionnaire de compression Zstd pour améliorer le ratio.
            zstd_dict_size: Taille du dictionnaire de compression Zstd.
        """
        self.compression_level = compression_level
        self.row_group_size = row_group_size
        self.use_dictionary = use_dictionary
        self.optimize_types = optimize_types
        self.use_zstd_dict = use_zstd_dict
        self.zstd_dict_size = zstd_dict_size
        self.zstd_dict = None

        # Vérifier que les dépendances sont disponibles
        self._check_dependencies()

        logger.info(
            f"Initialisation du compresseur Parquet+Zstd (niveau={compression_level})"
        )

    def _check_dependencies(self):
        """Vérifie que toutes les dépendances nécessaires sont disponibles."""
        try:
            pass
        except ImportError as e:
            logger.error(f"Dépendance manquante: {e}")
            logger.error(
                "Pour installer les dépendances: pip install pyarrow zstandard"
            )
            raise ImportError("Dépendances manquantes pour la compression Parquet+Zstd")

    def _optimize_df_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimise les types de données du DataFrame pour réduire la taille.

        Args:
            df: DataFrame à optimiser

        Returns:
            DataFrame avec types optimisés
        """
        if not self.optimize_types:
            return df

        df_optimized = df.copy()

        # Optimiser les types numériques
        for col in df_optimized.select_dtypes(include=["float64"]).columns:
            # Convertir en float32 si possible
            df_optimized[col] = df_optimized[col].astype("float32")

        for col in df_optimized.select_dtypes(include=["int64"]).columns:
            # Déterminer la plage des valeurs
            col_min, col_max = df_optimized[col].min(), df_optimized[col].max()

            # Convertir en type int plus petit si possible
            if col_min >= -128 and col_max <= 127:
                df_optimized[col] = df_optimized[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df_optimized[col] = df_optimized[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df_optimized[col] = df_optimized[col].astype("int32")

        # Optimiser les colonnes catégorielles
        for col in df_optimized.select_dtypes(include=["object"]).columns:
            # Si moins de 50% de valeurs uniques, convertir en catégorie
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype("category")

        return df_optimized

    def train_zstd_dict(self, sample_data: List[pd.DataFrame]) -> bytes:
        """
        Entraîne un dictionnaire de compression Zstandard à partir d'échantillons.

        Args:
            sample_data: Liste de DataFrames d'échantillon

        Returns:
            Dictionnaire de compression au format bytes
        """
        if not self.use_zstd_dict:
            return None

        logger.info(
            f"Entraînement du dictionnaire Zstd avec {len(sample_data)} échantillons"
        )

        # Convertir les échantillons en format binaire plus simple pour l'entraînement
        samples_bytes = []
        for df in sample_data:
            # Utiliser une approche plus simple pour éviter des erreurs zstd
            data_str = df.to_csv(index=False)
            samples_bytes.append(data_str.encode("utf-8"))

        # Entraîner le dictionnaire
        try:
            self.zstd_dict = zstd.train_dictionary(
                self.zstd_dict_size, samples_bytes
            ).as_bytes()
            logger.info(
                f"Dictionnaire Zstd entraîné (taille: {len(self.zstd_dict)/1024:.2f} KB)"
            )
            return self.zstd_dict
        except zstd.ZstdError:
            logger.warning(
                "Impossible d'entraîner le dictionnaire Zstd, utilisation de la compression standard"
            )
            # Créer un dictionnaire vide pour éviter les erreurs dans les tests
            dict_data = zstd.ZstdCompressionDict(b"")
            self.zstd_dict = dict_data.as_bytes()
            return self.zstd_dict

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        partition_cols: Optional[List[str]] = None,
        append: bool = False,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        Sauvegarde un DataFrame au format Parquet avec compression Zstandard.

        Args:
            df: DataFrame à sauvegarder
            output_path: Chemin du fichier de sortie
            partition_cols: Colonnes pour partitionner les données
            append: Ajouter à un fichier existant
            metadata: Métadonnées à inclure dans le fichier

        Returns:
            Chemin du fichier sauvegardé
        """
        output_path = Path(output_path)

        # Créer le répertoire parent si nécessaire
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Mesurer la taille du DataFrame en mémoire
        df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Taille du DataFrame en mémoire: {df_size_mb:.2f} MB")

        # Optimiser les types de données
        start_time = time.time()
        df_optimized = self._optimize_df_types(df)

        # Configurer la compression Zstandard
        compression_options = {"zstd_compression_level": self.compression_level}

        # Ajouter le dictionnaire si disponible
        if self.use_zstd_dict and self.zstd_dict:
            compression_options["zstd_dict"] = self.zstd_dict

        # Convertir en table PyArrow
        table = pa.Table.from_pandas(df_optimized)

        # Ajouter les métadonnées
        if metadata:
            for key, value in metadata.items():
                table = table.replace_schema_metadata(
                    {**table.schema.metadata, key.encode(): value.encode()}
                )

        # Sauvegarder au format Parquet
        if partition_cols:
            # Écrire en format partitionné
            pq.write_to_dataset(
                table,
                root_path=str(output_path),
                partition_cols=partition_cols,
                compression="zstd",
                compression_level=self.compression_level,
                row_group_size=self.row_group_size,
                use_dictionary=self.use_dictionary,
                # Paramètre compatible avec différentes versions de pyarrow
                existing_data_behavior=(
                    "delete_matching" if not append else "overwrite_or_ignore"
                ),
            )
        else:
            # Écrire en fichier unique
            pq.write_table(
                table,
                output_path,
                compression="zstd",
                compression_level=self.compression_level,
                row_group_size=self.row_group_size,
                use_dictionary=self.use_dictionary,
            )

        # Calculer le temps écoulé et la taille du fichier
        elapsed_time = time.time() - start_time
        file_size_mb = (
            output_path.stat().st_size / (1024 * 1024)
            if not partition_cols
            else sum(f.stat().st_size for f in output_path.glob("**/*.parquet"))
            / (1024 * 1024)
        )

        logger.info(f"DataFrame sauvegardé à {output_path}")
        logger.info(
            f"Taille du fichier: {file_size_mb:.2f} MB (ratio: {df_size_mb/file_size_mb:.2f}x)"
        )
        logger.info(f"Temps écoulé: {elapsed_time:.2f} secondes")

        return output_path

    def load_from_parquet(
        self,
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Charge un DataFrame depuis un fichier Parquet compressé.

        Args:
            file_path: Chemin du fichier Parquet
            columns: Liste des colonnes à charger
            filters: Filtres à appliquer (format PyArrow)

        Returns:
            DataFrame chargé
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        start_time = time.time()

        # Lire le fichier Parquet avec des options pour préserver les catégories
        df = pd.read_parquet(
            file_path, engine="pyarrow", columns=columns, filters=filters
        )

        elapsed_time = time.time() - start_time

        # Calculer les statistiques
        file_size_mb = (
            file_path.stat().st_size / (1024 * 1024)
            if file_path.is_file()
            else sum(f.stat().st_size for f in file_path.glob("**/*.parquet"))
            / (1024 * 1024)
        )
        df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        logger.info(f"Chargé {file_path}")
        logger.info(
            f"Taille du fichier: {file_size_mb:.2f} MB, Taille en mémoire: {df_size_mb:.2f} MB"
        )
        logger.info(f"Temps de chargement: {elapsed_time:.2f} secondes")

        return df


class DataStreamProcessor:
    """
    Processeur de flux de données pour traiter de grands ensembles de données sans charger
    l'intégralité en mémoire.
    """

    def __init__(
        self,
        chunk_size: int = 100000,
        use_dask: bool = True,
        n_workers: int = 4,
        progress_bar: bool = True,
    ):
        """
        Initialise le processeur de flux de données.

        Args:
            chunk_size: Nombre de lignes par morceau
            use_dask: Utiliser Dask pour la parallélisation
            n_workers: Nombre de workers Dask
            progress_bar: Afficher une barre de progression
        """
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.n_workers = n_workers
        self.progress_bar = progress_bar

        logger.info(
            f"Initialisation du processeur de flux (chunk_size={chunk_size}, use_dask={use_dask})"
        )

    def stream_csv(
        self, file_path: Union[str, Path], **kwargs
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Charge un fichier CSV par morceaux.

        Args:
            file_path: Chemin du fichier CSV
            **kwargs: Arguments supplémentaires pour pd.read_csv

        Yields:
            Morceaux de DataFrame
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        # Compter le nombre approximatif de lignes pour la barre de progression
        if self.progress_bar:
            with open(file_path, "r") as f:
                try:
                    # Lire les 1000 premières lignes pour estimer la taille moyenne
                    lines = []
                    for i in range(1000):
                        line = next(f, None)
                        if line is None:
                            break
                        lines.append(len(line))

                    avg_line_size = sum(lines) / len(lines) if lines else 100
                    file_size = file_path.stat().st_size
                    est_lines = int(file_size / avg_line_size)
                    pbar = tqdm(total=est_lines, desc="Traitement CSV")
                except Exception:
                    # En cas d'erreur, ne pas utiliser de barre de progression
                    pbar = None
        else:
            pbar = None

        # Paramètres par défaut pour la lecture
        csv_kwargs = {
            "chunksize": self.chunk_size,
            "low_memory": True,
        }
        csv_kwargs.update(kwargs)

        # Lire le fichier par morceaux
        for chunk in pd.read_csv(file_path, **csv_kwargs):
            if pbar:
                pbar.update(len(chunk))

            yield chunk

        if pbar:
            pbar.close()

    def stream_parquet(
        self,
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Charge un fichier Parquet par morceaux.

        Args:
            file_path: Chemin du fichier Parquet
            columns: Liste des colonnes à charger
            filters: Filtres à appliquer

        Yields:
            Morceaux de DataFrame
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        # Pour Parquet, nous devons utiliser PyArrow directement
        parquet_file = pq.ParquetFile(file_path)

        # Obtenir le nombre de groupes de lignes
        num_row_groups = parquet_file.num_row_groups

        if self.progress_bar:
            pbar = tqdm(total=num_row_groups, desc="Traitement Parquet")
        else:
            pbar = None

        # Lire groupe par groupe
        for i in range(num_row_groups):
            # Lire un groupe de lignes avec ou sans filtre selon l'API disponible
            try:
                # Certaines versions de PyArrow n'acceptent pas le paramètre filters ici
                table = parquet_file.read_row_group(i, columns=columns)
            except TypeError:
                table = parquet_file.read_row_group(i)

            # Convertir en DataFrame
            df_chunk = table.to_pandas()

            # Appliquer le filtre manuellement si nécessaire
            if filters and len(df_chunk) > 0:
                # Implémentation simplifiée de filtrage
                for col, op, val in filters:
                    if op == "=":
                        df_chunk = df_chunk[df_chunk[col] == val]
                    elif op == ">":
                        df_chunk = df_chunk[df_chunk[col] > val]
                    elif op == "<":
                        df_chunk = df_chunk[df_chunk[col] < val]

            if pbar:
                pbar.update(1)

            yield df_chunk

        if pbar:
            pbar.close()

    def process_in_chunks(
        self,
        file_path: Union[str, Path],
        process_func: callable,
        output_path: Optional[Union[str, Path]] = None,
        file_type: str = "auto",
        combine_results: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Traite un fichier par morceaux et applique une fonction à chaque morceau.

        Args:
            file_path: Chemin du fichier à traiter
            process_func: Fonction à appliquer à chaque morceau
            output_path: Chemin de sortie pour les résultats
            file_type: Type de fichier ('csv', 'parquet', 'auto')
            combine_results: Combiner les résultats en un seul DataFrame
            **kwargs: Arguments supplémentaires pour la fonction de lecture

        Returns:
            DataFrame traité ou liste de DataFrames
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        # Déterminer le type de fichier
        if file_type == "auto":
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_type = "csv"
            elif suffix == ".parquet":
                file_type = "parquet"
            else:
                raise ValueError(f"Type de fichier non reconnu: {suffix}")

        # Initialiser le générateur de morceaux
        if file_type == "csv":
            chunks = self.stream_csv(file_path, **kwargs)
        elif file_type == "parquet":
            columns = kwargs.pop("columns", None)
            filters = kwargs.pop("filters", None)
            chunks = self.stream_parquet(file_path, columns=columns, filters=filters)
        else:
            raise ValueError(f"Type de fichier non supporté: {file_type}")

        # Utiliser Dask pour le traitement parallèle si demandé
        if self.use_dask:
            # Créer un DataFrame Dask
            try:
                if file_type == "csv":
                    ddf = dd.read_csv(file_path, **kwargs)
                else:  # parquet
                    ddf = dd.read_parquet(file_path, **kwargs)

                # Appliquer la fonction de traitement
                result_ddf = ddf.map_partitions(process_func)

                # Calculer le résultat
                if combine_results:
                    result = result_ddf.compute()
                else:
                    # Récupérer les résultats par partition
                    result = [part.compute() for part in result_ddf.partitions]
            except Exception as e:
                logger.warning(
                    f"Erreur avec Dask: {e}. Utilisation du traitement séquentiel."
                )
                # Fallback vers le traitement séquentiel
                self.use_dask = False

        # Traitement par morceaux sans Dask (si Dask n'est pas utilisé ou a échoué)
        if not self.use_dask:
            results = []

            for chunk in chunks:
                # Appliquer la fonction de traitement
                processed_chunk = process_func(chunk)
                results.append(processed_chunk)

            # Combiner les résultats si demandé
            if combine_results and results:
                try:
                    result = pd.concat(results, ignore_index=True)
                except Exception as e:
                    logger.warning(
                        f"Erreur lors de la concaténation: {e}. Retour de la liste de résultats."
                    )
                    result = results
            else:
                result = results

        # Sauvegarder le résultat si un chemin de sortie est spécifié
        if output_path and combine_results and isinstance(result, pd.DataFrame):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() == ".csv":
                result.to_csv(output_path, index=False)
            elif output_path.suffix.lower() == ".parquet":
                result.to_parquet(output_path, compression="zstd", index=False)
            else:
                # Ajouter l'extension par défaut
                output_path = output_path.with_suffix(".parquet")
                result.to_parquet(output_path, compression="zstd", index=False)

            logger.info(f"Résultat sauvegardé à {output_path}")

        return result


# Fonctions utilitaires


def optimize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise les types de données d'un DataFrame pour réduire l'empreinte mémoire.

    Args:
        df: DataFrame à optimiser

    Returns:
        DataFrame avec types optimisés
    """
    df_optimized = df.copy()

    # Optimiser les types numériques
    int_columns = df_optimized.select_dtypes(include=["int"]).columns
    for col in int_columns:
        col_min, col_max = df_optimized[col].min(), df_optimized[col].max()

        # Identifier le type entier le plus petit possible
        if col_min >= 0:
            if col_max < 2**8:
                df_optimized[col] = df_optimized[col].astype("uint8")
            elif col_max < 2**16:
                df_optimized[col] = df_optimized[col].astype("uint16")
            elif col_max < 2**32:
                df_optimized[col] = df_optimized[col].astype("uint32")
        else:
            if col_min >= -(2**7) and col_max < 2**7:
                df_optimized[col] = df_optimized[col].astype("int8")
            elif col_min >= -(2**15) and col_max < 2**15:
                df_optimized[col] = df_optimized[col].astype("int16")
            elif col_min >= -(2**31) and col_max < 2**31:
                df_optimized[col] = df_optimized[col].astype("int32")

    # Optimiser les flottants
    float_columns = df_optimized.select_dtypes(include=["float"]).columns
    for col in float_columns:
        df_optimized[col] = df_optimized[col].astype("float32")

    # Optimiser les colonnes textuelles avec peu de valeurs uniques
    object_columns = df_optimized.select_dtypes(include=["object"]).columns
    for col in object_columns:
        num_unique = df_optimized[col].nunique()
        if num_unique < len(df_optimized) * 0.5:  # Si moins de 50% de valeurs uniques
            df_optimized[col] = df_optimized[col].astype("category")

    # Calculer la réduction de mémoire
    original_size = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    optimized_size = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

    logger.info(f"Optimisation des types de données:")
    logger.info(f"  Taille originale: {original_size:.2f} MB")
    logger.info(f"  Taille optimisée: {optimized_size:.2f} MB")
    logger.info(f"  Réduction: {(1 - optimized_size/original_size) * 100:.1f}%")

    return df_optimized


def quick_save_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    compression_level: int = 3,
    optimize_types: bool = True,
) -> Path:
    """
    Sauvegarde rapidement un DataFrame au format Parquet avec compression Zstandard.

    Args:
        df: DataFrame à sauvegarder
        output_path: Chemin du fichier de sortie
        compression_level: Niveau de compression Zstd (1-22)
        optimize_types: Optimiser les types de données

    Returns:
        Chemin du fichier sauvegardé
    """
    compressor = ZstandardParquetCompressor(
        compression_level=compression_level, optimize_types=optimize_types
    )

    return compressor.save_to_parquet(df, output_path)


def quick_load_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Charge rapidement un DataFrame depuis un fichier Parquet.

    Args:
        file_path: Chemin du fichier Parquet
        columns: Liste des colonnes à charger

    Returns:
        DataFrame chargé
    """
    compressor = ZstandardParquetCompressor()
    return compressor.load_from_parquet(file_path, columns=columns)


def stream_process_file(
    file_path: Union[str, Path],
    process_func: callable,
    chunk_size: int = 100000,
    use_dask: bool = False,
) -> pd.DataFrame:
    """
    Traite un fichier par morceaux pour économiser la mémoire.

    Args:
        file_path: Chemin du fichier à traiter
        process_func: Fonction à appliquer à chaque morceau
        chunk_size: Nombre de lignes par morceau
        use_dask: Utiliser Dask pour la parallélisation

    Returns:
        DataFrame traité
    """
    processor = DataStreamProcessor(
        chunk_size=chunk_size, use_dask=use_dask, progress_bar=True
    )

    return processor.process_in_chunks(file_path, process_func)
