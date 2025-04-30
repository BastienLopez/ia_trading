import io
import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zstandard as zstd

logger = logging.getLogger(__name__)

class CompressedStorage:
    """
    Classe pour gérer la compression et décompression des fichiers avec zstd.
    Permet de stocker et charger efficacement des fichiers volumineux.
    """
    
    def __init__(self, compression_level: int = 3, use_dict: bool = False, dict_size: int = 100000):
        """
        Initialise le gestionnaire de stockage compressé.
        
        Args:
            compression_level: Niveau de compression (1-22). Plus élevé = meilleure compression mais plus lent.
            use_dict: Utiliser un dictionnaire de compression pour améliorer le ratio.
            dict_size: Taille du dictionnaire de compression en octets.
        """
        self.compression_level = compression_level
        self.use_dict = use_dict
        self.dict_size = dict_size
        self.compression_dict = None
        
        # Extension de fichier pour les fichiers compressés
        self.extension = ".zst"
    
    def train_dictionary(self, sample_data: List[bytes]) -> bytes:
        """
        Entraîne un dictionnaire de compression à partir d'échantillons de données.
        
        Args:
            sample_data: Liste d'échantillons de données au format bytes.
            
        Returns:
            Le dictionnaire de compression au format bytes.
        """
        if not self.use_dict:
            logger.warning("Dictionary training requested but use_dict is False")
            return None
            
        logger.info(f"Training compression dictionary with {len(sample_data)} samples")
        dict_data = zstd.train_dictionary(self.dict_size, sample_data)
        # Stocker le dictionnaire sous forme de bytes (pas l'objet ZstdCompressionDict)
        self.compression_dict = dict_data.as_bytes()
        return self.compression_dict
    
    def save_dictionary(self, path: Union[str, Path]) -> None:
        """
        Sauvegarde le dictionnaire de compression dans un fichier.
        
        Args:
            path: Chemin du fichier où sauvegarder le dictionnaire.
        """
        if self.compression_dict is None:
            raise ValueError("No dictionary available to save")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(self.compression_dict)
        
        logger.info(f"Saved compression dictionary to {path}")
    
    def load_dictionary(self, path: Union[str, Path]) -> None:
        """
        Charge un dictionnaire de compression depuis un fichier.
        
        Args:
            path: Chemin du fichier contenant le dictionnaire.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {path}")
            
        with open(path, 'rb') as f:
            self.compression_dict = f.read()
            
        self.use_dict = True
        logger.info(f"Loaded compression dictionary from {path}")
    
    def _get_compressor(self) -> zstd.ZstdCompressor:
        """
        Crée et retourne un compresseur zstd avec les paramètres actuels.
        """
        if self.use_dict and self.compression_dict:
            dict_data = zstd.ZstdCompressionDict(self.compression_dict)
            return zstd.ZstdCompressor(level=self.compression_level, dict_data=dict_data)
        else:
            return zstd.ZstdCompressor(level=self.compression_level)
    
    def _get_decompressor(self) -> zstd.ZstdDecompressor:
        """
        Crée et retourne un décompresseur zstd avec les paramètres actuels.
        """
        if self.use_dict and self.compression_dict:
            dict_data = zstd.ZstdCompressionDict(self.compression_dict)
            return zstd.ZstdDecompressor(dict_data=dict_data)
        else:
            return zstd.ZstdDecompressor()
    
    def compress_data(self, data: bytes) -> bytes:
        """
        Comprime des données binaires.
        
        Args:
            data: Données à compresser au format bytes.
            
        Returns:
            Données compressées au format bytes.
        """
        # Utiliser un niveau de compression plus élevé pour les petites données
        # pour garantir une réduction de taille
        if len(data) < 100000:
            level = max(self.compression_level, 9)
            compressor = zstd.ZstdCompressor(level=level)
        else:
            compressor = self._get_compressor()
        
        return compressor.compress(data)
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        Décompresse des données binaires.
        
        Args:
            compressed_data: Données compressées au format bytes.
            
        Returns:
            Données décompressées au format bytes.
        """
        decompressor = self._get_decompressor()
        return decompressor.decompress(compressed_data)
    
    def compress_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Comprime un fichier.
        
        Args:
            input_path: Chemin du fichier à compresser.
            output_path: Chemin du fichier compressé. Si None, ajoute l'extension .zst au chemin d'entrée.
            
        Returns:
            Chemin du fichier compressé.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
            
        if output_path is None:
            output_path = input_path.with_suffix(self.extension)
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        original_size = input_path.stat().st_size
        
        # Utiliser un niveau de compression plus élevé pour les petits fichiers
        if original_size < 100000:
            level = max(self.compression_level, 9)
            compressor = zstd.ZstdCompressor(level=level)
        else:
            compressor = self._get_compressor()
            
        with open(input_path, 'rb') as in_file:
            with open(output_path, 'wb') as out_file:
                compressor.copy_stream(in_file, out_file)
        
        compressed_size = output_path.stat().st_size
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        elapsed_time = time.time() - start_time
        
        logger.info(f"Compressed {input_path} to {output_path}")
        logger.info(f"Original size: {original_size/1e6:.2f} MB, Compressed size: {compressed_size/1e6:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x, Time: {elapsed_time:.2f}s")
        
        return output_path
    
    def decompress_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Décompresse un fichier.
        
        Args:
            input_path: Chemin du fichier compressé.
            output_path: Chemin du fichier décompressé. Si None, enlève l'extension .zst du chemin d'entrée.
            
        Returns:
            Chemin du fichier décompressé.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
            
        if output_path is None:
            if input_path.suffix == self.extension:
                output_path = input_path.with_suffix('')
            else:
                output_path = input_path.with_suffix(input_path.suffix + '.decompressed')
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        compressed_size = input_path.stat().st_size
        
        decompressor = self._get_decompressor()
        with open(input_path, 'rb') as in_file:
            with open(output_path, 'wb') as out_file:
                decompressor.copy_stream(in_file, out_file)
        
        original_size = output_path.stat().st_size
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        elapsed_time = time.time() - start_time
        
        logger.info(f"Decompressed {input_path} to {output_path}")
        logger.info(f"Compressed size: {compressed_size/1e6:.2f} MB, Original size: {original_size/1e6:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x, Time: {elapsed_time:.2f}s")
        
        return output_path
    
    def save_dataframe(self, df: pd.DataFrame, path: Union[str, Path], format: str = 'parquet') -> Path:
        """
        Sauvegarde un DataFrame dans un fichier compressé.
        
        Args:
            df: DataFrame à sauvegarder.
            path: Chemin du fichier compressé.
            format: Format de sérialisation ('parquet', 'csv', 'pickle').
            
        Returns:
            Chemin du fichier compressé.
        """
        path = Path(path)
        if not path.suffix == self.extension:
            path = path.with_suffix(path.suffix + self.extension)
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sérialisation en mémoire selon le format choisi
        buffer = io.BytesIO()
        
        if format == 'parquet':
            df.to_parquet(buffer)
        elif format == 'csv':
            df.to_csv(buffer, index=False)
        elif format == 'pickle':
            df.to_pickle(buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        buffer.seek(0)
        data = buffer.read()
        
        # Compression et sauvegarde
        compressed_data = self.compress_data(data)
        with open(path, 'wb') as f:
            f.write(compressed_data)
            
        logger.info(f"Saved DataFrame to {path} (format: {format})")
        logger.info(f"Original size: {len(data)/1e6:.2f} MB, Compressed size: {len(compressed_data)/1e6:.2f} MB")
        
        return path
    
    def load_dataframe(self, path: Union[str, Path], format: str = 'parquet') -> pd.DataFrame:
        """
        Charge un DataFrame depuis un fichier compressé.
        
        Args:
            path: Chemin du fichier compressé.
            format: Format de sérialisation ('parquet', 'csv', 'pickle').
            
        Returns:
            DataFrame chargé.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        # Lecture et décompression
        with open(path, 'rb') as f:
            compressed_data = f.read()
            
        decompressed_data = self.decompress_data(compressed_data)
        buffer = io.BytesIO(decompressed_data)
        
        # Désérialisation selon le format
        if format == 'parquet':
            df = pd.read_parquet(buffer)
        elif format == 'csv':
            df = pd.read_csv(buffer)
        elif format == 'pickle':
            df = pd.read_pickle(buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Loaded DataFrame from {path} (format: {format})")
        
        return df
    
    def save_numpy(self, array: np.ndarray, path: Union[str, Path]) -> Path:
        """
        Sauvegarde un tableau NumPy dans un fichier compressé.
        
        Args:
            array: Tableau NumPy à sauvegarder.
            path: Chemin du fichier compressé.
            
        Returns:
            Chemin du fichier compressé.
        """
        path = Path(path)
        if not path.suffix == self.extension:
            path = path.with_suffix(path.suffix + self.extension)
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sérialisation en mémoire
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        data = buffer.read()
        
        # Compression et sauvegarde
        compressed_data = self.compress_data(data)
        with open(path, 'wb') as f:
            f.write(compressed_data)
            
        logger.info(f"Saved NumPy array to {path}")
        logger.info(f"Original size: {len(data)/1e6:.2f} MB, Compressed size: {len(compressed_data)/1e6:.2f} MB")
        
        return path
    
    def load_numpy(self, path: Union[str, Path]) -> np.ndarray:
        """
        Charge un tableau NumPy depuis un fichier compressé.
        
        Args:
            path: Chemin du fichier compressé.
            
        Returns:
            Tableau NumPy chargé.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        # Lecture et décompression
        with open(path, 'rb') as f:
            compressed_data = f.read()
            
        decompressed_data = self.decompress_data(compressed_data)
        buffer = io.BytesIO(decompressed_data)
        
        # Désérialisation
        array = np.load(buffer, allow_pickle=True)
        
        logger.info(f"Loaded NumPy array from {path}")
        
        return array
    
    def save_json(self, data: Dict[str, Any], path: Union[str, Path]) -> Path:
        """
        Sauvegarde des données JSON dans un fichier compressé.
        
        Args:
            data: Données à sauvegarder.
            path: Chemin du fichier compressé.
            
        Returns:
            Chemin du fichier compressé.
        """
        path = Path(path)
        if not path.suffix == self.extension:
            path = path.with_suffix(path.suffix + self.extension)
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sérialisation en JSON
        json_str = json.dumps(data).encode('utf-8')
        
        # Compression et sauvegarde
        compressed_data = self.compress_data(json_str)
        with open(path, 'wb') as f:
            f.write(compressed_data)
            
        logger.info(f"Saved JSON data to {path}")
        logger.info(f"Original size: {len(json_str)/1e6:.2f} MB, Compressed size: {len(compressed_data)/1e6:.2f} MB")
        
        return path
    
    def load_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Charge des données JSON depuis un fichier compressé.
        
        Args:
            path: Chemin du fichier compressé.
            
        Returns:
            Données JSON chargées.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        # Lecture et décompression
        with open(path, 'rb') as f:
            compressed_data = f.read()
            
        decompressed_data = self.decompress_data(compressed_data)
        
        # Désérialisation JSON
        data = json.loads(decompressed_data.decode('utf-8'))
        
        logger.info(f"Loaded JSON data from {path}")
        
        return data
    
    def read_compressed_chunks(self, path: Union[str, Path], chunk_size: int = 1024*1024) -> bytes:
        """
        Génère des chunks décompressés à partir d'un fichier compressé.
        Utile pour traiter de très grands fichiers sans les charger entièrement en mémoire.
        
        Args:
            path: Chemin du fichier compressé.
            chunk_size: Taille des chunks en octets.
            
        Yields:
            Chunks décompressés.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        decompressor = self._get_decompressor()
        
        with open(path, 'rb') as f:
            reader = decompressor.stream_reader(f)
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                
    def stream_dataframe_chunks(self, path: Union[str, Path], format: str = 'parquet', 
                               chunksize: int = 10000) -> pd.DataFrame:
        """
        Charge un DataFrame par chunks depuis un fichier compressé.
        Utile pour traiter de très grands DataFrames sans les charger entièrement en mémoire.
        
        Args:
            path: Chemin du fichier compressé.
            format: Format de sérialisation ('parquet', 'csv').
            chunksize: Nombre de lignes par chunk.
            
        Yields:
            Chunks de DataFrame.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        # Pour les formats qui supportent naturellement le streaming
        if format == 'csv':
            # Décompresser d'abord le fichier en mémoire ou dans un fichier temporaire
            with open(path, 'rb') as f:
                compressed_data = f.read()
                
            decompressed_data = self.decompress_data(compressed_data)
            buffer = io.BytesIO(decompressed_data)
            
            # Lire le CSV par chunks
            for chunk in pd.read_csv(buffer, chunksize=chunksize):
                yield chunk
        else:
            # Pour les formats qui ne supportent pas nativement le streaming,
            # on charge tout le DataFrame puis on le découpe en chunks
            df = self.load_dataframe(path, format=format)
            for i in range(0, len(df), chunksize):
                yield df.iloc[i:i+chunksize]


# Fonctions utilitaires pour travailler avec la compression en dehors de la classe
def get_compression_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Obtient des informations sur un fichier compressé zstd.
    
    Args:
        file_path: Chemin du fichier compressé.
        
    Returns:
        Dictionnaire d'informations sur le fichier.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Vérifier s'il s'agit bien d'un fichier zstd
    with open(file_path, 'rb') as f:
        header = f.read(4)
        if header[:4] != b'\x28\xB5\x2F\xFD':
            raise ValueError(f"File {file_path} is not a valid zstd compressed file")
    
    file_size = file_path.stat().st_size
    
    # Obtenir des informations supplémentaires sur le fichier compressé
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        try:
            frame_info = dctx.frame_header_size(f.read(18))
            decompressed_size = frame_info.decompressed_size
            is_checksum = frame_info.has_checksum
            is_dict = frame_info.dict_id != 0
        except Exception:
            decompressed_size = None
            is_checksum = None
            is_dict = None
    
    return {
        'file_path': str(file_path),
        'file_size': file_size,
        'compressed_size': file_size,
        'decompressed_size': decompressed_size,
        'compression_ratio': (decompressed_size / file_size) if decompressed_size else None,
        'has_checksum': is_checksum,
        'uses_dictionary': is_dict,
    }

def optimize_compression_level(data: bytes, test_levels: List[int] = None) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Teste différents niveaux de compression pour trouver le meilleur compromis.
    
    Args:
        data: Données à compresser.
        test_levels: Liste des niveaux de compression à tester. Par défaut [1, 3, 5, 9, 15, 22].
        
    Returns:
        Tuple (niveau_optimal, résultats_par_niveau)
    """
    if test_levels is None:
        test_levels = [1, 3, 5, 9, 15, 22]
    
    results = {}
    original_size = len(data)
    
    for level in test_levels:
        start_time = time.time()
        compressor = zstd.ZstdCompressor(level=level)
        compressed = compressor.compress(data)
        compression_time = time.time() - start_time
        
        # Éviter la division par zéro
        if compression_time == 0:
            compression_time = 0.001  # 1 ms minimum
        
        start_time = time.time()
        decompressor = zstd.ZstdDecompressor()
        decompressor.decompress(compressed, max_output_size=original_size)
        decompression_time = time.time() - start_time
        
        compressed_size = len(compressed)
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        results[level] = {
            'compression_ratio': ratio,
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compressed_size': compressed_size,
            'original_size': original_size,
            'efficiency': ratio / compression_time,  # Ratio de compression par unité de temps
        }
    
    # Trouver le niveau avec la meilleure efficacité
    optimal_level = max(results, key=lambda x: results[x]['efficiency'])
    
    return optimal_level, results

# Fonction utilitaire pour convertir rapidement un DataFrame en fichier compressé
def dataframe_to_compressed(df: pd.DataFrame, output_path: Union[str, Path], 
                          format: str = 'parquet', compression_level: int = 3) -> Path:
    """
    Convertit rapidement un DataFrame en fichier compressé zstd.
    
    Args:
        df: DataFrame à compresser.
        output_path: Chemin du fichier de sortie.
        format: Format de sérialisation ('parquet', 'csv', 'pickle').
        compression_level: Niveau de compression zstd (1-22).
        
    Returns:
        Chemin du fichier compressé.
    """
    storage = CompressedStorage(compression_level=compression_level)
    return storage.save_dataframe(df, output_path, format=format)

# Fonction utilitaire pour charger rapidement un DataFrame depuis un fichier compressé
def compressed_to_dataframe(input_path: Union[str, Path], format: str = 'parquet') -> pd.DataFrame:
    """
    Charge rapidement un DataFrame depuis un fichier compressé zstd.
    
    Args:
        input_path: Chemin du fichier compressé.
        format: Format de sérialisation ('parquet', 'csv', 'pickle').
        
    Returns:
        DataFrame chargé.
    """
    storage = CompressedStorage()
    return storage.load_dataframe(input_path, format=format) 