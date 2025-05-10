"""
Tests pour le module de cache intelligent.

Ce module teste les fonctionnalités de SmartCache et DataCache, notamment :
- Stratégie LRU (Least Recently Used)
- Préchargement des données
- Compression
- Gestion de la cohérence
"""

import os
import time
import tempfile
import shutil
import threading
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading.utils.smart_cache import SmartCache, DataCache

class TestSmartCache(unittest.TestCase):
    """Tests pour la classe SmartCache."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SmartCache(
            max_size=10,
            ttl=60,
            compression_level=3,
            preload_threshold=2,
            cache_dir=self.temp_dir,
            persist=True
        )
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        # Supprimer le répertoire temporaire
        shutil.rmtree(self.temp_dir)
    
    def test_basic_operations(self):
        """Teste les opérations de base du cache."""
        # Test de set/get
        self.cache.set("test_key", "test_value")
        self.assertEqual(self.cache.get("test_key"), "test_value")
        
        # Test de valeur par défaut
        self.assertEqual(self.cache.get("nonexistent_key", "default"), "default")
        
        # Test de suppression
        self.cache.delete("test_key")
        self.assertIsNone(self.cache.get("test_key"))
    
    def test_lru_eviction(self):
        """Teste l'éviction LRU (Least Recently Used)."""
        # Remplir le cache au-delà de sa capacité
        for i in range(15):
            self.cache.set(f"key_{i}", f"value_{i}")
        
        # Vérifier que les premières clés ont été évincées
        for i in range(5):
            self.assertIsNone(self.cache.get(f"key_{i}"))
        
        # Vérifier que les dernières clés sont toujours présentes
        for i in range(5, 15):
            self.assertEqual(self.cache.get(f"key_{i}"), f"value_{i}")
    
    def test_ttl_expiration(self):
        """Teste l'expiration des éléments basée sur le TTL."""
        # Créer un cache avec TTL court
        short_ttl_cache = SmartCache(
            max_size=10,
            ttl=1,  # 1 seconde
            cache_dir=self.temp_dir,
            persist=False
        )
        
        # Ajouter un élément
        short_ttl_cache.set("expiring_key", "expiring_value")
        
        # Vérifier qu'il est accessible immédiatement
        self.assertEqual(short_ttl_cache.get("expiring_key"), "expiring_value")
        
        # Attendre l'expiration
        time.sleep(1.5)
        
        # Vérifier qu'il a expiré
        self.assertIsNone(short_ttl_cache.get("expiring_key"))
    
    def test_compression(self):
        """Teste la compression des données volumineuses."""
        # Créer des données volumineuses
        large_data = pd.DataFrame({
            'A': np.random.randn(10000),
            'B': np.random.randn(10000)
        })
        
        # Ajouter au cache
        self.cache.set("large_data", large_data)
        
        # Récupérer et vérifier
        retrieved_data = self.cache.get("large_data")
        pd.testing.assert_frame_equal(large_data, retrieved_data)
        
        # Vérifier les statistiques
        stats = self.cache.get_stats()
        self.assertGreater(stats["compressed_items"], 0)
    
    def test_persistence(self):
        """Teste la persistance du cache sur disque."""
        # Ajouter des éléments au cache
        self.cache.set("persistent_key", "persistent_value")
        
        # Créer un nouveau cache avec le même répertoire
        new_cache = SmartCache(
            max_size=10,
            ttl=60,
            cache_dir=self.temp_dir,
            persist=True
        )
        
        # Vérifier que les éléments sont chargés
        self.assertEqual(new_cache.get("persistent_key"), "persistent_value")
    
    def test_preload(self):
        """Teste le préchargement des données."""
        # Définir une fonction de chargement
        def data_loader():
            return "preloaded_value"
        
        # Précharger les données
        self.cache.preload("preload_key", data_loader)
        
        # Vérifier qu'elles sont disponibles
        self.assertEqual(self.cache.get("preload_key"), "preloaded_value")
        
        # Tester le forçage du préchargement
        def updated_loader():
            return "updated_value"
        
        self.cache.preload("preload_key", updated_loader, force=True)
        self.assertEqual(self.cache.get("preload_key"), "updated_value")
    
    def test_access_stats(self):
        """Teste les statistiques d'accès."""
        # Accéder plusieurs fois à une clé
        self.cache.set("popular_key", "popular_value")
        
        for _ in range(5):
            self.cache.get("popular_key")
        
        # Vérifier les statistiques
        stats = self.cache.get_stats()
        popular_keys = dict(stats["popular_keys"])
        
        self.assertIn("popular_key", popular_keys)
        self.assertEqual(popular_keys["popular_key"], 5)


class TestDataCache(unittest.TestCase):
    """Tests pour la classe DataCache."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer des sources de données fictives
        def dummy_data_loader(symbol, start_date, end_date, interval):
            # Générer des données fictives pour la période demandée
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            data = pd.DataFrame({
                'date': date_range,
                'open': np.random.randn(len(date_range)) * 10 + 100,
                'high': np.random.randn(len(date_range)) * 10 + 105,
                'low': np.random.randn(len(date_range)) * 10 + 95,
                'close': np.random.randn(len(date_range)) * 10 + 100,
                'volume': np.random.randint(1000, 10000, len(date_range))
            })
            return data
        
        # Initialiser le cache avec la source de données
        self.data_cache = DataCache(
            max_size=20,
            ttl=3600,
            cache_dir=self.temp_dir,
            persist=True,
            data_sources={"dummy": dummy_data_loader}
        )
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_timeseries(self):
        """Teste la récupération de séries temporelles."""
        # Demander des données
        start_date = datetime.now() - timedelta(days=10)
        end_date = datetime.now()
        
        # Première requête (aucun cache)
        data1 = self.data_cache.get_timeseries(
            symbol="BTC",
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            source="dummy"
        )
        
        # Vérifier que les données sont correctes
        self.assertIsInstance(data1, pd.DataFrame)
        self.assertEqual(len(data1), 11)  # 11 jours de données
        
        # Deuxième requête (devrait utiliser le cache)
        data2 = self.data_cache.get_timeseries(
            symbol="BTC",
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            source="dummy"
        )
        
        # Les données devraient être identiques
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_partial_cache_hit(self):
        """Teste la récupération partielle depuis le cache."""
        # Charger des données initiales
        start_date = datetime.now() - timedelta(days=30)
        middle_date = datetime.now() - timedelta(days=15)
        end_date = datetime.now()
        
        # Charger la première moitié
        self.data_cache.get_timeseries(
            symbol="ETH",
            start_date=start_date,
            end_date=middle_date,
            interval="1d",
            source="dummy"
        )
        
        # Charger la plage complète
        full_data = self.data_cache.get_timeseries(
            symbol="ETH",
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            source="dummy"
        )
        
        # Vérifier que les données sont complètes
        self.assertEqual(len(full_data), 31)  # 31 jours
    
    def test_invalidate(self):
        """Teste l'invalidation des entrées du cache."""
        # Charger des données
        self.data_cache.get_timeseries(
            symbol="BTC",
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now(),
            interval="1d",
            source="dummy"
        )
        
        self.data_cache.get_timeseries(
            symbol="ETH",
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now(),
            interval="1d",
            source="dummy"
        )
        
        # Invalider toutes les entrées BTC
        count = self.data_cache.invalidate("timeseries:BTC:*")
        self.assertEqual(count, 1)
        
        # Vérifier que BTC a été invalidé
        cache_key = "timeseries:BTC:1d:dummy"
        self.assertIsNone(self.data_cache.get(cache_key))
        
        # Vérifier que ETH est toujours là
        cache_key = "timeseries:ETH:1d:dummy"
        self.assertIsNotNone(self.data_cache.get(cache_key))
    
    def test_validate_data(self):
        """Teste la validation de cohérence des données."""
        # Charger des données
        start_date = datetime.now() - timedelta(days=10)
        end_date = datetime.now()
        
        self.data_cache.get_timeseries(
            symbol="BTC",
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            source="dummy"
        )
        
        # Valider les données
        cache_key = "timeseries:BTC:1d:dummy"
        valid, message = self.data_cache.validate_data(cache_key)
        
        self.assertTrue(valid)
        self.assertIsNone(message)
        
        # Corrompre les métadonnées pour tester la validation négative
        self.data_cache.metadata[cache_key]['row_count'] = 999
        
        valid, message = self.data_cache.validate_data(cache_key)
        self.assertFalse(valid)
        self.assertIn("Nombre de lignes incohérent", message)


if __name__ == "__main__":
    unittest.main() 