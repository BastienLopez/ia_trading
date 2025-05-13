"""
Tests pour le cache distribué amélioré et le préchargement intelligent.
"""

import json
from unittest.mock import patch

import pytest

from ai_trading.utils.enhanced_cache import AccessPattern, EnhancedDistributedCache


@pytest.fixture
def access_pattern():
    """Fixture pour l'analyseur de modèles d'accès."""
    return AccessPattern(max_history=20, time_window=60)


@pytest.fixture
def mock_redis():
    """Mock pour Redis."""
    with patch("redis.Redis") as mock:
        # Simuler le comportement du cache Redis
        mock_instance = mock.return_value

        # Dictionnaire pour stocker les données du cache simulé
        mock_data = {}

        def mock_get(key):
            return mock_data.get(key)

        def mock_setex(key, ttl, value):
            mock_data[key] = value

        def mock_delete(key):
            if key in mock_data:
                del mock_data[key]

        # Configuration des mocks
        mock_instance.get.side_effect = mock_get
        mock_instance.setex.side_effect = mock_setex
        mock_instance.delete.side_effect = mock_delete

        yield mock


@pytest.fixture
def mock_redis_cluster():
    """Mock pour RedisCluster."""
    with patch("ai_trading.utils.enhanced_cache.RedisCluster") as mock:
        # Simuler le même comportement que le mock Redis
        mock_instance = mock.return_value

        # Dictionnaire pour stocker les données du cache simulé
        mock_data = {}

        def mock_get(key):
            return mock_data.get(key)

        def mock_setex(key, ttl, value):
            mock_data[key] = value

        def mock_delete(key):
            if key in mock_data:
                del mock_data[key]

        # Configuration des mocks
        mock_instance.get.side_effect = mock_get
        mock_instance.setex.side_effect = mock_setex
        mock_instance.delete.side_effect = mock_delete

        yield mock


@pytest.fixture
def enhanced_cache(mock_redis):
    """Fixture pour le cache distribué amélioré."""
    cache = EnhancedDistributedCache(
        host="localhost", port=6379, db=0, use_cluster=False, prefetch_enabled=True
    )
    return cache


@pytest.fixture
def enhanced_cache_cluster(mock_redis_cluster):
    """Fixture pour le cache distribué amélioré en mode cluster."""
    startup_nodes = [{"host": "127.0.0.1", "port": 7000}]
    return EnhancedDistributedCache(
        startup_nodes=startup_nodes, use_cluster=True, prefetch_enabled=True
    )


class TestAccessPattern:
    """Tests pour l'analyseur de modèles d'accès."""

    def test_record_access(self, access_pattern):
        """Test de l'enregistrement des accès."""
        # Enregistrer quelques accès
        access_pattern.record_access("key1")
        access_pattern.record_access("key2")
        access_pattern.record_access("key1")

        # Vérifier que l'historique est correct
        assert len(access_pattern.access_history) == 3
        assert access_pattern.access_history[0][1] == "key1"
        assert access_pattern.access_history[1][1] == "key2"
        assert access_pattern.access_history[2][1] == "key1"

        # Vérifier que les modèles de séquence sont corrects
        assert access_pattern.sequence_patterns["key1"]["key2"] == 1
        assert access_pattern.sequence_patterns["key2"]["key1"] == 1

    def test_get_frequent_patterns(self, access_pattern):
        """Test de la récupération des modèles fréquents."""
        # Créer un modèle d'accès fréquent: key1 -> key2 -> key3 -> key1 -> key2 -> key3
        for _ in range(3):
            access_pattern.record_access("key1")
            access_pattern.record_access("key2")
            access_pattern.record_access("key3")

        # Récupérer les modèles fréquents
        patterns = access_pattern.get_frequent_patterns(min_count=2)

        # Vérifier que les modèles sont corrects
        assert "key1" in patterns
        assert "key2" in patterns
        assert "key3" in patterns

        # key1 est suivi par key2 3 fois
        assert ("key2", 3) in patterns["key1"]
        # key2 est suivi par key3 3 fois
        assert ("key3", 3) in patterns["key2"]
        # key3 est suivi par key1 2 fois (le dernier accès n'a pas de suite)
        assert ("key1", 2) in patterns["key3"]

    def test_predict_next_accesses(self, access_pattern):
        """Test de la prédiction des prochains accès."""
        # Créer un modèle d'accès: key1 -> key2 -> key3, key1 -> key4
        access_pattern.record_access("key1")
        access_pattern.record_access("key2")
        access_pattern.record_access("key3")
        access_pattern.record_access("key1")
        access_pattern.record_access("key4")

        # Prédire les prochains accès après key1
        predictions = access_pattern.predict_next_accesses("key1", limit=2)

        # key2 et key4 devraient être prédits
        assert len(predictions) == 2
        assert "key2" in predictions
        assert "key4" in predictions

    def test_get_hot_keys(self, access_pattern):
        """Test de la récupération des clés les plus accédées."""
        # Enregistrer des accès avec des fréquences différentes
        for _ in range(5):
            access_pattern.record_access("key1")

        for _ in range(3):
            access_pattern.record_access("key2")

        for _ in range(1):
            access_pattern.record_access("key3")

        # Récupérer les clés les plus accédées
        hot_keys = access_pattern.get_hot_keys(limit=2)

        # Vérifier que les clés sont triées par fréquence
        assert len(hot_keys) == 2
        assert hot_keys[0][0] == "key1"
        assert hot_keys[0][1] == 5
        assert hot_keys[1][0] == "key2"
        assert hot_keys[1][1] == 3


class TestEnhancedDistributedCache:
    """Tests pour le cache distribué amélioré."""

    def test_get_set(self, enhanced_cache, mock_redis):
        """Test de get et set."""
        # Préparer les données
        key = "test_key"
        value = {"key": "value"}
        encoded_value = (
            json.dumps(value).encode()
            if isinstance(json.dumps(value), str)
            else json.dumps(value)
        )

        # Simuler un miss puis un hit
        mock_redis.return_value.get.return_value = None
        result1 = enhanced_cache.get(key)
        assert result1 is None

        # Set la valeur
        enhanced_cache.set(key, value)

        # Simuler un hit
        mock_redis.return_value.get.return_value = encoded_value
        result2 = enhanced_cache.get(key)

        # Vérifier les résultats
        assert result2 == value

        # Vérifier les statistiques
        assert enhanced_cache.hits == 1
        assert enhanced_cache.misses == 1

    def test_delete(self, enhanced_cache, mock_redis):
        """Test de delete."""
        # Préparer les données
        key = "test_key"

        # Delete la clé
        enhanced_cache.delete(key)

        # Vérifier que delete a été appelé
        mock_redis.return_value.delete.assert_any_call(key)
        mock_redis.return_value.delete.assert_any_call(f"stats:{key}")

    def test_prefetching_keys_management(self, enhanced_cache):
        """Test de la gestion des clés de préchargement."""
        # Tester l'ajout et la suppression des clés dans le set prefetching_keys
        test_key = "test_prefetch_key"

        # Vérifier que le set est initialement vide
        assert test_key not in enhanced_cache.prefetching_keys

        # Ajouter une clé
        enhanced_cache.prefetching_keys.add(test_key)
        assert test_key in enhanced_cache.prefetching_keys

        # Supprimer la clé
        enhanced_cache.prefetching_keys.remove(test_key)
        assert test_key not in enhanced_cache.prefetching_keys

    def test_mget_mset(self, enhanced_cache, mock_redis):
        """Test de mget et mset."""
        # Préparer les données
        keys = ["key1", "key2"]
        values = {"key1": {"data": 1}, "key2": {"data": 2}}

        # Mock pour mget
        def mock_mget(keys_list):
            return [
                json.dumps(values.get(k, None)).encode() if k in values else None
                for k in keys_list
            ]

        mock_redis.return_value.mget.side_effect = mock_mget

        # Test mset
        enhanced_cache.mset(values)

        # Test mget
        results = enhanced_cache.mget(keys)

        # Vérifier les résultats
        assert len(results) == 2
        assert results["key1"] == values["key1"]
        assert results["key2"] == values["key2"]

    def test_stats(self, enhanced_cache):
        """Test des statistiques."""
        # Simuler des hits et misses
        enhanced_cache.hits = 10
        enhanced_cache.misses = 5
        enhanced_cache.prefetch_hits = 3

        # Récupérer les statistiques
        stats = enhanced_cache.get_stats()

        # Vérifier les statistiques
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_ratio"] == 10 / 15
        assert stats["prefetch_hits"] == 3
        assert stats["prefetch_ratio"] == 3 / 10

    def test_cluster_mode(self, enhanced_cache_cluster, mock_redis_cluster):
        """Test du mode cluster."""
        # Vérifier que le mode cluster est activé
        assert enhanced_cache_cluster.use_cluster is True

        # Vérifier que RedisCluster a été appelé avec les bons paramètres
        mock_redis_cluster.assert_called_once()


class TestEnhancedDistributedCacheFallback:
    """Tests pour le fallback en cas d'erreur de connexion au cluster."""

    def test_cluster_fallback(self):
        """Test du fallback vers Redis standard en cas d'erreur."""
        with patch("ai_trading.utils.enhanced_cache.RedisCluster") as mock_cluster:
            # Simuler une erreur de connexion au cluster
            mock_cluster.side_effect = Exception("Erreur de connexion")

            with patch("redis.Redis") as mock_redis:
                # Créer le cache
                cache = EnhancedDistributedCache(
                    startup_nodes=[{"host": "127.0.0.1", "port": 7000}],
                    use_cluster=True,
                    prefetch_enabled=True,
                )

                # Vérifier que le fallback a été activé
                assert cache.use_cluster is False
                mock_redis.assert_called_once()
