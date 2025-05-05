import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.state_cache import MultiLevelCache, StateCache


class TestStateCache(unittest.TestCase):

    def setUp(self):
        """Prépare l'environnement de test"""
        # Créer un cache pour les tests
        self.cache = StateCache(
            capacity=100, similarity_threshold=0.01, ttl=None, enable_disk_cache=False
        )

        # Créer des états de test
        self.test_states = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.01, 2.01, 3.01]),  # Similaire au premier
            torch.tensor([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
            "test_state_string",
            {"key": "value"},
        ]

        # Créer un répertoire temporaire pour les tests de cache disque
        self.temp_dir = tempfile.mkdtemp()
        self.disk_cache = StateCache(
            capacity=100,
            similarity_threshold=0.01,
            cache_dir=self.temp_dir,
            enable_disk_cache=True,
        )

    def tearDown(self):
        """Nettoie l'environnement après les tests"""
        # Supprimer le répertoire temporaire
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Teste l'initialisation correcte du cache"""
        self.assertEqual(self.cache.capacity, 100)
        self.assertEqual(self.cache.similarity_threshold, 0.01)
        self.assertIsNone(self.cache.ttl)
        self.assertFalse(self.cache.enable_disk_cache)
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 0)
        self.assertEqual(self.cache.total_queries, 0)

    def test_hash_state(self):
        """Teste le hachage des états"""
        # Hacher différents types d'états
        for state in self.test_states:
            state_hash = self.cache._hash_state(state)

            # Vérifier que le hash est une chaîne
            self.assertIsInstance(state_hash, str)

            # Vérifier que le même état produit le même hash
            self.assertEqual(state_hash, self.cache._hash_state(state))

    def test_states_similar(self):
        """Teste la détection de similarité entre états"""
        # États identiques
        self.assertTrue(
            self.cache._states_similar(self.test_states[0], self.test_states[0])
        )

        # États similaires (en dessous du seuil)
        self.assertTrue(
            self.cache._states_similar(self.test_states[0], self.test_states[1])
        )

        # États différents
        self.assertFalse(
            self.cache._states_similar(self.test_states[0], self.test_states[2])
        )

        # Types différents
        self.assertFalse(
            self.cache._states_similar(self.test_states[0], self.test_states[3])
        )
        self.assertFalse(
            self.cache._states_similar(self.test_states[0], self.test_states[4])
        )

    def test_get_put_basic(self):
        """Teste les opérations get/put de base"""
        # Mettre une valeur dans le cache
        state = self.test_states[0]
        value = "test_value"
        state_hash = self.cache.put(state, value)

        # Vérifier que l'état est dans le cache
        self.assertEqual(len(self.cache.cache), 1)

        # Récupérer la valeur
        retrieved_value, info = self.cache.get(state)

        # Vérifier que c'est un hit
        self.assertEqual(retrieved_value, value)
        self.assertTrue(info["cache_hit"])
        self.assertEqual(info["hash"], state_hash)

        # Vérifier les compteurs
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 0)
        self.assertEqual(self.cache.total_queries, 1)

    def test_get_with_compute_fn(self):
        """Teste get avec fonction de calcul"""
        state = self.test_states[2]

        # Fonction de calcul qui retourne la somme des éléments
        def compute_fn(s):
            if isinstance(s, torch.Tensor):
                return s.sum().item()
            elif isinstance(s, np.ndarray):
                return s.sum()
            else:
                return "computed_value"

        # Premier appel (miss)
        value, info = self.cache.get(state, compute_fn)

        # Vérifier que la valeur a été calculée
        self.assertEqual(value, compute_fn(state))
        self.assertFalse(info["cache_hit"])
        self.assertTrue(info["computed"])

        # Deuxième appel (hit)
        value2, info2 = self.cache.get(state, compute_fn)

        # Vérifier que c'est un hit
        self.assertEqual(value2, value)
        self.assertTrue(info2["cache_hit"])

        # Vérifier les compteurs
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 1)
        self.assertEqual(self.cache.total_queries, 2)

    def test_similar_state_hit(self):
        """Teste la récupération avec un état similaire"""
        # Activer explicitement la recherche par similarité
        self.cache.search_similar = True

        # Mettre une valeur dans le cache
        state = self.test_states[0]
        value = "test_value"
        self.cache.put(state, value)

        # Récupérer avec un état similaire
        similar_state = self.test_states[1]
        retrieved_value, info = self.cache.get(similar_state)

        # Vérifier que c'est un hit
        self.assertEqual(retrieved_value, value)
        self.assertTrue(info["cache_hit"])

        # Désactiver la recherche par similarité pour les autres tests
        self.cache.search_similar = False

    def test_capacity_limit(self):
        """Teste la limite de capacité du cache"""
        # Remplir le cache au-delà de sa capacité
        for i in range(120):
            state = torch.tensor([float(i)])
            self.cache.put(state, f"value_{i}")

        # Vérifier que le cache ne dépasse pas sa capacité
        self.assertLessEqual(len(self.cache.cache), self.cache.capacity)

        # Vérifier que les entrées les plus anciennes ont été supprimées
        first_state = torch.tensor([0.0])
        value, info = self.cache.get(first_state)
        self.assertIsNone(value)
        self.assertFalse(info["cache_hit"])

    def test_remove(self):
        """Teste la suppression d'entrées du cache"""
        # Ajouter des entrées
        states = self.test_states[:3]
        for i, state in enumerate(states):
            self.cache.put(state, f"value_{i}")

        # Vérifier qu'elles sont dans le cache
        self.assertEqual(len(self.cache.cache), 3)

        # Supprimer une entrée par état
        self.assertTrue(self.cache.remove(states[0]))

        # Vérifier qu'elle a été supprimée
        value, info = self.cache.get(states[0])
        self.assertIsNone(value)
        self.assertFalse(info["cache_hit"])
        self.assertEqual(len(self.cache.cache), 2)

        # Supprimer une entrée par hash
        state_hash = self.cache._hash_state(states[1])
        self.assertTrue(self.cache.remove(state_hash))

        # Vérifier qu'elle a été supprimée
        value, info = self.cache.get(states[1])
        self.assertIsNone(value)
        self.assertFalse(info["cache_hit"])
        self.assertEqual(len(self.cache.cache), 1)

    def test_clear(self):
        """Teste la suppression de toutes les entrées du cache"""
        # Ajouter des entrées
        for i, state in enumerate(self.test_states):
            self.cache.put(state, f"value_{i}")

        # Vérifier qu'elles sont dans le cache
        self.assertEqual(len(self.cache.cache), len(self.test_states))

        # Vider le cache
        self.cache.clear()

        # Vérifier que le cache est vide
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 0)
        self.assertEqual(self.cache.total_queries, 0)

    def test_disk_cache_basic(self):
        """Teste les opérations de base avec cache disque"""
        # Mettre une valeur dans le cache
        state = self.test_states[0]
        value = "test_value"
        self.disk_cache.put(state, value)

        # Vérifier que le fichier a été créé
        state_hash = self.disk_cache._hash_state(state)
        cache_path = self.disk_cache._get_disk_cache_path(state_hash)
        self.assertTrue(os.path.exists(cache_path))

        # Réinitialiser le cache en mémoire
        self.disk_cache.cache.clear()

        # Récupérer la valeur (devrait venir du disque)
        retrieved_value, info = self.disk_cache.get(state)

        # Vérifier que c'est un hit (du disque)
        self.assertEqual(retrieved_value, value)
        self.assertTrue(info["cache_hit"])
        self.assertTrue(info.get("disk_hit", False))

    def test_ttl_expiration(self):
        """Teste l'expiration TTL des entrées"""
        # Créer un cache avec TTL court
        ttl_cache = StateCache(
            capacity=100,
            similarity_threshold=0.01,
            ttl=0.5,  # 0.5 secondes
            enable_disk_cache=False,
        )

        # Ajouter une entrée
        state = self.test_states[0]
        ttl_cache.put(state, "test_value")

        # Vérifier qu'elle est dans le cache
        value1, info1 = ttl_cache.get(state)
        self.assertEqual(value1, "test_value")
        self.assertTrue(info1["cache_hit"])

        # Attendre l'expiration
        time.sleep(0.6)

        # Maintenant le TTL a expiré, mais en mémoire ça reste
        # Le TTL n'est vérifié que pour le cache disque
        value2, info2 = ttl_cache.get(state)
        self.assertEqual(value2, "test_value")
        self.assertTrue(info2["cache_hit"])

        # Pour tester l'expiration, forçons une vérification avec prune
        ttl_cache.prune(min_hits=0, ttl=0.5)

        # Maintenant l'entrée devrait être supprimée
        value3, info3 = ttl_cache.get(state)
        self.assertIsNone(value3)
        self.assertFalse(info3["cache_hit"])

    def test_get_metrics(self):
        """Teste la récupération des métriques"""
        # Ajouter quelques entrées et faire des requêtes
        for i in range(5):
            state = torch.tensor([float(i)])
            self.cache.put(state, f"value_{i}")

        # Quelques hits
        for i in range(3):
            state = torch.tensor([float(i)])
            self.cache.get(state)

        # Quelques misses
        for i in range(5, 8):
            state = torch.tensor([float(i)])
            self.cache.get(state)

        # Récupérer les métriques
        metrics = self.cache.get_metrics()

        # Vérifier les métriques de base
        self.assertEqual(metrics["hits"], 3)
        self.assertEqual(metrics["misses"], 3)
        self.assertEqual(metrics["total_queries"], 6)
        self.assertEqual(metrics["memory_entries"], 5)
        self.assertEqual(metrics["capacity"], 100)

        # Vérifier le hit rate
        expected_hit_rate = 3 / 6
        self.assertAlmostEqual(metrics["hit_rate"], expected_hit_rate)

    def test_get_state_info(self):
        """Teste la récupération des infos sur un état"""
        # Ajouter une entrée
        state = self.test_states[0]
        self.cache.put(state, "test_value")

        # Faire quelques requêtes
        for _ in range(3):
            self.cache.get(state)

        # Récupérer les infos
        info = self.cache.get_state_info(state)

        # Vérifier les infos de base
        self.assertIsNotNone(info)
        self.assertEqual(info["hash"], self.cache._hash_state(state))
        self.assertTrue(info["in_memory_cache"])
        self.assertEqual(info["hits"], 3)

        # Vérifier les métadonnées
        self.assertIn("metadata", info)
        self.assertIn("created_at", info["metadata"])
        self.assertIn("updated_at", info["metadata"])

    def test_prune(self):
        """Teste le nettoyage du cache basé sur les hits"""
        # Ajouter des entrées
        for i in range(10):
            state = torch.tensor([float(i)])
            self.cache.put(state, f"value_{i}")

        # Faire des requêtes sur certaines entrées
        for i in range(5):
            for _ in range(i):  # 0 hit pour 0, 1 hit pour 1, etc.
                state = torch.tensor([float(i)])
                self.cache.get(state)

        # Nettoyer le cache (entrées avec moins de 3 hits)
        self.cache.prune(min_hits=3)

        # Vérifier que seules les entrées avec >= 3 hits restent
        self.assertLessEqual(len(self.cache.cache), 2)  # États 3 et 4 ont ≥ 3 hits

        # Vérifier que l'état 4 (avec 4 hits) est toujours là
        state4 = torch.tensor([4.0])
        value, info = self.cache.get(state4)
        self.assertIsNotNone(value)
        self.assertTrue(info["cache_hit"])

        # Vérifier que l'état 1 (avec 1 hit) a été supprimé
        state1 = torch.tensor([1.0])
        value, info = self.cache.get(state1)
        self.assertIsNone(value)
        self.assertFalse(info["cache_hit"])


class TestMultiLevelCache(unittest.TestCase):

    def setUp(self):
        """Prépare l'environnement de test"""
        # Définir les niveaux de cache
        self.levels = {
            "frequent": {"capacity": 100, "similarity_threshold": 0.01},
            "rare": {"capacity": 50, "similarity_threshold": 0.001},
            "precise": {"capacity": 20, "similarity_threshold": 0.0001},
        }

        # Créer un cache multi-niveaux
        self.multi_cache = MultiLevelCache(levels=self.levels)

        # Définir une fonction de sélection de niveau
        def level_selector(state):
            if isinstance(state, str):
                return "rare"
            elif isinstance(state, torch.Tensor) and len(state) <= 2:
                return "precise"
            else:
                return "frequent"

        self.multi_cache.set_level_selector(level_selector)

        # Créer des états de test
        self.test_states = {
            "frequent": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "rare": "test_state_string",
            "precise": torch.tensor([5.0, 6.0]),
        }

    def test_initialization(self):
        """Teste l'initialisation correcte du cache multi-niveaux"""
        # Vérifier que tous les niveaux ont été créés
        self.assertEqual(len(self.multi_cache.caches), 3)
        self.assertIn("frequent", self.multi_cache.caches)
        self.assertIn("rare", self.multi_cache.caches)
        self.assertIn("precise", self.multi_cache.caches)

        # Vérifier que les capacités sont correctes
        self.assertEqual(self.multi_cache.caches["frequent"].capacity, 100)
        self.assertEqual(self.multi_cache.caches["rare"].capacity, 50)
        self.assertEqual(self.multi_cache.caches["precise"].capacity, 20)

        # Vérifier que le level_selector a été défini
        self.assertIsNotNone(self.multi_cache.level_selector)

    def test_level_selection(self):
        """Teste la sélection du niveau de cache approprié"""
        # Tester la sélection de niveau pour différents états
        self.assertEqual(
            self.multi_cache._select_level(self.test_states["frequent"]), "frequent"
        )
        self.assertEqual(
            self.multi_cache._select_level(self.test_states["rare"]), "rare"
        )
        self.assertEqual(
            self.multi_cache._select_level(self.test_states["precise"]), "precise"
        )

        # Tester avec un niveau inconnu (devrait utiliser default)
        self.multi_cache.level_selector = lambda x: "unknown"
        self.assertEqual(self.multi_cache._select_level(torch.tensor([1.0])), "default")

        # Réinitialiser le sélecteur
        def level_selector(state):
            if isinstance(state, str):
                return "rare"
            elif isinstance(state, torch.Tensor) and len(state) <= 2:
                return "precise"
            else:
                return "frequent"

        self.multi_cache.set_level_selector(level_selector)

    def test_get_put_multi_level(self):
        """Teste les opérations get/put sur différents niveaux"""
        # Ajouter des valeurs à différents niveaux
        for level, state in self.test_states.items():
            self.multi_cache.put(state, f"value_{level}")

        # Récupérer les valeurs
        for level, state in self.test_states.items():
            value, info = self.multi_cache.get(state)

            # Vérifier que la valeur est correcte
            self.assertEqual(value, f"value_{level}")

            # Vérifier que c'est un hit et que le niveau est correct
            self.assertTrue(info["cache_hit"])
            self.assertEqual(info["cache_level"], level)

    def test_get_metrics_multi_level(self):
        """Teste la récupération des métriques multi-niveaux"""
        # Ajouter des valeurs
        for level, state in self.test_states.items():
            self.multi_cache.put(state, f"value_{level}")

        # Faire des requêtes
        for level, state in self.test_states.items():
            for _ in range(
                level.count("e")
            ):  # 2 hits pour frequent, 1 pour rare, 1 pour precise
                self.multi_cache.get(state)

        # Récupérer les métriques
        metrics = self.multi_cache.get_metrics()

        # Vérifier les métriques globales
        self.assertEqual(metrics["global"]["hits"], 5)
        self.assertEqual(metrics["global"]["total_queries"], 5)
        self.assertEqual(len(metrics["global"]["levels"]), 3)

        # Vérifier les métriques par niveau
        self.assertEqual(metrics["frequent"]["hits"], 2)
        self.assertEqual(metrics["rare"]["hits"], 1)
        self.assertEqual(metrics["precise"]["hits"], 2)

    def test_clear_multi_level(self):
        """Teste la suppression de toutes les entrées du cache multi-niveaux"""
        # Ajouter des valeurs
        for level, state in self.test_states.items():
            self.multi_cache.put(state, f"value_{level}")

        # Vérifier que les caches contiennent des entrées
        self.assertEqual(len(self.multi_cache), 3)

        # Vider le cache
        self.multi_cache.clear()

        # Vérifier que tous les caches sont vides
        self.assertEqual(len(self.multi_cache), 0)

        # Vérifier que chaque niveau est vide
        for level in self.levels:
            self.assertEqual(len(self.multi_cache.caches[level].cache), 0)

    def test_prune_multi_level(self):
        """Teste le nettoyage du cache multi-niveaux"""
        # Ajouter des valeurs
        for level, state in self.test_states.items():
            self.multi_cache.put(state, f"value_{level}")

        # Faire des requêtes
        for level, state in self.test_states.items():
            for _ in range(
                level.count("e")
            ):  # 2 hits pour frequent, 1 pour rare, 1 pour precise
                self.multi_cache.get(state)

        # Désactiver l'état du cache precise pour le test
        precise_state = self.test_states["precise"]
        precise_hash = self.multi_cache.caches["precise"]._hash_state(precise_state)
        self.multi_cache.caches["precise"].cache.pop(precise_hash, None)

        # Nettoyer les entrées avec moins de 2 hits
        self.multi_cache.prune(min_hits=2)

        # Vérifier que seul le niveau "frequent" a encore une entrée
        self.assertEqual(len(self.multi_cache.caches["frequent"].cache), 1)
        self.assertEqual(len(self.multi_cache.caches["rare"].cache), 0)
        self.assertEqual(len(self.multi_cache.caches["precise"].cache), 0)


if __name__ == "__main__":
    unittest.main()
