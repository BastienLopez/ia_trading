#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests pour le module d'optimisation Ray RLlib.
"""

import os
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ai_trading.utils.ray_rllib_optimizer import (
    RayRLlibOptimizer,
    optimize_rl_with_ray,
    HAVE_RAY
)

# Environnement simple pour les tests
class SimpleTestEnv(gym.Env):
    """Environnement simple pour tester Ray RLlib."""
    
    def __init__(self, config=None):
        config = config or {}
        self.max_steps = config.get("max_steps", 100)
        self.current_step = 0
        
        # Espace d'observation: 4 features
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(4,), dtype=np.float32
        )
        
        # Espace d'action: 2 actions discrètes
        self.action_space = spaces.Discrete(2)
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement."""
        super().reset(seed=seed)
        self.current_step = 0
        observation = self.observation_space.sample()
        info = {}
        return observation, info
    
    def step(self, action):
        """Exécute une étape dans l'environnement."""
        self.current_step += 1
        
        # Observation aléatoire
        observation = self.observation_space.sample()
        
        # Récompense simple basée sur l'action
        reward = 1.0 if action == 1 else -0.1
        
        # Fin de l'épisode
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Informations supplémentaires
        info = {"step": self.current_step}
        
        return observation, reward, done, truncated, info


@unittest.skipIf(not HAVE_RAY, "Ray n'est pas installé")
class TestRayRLlibOptimizer(unittest.TestCase):
    """Tests pour l'optimiseur Ray RLlib."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Fonction de création d'environnement
        self.env_creator = lambda config: SimpleTestEnv(config)
        
        # Nous allons patcher la méthode _create_algorithm pour éviter de créer
        # des algorithmes réels et les problèmes d'initialisation de Ray
        self.patcher = patch.object(RayRLlibOptimizer, '_create_algorithm')
        self.mock_create_algorithm = self.patcher.start()
        
        # Mock pour l'algorithme
        self.algorithm_mock = MagicMock()
        self.algorithm_mock.train.return_value = {
            "episode_reward_mean": 10.5,
            "episode_len_mean": 50
        }
        self.algorithm_mock.save.return_value = os.path.join(self.temp_dir, "checkpoint")
        self.algorithm_mock.compute_single_action.return_value = 1
        
        # Configurer le mock pour retourner l'algorithme fictif
        self.mock_create_algorithm.return_value = self.algorithm_mock
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        # Arrêter le patch
        self.patcher.stop()
        
        # Nettoyer les fichiers temporaires
        shutil.rmtree(self.temp_dir)
    
    @patch('ray.init')
    @patch('ray.tune.registry.register_env')
    def test_init(self, mock_register_env, mock_ray_init):
        """Teste l'initialisation de l'optimiseur Ray RLlib."""
        # Configurer les mocks
        mock_ray_init.return_value = None
        mock_register_env.return_value = None
        
        # Créer l'optimiseur
        optimizer = RayRLlibOptimizer(
            env_creator=self.env_creator,
            algorithm="PPO",
            num_workers=2,
            num_gpus=0,
            checkpoint_dir=self.temp_dir
        )
        
        # Vérifier que les attributs sont correctement initialisés
        self.assertEqual(optimizer.env_creator, self.env_creator)
        self.assertEqual(optimizer.algorithm_name, "PPO")
        self.assertEqual(optimizer.num_workers, 2)
        self.assertEqual(optimizer.num_gpus, 0)
        self.assertEqual(optimizer.checkpoint_dir, self.temp_dir)
        
        # Vérifier que ray.init a été appelé
        mock_ray_init.assert_called_once()
        
        # Vérifier que register_env a été appelé
        # Comme nous avons patché _create_algorithm, register_env n'est pas nécessairement appelé
        # dans ce test, mais cela n'affecte pas notre test de l'initialisation correcte
    
    @patch('ray.init')
    @patch('ray.tune.registry.register_env')
    def test_create_base_config(self, mock_register_env, mock_ray_init):
        """Teste la création de la configuration de base."""
        # Configurer les mocks
        mock_ray_init.return_value = None
        mock_register_env.return_value = None
        
        # Créer l'optimiseur
        optimizer = RayRLlibOptimizer(
            env_creator=self.env_creator,
            algorithm="PPO",
            num_workers=2,
            num_gpus=0.5,
            framework="torch",
            checkpoint_dir=self.temp_dir
        )
        
        # Créer la configuration de base
        config = optimizer._create_base_config(
            num_workers=2,
            num_gpus=0.5,
            num_envs_per_worker=4,
            framework="torch",
            seed=42
        )
        
        # Vérifier les clés de la configuration
        self.assertIn("env", config)
        self.assertIn("num_workers", config)
        self.assertIn("num_gpus", config)
        self.assertIn("num_envs_per_env_runner", config)
        self.assertIn("framework", config)
        self.assertIn("log_level", config)
        self.assertIn("gamma", config)
        self.assertIn("lr", config)
        
        # Vérifier les valeurs spécifiques
        self.assertEqual(config["num_workers"], 2)
        self.assertEqual(config["num_gpus"], 0.5)
        self.assertEqual(config["num_envs_per_env_runner"], 4)
        self.assertEqual(config["framework"], "torch")
        self.assertEqual(config["seed"], 42)
        
        # Vérifier les optimisations spécifiques au framework torch
        self.assertTrue(config["use_pytorch"])
        self.assertIn("torch_optimizer_options", config)
    
    @patch('ray.init')
    @patch('ray.tune.registry.register_env')
    def test_train(self, mock_register_env, mock_ray_init):
        """Teste l'entraînement de l'agent."""
        # Configurer les mocks
        mock_ray_init.return_value = None
        mock_register_env.return_value = None
        
        # Créer l'optimiseur
        optimizer = RayRLlibOptimizer(
            env_creator=self.env_creator,
            algorithm="PPO",
            num_workers=1,
            checkpoint_dir=self.temp_dir,
            checkpoint_freq=2
        )
        
        # Entraîner l'agent
        results = optimizer.train(num_iterations=3)
        
        # Vérifier que train a été appelé 3 fois
        self.assertEqual(self.algorithm_mock.train.call_count, 3)
        
        # Vérifier que save a été appelé 2 fois (1 à l'itération 2 et 1 à la fin)
        self.assertEqual(self.algorithm_mock.save.call_count, 2)
        
        # Vérifier les résultats
        self.assertIn("iteration_1", results)
        self.assertIn("iteration_2", results)
        self.assertIn("iteration_3", results)
        self.assertIn("final_checkpoint", results)
    
    @patch('ray.init')
    @patch('ray.tune.registry.register_env')
    def test_evaluate(self, mock_register_env, mock_ray_init):
        """Teste l'évaluation de l'agent."""
        # Configurer les mocks
        mock_ray_init.return_value = None
        mock_register_env.return_value = None
        
        # Créer un mock pour l'environnement d'évaluation
        env_mock = MagicMock()
        env_mock.reset.return_value = (np.array([0, 0, 0, 0], dtype=np.float32), {})
        env_mock.step.return_value = (np.array([0, 0, 0, 0], dtype=np.float32), 1.0, False, False, {})
        
        # Après 5 étapes, faire comme si l'épisode était terminé
        def step_side_effect(action):
            env_mock.step.side_effect = None  # Réinitialiser l'effet de côté
            return (np.array([0, 0, 0, 0], dtype=np.float32), 1.0, True, False, {})
        
        env_mock.step.side_effect = [
            (np.array([0, 0, 0, 0], dtype=np.float32), 1.0, False, False, {}),
            (np.array([0, 0, 0, 0], dtype=np.float32), 1.0, False, False, {}),
            (np.array([0, 0, 0, 0], dtype=np.float32), 1.0, False, False, {}),
            (np.array([0, 0, 0, 0], dtype=np.float32), 1.0, False, False, {}),
            (np.array([0, 0, 0, 0], dtype=np.float32), 1.0, True, False, {})
        ]
        
        # Modifier la fonction env_creator pour retourner le mock
        env_creator_mock = MagicMock(return_value=env_mock)
        
        # Créer l'optimiseur
        optimizer = RayRLlibOptimizer(
            env_creator=env_creator_mock,
            algorithm="PPO",
            num_workers=1
        )
        
        # Injecter manuellement le mock de l'algorithme dans l'optimiseur pour le test
        optimizer.agent = self.algorithm_mock
        
        # Évaluer l'agent
        eval_results = optimizer.evaluate(num_episodes=1)
        
        # Vérifier que les méthodes d'environnement ont été appelées
        env_mock.reset.assert_called_once()
        self.assertEqual(env_mock.step.call_count, 5)
        
        # Vérifier que compute_single_action a été appelé
        self.assertEqual(self.algorithm_mock.compute_single_action.call_count, 5)
        
        # Vérifier les résultats
        self.assertIn("mean_reward", eval_results)
        self.assertIn("std_reward", eval_results)
        self.assertIn("max_reward", eval_results)
        self.assertIn("min_reward", eval_results)
        self.assertIn("mean_episode_length", eval_results)
        self.assertIn("total_rewards", eval_results)
        self.assertIn("episode_lengths", eval_results)
    
    @patch('ray.init')
    @patch('ray.tune.registry.register_env')
    def test_save_load_model(self, mock_register_env, mock_ray_init):
        """Teste la sauvegarde et le chargement du modèle."""
        # Configurer les mocks
        mock_ray_init.return_value = None
        mock_register_env.return_value = None
        
        # Créer l'optimiseur
        optimizer = RayRLlibOptimizer(
            env_creator=self.env_creator,
            algorithm="PPO",
            num_workers=1
        )
        
        # Créer un chemin de sauvegarde
        save_path = os.path.join(self.temp_dir, "test_model")
        
        # Sauvegarder le modèle
        checkpoint_path = optimizer.save_model(save_path)
        
        # Vérifier que save a été appelé
        self.algorithm_mock.save.assert_called_once_with(save_path)
        
        # Charger le modèle
        optimizer.load_model(checkpoint_path)
        
        # Vérifier que restore a été appelé
        self.algorithm_mock.restore.assert_called_once_with(checkpoint_path)
    
    @patch('ray.init')
    @patch('ray.tune.registry.register_env')
    def test_get_policy(self, mock_register_env, mock_ray_init):
        """Teste la récupération de la politique."""
        # Configurer les mocks
        mock_ray_init.return_value = None
        mock_register_env.return_value = None
        
        policy_mock = MagicMock()
        self.algorithm_mock.get_policy.return_value = policy_mock
        
        # Créer l'optimiseur
        optimizer = RayRLlibOptimizer(
            env_creator=self.env_creator,
            algorithm="PPO",
            num_workers=1
        )
        
        # Récupérer la politique
        policy = optimizer.get_policy()
        
        # Vérifier que get_policy a été appelé
        self.algorithm_mock.get_policy.assert_called_once()
        
        # Vérifier que la politique est correcte
        self.assertEqual(policy, policy_mock)


@unittest.skipIf(not HAVE_RAY, "Ray n'est pas installé")
class TestOptimizeRLWithRay(unittest.TestCase):
    """Tests pour la fonction optimize_rl_with_ray."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Fonction de création d'environnement
        self.env_creator = lambda config: SimpleTestEnv(config)
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    @patch('ray.init')
    @patch('ray.tune.registry.register_env')
    def test_optimize_rl_with_ray(self, mock_register_env, mock_ray_init):
        """Teste la fonction utilitaire optimize_rl_with_ray."""
        # Configurer les mocks
        mock_ray_init.return_value = None
        mock_register_env.return_value = None
        
        # Créer un mock pour RayRLlibOptimizer
        optimizer_mock = MagicMock()
        optimizer_mock.train.return_value = {"training": "results"}
        optimizer_mock.evaluate.return_value = {"evaluation": "results"}
        
        with patch('ai_trading.utils.ray_rllib_optimizer.RayRLlibOptimizer', return_value=optimizer_mock):
            
            # Appeler la fonction
            results = optimize_rl_with_ray(
                env_creator=self.env_creator,
                algorithm="PPO",
                num_workers=1,
                num_gpus=0,
                train_iterations=5
            )
            
            # Vérifier que les méthodes ont été appelées
            optimizer_mock.train.assert_called_once_with(num_iterations=5)
            optimizer_mock.evaluate.assert_called_once_with(num_episodes=10)
            optimizer_mock.cleanup.assert_called_once()
            
            # Vérifier les résultats
            self.assertIn("train_results", results)
            self.assertIn("eval_results", results)
            self.assertIn("optimizer", results)
            self.assertEqual(results["train_results"], {"training": "results"})
            self.assertEqual(results["eval_results"], {"evaluation": "results"})
            self.assertEqual(results["optimizer"], optimizer_mock)


if __name__ == '__main__':
    unittest.main() 