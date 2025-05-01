#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests pour le module d'optimisation système.
"""

import os
import sys
import unittest
import tempfile
import json
import logging
import platform
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ajout du répertoire parent au path pour importer les modules du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_trading.utils.system_optimizer import SystemOptimizer, optimize_system


class TestSystemOptimizer(unittest.TestCase):
    """Tests pour la classe SystemOptimizer."""
    
    def setUp(self):
        """Initialiser les éléments nécessaires pour chaque test."""
        # Configuration de test dans un fichier temporaire
        self.test_config = {
            "env_vars": {
                "TEST_VAR": "test_value",
                "PYTHONHASHSEED": "1"
            },
            "system_limits": {
                "file_limit": 1024,
                "process_limit": 256,
                "memory_limit_mb": 0
            },
            "logging": {
                "level": "DEBUG",
                "max_file_size_mb": 5,
                "rotation_count": 3,
                "use_json_format": True
            }
        }
        
        # Créer un fichier de configuration temporaire
        self.temp_config_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        json.dump(self.test_config, self.temp_config_file)
        self.temp_config_file.close()
    
    def tearDown(self):
        """Nettoyer après chaque test."""
        # Supprimer le fichier temporaire
        if hasattr(self, 'temp_config_file'):
            os.unlink(self.temp_config_file.name)
    
    def test_init_with_default_config(self):
        """Tester l'initialisation avec la configuration par défaut."""
        optimizer = SystemOptimizer()
        self.assertIsNotNone(optimizer.system_info)
        self.assertIsNotNone(optimizer.config)
        self.assertEqual(optimizer.applied_optimizations, {})
    
    def test_init_with_custom_config(self):
        """Tester l'initialisation avec une configuration personnalisée."""
        optimizer = SystemOptimizer(self.temp_config_file.name)
        
        # Vérifier que la configuration a été correctement chargée
        self.assertEqual(optimizer.config["env_vars"]["TEST_VAR"], "test_value")
        self.assertEqual(optimizer.config["system_limits"]["file_limit"], 1024)
        self.assertEqual(optimizer.config["logging"]["level"], "DEBUG")
    
    @patch('os.environ')
    def test_optimize_environment_variables(self, mock_environ):
        """Tester l'optimisation des variables d'environnement."""
        mock_environ.__setitem__ = MagicMock()
        mock_environ.items = MagicMock(return_value=[])
        
        optimizer = SystemOptimizer(self.temp_config_file.name)
        result = optimizer.optimize_environment_variables()
        
        # Vérifier que les variables ont été définies
        mock_environ.__setitem__.assert_any_call('TEST_VAR', 'test_value')
        self.assertIn('environment_variables', optimizer.applied_optimizations)
    
    @unittest.skipIf(platform.system() == 'Windows', "Le module 'resource' n'est pas disponible sur Windows")
    def test_configure_system_limits_unix(self):
        """Tester la configuration des limites système sur les systèmes Unix."""
        # Ce test ne s'exécute que sur les systèmes Unix/Linux
        import resource
        
        with patch('resource.setrlimit') as mock_setrlimit, \
             patch('resource.getrlimit') as mock_getrlimit, \
             patch('ai_trading.utils.system_optimizer.SystemOptimizer._check_admin_privileges') as mock_admin:
            
            # Configurer les mocks
            mock_admin.return_value = True
            mock_getrlimit.return_value = (1000, 2000)
            
            optimizer = SystemOptimizer(self.temp_config_file.name)
            result = optimizer.configure_system_limits()
            
            # Vérifier les résultats
            self.assertIn('system_limits', optimizer.applied_optimizations)
    
    def test_configure_system_limits_windows(self):
        """Tester la configuration des limites système sur Windows."""
        # Ce test s'exécute sur tous les systèmes
        optimizer = SystemOptimizer(self.temp_config_file.name)
        result = optimizer.configure_system_limits()
        
        # Sur Windows, on s'attend à ce que la fonction ne fasse rien de particulier
        # mais retourne quand même un dictionnaire de limites (éventuellement vide)
        self.assertIsInstance(result, dict)
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('tempfile.mkdtemp')
    def test_optimize_disk_io(self, mock_mkdtemp, mock_makedirs, mock_exists):
        """Tester l'optimisation des E/S disque."""
        # Configurer les mocks
        mock_exists.return_value = True
        mock_mkdtemp.return_value = '/tmp/test_dir'
        
        optimizer = SystemOptimizer()
        result = optimizer.optimize_disk_io()
        
        # Vérifier que les opérations d'optimisation ont été effectuées
        self.assertIn('disk_optimization', optimizer.applied_optimizations)
    
    def test_setup_logging(self):
        """Tester la configuration du système de logging."""
        optimizer = SystemOptimizer(self.temp_config_file.name)
        logger = optimizer.setup_logging()
        
        # Vérifier que le logger a été correctement configuré
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertIn('logging', optimizer.applied_optimizations)
        
        # Vérifier que les handlers ont été ajoutés
        self.assertTrue(len(logger.handlers) > 0)
    
    @patch('ai_trading.utils.system_optimizer.SystemOptimizer.optimize_environment_variables')
    @patch('ai_trading.utils.system_optimizer.SystemOptimizer.configure_system_limits')
    @patch('ai_trading.utils.system_optimizer.SystemOptimizer.optimize_disk_io')
    @patch('ai_trading.utils.system_optimizer.SystemOptimizer.configure_memory_params')
    @patch('ai_trading.utils.system_optimizer.SystemOptimizer.setup_logging')
    def test_optimize_all(self, mock_logging, mock_memory, mock_disk, mock_limits, mock_env):
        """Tester l'application de toutes les optimisations."""
        optimizer = SystemOptimizer()
        result = optimizer.optimize_all()
        
        # Vérifier que toutes les méthodes d'optimisation ont été appelées
        mock_env.assert_called_once()
        mock_limits.assert_called_once()
        mock_disk.assert_called_once()
        mock_memory.assert_called_once()
        mock_logging.assert_called_once()
    
    def test_get_optimization_status(self):
        """Tester la récupération de l'état des optimisations."""
        optimizer = SystemOptimizer()
        optimizer.applied_optimizations = {"test": "value"}
        
        status = optimizer.get_optimization_status()
        
        # Vérifier le contenu du statut
        self.assertIn("system_info", status)
        self.assertIn("optimizations", status)
        self.assertIn("config", status)
        self.assertEqual(status["optimizations"]["test"], "value")
    
    @patch('ai_trading.utils.system_optimizer.SystemOptimizer.optimize_all')
    def test_optimize_system_helper(self, mock_optimize_all):
        """Tester la fonction utilitaire optimize_system."""
        optimizer = optimize_system(self.temp_config_file.name)
        
        # Vérifier que optimize_all a été appelé
        mock_optimize_all.assert_called_once()


if __name__ == '__main__':
    unittest.main() 