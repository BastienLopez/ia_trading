#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests pour le module d'optimisation DeepSpeed.
"""

import os
import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from ai_trading.utils.deepspeed_optimizer import (
    create_deepspeed_config,
    DeepSpeedOptimizer,
    HAVE_DEEPSPEED
)

# Modèle simple pour les tests
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def compute_loss(self, outputs, targets):
        return torch.nn.functional.mse_loss(outputs, targets)


@unittest.skipIf(not HAVE_DEEPSPEED, "DeepSpeed n'est pas installé")
class TestDeepSpeedOptimizer(unittest.TestCase):
    """Tests pour l'optimiseur DeepSpeed."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.model = SimpleModel()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    @patch('deepspeed.initialize')
    def test_init(self, mock_initialize):
        """Teste l'initialisation de l'optimiseur DeepSpeed."""
        # Configurer les mocks
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_config = MagicMock()
        mock_initialize.return_value = (mock_model, mock_optimizer, mock_scheduler, mock_config)
        
        # Créer l'optimiseur
        optimizer = DeepSpeedOptimizer(
            model=self.model,
            fp16=True,
            zero_stage=2,
            offload_optimizer=False,
            offload_parameters=False,
            local_rank=-1,
            train_batch_size=32,
            checkpoint_dir=self.temp_dir
        )
        
        # Vérifier que les attributs sont correctement initialisés
        self.assertEqual(optimizer.model, self.model)
        self.assertEqual(optimizer.fp16, True)
        self.assertEqual(optimizer.zero_stage, 2)
        self.assertEqual(optimizer.train_batch_size, 32)
        self.assertEqual(optimizer.checkpoint_dir, self.temp_dir)
        
        # Vérifier que deepspeed.initialize a été appelé
        mock_initialize.assert_called_once()
    
    def test_create_ds_config(self):
        """Teste la création de la configuration DeepSpeed."""
        # Créer l'optimiseur (avec mocks)
        with patch('deepspeed.initialize'):
            optimizer = DeepSpeedOptimizer(
                model=self.model,
                fp16=True,
                zero_stage=2,
                offload_optimizer=True,
                offload_parameters=False,
                checkpoint_dir=self.temp_dir
            )
        
        # Créer la configuration
        config = optimizer._create_ds_config(
            fp16=True,
            zero_stage=2,
            offload_optimizer=True,
            offload_parameters=False,
            train_batch_size=32,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            checkpoint_dir=self.temp_dir,
            save_interval=1000
        )
        
        # Vérifier les clés de la configuration
        self.assertIn("train_batch_size", config)
        self.assertIn("gradient_accumulation_steps", config)
        self.assertIn("optimizer", config)
        self.assertIn("scheduler", config)
        self.assertIn("gradient_clipping", config)
        self.assertIn("zero_optimization", config)
        self.assertIn("fp16", config)
        self.assertIn("checkpoint", config)
        
        # Vérifier les valeurs spécifiques
        self.assertEqual(config["train_batch_size"], 32)
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertEqual(config["gradient_clipping"], 1.0)
        self.assertEqual(config["zero_optimization"]["stage"], 2)
        self.assertEqual(config["zero_optimization"]["offload_optimizer"], True)
        self.assertTrue(config["fp16"]["enabled"])
        self.assertEqual(config["checkpoint"]["save_interval"], 1000)
    
    @patch('deepspeed.initialize')
    def test_train_step(self, mock_initialize):
        """Teste une étape d'entraînement."""
        # Configurer les mocks
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        mock_model.device = torch.device('cpu')
        mock_initialize.return_value = (mock_model, mock_optimizer, None, None)
        
        # Créer l'optimiseur
        optimizer = DeepSpeedOptimizer(model=self.model)
        
        # Créer des données factices
        batch = torch.randn(2, 10)
        labels = torch.randn(2, 2)
        
        # Simuler une valeur de perte
        mock_model.return_value = torch.tensor([1.0, 2.0])
        
        # Configurer la valeur de retour de compute_loss
        optimizer.model.compute_loss = MagicMock(return_value=torch.tensor(0.5))
        
        # Exécuter une étape d'entraînement
        loss = optimizer.train_step(batch, labels)
        
        # Vérifier que le forward, backward et step ont été appelés
        mock_model.assert_called_once()
        mock_model.backward.assert_called_once()
        mock_model.step.assert_called_once()
    
    @patch('deepspeed.initialize')
    def test_eval_step(self, mock_initialize):
        """Teste une étape d'inférence."""
        # Configurer les mocks
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        mock_model.device = torch.device('cpu')
        mock_initialize.return_value = (mock_model, mock_optimizer, None, None)
        
        # Créer l'optimiseur
        optimizer = DeepSpeedOptimizer(model=self.model)
        
        # Créer des données factices
        batch = torch.randn(2, 10)
        
        # Simuler une valeur de sortie
        expected_output = torch.tensor([1.0, 2.0])
        mock_model.return_value = expected_output
        
        # Exécuter une étape d'inférence
        output = optimizer.eval_step(batch)
        
        # Vérifier que le modèle a été appelé en mode évaluation
        mock_model.eval.assert_called_once()
        mock_model.assert_called_once()
        mock_model.train.assert_called_once()  # Revient au mode entraînement
    
    @patch('deepspeed.initialize')
    def test_save_load_checkpoint(self, mock_initialize):
        """Teste la sauvegarde et le chargement de checkpoints."""
        # Configurer les mocks
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        mock_initialize.return_value = (mock_model, mock_optimizer, None, None)
        
        # Créer l'optimiseur
        optimizer = DeepSpeedOptimizer(
            model=self.model,
            checkpoint_dir=self.temp_dir
        )
        
        # Créer un chemin de checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Sauvegarder le checkpoint
        optimizer.save_checkpoint(checkpoint_path)
        
        # Vérifier que save_checkpoint a été appelé
        mock_model.save_checkpoint.assert_called_once_with(checkpoint_path, client_state={'model_config': {}})
        
        # Configurer le mock pour load_checkpoint
        mock_model.load_checkpoint.return_value = (None, {'test': 'data'})
        
        # Charger le checkpoint
        client_state = optimizer.load_checkpoint(checkpoint_path)
        
        # Vérifier que load_checkpoint a été appelé
        mock_model.load_checkpoint.assert_called_once_with(checkpoint_path)
        self.assertEqual(client_state, {'test': 'data'})


class TestCreateDeepSpeedConfig(unittest.TestCase):
    """Tests pour la fonction create_deepspeed_config."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_config_basic(self):
        """Teste la création d'une configuration de base."""
        # Créer une configuration simple
        config = create_deepspeed_config(
            zero_stage=1,
            fp16=True,
            offload_optimizer=False,
            offload_parameters=False
        )
        
        # Vérifier les clés de base
        self.assertIn("train_batch_size", config)
        self.assertIn("gradient_accumulation_steps", config)
        self.assertIn("optimizer", config)
        self.assertIn("scheduler", config)
        self.assertIn("gradient_clipping", config)
        
        # Vérifier les optimisations ZeRO
        self.assertIn("zero_optimization", config)
        self.assertEqual(config["zero_optimization"]["stage"], 1)
        
        # Vérifier les optimisations FP16
        self.assertIn("fp16", config)
        self.assertTrue(config["fp16"]["enabled"])
    
    def test_create_config_with_offload(self):
        """Teste la création d'une configuration avec offload."""
        # Créer une configuration avec offload
        config = create_deepspeed_config(
            zero_stage=2,
            fp16=True,
            offload_optimizer=True,
            offload_parameters=False
        )
        
        # Vérifier les options d'offload
        self.assertTrue(config["zero_optimization"]["offload_optimizer"])
    
    def test_create_config_with_file(self):
        """Teste la création d'une configuration avec écriture dans un fichier."""
        # Créer un fichier de sortie
        output_file = os.path.join(self.temp_dir, "ds_config.json")
        
        # Créer une configuration
        config = create_deepspeed_config(
            zero_stage=3,
            fp16=True,
            offload_optimizer=True,
            offload_parameters=True,
            output_file=output_file
        )
        
        # Vérifier que le fichier a été créé
        self.assertTrue(os.path.exists(output_file))
        
        # Vérifier les options avancées pour ZeRO-3
        self.assertEqual(config["zero_optimization"]["stage"], 3)
        self.assertTrue(config["zero_optimization"]["offload_optimizer"])
        self.assertTrue(config["zero_optimization"]["offload_param"])


if __name__ == '__main__':
    unittest.main() 