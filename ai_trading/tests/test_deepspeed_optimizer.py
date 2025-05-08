#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests pour le module d'optimisation DeepSpeed.
"""

import json
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from ai_trading.utils.deepspeed_optimizer import (
    HAVE_DEEPSPEED,
    DeepSpeedOptimizer,
    create_deepspeed_config,
)

# Configuration du logger de test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class TestDeepSpeedOptimizer(unittest.TestCase):
    """Tests pour l'optimiseur DeepSpeed."""

    def setUp(self):
        """Initialisation avant chaque test."""
        # Informer l'utilisateur si DeepSpeed n'est pas disponible mais continuer les tests
        if not HAVE_DEEPSPEED:
            logger.warning(
                "DeepSpeed n'est pas installé. Utilisation de l'implémentation stub pour les tests."
            )
            # Note: On n'utilise plus self.skipTest() pour permettre aux tests de s'exécuter

        self.model = SimpleModel()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Nettoyage après chaque test."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Teste l'initialisation de l'optimiseur DeepSpeed."""
        # Si DeepSpeed est disponible, utiliser le mock, sinon utiliser directement le stub
        if HAVE_DEEPSPEED:
            with patch("deepspeed.initialize") as mock_initialize:
                # Configurer les mocks
                mock_model = MagicMock()
                mock_optimizer = MagicMock()
                mock_scheduler = MagicMock()
                mock_config = MagicMock()
                mock_initialize.return_value = (
                    mock_model,
                    mock_optimizer,
                    mock_scheduler,
                    mock_config,
                )

                # Créer l'optimiseur
                optimizer = DeepSpeedOptimizer(
                    model=self.model,
                    fp16=True,
                    zero_stage=2,
                    offload_optimizer=False,
                    offload_parameters=False,
                    local_rank=-1,
                    train_batch_size=32,
                    checkpoint_dir=self.temp_dir,
                )

                # Vérifier que les attributs sont correctement initialisés
                self.assertEqual(optimizer.model, self.model)
                self.assertEqual(optimizer.fp16, True)
                self.assertEqual(optimizer.zero_stage, 2)
                self.assertEqual(optimizer.train_batch_size, 32)
                self.assertEqual(optimizer.checkpoint_dir, self.temp_dir)

                # Vérifier que deepspeed.initialize a été appelé
                mock_initialize.assert_called_once()
        else:
            # Test direct avec le stub
            optimizer = DeepSpeedOptimizer(
                model=self.model,
                fp16=True,
                zero_stage=2,
                offload_optimizer=False,
                offload_parameters=False,
                local_rank=-1,
                train_batch_size=32,
                checkpoint_dir=self.temp_dir,
            )

            # Vérifier que les attributs sont correctement initialisés
            self.assertEqual(optimizer.model, self.model)
            self.assertEqual(optimizer.fp16, True)
            self.assertEqual(optimizer.zero_stage, 2)
            self.assertEqual(optimizer.train_batch_size, 32)
            self.assertEqual(optimizer.checkpoint_dir, self.temp_dir)

            # Vérifier que ds_model est une instance de DeepSpeedModelStub
            if not HAVE_DEEPSPEED:
                from ai_trading.utils.deepspeed_optimizer import DeepSpeedModelStub

                self.assertIsInstance(optimizer.ds_model, DeepSpeedModelStub)

    def test_create_ds_config(self):
        """Teste la création de la configuration DeepSpeed."""
        # Créer l'optimiseur (avec mocks si DeepSpeed est disponible)
        if HAVE_DEEPSPEED:
            with patch("deepspeed.initialize"):
                optimizer = DeepSpeedOptimizer(
                    model=self.model,
                    fp16=True,
                    zero_stage=2,
                    offload_optimizer=True,
                    offload_parameters=False,
                    checkpoint_dir=self.temp_dir,
                )
        else:
            optimizer = DeepSpeedOptimizer(
                model=self.model,
                fp16=True,
                zero_stage=2,
                offload_optimizer=True,
                offload_parameters=False,
                checkpoint_dir=self.temp_dir,
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
            save_interval=1000,
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

        # Adapter la vérification en fonction de la structure
        if isinstance(config["zero_optimization"]["offload_optimizer"], bool):
            self.assertEqual(config["zero_optimization"]["offload_optimizer"], True)
        else:
            # Si c'est un dict, vérifier que c'est bien configuré
            self.assertIsInstance(
                config["zero_optimization"]["offload_optimizer"], dict
            )
            self.assertIn("device", config["zero_optimization"]["offload_optimizer"])
            self.assertEqual(
                config["zero_optimization"]["offload_optimizer"]["device"], "cpu"
            )

        self.assertTrue(config["fp16"]["enabled"])
        self.assertEqual(config["checkpoint"]["save_interval"], 1000)

    def test_train_step(self):
        """Teste une étape d'entraînement."""
        if HAVE_DEEPSPEED:
            with patch("deepspeed.initialize") as mock_initialize:
                # Configurer les mocks
                mock_model = MagicMock()
                mock_optimizer = MagicMock()
                mock_model.device = torch.device("cpu")
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
        else:
            # Créer l'optimiseur avec le stub
            optimizer = DeepSpeedOptimizer(model=self.model)

            # Sauvegarder l'implémentation originale de compute_loss pour la restaurer plus tard
            original_compute_loss = self.model.compute_loss

            try:
                # Remplacer compute_loss par un mock qui retourne un tensor avec requires_grad=True
                mock_loss = torch.tensor(0.5, requires_grad=True)
                self.model.compute_loss = MagicMock(return_value=mock_loss)

                # Créer des données factices
                batch = torch.randn(2, 10)
                labels = torch.randn(2, 2)

                # Exécuter une étape d'entraînement
                loss = optimizer.train_step(batch, labels)

                # Vérifier que compute_loss a été appelé
                self.model.compute_loss.assert_called_once()

                # Vérifier que la perte a été calculée correctement
                self.assertEqual(loss, 0.5)
            finally:
                # Restaurer l'implémentation originale
                self.model.compute_loss = original_compute_loss

    def test_eval_step(self):
        """Teste une étape d'inférence."""
        if HAVE_DEEPSPEED:
            with patch("deepspeed.initialize") as mock_initialize:
                # Configurer les mocks
                mock_model = MagicMock()
                mock_optimizer = MagicMock()
                mock_model.device = torch.device("cpu")
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
        else:
            # Créer l'optimiseur avec le stub
            optimizer = DeepSpeedOptimizer(model=self.model)

            # Sauvegarder et remplacer forward pour vérification
            original_forward = self.model.forward
            try:
                # Simuler une valeur de sortie
                expected_output = torch.tensor([[1.0, 2.0]])
                self.model.forward = MagicMock(return_value=expected_output)

                # Créer des données factices
                batch = torch.randn(1, 10)

                # Exécuter une étape d'inférence
                output = optimizer.eval_step(batch)

                # Vérifier que forward a été appelé
                self.model.forward.assert_called_once()

                # Vérifier que l'output est celui attendu
                self.assertTrue(torch.allclose(output, expected_output))
            finally:
                # Restaurer l'implémentation originale
                self.model.forward = original_forward

    def test_save_load_checkpoint(self):
        """Teste la sauvegarde et le chargement de checkpoints."""
        if HAVE_DEEPSPEED:
            with patch("deepspeed.initialize") as mock_initialize:
                # Configurer les mocks
                mock_model = MagicMock()
                mock_optimizer = MagicMock()
                mock_initialize.return_value = (mock_model, mock_optimizer, None, None)

                # Créer l'optimiseur
                optimizer = DeepSpeedOptimizer(
                    model=self.model, checkpoint_dir=self.temp_dir
                )

                # Créer un chemin de checkpoint
                checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint")
                os.makedirs(checkpoint_path, exist_ok=True)

                # Sauvegarder le checkpoint
                optimizer.save_checkpoint(checkpoint_path)

                # Vérifier que save_checkpoint a été appelé
                mock_model.save_checkpoint.assert_called_once_with(
                    checkpoint_path, client_state={"model_config": {}}
                )

                # Configurer le mock pour load_checkpoint
                mock_model.load_checkpoint.return_value = (None, {"test": "data"})

                # Charger le checkpoint
                client_state = optimizer.load_checkpoint(checkpoint_path)

                # Vérifier que load_checkpoint a été appelé
                mock_model.load_checkpoint.assert_called_once_with(checkpoint_path)
                self.assertEqual(client_state, {"test": "data"})
        else:
            # Créer l'optimiseur avec le stub
            optimizer = DeepSpeedOptimizer(
                model=self.model, checkpoint_dir=self.temp_dir
            )

            # Créer un chemin de checkpoint
            checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Sauvegarder l'état du modèle avant modification
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.data.clone()

            # Sauvegarder le checkpoint
            optimizer.save_checkpoint(checkpoint_path, client_state={"test": "data"})

            # Vérifier que le fichier a été créé
            self.assertTrue(os.path.exists(os.path.join(checkpoint_path, "model.pt")))
            self.assertTrue(
                os.path.exists(os.path.join(checkpoint_path, "client_state.pt"))
            )

            # Modifier le modèle
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(0.1)

            # Charger le checkpoint
            client_state = optimizer.load_checkpoint(checkpoint_path)

            # Vérifier que les poids ont été restaurés
            for name, param in self.model.named_parameters():
                self.assertTrue(torch.allclose(param.data, original_params[name]))

            # Vérifier l'état client (adapter pour notre implémentation stub qui peut modifier l'état client)
            self.assertIn("test", client_state)
            self.assertEqual(client_state["test"], "data")


class TestCreateDeepSpeedConfig(unittest.TestCase):
    """Tests pour la fonction create_deepspeed_config."""

    def setUp(self):
        """Initialisation avant chaque test."""
        # Informer l'utilisateur si DeepSpeed n'est pas disponible mais continuer les tests
        if not HAVE_DEEPSPEED:
            logger.warning(
                "DeepSpeed n'est pas installé. Utilisation de l'implémentation stub pour les tests."
            )
            # Note: On n'utilise plus self.skipTest() pour permettre aux tests de s'exécuter

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Nettoyage après chaque test."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_config_basic(self):
        """Teste la création d'une configuration de base."""
        # Créer une configuration simple
        config = create_deepspeed_config(
            zero_stage=1, fp16=True, offload_optimizer=False, offload_parameters=False
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
            zero_stage=2, fp16=True, offload_optimizer=True, offload_parameters=False
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
            output_file=output_file,
        )

        # Vérifier que le fichier a été créé
        self.assertTrue(os.path.exists(output_file))

        # Vérifier le contenu du fichier
        with open(output_file, "r") as f:
            file_config = json.loads(f.read())

        # Vérifier quelques valeurs clés
        self.assertEqual(file_config["zero_optimization"]["stage"], 3)
        self.assertTrue(file_config["zero_optimization"]["offload_optimizer"])
        self.assertTrue(file_config["zero_optimization"]["offload_param"])


if __name__ == "__main__":
    unittest.main()
