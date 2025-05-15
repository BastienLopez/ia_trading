import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Configurer le chemin pour importer depuis le répertoire parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.utils.efficient_checkpointing import (
    AsyncCheckpointSaver,
    CheckpointManager,
    compare_checkpoint_sizes,
    save_model_weights_only,
)

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Un modèle simple pour les tests."""

    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class TestEfficientCheckpointing(unittest.TestCase):
    """Tests pour les fonctionnalités d'Efficient Checkpointing."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.model = SimpleTestModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # Créer un répertoire temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.test_dir)

    def tearDown(self):
        """Nettoyage après chaque test."""
        # Supprimer le répertoire temporaire
        shutil.rmtree(self.test_dir)

    def test_save_model_weights_only(self):
        """Teste la fonction de sauvegarde des poids uniquement."""
        save_path = self.temp_path / "model_weights.pt"
        result_path = save_model_weights_only(self.model, save_path)

        # Vérifier que le fichier a été créé
        self.assertTrue(save_path.exists())
        self.assertEqual(save_path, result_path)

        # Vérifier que le fichier contient uniquement les poids
        checkpoint = torch.load(save_path)
        self.assertTrue(isinstance(checkpoint, dict))

        # Le format devrait être le même que model.state_dict()
        for key in self.model.state_dict().keys():
            self.assertIn(key, checkpoint)

    def test_compare_checkpoint_sizes(self):
        """Teste la fonction de comparaison des tailles de checkpoint."""
        result = compare_checkpoint_sizes(self.model, self.optimizer, self.temp_path)

        # Vérifier les clés du résultat
        self.assertIn("full_checkpoint_mb", result)
        self.assertIn("model_only_mb", result)
        self.assertIn("optimizer_only_mb", result)
        self.assertIn("optimizer_model_ratio", result)

        # Vérifier que les tailles sont cohérentes
        self.assertGreater(result["full_checkpoint_mb"], 0)
        self.assertGreater(result["model_only_mb"], 0)
        self.assertGreater(result["optimizer_only_mb"], 0)

        # La taille du checkpoint complet devrait être supérieure ou égale à la somme des parties
        self.assertGreaterEqual(
            result["full_checkpoint_mb"],
            result["model_only_mb"]
            + result["optimizer_only_mb"]
            - 0.1,  # Marge d'erreur
        )

    def test_async_checkpoint_saver(self):
        """Teste le sauvegardeur de checkpoint asynchrone."""
        # Créer un sauvegardeur asynchrone
        async_saver = AsyncCheckpointSaver()

        # Tester le démarrage
        async_saver.start()
        self.assertTrue(async_saver.active)

        # Tester la sauvegarde asynchrone
        save_path = self.temp_path / "async_checkpoint.pt"
        result = async_saver.save_checkpoint(
            model=self.model,
            save_path=save_path,
            optimizer=self.optimizer,
            weights_only=False,
        )

        # Vérifier que la tâche a été enfilée
        self.assertTrue(result)

        # Attendre un peu pour que pending_tasks soit mis à jour
        time.sleep(0.5)  # Attendre que le thread traite la tâche

        # Vérifier que pending_tasks a bien été incrémenté
        # Si ce n'est pas le cas, accepter également que la tâche ait pu être traitée immédiatement
        self.assertIn(
            async_saver.pending_tasks,
            [0, 1],
            "pending_tasks devrait être 0 (si déjà traité) ou 1 (si en attente)",
        )

        # Attendre la fin de la sauvegarde
        completion = async_saver.wait_for_completion(timeout=5.0)
        self.assertTrue(completion)
        self.assertEqual(async_saver.pending_tasks, 0)

        # Vérifier que le fichier existe
        self.assertTrue(save_path.exists())

        # Tester l'arrêt
        async_saver.stop()
        self.assertFalse(async_saver.active)

    def test_checkpoint_manager_basic(self):
        """Teste les fonctionnalités de base du gestionnaire de checkpoints."""
        # Créer un gestionnaire avec sauvegarde synchrone
        checkpoint_mgr = CheckpointManager(
            base_dir=self.temp_path / "checkpoints",
            max_checkpoints=3,
            async_save=False,
            weights_only=True,
        )

        # Vérifier l'initialisation
        self.assertEqual(checkpoint_mgr.max_checkpoints, 3)
        self.assertFalse(checkpoint_mgr.async_save)
        self.assertTrue(checkpoint_mgr.weights_only)

        # Sauvegarder quelques checkpoints
        for epoch in range(5):
            save_path = checkpoint_mgr.save(
                model=self.model, optimizer=self.optimizer, epoch=epoch
            )
            self.assertTrue(save_path.exists())

        # Vérifier la rotation des checkpoints (ne devrait conserver que les 3 derniers)
        self.assertEqual(len(checkpoint_mgr.checkpoints), 3)

        # Les époques présentes devraient être 2, 3, 4 (pas 0 et 1)
        checkpoint_epochs = [
            int(path.stem.split("_")[-1]) for path in checkpoint_mgr.checkpoints
        ]
        self.assertIn(2, checkpoint_epochs)
        self.assertIn(3, checkpoint_epochs)
        self.assertIn(4, checkpoint_epochs)
        self.assertNotIn(0, checkpoint_epochs)
        self.assertNotIn(1, checkpoint_epochs)

    def test_checkpoint_manager_async(self):
        """Teste le gestionnaire de checkpoints avec sauvegarde asynchrone."""
        # Créer un gestionnaire avec sauvegarde asynchrone
        checkpoint_mgr = CheckpointManager(
            base_dir=self.temp_path / "async_checkpoints",
            max_checkpoints=2,
            async_save=True,
            weights_only=False,
        )

        # Vérifier l'initialisation
        self.assertTrue(checkpoint_mgr.async_save)
        self.assertIsNotNone(checkpoint_mgr.async_saver)

        # Sauvegarder des checkpoints
        for epoch in range(3):
            # Les sauvegardes asynchrones retournent None
            save_path = checkpoint_mgr.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                extra_data={"metric": 0.9 - epoch * 0.1},
            )
            self.assertIsNone(save_path)

        # Attendre la fin des sauvegardes asynchrones
        checkpoint_mgr.async_saver.wait_for_completion(timeout=5.0)

        # Vérifier que des fichiers ont été créés (après re-scan)
        checkpoint_mgr._scan_existing_checkpoints()
        self.assertGreater(len(checkpoint_mgr.checkpoints), 0)
        self.assertLessEqual(len(checkpoint_mgr.checkpoints), 2)  # max_checkpoints=2

    def test_restore_model(self):
        """Teste la restauration d'un modèle à partir d'un checkpoint."""
        # Créer un gestionnaire
        checkpoint_mgr = CheckpointManager(
            base_dir=self.temp_path / "restore_test",
            max_checkpoints=5,
            async_save=False,
            weights_only=False,  # Sauvegarder également l'optimiseur
        )

        # Sauvegarder l'état actuel du modèle pour comparaison
        original_state = {
            k: v.clone().detach() for k, v in self.model.state_dict().items()
        }

        # Optimizer state
        original_optimizer_state = {
            group_idx: {
                k: v if not isinstance(v, torch.Tensor) else v.clone().detach()
                for k, v in group.items()
            }
            for group_idx, group in enumerate(
                self.optimizer.state_dict()["param_groups"]
            )
        }

        # Sauvegarder un checkpoint avec des métadonnées
        metadata = {"epoch": 10, "accuracy": 0.85, "custom_value": "test"}
        save_path = checkpoint_mgr.save(
            model=self.model, optimizer=self.optimizer, epoch=10, extra_data=metadata
        )

        # Modifier les poids du modèle significativement
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 2.0)  # Modifier aléatoirement

        # Également modifier l'optimiseur
        self.optimizer.param_groups[0]["lr"] = 0.5  # Changer le learning rate

        # Vérifier que le modèle a été modifié (les poids devraient être différents)
        for name, param in self.model.named_parameters():
            original_param = original_state[name]
            self.assertFalse(
                torch.allclose(original_param, param),
                f"Paramètre {name} n'a pas été modifié",
            )

        # Restaurer à partir du checkpoint
        new_model = SimpleTestModel()
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)

        # Charger le checkpoint
        loaded_metadata = checkpoint_mgr.load_latest(
            model=new_model, optimizer=new_optimizer
        )

        # Vérifier que les métadonnées sont correctes
        self.assertEqual(loaded_metadata["epoch"], 10)
        self.assertEqual(loaded_metadata["accuracy"], 0.85)
        self.assertEqual(loaded_metadata["custom_value"], "test")

        # Vérifier que les poids sont restaurés correctement
        for name, param in new_model.named_parameters():
            # Vérifier que le paramètre restauré est proche de l'original
            self.assertTrue(
                torch.allclose(original_state[name], param),
                f"Paramètre {name} n'a pas été restauré correctement",
            )

        # Vérifier que le learning rate de l'optimiseur est restauré
        self.assertEqual(
            new_optimizer.param_groups[0]["lr"], original_optimizer_state[0]["lr"]
        )

    def test_weights_only_mode(self):
        """Teste le mode weights_only du gestionnaire de checkpoints."""
        # Créer un gestionnaire en mode weights_only
        checkpoint_mgr = CheckpointManager(
            base_dir=self.temp_path / "weights_only_test",
            max_checkpoints=1,
            async_save=False,
            weights_only=True,
        )

        # Sauvegarder un checkpoint
        save_path = checkpoint_mgr.save(
            model=self.model, optimizer=self.optimizer, epoch=1
        )

        # Charger directement le fichier pour vérifier son contenu
        checkpoint = torch.load(save_path)

        # Dans ce mode, le checkpoint devrait être exactement le model.state_dict()
        model_state = self.model.state_dict()
        self.assertEqual(len(checkpoint), len(model_state))

        for key in model_state:
            self.assertIn(key, checkpoint)
            self.assertTrue(torch.equal(checkpoint[key], model_state[key]))

        # Le checkpoint ne devrait pas contenir l'état de l'optimiseur
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            self.fail(
                "Le checkpoint en mode weights_only contient l'état de l'optimiseur"
            )

    def test_custom_filename(self):
        """Teste l'utilisation d'un nom de fichier personnalisé."""
        checkpoint_mgr = CheckpointManager(
            base_dir=self.temp_path / "custom_filename_test",
            max_checkpoints=1,
            async_save=False,
        )

        # Sauvegarder avec un nom personnalisé
        custom_name = "my_special_checkpoint.pt"
        save_path = checkpoint_mgr.save(
            model=self.model, optimizer=self.optimizer, custom_filename=custom_name
        )

        # Vérifier le nom du fichier
        self.assertEqual(save_path.name, custom_name)
        self.assertTrue(save_path.exists())


if __name__ == "__main__":
    unittest.main()
