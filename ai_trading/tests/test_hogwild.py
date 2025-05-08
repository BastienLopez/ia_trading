import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.hogwild import (
    AsyncAdvantageActorCritic,
    HogwildTrainer,
    HogwildWorker,
    a3c_loss_fn,
)


# Fonction factory globale pour l'optimiseur (évite les problèmes de pickle)
def create_optimizer(params, lr):
    return optim.SGD(params, lr=lr)


# Modèle simple pour les tests
class SimpleModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Buffer simple pour les tests
class SimpleBuffer:
    def __init__(self, size=1000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.size = size

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        # Maintenir la taille maximale
        if len(self.states) > self.size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)

    def sample(self, batch_size):
        indices = random.sample(
            range(len(self.states)), min(batch_size, len(self.states))
        )

        # Utiliser np.array pour convertir la liste avant de créer le tenseur
        states = torch.FloatTensor(np.array([self.states[i] for i in indices]))
        actions = torch.LongTensor(np.array([self.actions[i] for i in indices]))
        rewards = torch.FloatTensor(np.array([self.rewards[i] for i in indices]))
        next_states = torch.FloatTensor(
            np.array([self.next_states[i] for i in indices])
        )
        dones = torch.FloatTensor(np.array([float(self.dones[i]) for i in indices]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.states)


# Fonction de perte simple pour les tests
def simple_loss_fn(model, states, actions, rewards, next_states, dones):
    outputs = model(states)
    action_values = outputs.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = ((action_values - rewards) ** 2).mean()
    return loss


class TestHogwild(unittest.TestCase):

    def setUp(self):
        """Prépare l'environnement de test"""
        # Créer un modèle partagé
        self.shared_model = SimpleModel()

        # Partager la mémoire pour HOGWILD
        self.shared_model.share_memory()

        # Créer un buffer
        self.buffer = SimpleBuffer()

        # Remplir le buffer avec des données aléatoires
        for _ in range(100):
            state = np.random.rand(4)
            action = random.randint(0, 1)
            reward = random.random()
            next_state = np.random.rand(4)
            done = random.random() > 0.8

            self.buffer.add(state, action, reward, next_state, done)

        # Fonction factory pour l'optimiseur - utilise la fonction globale
        self.optimizer_factory = create_optimizer

    def test_hogwild_worker(self):
        """Teste le fonctionnement d'un travailleur HOGWILD!"""
        worker = HogwildWorker(
            shared_model=self.shared_model,
            worker_id=0,
            buffer=self.buffer,
            optimizer_factory=self.optimizer_factory,
            loss_fn=simple_loss_fn,
            batch_size=8,
            learning_rate=0.01,
            mini_batches_per_update=2,
        )

        # Tester l'initialisation
        self.assertEqual(worker.worker_id, 0)
        self.assertEqual(worker.device.type, "cpu")
        self.assertEqual(worker.batch_size, 8)

        # Tester une mise à jour
        loss = worker.train_mini_batches()

        # Vérifier que la perte est un nombre
        self.assertIsInstance(loss, float)

        # Tester plusieurs mises à jour
        worker.train(num_updates=5)

        # Vérifier les métriques
        metrics = worker.get_metrics()
        self.assertIn("worker_id", metrics)
        self.assertIn("updates_done", metrics)
        self.assertEqual(metrics["updates_done"], 5)

    def test_hogwild_trainer(self):
        """Teste le fonctionnement de l'entraîneur HOGWILD!"""
        trainer = HogwildTrainer(
            model=self.shared_model,
            buffer=self.buffer,
            optimizer_factory=create_optimizer,  # Utiliser la fonction globale directement
            loss_fn=simple_loss_fn,
            num_workers=2,
            batch_size=8,
            learning_rate=0.01,
        )

        # Tester l'initialisation
        self.assertEqual(len(trainer.workers), 2)
        self.assertEqual(trainer.num_workers, 2)

        # Au lieu d'utiliser multiprocessing, entraînons directement les workers
        for worker in trainer.workers:
            worker.train(num_updates=3)

        # Vérifier les métriques
        metrics = trainer.get_metrics()
        self.assertIn("global", metrics)
        self.assertIn("workers", metrics)
        self.assertEqual(len(metrics["workers"]), 2)

        # Tester la sauvegarde du modèle
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model_path = tmp.name

        trainer.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))

        # Nettoyer
        os.unlink(model_path)

    def test_a3c_model(self):
        """Teste le modèle A3C"""
        # Créer un modèle A3C
        model = AsyncAdvantageActorCritic(
            state_dim=4, action_dim=2, hidden_dim=32, shared_backbone=True
        )

        # Vérifier l'architecture
        self.assertIsInstance(model.backbone, nn.Sequential)
        self.assertIsInstance(
            model.policy_head, nn.Linear
        )  # Utiliser le nom correct: policy_head
        self.assertIsInstance(
            model.value_head, nn.Linear
        )  # Utiliser le nom correct: value_head

        # Tester le forward pass
        state = torch.rand(2, 4)
        policy, value = model(state)

        # Vérifier les dimensions
        self.assertEqual(policy.shape, (2, 2))
        self.assertEqual(value.shape, (2, 1))

        # Tester get_action
        state = torch.rand(4)
        action = model.get_action(state)

        # Vérifier les types
        self.assertIsInstance(action, int)

    def test_a3c_loss(self):
        """Teste la fonction de perte A3C"""
        # Créer un modèle A3C
        model = AsyncAdvantageActorCritic(state_dim=4, action_dim=2, hidden_dim=32)

        # Préparer les données
        states = torch.rand(5, 4)
        actions = torch.randint(0, 2, (5,))
        rewards = torch.rand(5)
        next_states = torch.rand(5, 4)
        dones = torch.randint(0, 2, (5,)).float()

        # Calculer la perte
        loss = a3c_loss_fn(model, states, actions, rewards, next_states, dones)

        # Vérifier que la perte est valide
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(torch.isnan(loss).item())


if __name__ == "__main__":
    unittest.main()
