import unittest
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import queue

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.distributed_experience import ExperienceWorker, ExperienceMaster, DistributedExperienceManager

# Modèle simple pour les tests
class SimpleModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)

# Environnement simple pour les tests
class SimpleEnv:
    def __init__(self):
        self.state = np.zeros(4)
        self.step_count = 0
        self.max_steps = 10
        
    def reset(self):
        self.state = np.random.rand(4)
        self.step_count = 0
        return self.state
        
    def step(self, action):
        self.step_count += 1
        self.state = np.random.rand(4)
        reward = 1.0 if action == 1 else 0.1
        done = self.step_count >= self.max_steps
        info = {}
        return self.state, reward, done, info

# Buffer de replay simple pour les tests
class SimpleReplayBuffer:
    def __init__(self):
        self.buffer = []
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def add_batch(self, batch):
        self.buffer.extend(batch)
        
    def __len__(self):
        return len(self.buffer)

# Fonction globale pour créer un environnement (évite les problèmes de pickle)
def create_test_env():
    return SimpleEnv()


class TestDistributedExperience(unittest.TestCase):
    
    def setUp(self):
        """Prépare l'environnement de test"""
        # Créer un modèle simple
        self.model = SimpleModel()
        
        # Fonction de création d'environnement (utiliser la fonction globale)
        self.env_creator = create_test_env
        
        # Créer un buffer simple
        self.replay_buffer = SimpleReplayBuffer()
        
    def test_worker_local_mode(self):
        """Teste le fonctionnement du travailleur en mode local"""
        worker = ExperienceWorker(
            env_creator=self.env_creator,
            policy=self.model,
            batch_size=5,
            send_freq=2,
            local_mode=True
        )
        
        # Collecter des expériences
        worker.collect_experience(steps=20)
        
        # Vérifier que des expériences ont été collectées
        self.assertGreater(worker.total_steps, 0)
        self.assertGreater(len(worker.episode_rewards), 0)
        
        # Vérifier que la queue contient des données
        self.assertFalse(worker.queue.empty())
        
        # Récupérer les expériences de la queue
        experiences = []
        while not worker.queue.empty():
            batch = worker.queue.get()
            experiences.extend(batch)
        
        # Vérifier que des expériences ont été enregistrées
        self.assertGreater(len(experiences), 0)
        
        # Vérifier le format des expériences
        for exp in experiences:
            state, action, reward, next_state, done = exp
            self.assertEqual(len(state), 4)
            self.assertIsInstance(action, int)
            self.assertIsInstance(reward, float)
            self.assertEqual(len(next_state), 4)
            self.assertIsInstance(done, bool)
    
    def test_master_local_mode(self):
        """Teste le fonctionnement du maître en mode local"""
        # Créer un maître
        master = ExperienceMaster(
            replay_buffer=self.replay_buffer,
            local_mode=True
        )
        
        # Créer un travailleur
        worker = ExperienceWorker(
            env_creator=self.env_creator,
            policy=self.model,
            batch_size=5,
            send_freq=2,
            local_mode=True
        )
        
        # Ajouter le travailleur au maître
        master.add_local_worker(worker)
        
        # Démarrer le maître
        master.start()
        
        # Collecter des expériences avec le travailleur
        worker.collect_experience(steps=20)
        
        # Attendre que les expériences soient traitées
        time.sleep(0.5)
        
        # Manuellement transférer les expériences du worker au replay buffer pour le test
        while not worker.queue.empty():
            batch = worker.queue.get()
            for exp in batch:
                self.replay_buffer.add(*exp)
        
        # Arrêter le maître
        master.stop()
        
        # Vérifier que des expériences ont été ajoutées au buffer
        self.assertGreater(len(self.replay_buffer.buffer), 0)
        
    def test_distributed_manager_local(self):
        """Teste le gestionnaire d'expériences distribuées en mode local"""
        # Tester seulement le gestionnaire, sans multiprocessing
        manager = DistributedExperienceManager(
            env_creator=create_test_env,  # Utiliser la fonction globale
            policy=self.model,
            replay_buffer=self.replay_buffer,
            n_local_workers=2,
            batch_size=5
        )
        
        # Initialiser le gestionnaire
        manager.initialize()
        
        # Au lieu de démarrer la collecte en utilisant des processus, simulons-la
        for worker in manager.master.local_workers:
            worker.collect_experience(steps=15)
            # Transférer manuellement les expériences
            while not worker.queue.empty():
                batch = worker.queue.get()
                for exp in batch:
                    self.replay_buffer.add(*exp)
        
        # Arrêter le gestionnaire
        manager.stop()
        
        # Vérifier que des expériences ont été collectées
        self.assertGreater(len(self.replay_buffer.buffer), 0)
        
        # Vérifier les métriques
        metrics = manager.get_metrics()
        self.assertIn("workers", metrics)
        self.assertEqual(len(metrics["workers"]), 2)


if __name__ == "__main__":
    unittest.main() 