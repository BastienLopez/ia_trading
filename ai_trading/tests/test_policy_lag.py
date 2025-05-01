import unittest
import sys
import time
import torch
import numpy as np
from pathlib import Path
import threading

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.policy_lag import PolicyLag, DecoupledPolicyTrainer

# Définir un modèle simple pour les tests
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=2):
        super(SimpleModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        # Initialiser avec des poids aléatoires
        self.random_value = torch.rand(1).item()
    
    def forward(self, x):
        return self.layers(x)
        
    def get_random_value(self):
        return self.random_value
        
# Fonction de perte simple pour les tests
def simple_loss_fn(train_model, target_model, states, actions, rewards, next_states, dones):
    # Une fonction de perte simplifiée pour les tests
    q_values = train_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_states).max(1)[0]
    expected_q_values = rewards + 0.99 * next_q_values * (1 - dones)
    return torch.nn.functional.mse_loss(q_values, expected_q_values.detach())

class TestPolicyLag(unittest.TestCase):
    
    def setUp(self):
        """Prépare les objets de test"""
        # Créer un modèle simple pour les tests
        self.model = SimpleModel()
        
        # Initialiser le policy lag avec différentes configurations
        self.sync_policy_lag = PolicyLag(
            model=self.model,
            update_frequency=10,
            target_update_freq=20,
            async_update=False
        )
        
        self.async_policy_lag = PolicyLag(
            model=self.model,
            update_frequency=10,
            target_update_freq=20,
            async_update=True
        )
    
    def test_model_copy(self):
        """Teste la copie initiale des modèles"""
        # Vérifier que les modèles ont bien été copiés
        train_model = self.sync_policy_lag.get_train_model()
        collect_model = self.sync_policy_lag.get_collect_model()
        target_model = self.sync_policy_lag.get_target_model()
        
        # Les modèles doivent être des objets différents
        self.assertIsNot(train_model, collect_model)
        self.assertIsNot(train_model, target_model)
        self.assertIsNot(collect_model, target_model)
        
        # Mais ils doivent avoir le même comportement initial
        self.assertEqual(train_model.get_random_value(), collect_model.get_random_value())
        self.assertEqual(train_model.get_random_value(), target_model.get_random_value())
    
    def test_sync_update(self):
        """Teste la mise à jour synchrone des modèles"""
        # Créer un nouveau modèle pour ce test
        model = SimpleModel()
        
        # Initialiser le policy lag
        policy_lag = PolicyLag(
            model=model,
            update_frequency=5,
            target_update_freq=10,
            async_update=False
        )
        
        # Stocker la valeur aléatoire initiale
        initial_value = model.get_random_value()
        
        # Modifier la valeur aléatoire du modèle d'entraînement
        model.random_value = 0.9876
        
        # Vérifier que la modification n'a pas affecté les autres modèles
        self.assertEqual(model.get_random_value(), 0.9876)
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), initial_value)
        self.assertEqual(policy_lag.get_target_model().get_random_value(), initial_value)
        
        # Faire quelques pas de collecte
        for _ in range(4):
            policy_lag.collect_step()
        
        # Vérifier que le modèle de collecte n'a pas encore été mis à jour
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), initial_value)
        
        # Faire un pas de plus pour déclencher la mise à jour
        policy_lag.collect_step()
        
        # Vérifier que le modèle de collecte a été mis à jour
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), 0.9876)
        
        # Vérifier que le modèle cible n'a pas encore été mis à jour
        self.assertEqual(policy_lag.get_target_model().get_random_value(), initial_value)
        
        # Faire 10 pas d'entraînement pour déclencher la mise à jour du modèle cible
        for _ in range(10):
            policy_lag.train_step()
        
        # Vérifier que le modèle cible a été mis à jour
        self.assertEqual(policy_lag.get_target_model().get_random_value(), 0.9876)
    
    def test_async_update(self):
        """Teste la mise à jour asynchrone des modèles"""
        # Créer un nouveau modèle pour ce test
        model = SimpleModel()
        
        # Initialiser le policy lag avec mise à jour asynchrone
        policy_lag = PolicyLag(
            model=model,
            update_frequency=5,
            target_update_freq=10,
            async_update=True
        )
        
        # Stocker la valeur aléatoire initiale
        initial_value = model.get_random_value()
        
        # Modifier la valeur aléatoire du modèle d'entraînement
        model.random_value = 0.9876
        
        # Faire quelques pas de collecte
        for _ in range(4):
            policy_lag.collect_step()
        
        # Faire un pas de plus pour déclencher la mise à jour asynchrone
        policy_lag.collect_step()
        
        # Attendre un peu que la mise à jour asynchrone se termine
        time.sleep(0.5)
        
        # Vérifier que le modèle de collecte a été mis à jour
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), 0.9876)
        
        # Faire 10 pas d'entraînement pour déclencher la mise à jour du modèle cible
        for _ in range(10):
            policy_lag.train_step()
        
        # Attendre un peu que la mise à jour asynchrone se termine
        time.sleep(0.5)
        
        # Vérifier que le modèle cible a été mis à jour
        self.assertEqual(policy_lag.get_target_model().get_random_value(), 0.9876)
        
        # Arrêter le thread de mise à jour
        policy_lag.shutdown()
    
    def test_freeze_collect_model(self):
        """Teste le gel du modèle de collecte"""
        # Initialiser le policy lag
        policy_lag = PolicyLag(
            model=self.model,
            update_frequency=5,
            target_update_freq=10,
            async_update=False
        )
        
        # Stocker la valeur aléatoire initiale
        initial_value = self.model.get_random_value()
        
        # Modifier la valeur aléatoire du modèle d'entraînement
        self.model.random_value = 0.9876
        
        # Geler le modèle de collecte
        policy_lag.freeze_collect_model(True)
        
        # Faire suffisamment de pas pour déclencher une mise à jour
        for _ in range(5):
            policy_lag.collect_step()
        
        # Vérifier que le modèle de collecte n'a pas été mis à jour (car gelé)
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), initial_value)
        
        # Dégeler le modèle de collecte
        policy_lag.freeze_collect_model(False)
        
        # Faire suffisamment de pas pour déclencher une mise à jour
        for _ in range(5):
            policy_lag.collect_step()
        
        # Vérifier que le modèle de collecte a été mis à jour
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), 0.9876)
    
    def test_force_update(self):
        """Teste la mise à jour forcée des modèles"""
        # Initialiser le policy lag
        policy_lag = PolicyLag(
            model=self.model,
            update_frequency=1000,  # Valeur élevée pour ne pas déclencher de mise à jour automatique
            target_update_freq=1000,
            async_update=False
        )
        
        # Stocker la valeur aléatoire initiale
        initial_value = self.model.get_random_value()
        
        # Modifier la valeur aléatoire du modèle d'entraînement
        self.model.random_value = 0.9876
        
        # Vérifier que les modèles n'ont pas été mis à jour
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), initial_value)
        self.assertEqual(policy_lag.get_target_model().get_random_value(), initial_value)
        
        # Forcer la mise à jour du modèle de collecte
        policy_lag.force_update("collect")
        
        # Vérifier que le modèle de collecte a été mis à jour
        self.assertEqual(policy_lag.get_collect_model().get_random_value(), 0.9876)
        self.assertEqual(policy_lag.get_target_model().get_random_value(), initial_value)
        
        # Forcer la mise à jour du modèle cible
        policy_lag.force_update("target")
        
        # Vérifier que le modèle cible a été mis à jour
        self.assertEqual(policy_lag.get_target_model().get_random_value(), 0.9876)
    
    def test_get_metrics(self):
        """Teste la récupération des métriques"""
        # Faire quelques pas
        for _ in range(15):
            self.sync_policy_lag.collect_step()
            self.sync_policy_lag.train_step()
        
        # Récupérer les métriques
        metrics = self.sync_policy_lag.get_metrics()
        
        # Vérifier la présence des métriques importantes
        self.assertIn("collect_steps", metrics)
        self.assertIn("train_steps", metrics)
        self.assertIn("update_frequency", metrics)
        self.assertIn("target_update_freq", metrics)
        self.assertIn("is_async", metrics)
        self.assertIn("is_frozen", metrics)
        
        # Vérifier les valeurs
        self.assertEqual(metrics["collect_steps"], 15)
        self.assertEqual(metrics["train_steps"], 15)
        self.assertEqual(metrics["update_frequency"], 10)
        self.assertEqual(metrics["target_update_freq"], 20)
        self.assertFalse(metrics["is_async"])
        self.assertFalse(metrics["is_frozen"])
    
    def test_shutdown(self):
        """Teste l'arrêt propre"""
        # Initialiser le policy lag avec mise à jour asynchrone
        policy_lag = PolicyLag(
            model=self.model,
            update_frequency=5,
            target_update_freq=10,
            async_update=True
        )
        
        # Vérifier que le thread est en cours d'exécution
        self.assertTrue(policy_lag.update_thread.is_alive())
        
        # Arrêter
        policy_lag.shutdown()
        
        # Attendre un peu
        time.sleep(0.5)
        
        # Vérifier que le thread s'est arrêté
        self.assertFalse(policy_lag.running)
        
        # Vérifier qu'il n'y a pas d'erreur si on appelle shutdown plusieurs fois
        policy_lag.shutdown()


class TestDecoupledPolicyTrainer(unittest.TestCase):
    
    def setUp(self):
        """Prépare les objets de test"""
        # Créer un modèle simple
        self.model = SimpleModel()
        
        # Créer un optimiseur
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialiser le trainer
        self.trainer = DecoupledPolicyTrainer(
            model=self.model,
            optimizer=self.optimizer,
            update_frequency=10,
            target_update_freq=20,
            async_update=False,
            batch_queue_size=5
        )
        
        # Données de test
        self.states = torch.rand(32, 4)
        self.actions = torch.randint(0, 2, (32,))
        self.rewards = torch.rand(32)
        self.next_states = torch.rand(32, 4)
        self.dones = torch.zeros(32, dtype=torch.float32)
        
        # Batch de test
        self.batch = (self.states, self.actions, self.rewards, self.next_states, self.dones)
    
    def test_collect(self):
        """Teste la collecte d'actions"""
        # Créer un état de test
        state = torch.rand(4)
        
        # Collecter une action
        action, info = self.trainer.collect(state)
        
        # Vérifier que l'action est un tenseur
        self.assertIsInstance(action, torch.Tensor)
        
        # Vérifier que l'info contient les métriques
        self.assertIn("metrics", info)
        self.assertIn("updated", info)
        
        # Faire plusieurs collectes pour déclencher une mise à jour
        # Note: nous voulons atteindre update_frequency exactement
        # donc il faut calculer combien d'étapes restantes
        current_steps = self.trainer.policy_lag.collect_steps
        steps_needed = self.trainer.policy_lag.update_frequency - (current_steps % self.trainer.policy_lag.update_frequency)
        if steps_needed == self.trainer.policy_lag.update_frequency:
            steps_needed = 0
        
        for _ in range(steps_needed - 1):  # -1 car on va faire une dernière collecte après
            self.trainer.collect(state)
        
        # Cette collecte devrait déclencher une mise à jour
        action, info = self.trainer.collect(state)
        
        # Vérifier que la mise à jour a été déclenchée
        self.assertTrue(info["updated"])
    
    def test_train_sync(self):
        """Teste l'entraînement synchrone"""
        # Entraîner le modèle sur un batch
        loss = self.trainer.train_sync(self.batch, simple_loss_fn)
        
        # Vérifier que la perte est un nombre
        self.assertIsInstance(loss, float)
        
        # Vérifier que la perte a été ajoutée aux métriques
        self.assertEqual(len(self.trainer.metrics["train_loss"]), 1)
        self.assertEqual(self.trainer.metrics["train_loss"][0], loss)
    
    def test_async_training(self):
        """Teste l'entraînement asynchrone"""
        # Démarrer l'entraînement asynchrone
        self.trainer.start_training(simple_loss_fn)
        
        # Vérifier que l'entraînement est en cours
        self.assertTrue(self.trainer.is_training)
        
        # Ajouter quelques batchs
        for _ in range(3):
            self.trainer.add_batch(self.batch)
        
        # Attendre un peu que les batchs soient traités
        time.sleep(1.0)
        
        # Vérifier que des pertes ont été enregistrées
        self.assertGreater(len(self.trainer.metrics["train_loss"]), 0)
        
        # Arrêter l'entraînement
        self.trainer.stop_training()
        
        # Vérifier que l'entraînement est arrêté
        self.assertFalse(self.trainer.is_training)
    
    def test_add_batch(self):
        """Teste l'ajout de batchs à la queue"""
        # Démarrer l'entraînement
        self.trainer.start_training(simple_loss_fn)
        
        # Ajouter quelques batchs
        for _ in range(self.trainer.batch_queue.maxsize):
            result = self.trainer.add_batch(self.batch)
            self.assertTrue(result)
        
        # La queue est maintenant pleine, donc add_batch devrait échouer
        result = self.trainer.add_batch(self.batch)
        self.assertFalse(result)
        
        # Attendre que les batchs soient traités
        time.sleep(1.0)
        
        # Maintenant il devrait y avoir de la place
        result = self.trainer.add_batch(self.batch)
        self.assertTrue(result)
        
        # Arrêter l'entraînement
        self.trainer.stop_training()
    
    def test_get_metrics(self):
        """Teste la récupération des métriques combinées"""
        # Entraîner le modèle sur quelques batchs
        for _ in range(3):
            self.trainer.train_sync(self.batch, simple_loss_fn)
        
        # Récupérer les métriques
        metrics = self.trainer.get_metrics()
        
        # Vérifier la présence des métriques importantes
        self.assertIn("train_loss_avg", metrics)
        self.assertIn("collect_steps", metrics)
        self.assertIn("train_steps", metrics)
        self.assertIn("is_training", metrics)
        
        # Vérifier les valeurs
        self.assertFalse(metrics["is_training"])
        self.assertEqual(metrics["train_steps"], 3)
    
    def test_shutdown(self):
        """Teste l'arrêt propre"""
        # Démarrer l'entraînement
        self.trainer.start_training(simple_loss_fn)
        
        # Vérifier que l'entraînement est en cours
        self.assertTrue(self.trainer.is_training)
        
        # Arrêter
        self.trainer.shutdown()
        
        # Vérifier que l'entraînement est arrêté
        self.assertFalse(self.trainer.is_training)


if __name__ == "__main__":
    unittest.main() 