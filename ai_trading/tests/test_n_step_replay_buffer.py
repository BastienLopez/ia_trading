import unittest
import numpy as np
import pandas as pd
from ai_trading.rl.agents.n_step_replay_buffer import NStepReplayBuffer

class TestNStepReplayBuffer(unittest.TestCase):
    """Tests pour le tampon de replay avec retours multi-étapes"""
    
    def setUp(self):
        """Configuration pour les tests"""
        self.buffer_size = 1000
        self.n_steps = 3
        self.gamma = 0.9
        
        # Créer un tampon de replay
        self.buffer = NStepReplayBuffer(
            buffer_size=self.buffer_size,
            n_steps=self.n_steps,
            gamma=self.gamma
        )
        
        # Valeurs pour les tests
        self.state_size = 4
        self.action_size = 2
        
    def test_initialization(self):
        """Teste l'initialisation correcte du tampon"""
        self.assertEqual(self.buffer.n_steps, self.n_steps)
        self.assertEqual(self.buffer.gamma, self.gamma)
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(len(self.buffer.n_step_buffer), 0)
    
    def test_add_single_experience(self):
        """Teste l'ajout d'une expérience unique"""
        # Créer une expérience
        state = np.random.random(self.state_size)
        action = np.random.random(self.action_size)
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False
        
        # Ajouter l'expérience
        self.buffer.add(state, action, reward, next_state, done)
        
        # Le tampon n'est pas encore plein, donc rien ne devrait être ajouté au tampon principal
        self.assertEqual(len(self.buffer), 0)
        
        # Mais le tampon temporaire devrait contenir l'expérience
        self.assertEqual(len(self.buffer.n_step_buffer), 1)
    
    def test_n_step_return_calculation(self):
        """Teste le calcul du retour sur n étapes"""
        # Ajouter des expériences
        states = [np.random.random(self.state_size) for _ in range(self.n_steps)]
        actions = [np.random.random(self.action_size) for _ in range(self.n_steps)]
        rewards = [1.0, 2.0, 3.0]  # Récompenses croissantes pour faciliter la vérification
        next_states = [np.random.random(self.state_size) for _ in range(self.n_steps)]
        dones = [False, False, False]
        
        # Ajouter les n-1 premières expériences
        for i in range(self.n_steps - 1):
            self.buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
            # Le tampon principal devrait être vide
            self.assertEqual(len(self.buffer), 0)
        
        # Ajouter la dernière expérience qui déclenchera le calcul du retour
        self.buffer.add(states[-1], actions[-1], rewards[-1], next_states[-1], dones[-1])
        
        # Maintenant le tampon principal devrait contenir une expérience
        self.assertEqual(len(self.buffer), 1)
        
        # Vérifier que l'expérience dans le tampon principal a la récompense accumulée correcte
        expected_reward = rewards[0] + self.gamma * rewards[1] + (self.gamma ** 2) * rewards[2]
        
        # Échantillonner l'expérience et vérifier la récompense
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.buffer.sample(1)
        self.assertAlmostEqual(sampled_rewards[0], expected_reward)
        
        # Vérifier que l'état et l'action sont ceux de la première expérience
        np.testing.assert_array_equal(sampled_states[0], states[0])
        np.testing.assert_array_equal(sampled_actions[0], actions[0])
        
        # Vérifier que l'état suivant est celui de la dernière expérience
        np.testing.assert_array_equal(sampled_next_states[0], next_states[-1])
    
    def test_early_termination(self):
        """Teste que le calcul s'arrête correctement si un épisode se termine avant n étapes"""
        # Ajouter des expériences avec une terminaison au milieu
        states = [np.random.random(self.state_size) for _ in range(self.n_steps)]
        actions = [np.random.random(self.action_size) for _ in range(self.n_steps)]
        rewards = [1.0, 2.0, 3.0]
        next_states = [np.random.random(self.state_size) for _ in range(self.n_steps)]
        dones = [False, True, False]  # L'épisode se termine à la deuxième étape
        
        # Ajouter toutes les expériences
        for i in range(self.n_steps):
            self.buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # Le tampon principal devrait contenir une expérience
        self.assertEqual(len(self.buffer), 1)
        
        # Vérifier que l'expérience dans le tampon a la récompense accumulée correcte et done=True
        expected_reward = rewards[0] + self.gamma * rewards[1]  # Seulement 2 récompenses car l'épisode se termine
        
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.buffer.sample(1)
        self.assertAlmostEqual(sampled_rewards[0], expected_reward)
        self.assertTrue(sampled_dones[0])  # Le flag done doit être True
    
    def test_handle_episode_end(self):
        """Teste la gestion de la fin d'un épisode"""
        # Ajouter quelques expériences mais pas assez pour remplir le tampon temporaire
        states = [np.random.random(self.state_size) for _ in range(2)]  # Seulement 2 expériences
        actions = [np.random.random(self.action_size) for _ in range(2)]
        rewards = [1.0, 2.0]
        next_states = [np.random.random(self.state_size) for _ in range(2)]
        dones = [False, False]
        
        for i in range(2):
            self.buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # Le tampon principal devrait être vide
        self.assertEqual(len(self.buffer), 0)
        
        # Appel à handle_episode_end qui devrait traiter les expériences restantes
        self.buffer.handle_episode_end()
        
        # Le tampon principal devrait maintenant contenir les expériences
        self.assertEqual(len(self.buffer), 2)
        
        # Vérifier les récompenses
        expected_reward_1 = rewards[0] + self.gamma * rewards[1]
        expected_reward_2 = rewards[1]
        
        # Échantillonner les expériences
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.buffer.sample(2)
        
        # Trier les expériences par récompense pour les comparer
        sorted_indices = np.argsort(sampled_rewards)
        sorted_rewards = sampled_rewards[sorted_indices]
        
        # Vérifier les récompenses
        self.assertAlmostEqual(sorted_rewards[1], expected_reward_1)
        self.assertAlmostEqual(sorted_rewards[0], expected_reward_2)
    
    def test_clear_n_step_buffer(self):
        """Teste que la méthode clear_n_step_buffer vide correctement le tampon temporaire"""
        # Ajouter quelques expériences
        state = np.random.random(self.state_size)
        action = np.random.random(self.action_size)
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False
        
        self.buffer.add(state, action, reward, next_state, done)
        self.assertEqual(len(self.buffer.n_step_buffer), 1)
        
        # Vider le tampon temporaire
        self.buffer.clear_n_step_buffer()
        self.assertEqual(len(self.buffer.n_step_buffer), 0)
    
    def test_large_buffer(self):
        """Teste le comportement avec un grand nombre d'expériences"""
        # Générer un grand nombre d'expériences
        num_experiences = 100
        
        for i in range(num_experiences):
            state = np.random.random(self.state_size)
            action = np.random.random(self.action_size)
            reward = float(i % 5)  # Récompenses cycliques
            next_state = np.random.random(self.state_size)
            done = (i % 20 == 0)  # Terminaison tous les 20 pas
            
            self.buffer.add(state, action, reward, next_state, done)
            
            # Si l'épisode se termine, traiter les expériences restantes
            if done:
                self.buffer.handle_episode_end()
        
        # Traiter les expériences restantes à la fin
        self.buffer.handle_episode_end()
        
        # Vérifier que le tampon contient le bon nombre d'expériences
        # Note: Le tampon peut contenir plus d'expériences que num_experiences
        # car handle_episode_end() peut générer des entrées supplémentaires
        self.assertGreater(len(self.buffer), 0)
        
        # Tester l'échantillonnage d'un lot
        batch_size = 32
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Vérifier les dimensions du lot
        self.assertEqual(states.shape[0], min(batch_size, len(self.buffer)))
        self.assertEqual(actions.shape[0], min(batch_size, len(self.buffer)))
        self.assertEqual(rewards.shape[0], min(batch_size, len(self.buffer)))
        self.assertEqual(next_states.shape[0], min(batch_size, len(self.buffer)))
        self.assertEqual(dones.shape[0], min(batch_size, len(self.buffer)))

if __name__ == '__main__':
    unittest.main() 