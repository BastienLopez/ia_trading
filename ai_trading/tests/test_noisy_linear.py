import unittest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(str(Path(__file__).parent.parent))

from rl.agents.layers.noisy_linear import NoisyLinear


class TestNoisyLinear(unittest.TestCase):
    """Tests unitaires pour la couche NoisyLinear."""
    
    def setUp(self):
        """Initialisation des tests."""
        # Fixer la graine aléatoire pour la reproductibilité
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Paramètres par défaut pour les tests
        self.in_features = 10
        self.out_features = 5
        self.sigma_init = 0.5
        
        # Créer une instance de la couche avec factorised_noise=False pour garder la compatibilité avec les tests existants
        self.layer = NoisyLinear(
            in_features=self.in_features, 
            out_features=self.out_features,
            sigma_init=self.sigma_init,
            factorised_noise=False  # Utiliser l'implémentation avec bruit direct plutôt que factoriel
        )
        
        # Données d'entrée de test
        self.batch_size = 3
        self.x = torch.randn(self.batch_size, self.in_features)
    
    def test_initialization(self):
        """Vérifie que l'initialisation de la couche est correcte."""
        # Vérifier que les dimensions sont correctes
        self.assertEqual(self.layer.weight_mu.shape, (self.out_features, self.in_features))
        self.assertEqual(self.layer.bias_mu.shape, (self.out_features,))
        self.assertEqual(self.layer.weight_sigma.shape, (self.out_features, self.in_features))
        self.assertEqual(self.layer.bias_sigma.shape, (self.out_features,))
        
        # Vérifier que les paramètres de bruit sont initialisés correctement
        # Note: avec factorised_noise=False, l'initialisation est légèrement différente
        bound_sigma = self.sigma_init / np.sqrt(self.in_features) / np.sqrt(self.out_features)
        self.assertTrue(torch.allclose(self.layer.weight_sigma, torch.full_like(self.layer.weight_sigma, bound_sigma)))
        
        # Pour le biais, l'initialisation reste la même
        bound_sigma_bias = self.sigma_init / np.sqrt(self.in_features)
        self.assertTrue(torch.allclose(self.layer.bias_sigma, torch.full_like(self.layer.bias_sigma, bound_sigma_bias)))
    
    def test_deterministic_forward(self):
        """Vérifie que le forward déterministe donne toujours le même résultat."""
        # Premier passage
        output1 = self.layer(self.x, deterministic=True)
        
        # Réinitialiser le bruit ne devrait pas affecter le résultat déterministe
        self.layer.reset_noise()
        output2 = self.layer(self.x, deterministic=True)
        
        # Les deux sorties doivent être identiques
        self.assertTrue(torch.allclose(output1, output2))
        
        # Vérifier les dimensions de sortie
        self.assertEqual(output1.shape, (self.batch_size, self.out_features))
    
    def test_stochastic_forward(self):
        """Vérifie que le forward stochastique donne des résultats différents après reset_noise."""
        # Premier passage avec bruit
        output1 = self.layer(self.x, deterministic=False)
        
        # Réinitialiser le bruit
        self.layer.reset_noise()
        
        # Deuxième passage avec nouveau bruit
        output2 = self.layer(self.x, deterministic=False)
        
        # Les deux sorties doivent être différentes
        self.assertFalse(torch.allclose(output1, output2))
        
        # Vérifier les dimensions de sortie
        self.assertEqual(output1.shape, (self.batch_size, self.out_features))
        self.assertEqual(output2.shape, (self.batch_size, self.out_features))
    
    def test_noise_consistency(self):
        """Vérifie que le bruit est cohérent pour différents lots avec le même bruit."""
        # Premier lot avec bruit
        output1 = self.layer(self.x, deterministic=False)
        
        # Deuxième lot d'entrée différent
        x2 = torch.randn(self.batch_size, self.in_features)
        
        # Sans reset_noise, le bruit doit être le même
        output2 = self.layer(x2, deterministic=False)
        
        # Les paramètres bruités doivent être les mêmes pour les deux passages
        weight_epsilon, bias_epsilon = self.layer.get_noise()
        weight = self.layer.weight_mu + self.layer.weight_sigma * weight_epsilon
        bias = self.layer.bias_mu + self.layer.bias_sigma * bias_epsilon
        
        # Calculer manuellement la sortie attendue pour x2
        expected_output = torch.matmul(x2, weight.t()) + bias
        
        # La sortie calculée doit correspondre à la sortie attendue
        self.assertTrue(torch.allclose(output2, expected_output))
    
    def test_training_parameters(self):
        """Vérifie que les paramètres sont bien entraînables."""
        # Tous les paramètres mu et sigma doivent nécessiter des gradients
        self.assertTrue(self.layer.weight_mu.requires_grad)
        self.assertTrue(self.layer.bias_mu.requires_grad)
        self.assertTrue(self.layer.weight_sigma.requires_grad)
        self.assertTrue(self.layer.bias_sigma.requires_grad)
        
        # Les epsilon ne doivent pas nécessiter de gradients
        weight_epsilon, bias_epsilon = self.layer.get_noise()
        self.assertFalse(weight_epsilon.requires_grad)
        self.assertFalse(bias_epsilon.requires_grad)
    
    def test_backpropagation(self):
        """Vérifie que la rétropropagation fonctionne correctement."""
        # Définir un optimiseur simple
        optimizer = torch.optim.SGD([
            {'params': self.layer.parameters()}
        ], lr=0.01)
        
        # Enregistrer les valeurs initiales
        weight_mu_init = self.layer.weight_mu.clone().detach()
        bias_mu_init = self.layer.bias_mu.clone().detach()
        weight_sigma_init = self.layer.weight_sigma.clone().detach()
        bias_sigma_init = self.layer.bias_sigma.clone().detach()
        
        # Forward pass
        output = self.layer(self.x, deterministic=False)
        
        # Une fonction de perte simple
        loss = output.sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Vérifier que les paramètres ont été mis à jour
        self.assertFalse(torch.allclose(self.layer.weight_mu, weight_mu_init))
        self.assertFalse(torch.allclose(self.layer.bias_mu, bias_mu_init))
        self.assertFalse(torch.allclose(self.layer.weight_sigma, weight_sigma_init))
        self.assertFalse(torch.allclose(self.layer.bias_sigma, bias_sigma_init))


if __name__ == "__main__":
    unittest.main() 