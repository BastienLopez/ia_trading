import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class NoisyLinear(nn.Module):
    """
    Implémentation PyTorch d'une couche linéaire bruitée pour l'exploration
    paramétrique dans l'apprentissage par renforcement.
    
    Cette couche ajoute du bruit aux poids et aux biais pour favoriser l'exploration
    sans avoir besoin d'une politique d'exploration externe (comme ε-greedy).
    
    Cette implémentation suit l'approche décrite dans le papier:
    "Noisy Networks for Exploration" (Fortunato et al., 2018)
    
    Deux types de bruit sont supportés:
    - Factoriel (default): utilise un bruit factoriel plus efficace en calcul
    - Indépendant: utilise un bruit indépendant pour chaque paramètre
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5, 
                 bias: bool = True, factorised_noise: bool = True, device: Optional[torch.device] = None):
        """
        Initialise une couche linéaire bruitée.
        
        Args:
            in_features (int): Nombre de features d'entrée
            out_features (int): Nombre de features de sortie
            sigma_init (float): Facteur d'initialisation pour les paramètres de bruit
            bias (bool): Si True, ajoute un biais à la sortie
            factorised_noise (bool): Si True, utilise du bruit factoriel (plus efficace)
            device (torch.device, optional): Périphérique sur lequel créer les tenseurs
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.factorised_noise = factorised_noise
        self.use_bias = bias
        
        # Paramètres déterministes (μ)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias_mu = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        
        # Paramètres de bruit (σ)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias_sigma = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        
        # Variables pour stocker le bruit généré
        if factorised_noise:
            self.register_buffer('epsilon_in', torch.empty(in_features, device=device))
            self.register_buffer('epsilon_out', torch.empty(out_features, device=device))
        else:
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features, device=device))
            self.register_buffer('bias_epsilon', torch.empty(out_features, device=device) if bias else None)
        
        # Initialisation des paramètres
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialise les paramètres déterministes et de bruit."""
        # Initialisation des μ avec des valeurs uniformes bornées (Kaiming)
        bound_mu = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound_mu, bound_mu)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-bound_mu, bound_mu)
        
        # Initialisation des σ avec une petite valeur constante
        bound_sigma = self.sigma_init / math.sqrt(self.in_features)
        
        # Initialisation plus optimale pour les facteurs de bruit
        if self.factorised_noise:
            self.weight_sigma.data.fill_(bound_sigma)
        else:
            # Approche plus stable avec un bruit supplémentaire initial
            self.weight_sigma.data.fill_(bound_sigma / math.sqrt(self.out_features))
            
        if self.bias_sigma is not None:
            self.bias_sigma.data.fill_(bound_sigma)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """
        Génère du bruit corrélé factoriel.
        
        Args:
            size (int): Taille du vecteur de bruit
        
        Returns:
            torch.Tensor: Vecteur de bruit transformé
        """
        # Utilise la transformation signed square root pour avoir un bruit approprié
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        """Génère un nouveau bruit pour les paramètres."""
        if self.factorised_noise:
            # Approche factorielle (plus efficace en mémoire et calcul)
            self.epsilon_in.copy_(self._scale_noise(self.in_features))
            self.epsilon_out.copy_(self._scale_noise(self.out_features))
        else:
            # Approche avec bruit indépendant pour chaque paramètre
            self.weight_epsilon.copy_(torch.randn_like(self.weight_epsilon))
            if self.bias_epsilon is not None:
                self.bias_epsilon.copy_(torch.randn_like(self.bias_epsilon))
    
    def get_noise(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Récupère les tenseurs de bruit actuels.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Bruit des poids et du biais
        """
        if self.factorised_noise:
            # Recalcul du bruit factoriel pour les poids
            weight_epsilon = torch.outer(self.epsilon_out, self.epsilon_in)
            bias_epsilon = self.epsilon_out if self.use_bias else None
        else:
            weight_epsilon = self.weight_epsilon
            bias_epsilon = self.bias_epsilon if self.use_bias else None
            
        return weight_epsilon, bias_epsilon
    
    def forward(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Propage l'entrée à travers la couche.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée
            deterministic (bool): Si True, n'utilise pas de bruit (pour l'évaluation)
        
        Returns:
            torch.Tensor: Sortie de la couche
        """
        if deterministic:
            return F.linear(x, self.weight_mu, self.bias_mu)
        
        # Calcule les poids bruités: μ + σ·ε
        if self.factorised_noise:
            # Produit vectoriel pour le bruit factoriel
            weight_epsilon, bias_epsilon = self.get_noise()
        else:
            weight_epsilon, bias_epsilon = self.weight_epsilon, self.bias_epsilon
            
        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        
        bias = None
        if self.bias_mu is not None:
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        
        return F.linear(x, weight, bias)
    
    def sample_noise(self, batch_size: int = 1) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Génère plusieurs échantillons de bruit pour les ensembles.
        
        Args:
            batch_size (int): Nombre d'échantillons de bruit à générer
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Bruit des poids et du biais pour chaque échantillon
        """
        if self.factorised_noise:
            epsilon_in_samples = torch.stack([self._scale_noise(self.in_features) 
                                              for _ in range(batch_size)])
            epsilon_out_samples = torch.stack([self._scale_noise(self.out_features) 
                                               for _ in range(batch_size)])
            
            # Reshape pour broadcast
            # [batch_size, out_features, 1] * [batch_size, 1, in_features]
            weight_epsilons = torch.bmm(
                epsilon_out_samples.unsqueeze(2),
                epsilon_in_samples.unsqueeze(1)
            )
            bias_epsilons = epsilon_out_samples if self.use_bias else None
        else:
            weight_epsilons = torch.randn(batch_size, self.out_features, self.in_features, 
                                          device=self.weight_mu.device)
            bias_epsilons = torch.randn(batch_size, self.out_features, 
                                        device=self.weight_mu.device) if self.use_bias else None
                                        
        return weight_epsilons, bias_epsilons
        
    def extra_repr(self) -> str:
        """Représentation textuelle des paramètres de la couche."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'sigma_init={self.sigma_init}, factorised_noise={self.factorised_noise}')


if __name__ == "__main__":
    # Test simple
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique: {device}")
    
    # Test des deux modes de bruit
    for factorised in [True, False]:
        print(f"\nTest avec bruit {'factoriel' if factorised else 'indépendant'}")
        layer = NoisyLinear(10, 5, factorised_noise=factorised, device=device)
        x = torch.randn(3, 10, device=device)  # Batch de 3, 10 features
        
        # Sortie déterministe
        y_det = layer(x, deterministic=True)
        print(f"Sortie déterministe: {y_det.shape}")
        
        # Sortie avec bruit
        y_noisy = layer(x, deterministic=False)
        print(f"Sortie bruitée: {y_noisy.shape}")
        
        # Réinitialiser le bruit et obtenir une nouvelle sortie bruitée
        layer.reset_noise()
        y_noisy2 = layer(x, deterministic=False)
        
        # Vérifier que les sorties bruitées sont différentes
        diff = (y_noisy - y_noisy2).abs().mean().item()
        print(f"Différence entre deux sorties bruitées: {diff}")
        
    # Test de l'échantillonnage multiple de bruit
    layer = NoisyLinear(10, 5, device=device)
    batch_size = 4
    weight_eps, bias_eps = layer.sample_noise(batch_size)
    print(f"\nÉchantillons de bruit multiples - forme poids: {weight_eps.shape}")
    if bias_eps is not None:
        print(f"Échantillons de bruit multiples - forme biais: {bias_eps.shape}") 