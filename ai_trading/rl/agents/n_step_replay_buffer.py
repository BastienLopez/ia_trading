import numpy as np
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class NStepReplayBuffer:
    """
    Tampon de replay qui stocke les transitions et calcule les retours sur n étapes.
    
    Cette implémentation accumule les récompenses sur n étapes et utilise un 
    facteur d'actualisation pour calculer les retours multi-étapes, ce qui peut
    aider à propager les récompenses plus rapidement et améliorer l'apprentissage.
    
    Référence:
        "Rainbow: Combining Improvements in Deep Reinforcement Learning"
        https://arxiv.org/abs/1710.02298
    """
    
    def __init__(self, buffer_size=100000, n_steps=3, gamma=0.99):
        """
        Initialise le tampon de replay avec retours multi-étapes.
        
        Args:
            buffer_size (int): Taille maximale du tampon
            n_steps (int): Nombre d'étapes pour accumuler les récompenses
            gamma (float): Facteur d'actualisation pour les récompenses futures
        """
        self.buffer = deque(maxlen=buffer_size)
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Tampon temporaire pour accumuler les n transitions les plus récentes
        self.n_step_buffer = deque(maxlen=n_steps)
        
        logger.info(f"Tampon de replay à {n_steps} étapes initialisé: buffer_size={buffer_size}, gamma={gamma}")
    
    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience au tampon temporaire et calcule les retours sur n étapes
        si nécessaire.
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        # Stocker l'expérience dans le tampon temporaire
        experience = (state, action, reward, next_state, done)
        self.n_step_buffer.append(experience)
        
        # Si le tampon temporaire n'est pas encore plein, on ne fait rien de plus
        if len(self.n_step_buffer) < self.n_steps:
            return
        
        # Calculer le retour sur n étapes
        reward, next_state, done = self._calculate_n_step_return()
        
        # Récupérer l'état et l'action initiale
        state, action, _, _, _ = self.n_step_buffer[0]
        
        # Ajouter l'expérience avec retour sur n étapes au tampon principal
        self.buffer.append((state, action, reward, next_state, done))
    
    def _calculate_n_step_return(self):
        """
        Calcule le retour sur n étapes pour le tampon temporaire actuel.
        
        Returns:
            tuple: (récompense accumulée, état final, indicateur de fin)
        """
        # Initialiser avec les valeurs du dernier état
        _, _, _, last_next_state, last_done = self.n_step_buffer[-1]
        
        # Calculer la récompense accumulée
        n_step_reward = 0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            # Accumuler les récompenses avec actualisation
            n_step_reward += r * (self.gamma ** i)
            
            # Si un épisode se termine avant les n étapes, on s'arrête là
            if d:
                return n_step_reward, last_next_state, True
        
        # Retourner la récompense accumulée, le dernier état et l'indicateur de fin
        return n_step_reward, last_next_state, last_done
    
    def sample(self, batch_size):
        """
        Échantillonne un lot d'expériences aléatoirement.
        
        Args:
            batch_size (int): Taille du lot à échantillonner
            
        Returns:
            tuple: Contient (états, actions, récompenses, états suivants, indicateurs de fin)
        """
        # S'assurer que nous avons suffisamment d'expériences
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Échantillonner aléatoirement
        batch = random.sample(self.buffer, batch_size)
        
        # Séparer les composants
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def clear_n_step_buffer(self):
        """
        Vide le tampon temporaire de n étapes.
        Utile entre les épisodes ou en cas de changement d'environnement.
        """
        self.n_step_buffer.clear()
    
    def handle_episode_end(self):
        """
        Gère la fin d'un épisode en traitant les expériences restantes dans le tampon temporaire.
        Cette méthode doit être appelée à la fin de chaque épisode.
        """
        # Traiter chaque expérience restante dans le tampon temporaire
        while len(self.n_step_buffer) > 0:
            # Calculer le retour pour les expériences restantes
            reward, next_state, done = self._calculate_n_step_return()
            
            # Récupérer l'état et l'action initiale
            state, action, _, _, _ = self.n_step_buffer[0]
            
            # Ajouter l'expérience au tampon principal
            self.buffer.append((state, action, reward, next_state, done))
            
            # Retirer la première expérience du tampon temporaire
            self.n_step_buffer.popleft()
    
    def __len__(self):
        """
        Retourne la taille du tampon principal.
        
        Returns:
            int: Nombre d'expériences dans le tampon
        """
        return len(self.buffer) 