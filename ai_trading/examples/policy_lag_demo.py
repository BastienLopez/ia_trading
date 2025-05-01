import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import threading

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.policy_lag import PolicyLag, DecoupledPolicyTrainer

# Créer un environnement simple pour la démonstration
class SimpleEnv:
    """
    Environnement simple pour tester le RL
    L'agent doit atteindre une position cible en 1D
    """
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.position = 0
        self.target = 10
        self.steps = 0
        
    def reset(self):
        self.position = 0
        self.steps = 0
        return self._get_state()
    
    def step(self, action):
        # Action -1: gauche, 0: rester, 1: droite
        self.position += action
        self.steps += 1
        
        # Calculer la récompense
        distance = abs(self.position - self.target)
        reward = -0.1 - 0.1 * distance  # Pénalité pour chaque pas + distance
        
        if distance < 0.5:
            reward += 10  # Bonus pour atteindre la cible
            done = True
        elif self.steps >= self.max_steps:
            done = True
        else:
            done = False
            
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        # État: position relative à la cible et vitesse
        return torch.tensor([self.position, self.target - self.position], dtype=torch.float32)

# Modèle de policy simple
class PolicyModel(torch.nn.Module):
    def __init__(self, state_dim=2, action_dim=3, hidden_dim=64):
        super(PolicyModel, self).__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

# Fonction de perte pour Q-learning
def q_loss_fn(train_model, target_model, states, actions, rewards, next_states, dones):
    # Q-valeurs pour les actions prises
    q_values = train_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Q-valeurs cibles
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]
        targets = rewards + 0.99 * next_q_values * (1 - dones)
    
    # Perte MSE
    loss = torch.nn.functional.mse_loss(q_values, targets)
    return loss

# Fonction d'exploration epsilon-greedy
def epsilon_greedy(q_values, epsilon=0.1):
    """
    Stratégie epsilon-greedy pour l'exploration.
    
    Args:
        q_values: Valeurs Q pour chaque action
        epsilon: Probabilité d'exploration
        
    Returns:
        Action choisie
    """
    if np.random.random() < epsilon:
        # Exploration: action aléatoire
        return torch.randint(0, q_values.shape[-1], (1,)).item()
    else:
        # Exploitation: meilleure action
        return q_values.argmax().item()

# Fonction pour exécuter un épisode
def run_episode(env, trainer, epsilon=0.1, render=False):
    """
    Exécute un épisode complet avec le trainer donné.
    
    Args:
        env: Environnement
        trainer: DecoupledPolicyTrainer
        epsilon: Probabilité d'exploration
        render: Afficher l'environnement
        
    Returns:
        (total_reward, episode_length, metrics)
    """
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    metrics_history = []
    
    while not done:
        # Prédire l'action avec le modèle de collecte
        q_values, info = trainer.collect(state)
        
        # Appliquer epsilon-greedy
        action_idx = epsilon_greedy(q_values, epsilon)
        
        # Convertir l'indice d'action en action réelle (-1, 0, 1)
        action = action_idx - 1
        
        # Exécuter l'action
        next_state, reward, done, _ = env.step(action)
        
        # Stocker l'expérience
        batch = (
            state.unsqueeze(0),
            torch.tensor([action_idx], dtype=torch.long),
            torch.tensor([reward], dtype=torch.float32),
            next_state.unsqueeze(0),
            torch.tensor([float(done)], dtype=torch.float32)
        )
        
        # Ajouter le batch à la queue d'entraînement ou entraîner directement
        if trainer.is_training:
            trainer.add_batch(batch)
        else:
            trainer.train_sync(batch, q_loss_fn)
        
        # Mise à jour pour la prochaine étape
        state = next_state
        total_reward += reward
        step += 1
        
        # Enregistrer les métriques
        metrics = trainer.get_metrics()
        metrics_history.append(metrics)
        
        # Afficher l'environnement si demandé
        if render:
            print(f"Step {step}: Position {env.position:.1f}, Action {action}, Reward {reward:.2f}")
    
    return total_reward, step, metrics_history

# Fonction pour entraîner l'agent
def train_agent(args):
    """
    Entraîne un agent avec différentes configurations de policy lag.
    
    Args:
        args: Arguments de ligne de commande
    """
    # Créer l'environnement
    env = SimpleEnv(max_steps=args.max_steps)
    
    # Créer le modèle et l'optimiseur
    model = PolicyModel(state_dim=2, action_dim=3, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Sélectionner le mode de mise à jour
    if args.mode == "sync":
        # Mise à jour synchrone
        trainer = DecoupledPolicyTrainer(
            model=model,
            optimizer=optimizer,
            update_frequency=args.update_freq,
            target_update_freq=args.target_update_freq,
            async_update=False
        )
        # Entraînement synchrone
        async_training = False
    elif args.mode == "async":
        # Mise à jour asynchrone
        trainer = DecoupledPolicyTrainer(
            model=model,
            optimizer=optimizer,
            update_frequency=args.update_freq,
            target_update_freq=args.target_update_freq,
            async_update=True
        )
        # Entraînement asynchrone
        async_training = True
    else:
        # Mode de référence: pas de policy lag (mise à jour à chaque étape)
        trainer = DecoupledPolicyTrainer(
            model=model,
            optimizer=optimizer,
            update_frequency=1,  # Mise à jour à chaque étape
            target_update_freq=args.target_update_freq,
            async_update=False
        )
        # Entraînement synchrone
        async_training = False
    
    # Démarrer l'entraînement asynchrone si demandé
    if async_training:
        trainer.start_training(q_loss_fn)
    
    # Boucle d'entraînement
    rewards = []
    steps = []
    metrics_history = []
    
    print(f"Entraînement en mode: {args.mode}")
    print(f"Fréquence de mise à jour: {args.update_freq}")
    print(f"Fréquence de mise à jour du modèle cible: {args.target_update_freq}")
    
    # Entraîner pour le nombre d'épisodes demandé
    for episode in range(args.episodes):
        # Décroissance d'epsilon
        epsilon = max(0.05, args.epsilon * (1 - episode / args.episodes))
        
        # Exécuter un épisode
        episode_reward, episode_steps, episode_metrics = run_episode(
            env, trainer, epsilon=epsilon, render=(episode % args.render_freq == 0)
        )
        
        # Enregistrer les résultats
        rewards.append(episode_reward)
        steps.append(episode_steps)
        metrics_history.extend(episode_metrics)
        
        # Afficher les résultats périodiquement
        if episode % args.log_freq == 0 or episode == args.episodes - 1:
            avg_reward = sum(rewards[-args.log_freq:]) / min(args.log_freq, len(rewards[-args.log_freq:]))
            avg_steps = sum(steps[-args.log_freq:]) / min(args.log_freq, len(steps[-args.log_freq:]))
            print(f"Episode {episode+1}/{args.episodes}, "
                  f"Reward {episode_reward:.2f}, "
                  f"Avg Reward {avg_reward:.2f}, "
                  f"Steps {episode_steps}, "
                  f"Avg Steps {avg_steps:.1f}, "
                  f"Epsilon {epsilon:.3f}")
    
    # Arrêter l'entraînement asynchrone si nécessaire
    if async_training:
        trainer.stop_training()
    
    # Arrêter proprement
    trainer.shutdown()
    
    return rewards, steps, metrics_history

# Fonction pour comparer les différentes configurations
def compare_configurations(args):
    """
    Compare différentes configurations de policy lag.
    
    Args:
        args: Arguments de ligne de commande
    """
    # Configurations à comparer
    configs = [
        {"name": "Sans Policy Lag", "mode": "baseline", "update_freq": 1},
        {"name": "Policy Lag Sync", "mode": "sync", "update_freq": args.update_freq},
        {"name": "Policy Lag Async", "mode": "async", "update_freq": args.update_freq}
    ]
    
    # Conserver les résultats
    all_rewards = {}
    all_times = {}
    
    # Exécuter l'entraînement pour chaque configuration
    for config in configs:
        print(f"\n=== Configuration: {config['name']} ===")
        
        # Configurer les arguments
        args.mode = config["mode"]
        args.update_freq = config["update_freq"]
        
        # Mesurer le temps d'exécution
        start_time = time.time()
        
        # Entraîner l'agent
        rewards, steps, metrics = train_agent(args)
        
        # Enregistrer le temps d'exécution
        execution_time = time.time() - start_time
        
        # Stocker les résultats
        all_rewards[config["name"]] = rewards
        all_times[config["name"]] = execution_time
        
        print(f"Temps d'exécution: {execution_time:.2f} secondes")
    
    # Afficher les résultats
    plt.figure(figsize=(12, 8))
    
    # Graphique des récompenses
    plt.subplot(2, 1, 1)
    for name, rewards in all_rewards.items():
        plt.plot(rewards, label=name)
    
    plt.title("Comparaison des récompenses par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense")
    plt.legend()
    plt.grid(True)
    
    # Graphique des temps d'exécution
    plt.subplot(2, 1, 2)
    names = list(all_times.keys())
    times = [all_times[name] for name in names]
    
    plt.bar(names, times)
    plt.title("Temps d'exécution total")
    plt.ylabel("Secondes")
    plt.xticks(rotation=30)
    
    plt.tight_layout()
    plt.savefig("policy_lag_comparison.png")
    
    print(f"Graphiques enregistrés dans policy_lag_comparison.png")
    if args.show_plots:
        plt.show()

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Démonstration du Policy Lag")
    
    # Configuration générale
    parser.add_argument("--episodes", type=int, default=100, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--max-steps", type=int, default=100, help="Nombre maximum d'étapes par épisode")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Epsilon initial pour l'exploration")
    
    # Configuration du policy lag
    parser.add_argument("--mode", type=str, default="async", choices=["sync", "async", "baseline"],
                      help="Mode de mise à jour (sync: synchrone, async: asynchrone, baseline: sans policy lag)")
    parser.add_argument("--update-freq", type=int, default=10, help="Fréquence de mise à jour du modèle de collecte")
    parser.add_argument("--target-update-freq", type=int, default=100, help="Fréquence de mise à jour du modèle cible")
    
    # Configuration de l'entraînement
    parser.add_argument("--batch-size", type=int, default=32, help="Taille des batchs d'entraînement")
    parser.add_argument("--log-freq", type=int, default=10, help="Fréquence d'affichage des logs")
    parser.add_argument("--render-freq", type=int, default=20, help="Fréquence d'affichage de l'environnement")
    
    # Configuration de l'expérience
    parser.add_argument("--compare", action="store_true", help="Comparer différentes configurations")
    parser.add_argument("--show-plots", action="store_true", help="Afficher les graphiques")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations(args)
    else:
        rewards, steps, metrics = train_agent(args)
        
        # Afficher les résultats
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title("Récompenses par épisode")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps)
        plt.title("Nombre d'étapes par épisode")
        plt.xlabel("Épisode")
        plt.ylabel("Étapes")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("policy_lag_results.png")
        
        print(f"Graphiques enregistrés dans policy_lag_results.png")
        if args.show_plots:
            plt.show()

if __name__ == "__main__":
    main() 