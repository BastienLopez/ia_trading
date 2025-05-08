"""
Démonstration de l'utilisation des tampons de replay optimisés.

Ce script compare les performances des différents tampons de replay:
1. ReplayBuffer standard
2. PrioritizedReplayBuffer classique
3. DiskReplayBuffer pour les grands volumes de données
4. EnhancedPrioritizedReplay avec optimisations avancées
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Configurer le logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ReplayDemo")

from ai_trading.rl.disk_replay_buffer import DiskReplayBuffer
from ai_trading.rl.enhanced_prioritized_replay import EnhancedPrioritizedReplay

# Importer les tampons de replay
from ai_trading.rl.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


# Définir un modèle simple pour les tests
class SimpleModel(nn.Module):
    """
    Modèle simple pour démontrer l'apprentissage avec différents tampons.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class SimpleStateEncoder(nn.Module):
    """
    Encodeur d'état simple pour l'enhanced replay buffer.
    """

    def __init__(self, input_dim, encoding_dim=8):
        super(SimpleStateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
        )

    def forward(self, x):
        return self.encoder(x)


def generate_transitions(num_transitions, state_dim, action_dim, noise=0.1):
    """
    Génère des transitions aléatoires pour tester les tampons.

    Args:
        num_transitions: Nombre de transitions à générer
        state_dim: Dimension des états
        action_dim: Dimension des actions
        noise: Niveau de bruit dans les données

    Returns:
        tuple: Arrays de (states, actions, rewards, next_states, dones)
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    # Créer un modèle qui détermine les récompenses pour la cohérence
    reward_model = SimpleModel(state_dim + action_dim, 1)

    for i in range(num_transitions):
        # Générer un état aléatoire avec quelques motifs
        if i % 100 < 50:
            # Premier type d'état
            state = np.random.normal(0.5, noise, state_dim)
        else:
            # Second type d'état
            state = np.random.normal(-0.5, noise, state_dim)

        # Générer une action qui a une légère corrélation avec l'état
        if np.mean(state) > 0:
            action = np.random.normal(0.5, noise, action_dim)
        else:
            action = np.random.normal(-0.5, noise, action_dim)

        # Calculer la récompense basée sur l'état et l'action
        combined = np.concatenate([state, action])
        reward = reward_model(torch.FloatTensor(combined)).item()

        # Générer l'état suivant
        next_state = state + action * 0.1 + np.random.normal(0, noise, state_dim)

        # Déterminer si l'épisode est terminé
        done = i % 20 == 19  # Terminer tous les 20 pas

        states.append(state.astype(np.float32))
        actions.append(action.astype(np.float32))
        rewards.append(reward)
        next_states.append(next_state.astype(np.float32))
        dones.append(done)

    return (
        np.array(states),
        np.array(actions),
        np.array(rewards),
        np.array(next_states),
        np.array(dones),
    )


def train_model_with_buffer(
    buffer, model, target_model, num_updates, batch_size, gamma=0.99
):
    """
    Entraîne un modèle en échantillonnant depuis un tampon de replay.

    Args:
        buffer: Tampon de replay à utiliser
        model: Modèle à entraîner
        target_model: Modèle cible pour les calculs de Q-learning
        num_updates: Nombre de mises à jour
        batch_size: Taille du batch
        gamma: Facteur d'actualisation

    Returns:
        tuple: (pertes, temps d'échantillonnage, taille effective du buffer)
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    sampling_times = []
    buffer_sizes = [len(buffer)]

    for update in range(num_updates):
        # Mesurer le temps d'échantillonnage
        t_start = time.time()

        # Échantillonner depuis le buffer
        if isinstance(buffer, (PrioritizedReplayBuffer, EnhancedPrioritizedReplay)):
            indices, batch, is_weights = buffer.sample(batch_size)
            # Extraire les composants du batch
            states = np.array([e.state for e in batch])
            actions = np.array([e.action for e in batch])
            rewards = np.array([e.reward for e in batch])
            next_states = np.array([e.next_state for e in batch])
            dones = np.array([e.done for e in batch])

            # Convertir en tenseurs
            states_t = torch.FloatTensor(states)
            actions_t = torch.FloatTensor(actions)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones).unsqueeze(1)
            is_weights_t = torch.FloatTensor(is_weights).unsqueeze(1)
        else:
            # ReplayBuffer standard ou DiskReplayBuffer
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # Convertir en tenseurs
            states_t = torch.FloatTensor(states)
            actions_t = torch.FloatTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)
            is_weights_t = torch.ones_like(rewards_t)  # Pas de pondération

        sampling_times.append(time.time() - t_start)

        # Prédictions du modèle actuel
        q_values = model(states_t)

        # Prédictions du modèle cible pour les états suivants
        with torch.no_grad():
            next_q_values = target_model(next_states_t)
            next_q_values_max = next_q_values.max(1)[0].unsqueeze(1)
            # Calcul des cibles Q (Bellman)
            q_targets = rewards_t + gamma * next_q_values_max * (1 - dones_t)

        # Calcul de la perte
        if isinstance(buffer, (PrioritizedReplayBuffer, EnhancedPrioritizedReplay)):
            # Avec prioritized replay, on utilise les poids d'importance sampling
            td_errors = torch.abs(q_values - q_targets)
            loss = (td_errors * is_weights_t).mean()

            # Mettre à jour les priorités
            buffer.update_priorities(indices, td_errors.detach().numpy())
        else:
            # Perte MSE standard
            loss = nn.MSELoss()(q_values, q_targets)

        # Mise à jour du modèle
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Si on utilise EnhancedPrioritizedReplay, optimiser périodiquement
        if isinstance(buffer, EnhancedPrioritizedReplay) and update % 10 == 0:
            buffer.optimize_buffer()

        # Enregistrer la taille du buffer
        buffer_sizes.append(len(buffer))

        # Afficher les progrès
        if (update + 1) % 10 == 0:
            logger.info(
                f"Update {update+1}/{num_updates}, Loss: {loss.item():.4f}, "
                f"Buffer size: {len(buffer)}"
            )

    return losses, sampling_times, buffer_sizes


def compare_buffers(
    state_dim,
    action_dim,
    num_transitions,
    buffer_size,
    num_updates,
    batch_size,
    output_dir="outputs",
):
    """
    Compare les performances des différents tampons de replay.

    Args:
        state_dim: Dimension des états
        action_dim: Dimension des actions
        num_transitions: Nombre de transitions à générer
        buffer_size: Taille des tampons
        num_updates: Nombre de mises à jour pour l'entraînement
        batch_size: Taille du batch pour l'entraînement
        output_dir: Répertoire de sortie pour les graphiques
    """
    logger.info("Génération des transitions...")
    states, actions, rewards, next_states, dones = generate_transitions(
        num_transitions, state_dim, action_dim
    )

    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Liste des tampons à tester
    buffers = [
        ("Standard", ReplayBuffer(buffer_size)),
        ("Prioritized", PrioritizedReplayBuffer(buffer_size)),
        (
            "Disk",
            DiskReplayBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                action_dim=action_dim,
                storage_path=os.path.join(output_dir, "disk_buffer"),
                cache_size=100,
            ),
        ),
        (
            "Enhanced",
            EnhancedPrioritizedReplay(
                capacity=buffer_size,
                n_step=3,
                redundancy_threshold=0.9,
                state_encoder=SimpleStateEncoder(state_dim),
            ),
        ),
    ]

    # Résultats pour chaque tampon
    results = {}

    # Test de chaque tampon
    for buffer_name, buffer in buffers:
        logger.info(f"Test du tampon: {buffer_name}")

        # Remplir le tampon
        logger.info("Remplissage du tampon...")
        t_start_fill = time.time()
        for i in range(num_transitions):
            buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        fill_time = time.time() - t_start_fill

        # Créer un modèle pour ce tampon
        model = SimpleModel(state_dim, action_dim)
        target_model = SimpleModel(state_dim, action_dim)
        target_model.load_state_dict(model.state_dict())

        # Entraîner le modèle
        logger.info(f"Entraînement avec {buffer_name}...")
        losses, sampling_times, buffer_sizes = train_model_with_buffer(
            buffer, model, target_model, num_updates, batch_size
        )

        # Enregistrer les résultats
        if isinstance(buffer, DiskReplayBuffer):
            metrics = buffer.get_performance_metrics()
        elif isinstance(buffer, EnhancedPrioritizedReplay):
            metrics = buffer.get_metrics()
        else:
            metrics = {}

        results[buffer_name] = {
            "fill_time": fill_time,
            "losses": losses,
            "sampling_times": sampling_times,
            "buffer_sizes": buffer_sizes,
            "final_size": len(buffer),
            "metrics": metrics,
        }

        logger.info(f"Résultats pour {buffer_name}:")
        logger.info(f"  Temps de remplissage: {fill_time:.2f}s")
        logger.info(f"  Temps d'échantillonnage moyen: {np.mean(sampling_times):.6f}s")
        logger.info(f"  Perte finale: {losses[-1]:.4f}")

        # Nettoyer le tampon
        buffer.clear()

    # Tracer les résultats
    plot_results(results, output_dir)

    return results


def plot_results(results, output_dir):
    """
    Trace les résultats de comparaison.

    Args:
        results: Dictionnaire des résultats par tampon
        output_dir: Répertoire de sortie
    """
    # 1. Courbes de perte
    plt.figure(figsize=(10, 6))
    for buffer_name, data in results.items():
        plt.plot(data["losses"], label=buffer_name)
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.title("Comparaison des courbes de perte")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_comparison.png"))

    # 2. Temps d'échantillonnage
    plt.figure(figsize=(10, 6))
    for buffer_name, data in results.items():
        plt.plot(data["sampling_times"], label=buffer_name)
    plt.xlabel("Updates")
    plt.ylabel("Sampling Time (s)")
    plt.title("Comparaison des temps d'échantillonnage")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sampling_time_comparison.png"))

    # 3. Taille du buffer
    plt.figure(figsize=(10, 6))
    for buffer_name, data in results.items():
        plt.plot(data["buffer_sizes"], label=buffer_name)
    plt.xlabel("Updates")
    plt.ylabel("Buffer Size")
    plt.title("Évolution de la taille du buffer")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "buffer_size_comparison.png"))

    # 4. Tableau récapitulatif
    plt.figure(figsize=(12, 8))
    buffer_names = list(results.keys())

    # Métriques à comparer
    metrics = [
        "Temps de remplissage (s)",
        "Temps d'échantillonnage moyen (s)",
        "Perte finale",
        "Taille finale",
    ]

    values = []
    for buffer_name in buffer_names:
        data = results[buffer_name]
        values.append(
            [
                data["fill_time"],
                np.mean(data["sampling_times"]),
                data["losses"][-1],
                data["final_size"],
            ]
        )

    # Tracer le tableau
    table_data = []
    for i, metric in enumerate(metrics):
        row = [metric]
        for j, buffer_name in enumerate(buffer_names):
            row.append(f"{values[j][i]:.4f}")
        table_data.append(row)

    plt.axis("off")
    table = plt.table(
        cellText=table_data,
        colLabels=["Métrique"] + buffer_names,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("Tableau comparatif des performances")
    plt.savefig(os.path.join(output_dir, "performance_table.png"))

    logger.info(f"Graphiques sauvegardés dans {output_dir}")


if __name__ == "__main__":
    # Paramètres
    state_dim = 8
    action_dim = 2
    num_transitions = 10000
    buffer_size = 5000
    num_updates = 200
    batch_size = 64

    # Comparer les tampons
    logger.info("Démarrage de la comparaison des tampons de replay...")
    results = compare_buffers(
        state_dim=state_dim,
        action_dim=action_dim,
        num_transitions=num_transitions,
        buffer_size=buffer_size,
        num_updates=num_updates,
        batch_size=batch_size,
    )

    logger.info("Comparaison terminée!")
