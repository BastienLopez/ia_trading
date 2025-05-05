"""
Module d'entraînement RL avec accumulation de gradient.

Ce script applique l'accumulation de gradient à l'entraînement des agents
de Reinforcement Learning pour permettre d'utiliser des batchs plus grands
virtuellement sans augmenter la consommation de mémoire.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ai_trading.rl.trading_system import RLTradingSystem
from ai_trading.utils.gpu_cleanup import cleanup_gpu_memory
from ai_trading.utils.gpu_rtx_optimizer import setup_rtx_optimization
from ai_trading.utils.gradient_accumulation import GradientAccumulator

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chemin pour les logs et modèles
LOG_DIR = Path(__file__).parent.parent / "logs" / "gradient_accumulation"
MODEL_DIR = Path(__file__).parent.parent / "models" / "gradient_accumulation"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class AccumulatedSACAgent:
    """
    Wrapper pour un agent SAC qui utilise l'accumulation de gradient.
    Adapte la méthode update pour utiliser l'accumulation de gradient.
    """

    def __init__(self, sac_agent, accumulation_steps=2, gradient_clip=1.0, device=None):
        """
        Initialise l'agent SAC avec accumulation de gradient.

        Args:
            sac_agent: L'agent SAC original
            accumulation_steps: Nombre d'étapes d'accumulation de gradient
            gradient_clip: Valeur pour le clipping de gradient
            device: Périphérique de calcul (CPU/GPU)
        """
        self.agent = sac_agent
        self.accumulation_steps = accumulation_steps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Créer les accumulateurs de gradient pour chaque partie de l'agent
        self.accumulators = {
            "policy": GradientAccumulator(
                model=self.agent.policy,
                optimizer=self.agent.policy_optimizer,
                accumulation_steps=accumulation_steps,
                gradient_clip=gradient_clip,
            ),
            "q1": GradientAccumulator(
                model=self.agent.q1,
                optimizer=self.agent.q_optimizer,
                accumulation_steps=accumulation_steps,
                gradient_clip=gradient_clip,
            ),
            "q2": GradientAccumulator(
                model=self.agent.q2,
                optimizer=self.agent.q_optimizer,
                accumulation_steps=accumulation_steps,
                gradient_clip=gradient_clip,
            ),
        }

        # Attributs pour délégation
        self.state_size = self.agent.state_size
        self.action_size = self.agent.action_size
        self.buffer = self.agent.buffer

        logger.info(
            f"Agent SAC avec accumulation de gradient configuré: "
            f"{accumulation_steps} étapes d'accumulation"
        )

    def update(self, batch_size):
        """
        Mise à jour de l'agent avec accumulation de gradient.

        Args:
            batch_size: Taille du batch pour l'échantillonnage

        Returns:
            Un dictionnaire contenant les pertes
        """
        # Si le buffer n'est pas assez rempli, ne rien faire
        if len(self.agent.buffer) < batch_size:
            return {"policy_loss": 0, "q1_loss": 0, "q2_loss": 0, "alpha_loss": 0}

        total_policy_loss = 0
        total_q1_loss = 0
        total_q2_loss = 0
        total_alpha_loss = 0

        for _ in range(self.accumulation_steps):
            # Échantillonnage depuis le buffer
            states, actions, rewards, next_states, dones = self.agent.buffer.sample(
                batch_size
            )

            # Conversion en tensors et déplacement vers le périphérique
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # Calcul de la valeur cible pour Q
            with torch.no_grad():
                next_actions, next_log_probs = self.agent.policy.sample(next_states)
                q1_next = self.agent.target_q1(next_states, next_actions)
                q2_next = self.agent.target_q2(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next)
                value_target = rewards + self.agent.gamma * (1 - dones) * (
                    q_next - self.agent.alpha * next_log_probs
                )

            # Mise à jour des fonctions Q
            q1_value = self.agent.q1(states, actions)
            q2_value = self.agent.q2(states, actions)

            q1_loss = nn.MSELoss()(q1_value, value_target)
            q2_loss = nn.MSELoss()(q2_value, value_target)
            q_loss = q1_loss + q2_loss

            # Accumulation des gradients pour Q
            self.accumulators["q1"].backward(q_loss)
            self.accumulators["q2"].backward(q_loss)

            # Mise à jour de la politique
            new_actions, log_probs = self.agent.policy.sample(states)
            q1_new = self.agent.q1(states, new_actions)
            q2_new = self.agent.q2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            policy_loss = (self.agent.alpha * log_probs - q_new).mean()

            # Accumulation des gradients pour la politique
            self.accumulators["policy"].backward(policy_loss)

            # Mise à jour d'alpha (coefficient d'entropie)
            if self.agent.auto_entropy:
                alpha_loss = -(
                    self.agent.log_alpha
                    * (log_probs + self.agent.target_entropy).detach()
                ).mean()
                self.agent.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.agent.alpha_optimizer.step()
                self.agent.alpha = torch.exp(self.agent.log_alpha.detach())
                total_alpha_loss += alpha_loss.item()
            else:
                alpha_loss = torch.tensor(0.0).to(self.device)

            # Accumulation des pertes pour le reporting
            total_policy_loss += policy_loss.item()
            total_q1_loss += q1_loss.item()
            total_q2_loss += q2_loss.item()

        # Effectuer les mises à jour des poids si on a atteint le nombre d'étapes
        self.accumulators["policy"].step()
        self.accumulators["q1"].step()
        self.accumulators["q2"].step()

        # Mise à jour douce des modèles cibles
        for target_param, param in zip(
            self.agent.target_q1.parameters(), self.agent.q1.parameters()
        ):
            target_param.data.copy_(
                self.agent.tau * param.data + (1 - self.agent.tau) * target_param.data
            )

        for target_param, param in zip(
            self.agent.target_q2.parameters(), self.agent.q2.parameters()
        ):
            target_param.data.copy_(
                self.agent.tau * param.data + (1 - self.agent.tau) * target_param.data
            )

        # Calculer les moyennes
        avg_policy_loss = total_policy_loss / self.accumulation_steps
        avg_q1_loss = total_q1_loss / self.accumulation_steps
        avg_q2_loss = total_q2_loss / self.accumulation_steps
        avg_alpha_loss = (
            total_alpha_loss / self.accumulation_steps if self.agent.auto_entropy else 0
        )

        return {
            "policy_loss": avg_policy_loss,
            "q1_loss": avg_q1_loss,
            "q2_loss": avg_q2_loss,
            "alpha_loss": avg_alpha_loss,
        }

    def act(self, state, deterministic=False):
        """
        Détermine l'action à prendre dans un état donné.

        Args:
            state: L'état actuel
            deterministic: Si True, choisit l'action la plus probable

        Returns:
            L'action choisie
        """
        return self.agent.act(state, deterministic)

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke une transition dans le buffer de replay.

        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        self.agent.remember(state, action, reward, next_state, done)

    def replay(self, batch_size):
        """
        Effectue une mise à jour depuis le buffer de replay.

        Args:
            batch_size: Taille du batch à utiliser

        Returns:
            Un dictionnaire contenant les pertes
        """
        return self.update(batch_size)

    def save(self, path):
        """
        Sauvegarde le modèle.

        Args:
            path: Chemin où sauvegarder
        """
        self.agent.save(path)

    def load(self, path):
        """
        Charge le modèle.

        Args:
            path: Chemin d'où charger
        """
        self.agent.load(path)


def train_with_accumulation():
    """
    Entraîne un agent RL en utilisant l'accumulation de gradient.
    Compare les performances avec et sans accumulation.
    """
    # Configurer l'optimisation GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        setup_rtx_optimization()
        logger.info(f"Utilisation de CUDA: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA non disponible, utilisation du CPU")

    # Configuration des hyperparamètres
    config = {
        "env_name": "trading",
        "agent_type": "sac",
        "window_size": 20,
        "batch_size": 64,
        "initial_balance": 10000,
        "learning_rate": 0.0003,
        "episodes": 50,
    }

    # Initialiser TensorBoard pour le suivi
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Systèmes de trading avec et sans accumulation
    trading_systems = {
        "standard": RLTradingSystem(config=config),
        "accumulated_2": RLTradingSystem(config=config),
        "accumulated_4": RLTradingSystem(config=config),
    }

    # Intégrer les données (les mêmes pour tous les systèmes)
    data = trading_systems["standard"].integrate_data(
        symbol="BTC/USDT",
        start_date="2022-01-01",
        end_date="2023-01-01",
        timeframe="1d",
    )

    # Environnements (identiques pour tous les systèmes)
    for name, system in trading_systems.items():
        env = system.create_environment(
            data=data.copy(),
            initial_balance=config["initial_balance"],
            window_size=config["window_size"],
        )

    # Créer les agents
    for name, system in trading_systems.items():
        env = system._env  # Tous les environnements sont identiques

        # Créer l'agent SAC de base
        agent = system.create_agent(
            agent_type="sac",
            state_size=env.observation_space.shape[0],
            action_size=1,  # Pour trading continu
            action_bounds=(-1, 1),
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )

        # Pour les versions avec accumulation, wrapper l'agent
        if name == "accumulated_2":
            agent = AccumulatedSACAgent(
                sac_agent=agent,
                accumulation_steps=2,
                gradient_clip=1.0,
                device=device,
            )
        elif name == "accumulated_4":
            agent = AccumulatedSACAgent(
                sac_agent=agent,
                accumulation_steps=4,
                gradient_clip=1.0,
                device=device,
            )

        # Assigner l'agent au système
        system._agent = agent

    # Entraînement des différents systèmes
    results = {}

    for name, system in trading_systems.items():
        logger.info(f"\nEntraînement du système: {name}")

        # Nettoyer la mémoire avant chaque entraînement
        cleanup_gpu_memory()

        # Entraînement
        for episode in range(config["episodes"]):
            state = system._env.reset()
            if isinstance(state, tuple):
                state = state[0]

            done = False
            total_reward = 0
            losses = {"policy_loss": 0, "q1_loss": 0, "q2_loss": 0, "alpha_loss": 0}
            steps = 0

            while not done:
                action = system._agent.act(state)
                next_state, reward, terminated, truncated, info = system._env.step(
                    action
                )
                done = terminated or truncated

                system._agent.remember(state, action, reward, next_state, done)

                # Entraînement avec le batch
                if len(system._agent.buffer) >= config["batch_size"]:
                    step_losses = system._agent.replay(config["batch_size"])

                    # Accumulation des pertes
                    for key in losses:
                        if key in step_losses:
                            losses[key] += step_losses[key]

                state = next_state
                total_reward += reward
                steps += 1

            # Calculer les moyennes des pertes
            for key in losses:
                losses[key] = losses[key] / max(1, steps)

            # Logging
            writer.add_scalar(f"{name}/reward", total_reward, episode)
            writer.add_scalar(
                f"{name}/portfolio_value", info.get("portfolio_value", 0), episode
            )

            for loss_name, loss_value in losses.items():
                writer.add_scalar(f"{name}/{loss_name}", loss_value, episode)

            if episode % 5 == 0:
                logger.info(
                    f"Système {name}, Épisode {episode+1}/{config['episodes']}, "
                    f"Récompense: {total_reward:.2f}, "
                    f"Portfolio: {info.get('portfolio_value', 0):.2f}, "
                    f"Policy Loss: {losses['policy_loss']:.4f}"
                )

                # Sauvegarder périodiquement
                save_path = MODEL_DIR / f"{name}_episode_{episode}.pt"
                system._agent.save(str(save_path))

        # Évaluation finale
        eval_results = system.evaluate(num_episodes=10)
        results[name] = eval_results

        logger.info(
            f"\nRésultats pour {name}:\n"
            f"Valeur finale: {eval_results['final_value']:.2f}\n"
            f"Ratio de Sharpe: {eval_results['sharpe_ratio']:.4f}\n"
            f"Drawdown max: {eval_results['max_drawdown']:.4f}"
        )

        # Sauvegarder le modèle final
        final_path = MODEL_DIR / f"{name}_final.pt"
        system._agent.save(str(final_path))

    writer.close()

    # Comparaison des résultats
    logger.info("\n===== Comparaison des performances =====")
    for name, result in results.items():
        effective_batch = config["batch_size"]
        if name == "accumulated_2":
            effective_batch = config["batch_size"] * 2
        elif name == "accumulated_4":
            effective_batch = config["batch_size"] * 4

        logger.info(
            f"{name} (batch effectif: {effective_batch}):\n"
            f"  - Valeur finale: {result['final_value']:.2f}\n"
            f"  - Ratio de Sharpe: {result['sharpe_ratio']:.4f}\n"
            f"  - Drawdown max: {result['max_drawdown']:.4f}"
        )

    return results


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Entraînement RL avec accumulation de gradient"
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Forcer l'utilisation du GPU si disponible"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Forcer l'utilisation du CPU"
    )

    args = parser.parse_args()

    # Configuration du périphérique
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Utilisation forcée du CPU")
    elif args.gpu and torch.cuda.is_available():
        logger.info("Utilisation forcée du GPU")

    # Exécution de l'entraînement
    try:
        results = train_with_accumulation()
        logger.info("Entraînement terminé avec succès!")

        # Retourner les meilleurs résultats
        best_system = max(results.items(), key=lambda x: x[1]["sharpe_ratio"])
        logger.info(
            f"\nMeilleur système: {best_system[0]} avec un ratio de Sharpe de {best_system[1]['sharpe_ratio']:.4f}"
        )

    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {str(e)}")
        raise


if __name__ == "__main__":
    main()
