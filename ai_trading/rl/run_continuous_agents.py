import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from trading_environment import TradingEnvironment


def evaluate_agent(
    agent, env: TradingEnvironment, n_episodes: int = 10
) -> Dict[str, List[float]]:
    returns = []
    sharpe_ratios = []
    max_drawdowns = []

    for episode in range(n_episodes):
        state, _ = env.reset()  # Ignorer les infos supplémentaires
        episode_return = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Récupérer l'action et le log_prob pour PPO, ou juste l'action pour SAC
            if isinstance(agent, PPOAgent):
                action, _ = agent.get_action(
                    state, deterministic=True
                )  # Ignorer le log_prob
            else:
                action = agent.get_action(state, deterministic=True)

            next_state, reward, done, truncated, _ = env.step(
                action
            )  # Ignorer les infos supplémentaires
            episode_return += reward
            state = next_state

        returns.append(episode_return)
        sharpe_ratios.append(env.calculate_sharpe_ratio())
        max_drawdowns.append(env.calculate_max_drawdown())

    return {
        "returns": returns,
        "sharpe_ratios": sharpe_ratios,
        "max_drawdowns": max_drawdowns,
    }


def train_agent(
    agent, env: TradingEnvironment, n_episodes: int = 1000
) -> Dict[str, List[float]]:
    returns = []
    actor_losses = []
    critic_losses = []

    for episode in range(n_episodes):
        state, _ = env.reset()  # Ignorer les infos supplémentaires
        episode_return = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Récupérer l'action et le log_prob pour PPO, ou juste l'action pour SAC
            if isinstance(agent, PPOAgent):
                action, log_prob = agent.get_action(state)
            else:
                action = agent.get_action(state)

            next_state, reward, done, truncated, _ = env.step(
                action
            )  # Ignorer les infos supplémentaires

            if isinstance(agent, PPOAgent):
                # Pour PPO, on accumule les transitions
                if done or truncated:
                    losses = agent.update(
                        np.array(env.states),
                        np.array(env.actions),
                        np.array(env.rewards),
                        np.array(env.next_states),
                        np.array(env.dones),
                    )
                    episode_actor_loss += losses["actor_loss"]
                    episode_critic_loss += losses["critic_loss"]
            else:
                # Pour SAC, on met à jour à chaque étape
                losses = agent.update(state, action, reward, next_state, done)
                episode_actor_loss += losses["actor_loss"]
                episode_critic_loss += losses["critic_loss"]

            episode_return += reward
            state = next_state

        returns.append(episode_return)
        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"Return: {episode_return:.2f}")
            print(f"Actor Loss: {episode_actor_loss:.4f}")
            print(f"Critic Loss: {episode_critic_loss:.4f}")
            print("------------------------")

    return {
        "returns": returns,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
    }


def plot_results(
    ppo_results: Dict[str, List[float]], sac_results: Dict[str, List[float]]
):
    plt.figure(figsize=(15, 10))

    # Returns
    plt.subplot(3, 1, 1)
    plt.plot(ppo_results["returns"], label="PPO")
    plt.plot(sac_results["returns"], label="SAC")
    plt.title("Returns during training")
    plt.legend()

    # Actor Losses
    plt.subplot(3, 1, 2)
    plt.plot(ppo_results["actor_losses"], label="PPO")
    plt.plot(sac_results["actor_losses"], label="SAC")
    plt.title("Actor Losses during training")
    plt.legend()

    # Critic Losses
    plt.subplot(3, 1, 3)
    plt.plot(ppo_results["critic_losses"], label="PPO")
    plt.plot(sac_results["critic_losses"], label="SAC")
    plt.title("Critic Losses during training")
    plt.legend()

    plt.tight_layout()
    plt.savefig("continuous_agents_results.png")
    plt.close()


def main():
    # Chargement des données
    data_path = os.path.join(
        os.path.dirname(__file__), "../data/processed/btc_usd_1h.csv"
    )
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Configuration de l'environnement
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=30,
        action_type="continuous",
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Création des agents
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
    )

    sac_agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )

    # Entraînement
    print("Training PPO agent...")
    ppo_results = train_agent(ppo_agent, env, n_episodes=1000)

    print("\nTraining SAC agent...")
    sac_results = train_agent(sac_agent, env, n_episodes=1000)

    # Évaluation
    print("\nEvaluating PPO agent...")
    ppo_eval = evaluate_agent(ppo_agent, env)

    print("\nEvaluating SAC agent...")
    sac_eval = evaluate_agent(sac_agent, env)

    # Affichage des résultats
    print("\nPPO Evaluation Results:")
    print(f"Average Return: {np.mean(ppo_eval['returns']):.2f}")
    print(f"Average Sharpe Ratio: {np.mean(ppo_eval['sharpe_ratios']):.2f}")
    print(f"Average Max Drawdown: {np.mean(ppo_eval['max_drawdowns']):.2f}")

    print("\nSAC Evaluation Results:")
    print(f"Average Return: {np.mean(sac_eval['returns']):.2f}")
    print(f"Average Sharpe Ratio: {np.mean(sac_eval['sharpe_ratios']):.2f}")
    print(f"Average Max Drawdown: {np.mean(sac_eval['max_drawdowns']):.2f}")

    # Plot des résultats
    plot_results(ppo_results, sac_results)


if __name__ == "__main__":
    main()
