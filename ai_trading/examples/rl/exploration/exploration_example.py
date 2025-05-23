import numpy as np
from ai_trading.rl.exploration import UCBExploration, AdaptiveExploration


def exemple_ucb_strategy():
    print("\n--- Exemple : UCBExploration ---")
    action_size = 3
    ucb = UCBExploration(action_size=action_size, c=2.0)
    q_values = np.array([1.0, 2.0, 3.0])
    print(f"Q-values initiales : {q_values}")
    for step in range(10):
        action = ucb.select_action(q_values)
        reward = np.random.rand() * 2  # Récompense aléatoire
        ucb.update(action, reward)
        print(f"Étape {step+1}: action={action}, reward={reward:.2f}, counts={ucb.action_counts}, values={ucb.action_values}")
    ucb.reset()
    print("Explorateur UCB réinitialisé.")


def exemple_adaptive_rates():
    print("\n--- Exemple : AdaptiveExploration ---")
    adaptive = AdaptiveExploration(initial_epsilon=0.2, min_epsilon=0.01, decay=0.9)
    q_values = np.array([0.5, 1.5, 2.5])
    state_str = "etat_test"
    for step in range(10):
        should_explore = adaptive.should_explore(state_str, market_volatility=0.3)
        if should_explore:
            action = np.random.choice(len(q_values))
            print(f"Étape {step+1}: Exploration aléatoire, action={action}, epsilon={adaptive.epsilon:.3f}")
        else:
            action = adaptive.get_ucb_action(state_str, q_values)
            print(f"Étape {step+1}: Exploitation UCB, action={action}, epsilon={adaptive.epsilon:.3f}")
    adaptive.reset()
    print("Explorateur adaptatif réinitialisé.")


if __name__ == "__main__":
    exemple_ucb_strategy()
    exemple_adaptive_rates() 