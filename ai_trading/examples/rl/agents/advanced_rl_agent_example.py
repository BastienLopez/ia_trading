"""
Exemple d'agent d'apprentissage par renforcement avancé pour le trading.

Ce script montre comment configurer et entraîner un agent RL avancé
avec différentes architectures et mécanismes d'optimisation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Import des modules RL
from ai_trading.rl.agents.dqn_agent import DQNAgent
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.agents.layers.noisy_linear import NoisyLinear
from ai_trading.rl.environments.trading_environment import TradingEnvironment
from ai_trading.rl.memory.prioritized_replay import PrioritizedReplayBuffer

def load_market_data():
    """Charge des données de marché pour l'entraînement."""
    # Période de simulation
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Générer des données OHLCV simulées
    data = pd.DataFrame(index=dates)
    
    # Simuler différentes phases de marché
    n = len(dates)
    trend = np.concatenate([
        np.linspace(10000, 12000, n//4),  # Hausse
        np.linspace(12000, 9000, n//4),   # Baisse
        np.linspace(9000, 9500, n//4),    # Consolidation
        np.linspace(9500, 14000, n//4)    # Hausse forte
    ])
    
    # Ajouter du bruit pour simuler la volatilité
    noise = np.random.normal(0, 200, n)
    close_prices = trend + noise
    
    # Créer les données OHLCV
    data['open'] = close_prices - np.random.uniform(0, 100, n)
    data['high'] = close_prices + np.random.uniform(50, 200, n)
    data['low'] = close_prices - np.random.uniform(50, 200, n)
    data['close'] = close_prices
    data['volume'] = np.random.uniform(1000, 5000, n)
    
    return data

def create_technical_features(data):
    """Ajoute des indicateurs techniques aux données de marché."""
    # Copier les données pour éviter de modifier l'original
    df = data.copy()
    
    # Moyennes mobiles
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bandes de Bollinger
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * std
    df['bb_lower'] = df['bb_middle'] - 2 * std
    
    # Momentum et volatilité
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['volatility'] = df['close'].rolling(window=10).std() / df['close'].rolling(window=10).mean()
    
    # Volumes
    df['volume_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Supprimer les lignes avec des NaN
    df = df.dropna()
    
    return df

def create_advanced_dqn():
    """Crée un agent DQN avancé avec diverses améliorations."""
    class AdvancedDQNNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(AdvancedDQNNetwork, self).__init__()
            
            # Utiliser des couches noisy pour une exploration paramétrique
            self.fc1 = NoisyLinear(input_dim, 128)
            self.gru = nn.GRU(128, 128, batch_first=True)
            self.fc_v = NoisyLinear(128, 64)
            self.fc_a = NoisyLinear(128, 64)
            self.v = NoisyLinear(64, 1)
            self.a = NoisyLinear(64, output_dim)
            
            # Activation et normalisation
            self.relu = nn.ReLU()
            self.ln1 = nn.LayerNorm(128)
            self.ln_v = nn.LayerNorm(64)
            self.ln_a = nn.LayerNorm(64)
        
        def forward(self, x):
            # Séquence d'entrée pour le GRU
            batch_size = x.size(0)
            seq_len = 10  # Supposons 10 pas de temps
            
            # Reformater pour le GRU si nécessaire
            if len(x.shape) == 2:
                x = self.fc1(x)
                x = self.relu(self.ln1(x))
                x = x.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, features)
            else:
                # Si déjà 3D
                x = x.view(-1, x.size(-1))  # (batch*seq_len, features)
                x = self.fc1(x)
                x = self.relu(self.ln1(x))
                x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, features)
            
            # Traiter la séquence avec GRU
            x, _ = self.gru(x)
            x = x[:, -1, :]  # Prendre le dernier pas de temps
            
            # Dueling DQN: séparer la valeur et les avantages
            v = self.relu(self.ln_v(self.fc_v(x)))
            a = self.relu(self.ln_a(self.fc_a(x)))
            
            v = self.v(v)
            a = self.a(a)
            
            # Combiner pour l'estimation Q
            q = v + (a - a.mean(dim=1, keepdim=True))
            
            return q
    
    # Configuration du buffer de replay prioritaire
    memory = PrioritizedReplayBuffer(
        capacity=100000,
        alpha=0.6,  # Priorité d'échantillonnage
        beta=0.4,   # Correction d'importance-sampling
        beta_increment=0.001  # Augmenter beta vers 1 pendant l'entraînement
    )
    
    # Créer l'agent DQN avancé
    agent = DQNAgent(
        state_dim=20,       # Dimension de l'état (nombre de caractéristiques)
        action_dim=5,       # Dimension de l'action (par exemple, -2, -1, 0, 1, 2 pour différents niveaux d'achat/vente)
        hidden_dim=128,
        network=AdvancedDQNNetwork,
        memory=memory,
        batch_size=64,
        gamma=0.99,         # Facteur d'actualisation
        epsilon_start=1.0,  # Exploration initiale
        epsilon_final=0.1,  # Exploration finale
        epsilon_decay=5000, # Vitesse de décroissance
        target_update=100,  # Fréquence de mise à jour du réseau cible
        double_dqn=True,    # Utiliser Double DQN
        n_step_returns=3,   # Utiliser des retours à n étapes
        learning_rate=1e-4,
        gradient_clip=1.0   # Limiter la norme du gradient
    )
    
    return agent

def create_sac_agent():
    """Crée un agent SAC (Soft Actor-Critic) pour les actions continues."""
    # Configurer l'agent SAC
    agent = SACAgent(
        state_dim=20,        # Dimension de l'état
        action_dim=1,        # Dimension de l'action continue (pourcentage d'allocation)
        hidden_dim=256,      # Dimension des couches cachées
        actor_lr=3e-4,       # Taux d'apprentissage pour l'acteur
        critic_lr=3e-4,      # Taux d'apprentissage pour le critique
        alpha_lr=3e-4,       # Taux d'apprentissage pour l'entropie
        batch_size=256,      # Taille du batch
        gamma=0.99,          # Facteur d'actualisation
        tau=0.005,           # Mise à jour progressive
        buffer_size=100000,  # Taille du buffer de replay
        reward_scale=5.0,    # Échelle de récompense
        auto_entropy=True    # Ajustement automatique du coefficient d'entropie
    )
    
    return agent

def train_agent(agent, env, episodes=100, steps_per_episode=252):
    """Entraîne l'agent dans l'environnement spécifié."""
    rewards = []
    portfolio_values = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        portfolio_value = []
        
        for step in range(steps_per_episode):
            # Sélectionner une action
            action = agent.select_action(state)
            
            # Exécuter l'action dans l'environnement
            next_state, reward, done, info = env.step(action)
            
            # Stocker la transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Mettre à jour l'agent
            loss = agent.update()
            
            # Mettre à jour l'état courant
            state = next_state
            episode_reward += reward
            portfolio_value.append(info.get('portfolio_value', 0))
            
            if done:
                break
        
        rewards.append(episode_reward)
        portfolio_values.append(portfolio_value)
        
        # Afficher les progrès
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Épisode {episode + 1}/{episodes}, Récompense moyenne: {avg_reward:.2f}")
            
            # Afficher les métriques de performance
            if len(portfolio_value) > 0:
                returns = np.diff(portfolio_value) / portfolio_value[:-1]
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                max_drawdown = (np.min(portfolio_value) / np.max(portfolio_value) - 1) if np.max(portfolio_value) > 0 else 0
                print(f"  Sharpe: {sharpe:.2f}, Max Drawdown: {max_drawdown:.2%}")
    
    return rewards, portfolio_values

def rl_agent_example():
    """Exemple principal d'agent RL avancé pour le trading."""
    # Charger et préparer les données
    market_data = load_market_data()
    features_data = create_technical_features(market_data)
    
    # Créer l'environnement de trading
    env = TradingEnvironment(
        data=features_data,
        features=[
            'sma_10', 'sma_30', 'ema_10', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower', 'momentum', 'volatility',
            'volume_sma', 'volume_ratio'
        ],
        window_size=10,
        initial_balance=10000.0,
        commission=0.001,
        reward_function='sharpe',  # Utiliser le ratio de Sharpe comme récompense
        trading_cost_ratio=True,   # Inclure les coûts de transaction
        reward_scaling=True,       # Mettre à l'échelle les récompenses
        early_stopping=True        # Arrêt précoce en cas de perte importante
    )
    
    # Créer et configurer l'agent
    # Décommenter la ligne appropriée pour choisir le type d'agent
    agent = create_advanced_dqn()
    # agent = create_sac_agent()  # Pour les actions continues
    
    # Entraîner l'agent
    print("Début de l'entraînement...")
    rewards, portfolio_values = train_agent(agent, env, episodes=50, steps_per_episode=252)
    print("Entraînement terminé!")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 8))
    
    # Tracer les récompenses
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense cumulée')
    plt.grid(True)
    
    # Tracer la valeur du portefeuille du dernier épisode
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values[-1])
    plt.title('Valeur du portefeuille (dernier épisode)')
    plt.xlabel('Pas de temps')
    plt.ylabel('Valeur du portefeuille')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Enregistrer le graphique
    plt.savefig("examples/rl/agents/advanced_rl_performance.png")
    plt.close()
    
    print("Graphique des performances enregistré sous 'examples/rl/agents/advanced_rl_performance.png'")
    
    # Sauvegarder le modèle entraîné
    torch.save(agent.qnet.state_dict(), "examples/rl/agents/advanced_rl_model.pt")
    print("Modèle entraîné sauvegardé sous 'examples/rl/agents/advanced_rl_model.pt'")
    
    return agent, rewards, portfolio_values

if __name__ == "__main__":
    agent, rewards, portfolio_values = rl_agent_example() 