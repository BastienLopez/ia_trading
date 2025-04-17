import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_sac_agent")

def debug_environment(env):
    """
    Effectue un débogage de l'environnement pour vérifier sa dimensionnalité et comportement.
    
    Args:
        env: L'environnement de trading à déboguer
    """
    # Réinitialiser l'environnement
    state, _ = env.reset()
    
    # Informations sur l'état
    logger.info(f"État: forme={state.shape}, type={state.dtype}")
    logger.info(f"État min={state.min()}, max={state.max()}")
    logger.info(f"Longueur d'état réelle = {len(state)}")
    
    # Vérifier l'espace d'observation déclaré
    obs_shape = env.observation_space.shape
    logger.info(f"Forme d'espace d'observation déclarée = {obs_shape}")
    
    # Vérifier l'espace d'action
    act_shape = env.action_space.shape
    logger.info(f"Forme d'espace d'action = {act_shape}")
    
    # Tester une action aléatoire
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    
    logger.info(f"Action aléatoire = {action}")
    logger.info(f"Récompense = {reward}")
    logger.info(f"État suivant: forme={next_state.shape}, longueur={len(next_state)}")
    logger.info(f"Info = {info}")
    
    return len(state)

def train_sac_agent(env, agent, n_episodes=100, max_steps=1000):
    """
    Entraîne un agent SAC sur l'environnement de trading.
    
    Args:
        env (TradingEnvironment): Environnement de trading
        agent (SACAgent): Agent SAC à entraîner
        n_episodes (int): Nombre d'épisodes d'entraînement
        max_steps (int): Nombre maximum d'étapes par épisode
        
    Returns:
        dict: Statistiques d'entraînement
    """
    stats = {
        'episode_returns': [],
        'episode_lengths': [],
        'portfolio_values': [],
        'actor_losses': [],
        'critic_losses': [],
        'sharpe_ratios': [],
        'max_drawdowns': []
    }
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False
        truncated = False
        
        # Pour stocker les valeurs du portefeuille
        portfolio_values = []
        
        # Pour mesurer les pertes
        episode_actor_losses = []
        episode_critic_losses = []
        
        for step in range(max_steps):
            try:
                # Sélectionner une action
                action = agent.get_action(state)
                
                # Exécuter l'action
                next_state, reward, done, truncated, info = env.step(action)
                
                # Mettre à jour l'agent
                losses = agent.update(state, action, reward, next_state, done)
                
                # Enregistrer les pertes
                episode_actor_losses.append(losses['actor_loss'])
                episode_critic_losses.append(losses['critic_loss'])
                
                # Enregistrer la valeur du portefeuille
                portfolio_values.append(env.get_portfolio_value())
                
                # Accumuler la récompense
                episode_return += reward
                episode_length += 1
                
                # Mise à jour de l'état
                state = next_state
                
                if done or truncated:
                    break
            except Exception as e:
                logger.error(f"Erreur pendant l'entraînement (épisode {episode+1}, étape {step+1}): {str(e)}")
                # Continuer avec l'étape suivante plutôt que d'abandonner tout l'entraînement
                continue
        
        # Calculer les métriques de performance
        try:
            sharpe_ratio = env.calculate_sharpe_ratio()
            max_drawdown = env.calculate_max_drawdown()
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques: {str(e)}")
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        # Enregistrer les statistiques de l'épisode
        stats['episode_returns'].append(episode_return)
        stats['episode_lengths'].append(episode_length)
        stats['portfolio_values'].append(portfolio_values)
        stats['actor_losses'].append(np.mean(episode_actor_losses) if episode_actor_losses else 0.0)
        stats['critic_losses'].append(np.mean(episode_critic_losses) if episode_critic_losses else 0.0)
        stats['sharpe_ratios'].append(sharpe_ratio)
        stats['max_drawdowns'].append(max_drawdown)
        
        # Afficher les résultats
        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Episode {episode + 1}/{n_episodes} terminé")
            logger.info(f"Rendement: {episode_return:.2f}")
            logger.info(f"Longueur: {episode_length}")
            logger.info(f"Valeur finale du portefeuille: {portfolio_values[-1] if portfolio_values else 0:.2f}")
            logger.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"Perte moyenne acteur: {np.mean(episode_actor_losses) if episode_actor_losses else 0:.4f}")
            logger.info(f"Perte moyenne critique: {np.mean(episode_critic_losses) if episode_critic_losses else 0:.4f}")
            logger.info(f"Temps écoulé: {elapsed_time:.0f} secondes")
            logger.info("-" * 50)
    
    logger.info(f"Entraînement terminé en {time.time() - start_time:.0f} secondes")
    
    return stats

def evaluate_sac_agent(env, agent, n_episodes=10):
    """
    Évalue un agent SAC sur l'environnement de trading.
    
    Args:
        env (TradingEnvironment): Environnement de trading
        agent (SACAgent): Agent SAC à évaluer
        n_episodes (int): Nombre d'épisodes d'évaluation
        
    Returns:
        dict: Statistiques d'évaluation
    """
    stats = {
        'returns': [],
        'portfolio_values': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'sortino_ratios': []
    }
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0
        
        # Pour stocker les valeurs du portefeuille
        portfolio_values = []
        
        while not (done or truncated):
            # Sélectionner une action (déterministe pour l'évaluation)
            action = agent.get_action(state, deterministic=True)
            
            # Exécuter l'action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Enregistrer la valeur du portefeuille
            portfolio_values.append(env.get_portfolio_value())
            
            # Accumuler la récompense
            episode_return += reward
            
            # Mise à jour de l'état
            state = next_state
        
        # Calculer les métriques de performance
        sharpe_ratio = env.calculate_sharpe_ratio()
        max_drawdown = env.calculate_max_drawdown()
        sortino_ratio = env.calculate_sortino_ratio()
        
        # Enregistrer les statistiques de l'épisode
        stats['returns'].append(episode_return)
        stats['portfolio_values'].append(portfolio_values)
        stats['sharpe_ratios'].append(sharpe_ratio)
        stats['max_drawdowns'].append(max_drawdown)
        stats['sortino_ratios'].append(sortino_ratio)
        
        logger.info(f"Évaluation épisode {episode + 1}/{n_episodes}")
        logger.info(f"Rendement: {episode_return:.2f}")
        logger.info(f"Valeur finale du portefeuille: {portfolio_values[-1]:.2f}")
        logger.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        logger.info(f"Ratio de Sortino: {sortino_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info("-" * 50)
    
    # Calculer les moyennes
    avg_return = np.mean(stats['returns'])
    avg_sharpe = np.mean(stats['sharpe_ratios'])
    avg_sortino = np.mean(stats['sortino_ratios'])
    avg_max_dd = np.mean(stats['max_drawdowns'])
    
    logger.info("Résultats moyens d'évaluation :")
    logger.info(f"Rendement moyen: {avg_return:.2f}")
    logger.info(f"Ratio de Sharpe moyen: {avg_sharpe:.2f}")
    logger.info(f"Ratio de Sortino moyen: {avg_sortino:.2f}")
    logger.info(f"Max Drawdown moyen: {avg_max_dd:.2%}")
    
    return stats

def plot_training_results(stats, save_path="sac_training_results.png"):
    """
    Trace les graphiques des résultats d'entraînement.
    
    Args:
        stats (dict): Statistiques d'entraînement
        save_path (str): Chemin où sauvegarder l'image
    """
    plt.figure(figsize=(15, 20))
    
    # Rendement par épisode
    plt.subplot(5, 1, 1)
    plt.plot(stats['episode_returns'])
    plt.title('Rendement par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Rendement')
    plt.grid(True)
    
    # Pertes de l'acteur et du critique
    plt.subplot(5, 1, 2)
    plt.plot(stats['actor_losses'], label='Acteur')
    plt.plot(stats['critic_losses'], label='Critique')
    plt.title('Pertes moyennes par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)
    
    # Ratio de Sharpe par épisode
    plt.subplot(5, 1, 3)
    plt.plot(stats['sharpe_ratios'])
    plt.title('Ratio de Sharpe par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Ratio de Sharpe')
    plt.grid(True)
    
    # Max Drawdown par épisode
    plt.subplot(5, 1, 4)
    plt.plot(stats['max_drawdowns'])
    plt.title('Max Drawdown par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Max Drawdown')
    plt.grid(True)
    
    # Exemple de courbe d'équité (dernier épisode)
    plt.subplot(5, 1, 5)
    plt.plot(stats['portfolio_values'][-1])
    plt.title('Courbe d\'équité (dernier épisode)')
    plt.xlabel('Étape')
    plt.ylabel('Valeur du portefeuille')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_evaluation_results(stats, save_path="sac_evaluation_results.png"):
    """
    Trace les graphiques des résultats d'évaluation.
    
    Args:
        stats (dict): Statistiques d'évaluation
        save_path (str): Chemin où sauvegarder l'image
    """
    plt.figure(figsize=(15, 15))
    
    # Rendement par épisode
    plt.subplot(4, 1, 1)
    plt.bar(range(len(stats['returns'])), stats['returns'])
    plt.title('Rendement par épisode d\'évaluation')
    plt.xlabel('Épisode')
    plt.ylabel('Rendement')
    plt.grid(True)
    
    # Ratio de Sharpe par épisode
    plt.subplot(4, 1, 2)
    plt.bar(range(len(stats['sharpe_ratios'])), stats['sharpe_ratios'])
    plt.title('Ratio de Sharpe par épisode d\'évaluation')
    plt.xlabel('Épisode')
    plt.ylabel('Ratio de Sharpe')
    plt.grid(True)
    
    # Ratio de Sortino par épisode
    plt.subplot(4, 1, 3)
    plt.bar(range(len(stats['sortino_ratios'])), stats['sortino_ratios'])
    plt.title('Ratio de Sortino par épisode d\'évaluation')
    plt.xlabel('Épisode')
    plt.ylabel('Ratio de Sortino')
    plt.grid(True)
    
    # Exemple de courbe d'équité (premier épisode)
    plt.subplot(4, 1, 4)
    plt.plot(stats['portfolio_values'][0])
    plt.title('Courbe d\'équité (premier épisode d\'évaluation)')
    plt.xlabel('Étape')
    plt.ylabel('Valeur du portefeuille')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Fonction principale pour entraîner et évaluer l'agent SAC."""
    # Chemins des données synthétiques
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/synthetic_mixed_1h.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Le fichier de données {data_path} n'existe pas.")
        # Essayer de trouver d'autres fichiers de données
        data_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            logger.info(f"Fichiers disponibles dans {data_dir}: {files}")
            
            # Prendre le premier fichier CSV si disponible
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                data_path = os.path.join(data_dir, csv_files[0])
                logger.info(f"Utilisation du fichier {data_path}")
            else:
                logger.error("Aucun fichier CSV trouvé")
                return
        else:
            logger.error(f"Le dossier {data_dir} n'existe pas")
            return
    
    # Chargement des données
    df = pd.read_csv(data_path)
    
    # Vérification de la présence de la colonne timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Aperçu des données
    logger.info(f"Données chargées: {len(df)} entrées")
    logger.info(f"Colonnes disponibles: {df.columns.tolist()}")
    logger.info(f"Période: {df.index.min()} à {df.index.max()}")
    
    # Nous utilisons un sous-ensemble des données pour l'entraînement
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    logger.info(f"Taille des données d'entraînement: {len(train_df)}")
    logger.info(f"Taille des données de test: {len(test_df)}")
    
    # Création de l'environnement d'entraînement
    train_env = TradingEnvironment(
        df=train_df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        action_type="continuous",
        reward_function="sharpe"  # Utiliser la fonction de récompense basée sur Sharpe
    )
    
    # Déboguer l'environnement pour vérifier sa dimensionnalité
    logger.info("Débogage de l'environnement d'entraînement...")
    real_state_dim = debug_environment(train_env)
    
    # Création de l'environnement de test
    test_env = TradingEnvironment(
        df=test_df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        action_type="continuous",
        reward_function="sharpe"
    )
    
    # Obtenir un exemple d'observation pour déterminer la dimension réelle
    observation, _ = train_env.reset()
    action_dim = train_env.action_space.shape[0]
    
    logger.info(f"Dimension réelle de l'espace d'état: {real_state_dim}")
    logger.info(f"Dimension de l'espace d'action: {action_dim}")
    
    # Création de l'agent SAC avec la dimension d'état correcte
    agent = SACAgent(
        state_dim=real_state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        buffer_size=10000,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True
    )
    
    # Entraînement de l'agent
    logger.info("Début de l'entraînement de l'agent SAC...")
    train_stats = train_sac_agent(
        env=train_env,
        agent=agent,
        n_episodes=30,  # Réduire le nombre d'épisodes pour test
        max_steps=500   # Réduire le nombre d'étapes par épisode pour test
    )
    
    # Évaluation de l'agent
    logger.info("Début de l'évaluation de l'agent SAC...")
    eval_stats = evaluate_sac_agent(
        env=test_env,
        agent=agent,
        n_episodes=5  # Réduire le nombre d'épisodes d'évaluation pour test
    )
    
    # Sauvegarde de l'agent
    model_dir = os.path.join(os.path.dirname(__file__), 'models/sac')
    os.makedirs(model_dir, exist_ok=True)
    agent.save(model_dir)
    logger.info(f"Agent sauvegardé dans {model_dir}")
    
    # Afficher les résultats finaux
    logger.info("Résultats finaux:")
    logger.info(f"Rendement moyen sur l'entraînement: {np.mean(train_stats['episode_returns']):.2f}")
    logger.info(f"Ratio de Sharpe moyen sur l'entraînement: {np.mean(train_stats['sharpe_ratios']):.2f}")
    logger.info(f"Rendement moyen sur l'évaluation: {np.mean(eval_stats['returns']):.2f}")
    logger.info(f"Ratio de Sharpe moyen sur l'évaluation: {np.mean(eval_stats['sharpe_ratios']):.2f}")
    logger.info(f"Ratio de Sortino moyen sur l'évaluation: {np.mean(eval_stats['sortino_ratios']):.2f}")

if __name__ == "__main__":
    main() 