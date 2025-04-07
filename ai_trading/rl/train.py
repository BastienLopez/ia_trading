import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import time
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf

# Configuration du logger
logger = logging.getLogger('TrainRL')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TrainingMonitor:
    """
    Classe pour surveiller et visualiser l'entraînement en temps réel.
    """
    
    def __init__(self, save_dir=None, plot_interval=10):
        """
        Initialise le moniteur d'entraînement.
        
        Args:
            save_dir (str, optional): Répertoire pour sauvegarder les visualisations
            plot_interval (int): Intervalle d'épisodes entre les mises à jour des graphiques
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.plot_interval = plot_interval
        self.history = {
            'episode_rewards': [],
            'episode_portfolio_values': [],
            'episode_returns': [],
            'losses': [],
            'epsilon': []
        }
        
        # Créer les figures pour les visualisations en temps réel
        self.figs = {}
        self.axes = {}
        
        # Initialiser les figures si matplotlib est en mode interactif
        if plt.isinteractive():
            self._init_figures()
    
    def _init_figures(self):
        """Initialise les figures pour la visualisation en temps réel."""
        # Figure pour les récompenses
        self.figs['rewards'] = plt.figure(figsize=(10, 6))
        self.axes['rewards'] = self.figs['rewards'].add_subplot(111)
        self.axes['rewards'].set_title('Récompenses par épisode')
        self.axes['rewards'].set_xlabel('Épisode')
        self.axes['rewards'].set_ylabel('Récompense totale')
        self.axes['rewards'].grid(True)
        
        # Figure pour la valeur du portefeuille
        self.figs['portfolio'] = plt.figure(figsize=(10, 6))
        self.axes['portfolio'] = self.figs['portfolio'].add_subplot(111)
        self.axes['portfolio'].set_title('Valeur du portefeuille par épisode')
        self.axes['portfolio'].set_xlabel('Épisode')
        self.axes['portfolio'].set_ylabel('Valeur ($)')
        self.axes['portfolio'].grid(True)
        
        # Figure pour les rendements
        self.figs['returns'] = plt.figure(figsize=(10, 6))
        self.axes['returns'] = self.figs['returns'].add_subplot(111)
        self.axes['returns'].set_title('Rendements par épisode')
        self.axes['returns'].set_xlabel('Épisode')
        self.axes['returns'].set_ylabel('Rendement (%)')
        self.axes['returns'].grid(True)
        
        # Figure pour les pertes
        self.figs['losses'] = plt.figure(figsize=(10, 6))
        self.axes['losses'] = self.figs['losses'].add_subplot(111)
        self.axes['losses'].set_title('Perte moyenne par épisode')
        self.axes['losses'].set_xlabel('Épisode')
        self.axes['losses'].set_ylabel('Perte')
        self.axes['losses'].grid(True)
        
        # Figure pour epsilon
        self.figs['epsilon'] = plt.figure(figsize=(10, 6))
        self.axes['epsilon'] = self.figs['epsilon'].add_subplot(111)
        self.axes['epsilon'].set_title('Epsilon par épisode')
        self.axes['epsilon'].set_xlabel('Épisode')
        self.axes['epsilon'].set_ylabel('Epsilon')
        self.axes['epsilon'].grid(True)
        
        # Afficher les figures
        for fig in self.figs.values():
            fig.show()
    
    def update(self, episode, reward, portfolio_value, returns, loss, epsilon):
        """
        Met à jour l'historique d'entraînement et les visualisations.
        
        Args:
            episode (int): Numéro de l'épisode
            reward (float): Récompense totale de l'épisode
            portfolio_value (float): Valeur finale du portefeuille
            returns (float): Rendement de l'épisode
            loss (float): Perte moyenne de l'épisode
            epsilon (float): Valeur actuelle d'epsilon
        """
        # Mettre à jour l'historique
        self.history['episode_rewards'].append(reward)
        self.history['episode_portfolio_values'].append(portfolio_value)
        self.history['episode_returns'].append(returns)
        self.history['losses'].append(loss)
        self.history['epsilon'].append(epsilon)
        
        # Mettre à jour les visualisations en temps réel si l'intervalle est atteint
        if plt.isinteractive() and episode % self.plot_interval == 0:
            self._update_plots()
    
    def _update_plots(self):
        """Met à jour les graphiques en temps réel."""
        # Mettre à jour le graphique des récompenses
        self.axes['rewards'].clear()
        self.axes['rewards'].plot(self.history['episode_rewards'])
        self.axes['rewards'].set_title('Récompenses par épisode')
        self.axes['rewards'].set_xlabel('Épisode')
        self.axes['rewards'].set_ylabel('Récompense totale')
        self.axes['rewards'].grid(True)
        
        # Mettre à jour le graphique de la valeur du portefeuille
        self.axes['portfolio'].clear()
        self.axes['portfolio'].plot(self.history['episode_portfolio_values'])
        self.axes['portfolio'].set_title('Valeur du portefeuille par épisode')
        self.axes['portfolio'].set_xlabel('Épisode')
        self.axes['portfolio'].set_ylabel('Valeur ($)')
        self.axes['portfolio'].grid(True)
        
        # Mettre à jour le graphique des rendements
        self.axes['returns'].clear()
        self.axes['returns'].plot(self.history['episode_returns'])
        self.axes['returns'].set_title('Rendements par épisode')
        self.axes['returns'].set_xlabel('Épisode')
        self.axes['returns'].set_ylabel('Rendement (%)')
        self.axes['returns'].grid(True)
        
        # Mettre à jour le graphique des pertes
        self.axes['losses'].clear()
        self.axes['losses'].plot(self.history['losses'])
        self.axes['losses'].set_title('Perte moyenne par épisode')
        self.axes['losses'].set_xlabel('Épisode')
        self.axes['losses'].set_ylabel('Perte')
        self.axes['losses'].grid(True)
        
        # Mettre à jour le graphique d'epsilon
        self.axes['epsilon'].clear()
        self.axes['epsilon'].plot(self.history['epsilon'])
        self.axes['epsilon'].set_title('Epsilon par épisode')
        self.axes['epsilon'].set_xlabel('Épisode')
        self.axes['epsilon'].set_ylabel('Epsilon')
        self.axes['epsilon'].grid(True)
        
        # Rafraîchir les figures
        for fig in self.figs.values():
            fig.canvas.draw()
            fig.canvas.flush_events()
    
    def save_plots(self):
        """Sauvegarde les graphiques finaux."""
        if not self.save_dir:
            return
        
        # Créer les graphiques finaux
        self._save_plot('rewards', 'Récompenses par épisode', 'Épisode', 'Récompense totale')
        self._save_plot('portfolio', 'Valeur du portefeuille par épisode', 'Épisode', 'Valeur ($)')
        self._save_plot('returns', 'Rendements par épisode', 'Épisode', 'Rendement (%)')
        self._save_plot('losses', 'Perte moyenne par épisode', 'Épisode', 'Perte')
        self._save_plot('epsilon', 'Epsilon par épisode', 'Épisode', 'Epsilon')
    
    def _save_plot(self, key, title, xlabel, ylabel):
        """
        Sauvegarde un graphique spécifique.
        
        Args:
            key (str): Clé du graphique dans l'historique
            title (str): Titre du graphique
            xlabel (str): Étiquette de l'axe X
            ylabel (str): Étiquette de l'axe Y
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history[f'episode_{key}'] if key != 'losses' and key != 'epsilon' else self.history[key])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{key}.png'))
        plt.close()
    
    def get_history(self):
        """
        Retourne l'historique d'entraînement.
        
        Returns:
            dict: Historique d'entraînement
        """
        return self.history

def train_agent(agent, env, episodes=100, batch_size=32, update_target_every=5,
                save_path=None, visualize=True, checkpoint_interval=10,
                early_stopping=None, max_steps_per_episode=None,
                use_tensorboard=False, tensorboard_log_dir='./logs'):
    """
    Entraîne l'agent RL avec des fonctionnalités avancées.
    
    Args:
        agent: L'agent à entraîner (DQNAgent)
        env: L'environnement de trading (TradingEnvironment)
        episodes (int): Nombre d'épisodes d'entraînement
        batch_size (int): Taille du batch pour l'entraînement
        update_target_every (int): Fréquence de mise à jour du modèle cible
        save_path (str, optional): Chemin pour sauvegarder le modèle
        visualize (bool): Si True, génère des visualisations
        checkpoint_interval (int): Intervalle d'épisodes entre les sauvegardes
        early_stopping (dict, optional): Paramètres pour l'arrêt anticipé
            - 'patience': Nombre d'épisodes sans amélioration
            - 'min_delta': Amélioration minimale requise
            - 'metric': Métrique à surveiller ('reward', 'portfolio_value', 'returns')
        max_steps_per_episode (int, optional): Nombre maximum d'étapes par épisode
        use_tensorboard (bool): Si True, utilise TensorBoard pour le suivi
        tensorboard_log_dir (str): Répertoire pour les logs TensorBoard
        
    Returns:
        dict: Historique d'entraînement
    """
    # Configurer TensorBoard si demandé
    if use_tensorboard:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(tensorboard_log_dir, current_time)
        summary_writer = tf.summary.create_file_writer(log_dir)
        logger.info(f"Logs TensorBoard disponibles dans {log_dir}")
    
    # Créer le dossier de sauvegarde si nécessaire
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialiser le moniteur d'entraînement
    viz_dir = os.path.join(os.path.dirname(save_path), 'visualizations') if save_path else 'visualizations'
    monitor = TrainingMonitor(save_dir=viz_dir, plot_interval=5)
    
    # Paramètres pour l'arrêt anticipé
    if early_stopping:
        best_value = float('-inf')
        patience_counter = 0
        patience = early_stopping.get('patience', 10)
        min_delta = early_stopping.get('min_delta', 0.0)
        metric = early_stopping.get('metric', 'returns')
    
    # Temps de début de l'entraînement
    start_time = time.time()
    
    # Boucle d'entraînement avec barre de progression
    for e in tqdm(range(episodes), desc="Entraînement"):
        # Réinitialiser l'environnement
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        
        # Variables pour suivre les performances de l'épisode
        total_reward = 0
        episode_losses = []
        step_count = 0
        
        done = False
        while not done:
            # Vérifier si le nombre maximum d'étapes est atteint
            if max_steps_per_episode and step_count >= max_steps_per_episode:
                break
            
            # Choisir une action
            action = agent.act(state)
            
            # Exécuter l'action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            
            # Stocker l'expérience dans la mémoire
            agent.remember(state, action, reward, next_state, done)
            
            # Passer à l'état suivant
            state = next_state
            
            # Accumuler la récompense
            total_reward += reward
            
            # Entraîner l'agent
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                episode_losses.append(loss)
            
            step_count += 1
        
        # Mettre à jour le modèle cible périodiquement
        if e % update_target_every == 0:
            agent.update_target_model()
        
        # Calculer les métriques de l'épisode
        portfolio_value = env.get_portfolio_value()
        returns = (portfolio_value / env.initial_balance) - 1
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        # Mettre à jour le moniteur
        monitor.update(e, total_reward, portfolio_value, returns, avg_loss, agent.epsilon)
        
        # Ajouter la récompense à l'historique de l'agent
        agent.reward_history.append(total_reward)
        
        # Afficher les progrès
        if (e + 1) % 10 == 0 or e == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Épisode {e+1}/{episodes} - Récompense: {total_reward:.4f}, "
                       f"Valeur du portefeuille: ${portfolio_value:.2f}, "
                       f"Rendement: {returns*100:.2f}%, "
                       f"Epsilon: {agent.epsilon:.4f}, "
                       f"Perte moyenne: {avg_loss:.4f}, "
                       f"Temps écoulé: {elapsed_time:.1f}s")
        
        # Enregistrer les métriques dans TensorBoard
        if use_tensorboard:
            with summary_writer.as_default():
                tf.summary.scalar('reward', total_reward, step=e)
                tf.summary.scalar('portfolio_value', portfolio_value, step=e)
                tf.summary.scalar('returns', returns, step=e)
                tf.summary.scalar('loss', avg_loss, step=e)
                tf.summary.scalar('epsilon', agent.epsilon, step=e)
        
        # Sauvegarder le modèle périodiquement
        if save_path and (e+1) % checkpoint_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"{save_path}_ep{e+1}_{timestamp}.h5"
            agent.save(save_file)
            logger.info(f"Modèle sauvegardé dans {save_file}")
        
        # Vérifier l'arrêt anticipé
        if early_stopping:
            current_value = total_reward if metric == 'reward' else portfolio_value if metric == 'portfolio_value' else returns
            
            if current_value > best_value + min_delta:
                best_value = current_value
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                if save_path:
                    best_model_path = f"{save_path}_best.h5"
                    agent.save(best_model_path)
                    logger.info(f"Meilleur modèle sauvegardé dans {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Arrêt anticipé à l'épisode {e+1} après {patience} épisodes sans amélioration")
                    break
    
    # Temps total d'entraînement
    total_time = time.time() - start_time
    logger.info(f"Entraînement terminé en {total_time:.1f} secondes")
    
    # Sauvegarder le modèle final
    if save_path:
        final_save_path = f"{save_path}_final.h5"
        agent.save(final_save_path)
        logger.info(f"Modèle final sauvegardé dans {final_save_path}")
    
    # Générer des visualisations finales
    if visualize:
        monitor.save_plots()
    
    return monitor.get_history() 