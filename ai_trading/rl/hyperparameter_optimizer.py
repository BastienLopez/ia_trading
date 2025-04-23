import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.config import INFO_RETOUR_DIR


# Définition des fonctions de métriques directement dans ce fichier
# (à remplacer par l'import approprié quand le module metrics sera disponible)
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calcule le ratio de Sharpe à partir d'une série de rendements.

    Args:
        returns: Série de rendements (en pourcentage)
        risk_free_rate: Taux sans risque (par défaut 0)

    Returns:
        float: Ratio de Sharpe annualisé
    """
    if len(returns) < 2:
        return 0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0
    return (
        (mean_return - risk_free_rate) / std_return * np.sqrt(252)
    )  # Annualisation (252 jours de trading)


def calculate_max_drawdown(portfolio_values):
    """
    Calcule le drawdown maximum à partir d'une série de valeurs de portefeuille.

    Args:
        portfolio_values: Série de valeurs du portefeuille

    Returns:
        float: Drawdown maximum (valeur positive)
    """
    if len(portfolio_values) < 2:
        return 0
    peak = portfolio_values[0]
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd


def calculate_win_rate(trades):
    """
    Calcule le taux de réussite des trades.

    Args:
        trades: Liste des informations de trades

    Returns:
        float: Pourcentage de trades gagnants (0-1)
    """
    if not trades:
        return 0
    winning_trades = sum(1 for trade in trades if trade.get("profit", 0) > 0)
    return winning_trades / len(trades)


# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HyperparameterOptimizer:
    """
    Optimiseur d'hyperparamètres pour les agents d'apprentissage par renforcement.
    Implémente différentes méthodes de recherche d'hyperparamètres et métriques
    d'évaluation pour trouver la meilleure configuration.
    """

    def __init__(
        self,
        env_creator,
        agent_class,
        param_grid,
        n_episodes=50,
        max_steps=None,
        eval_episodes=10,
        metrics=["total_reward", "sharpe_ratio", "max_drawdown", "win_rate"],
        save_dir=None,
        n_jobs=1,
        verbose=1,
    ):
        """
        Initialise l'optimiseur d'hyperparamètres.

        Args:
            env_creator: Fonction qui crée et retourne un environnement de trading
            agent_class: Classe de l'agent à optimiser (ex: SACAgent)
            param_grid: Dictionnaire avec les hyperparamètres à optimiser et leurs valeurs
            n_episodes: Nombre d'épisodes pour l'entraînement de chaque configuration
            max_steps: Nombre maximal d'étapes par épisode (None = pas de limite)
            eval_episodes: Nombre d'épisodes pour l'évaluation de chaque configuration
            metrics: Liste des métriques à calculer pour l'évaluation
            save_dir: Répertoire pour sauvegarder les résultats
            n_jobs: Nombre de processus parallèles (1 = séquentiel)
            verbose: Niveau de détail des logs (0 = silencieux, 1 = normal, 2 = détaillé)
        """
        self.env_creator = env_creator
        self.agent_class = agent_class
        self.param_grid = param_grid
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.eval_episodes = eval_episodes
        self.metrics = metrics
        
        # Utiliser INFO_RETOUR_DIR/hyperopt par défaut si save_dir n'est pas spécifié
        if save_dir is None:
            self.save_dir = INFO_RETOUR_DIR / "hyperopt"
        else:
            self.save_dir = save_dir
            
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Créer le répertoire de sauvegarde s'il n'existe pas
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Historique des résultats
        self.results = []
        self.best_params = None
        self.best_score = float("-inf")

        logger.info(
            f"HyperparameterOptimizer initialisé avec {len(self._get_param_combinations())} "
            f"combinaisons de paramètres"
        )

    def _get_param_combinations(self):
        """
        Génère toutes les combinaisons d'hyperparamètres possibles.

        Returns:
            list: Liste de dictionnaires, chacun contenant une combinaison d'hyperparamètres
        """
        # Créer les listes de valeurs pour chaque paramètre
        param_values = list(self.param_grid.values())
        param_names = list(self.param_grid.keys())

        # Générer toutes les combinaisons possibles
        combinations = list(product(*param_values))

        # Convertir en liste de dictionnaires
        param_combinations = [dict(zip(param_names, combo)) for combo in combinations]

        return param_combinations

    def grid_search(self):
        """
        Effectue une recherche par grille sur les hyperparamètres.

        Returns:
            dict: Meilleurs hyperparamètres trouvés
            float: Score de la meilleure configuration
        """
        param_combinations = self._get_param_combinations()
        total_combinations = len(param_combinations)

        logger.info(
            f"Démarrage de la recherche par grille avec {total_combinations} combinaisons"
        )

        # Exécution parallèle ou séquentielle selon n_jobs
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i, params in enumerate(param_combinations):
                    future = executor.submit(
                        self._evaluate_params, params, i, total_combinations
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    score, params, metrics = future.result()
                    self._update_best(score, params, metrics)
        else:
            for i, params in enumerate(param_combinations):
                score, params, metrics = self._evaluate_params(
                    params, i, total_combinations
                )
                self._update_best(score, params, metrics)

        # Sauvegarder les résultats finaux
        self._save_results()

        logger.info(f"Recherche par grille terminée. Meilleur score: {self.best_score}")
        logger.info(f"Meilleurs paramètres: {self.best_params}")

        return self.best_params, self.best_score

    def _evaluate_params(self, params, index, total):
        """
        Évalue une combinaison d'hyperparamètres.

        Args:
            params: Dictionnaire des hyperparamètres à évaluer
            index: Index de la combinaison (pour le logging)
            total: Nombre total de combinaisons (pour le logging)

        Returns:
            tuple: (score global, params, dictionnaire des métriques)
        """
        if self.verbose > 0:
            logger.info(f"Évaluation de la combinaison {index+1}/{total}: {params}")

        # Créer l'environnement et l'agent
        env = self.env_creator()

        # Extraire les paramètres spécifiques à l'agent
        # (filtrer ceux qui ne sont pas des arguments du constructeur de l'agent)
        agent_params = {}

        for param, value in params.items():
            # Ajouter les paramètres spécifiques à l'agent (nous pourrions ajouter une validation plus stricte)
            agent_params[param] = value

        # Créer l'agent avec les paramètres
        agent = self.agent_class(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            action_bounds=[
                float(env.action_space.low[0]),
                float(env.action_space.high[0]),
            ],
            **agent_params,
        )

        # Entraîner l'agent
        train_metrics = self._train_agent(
            env, agent, params, self.n_episodes, self.max_steps
        )

        # Évaluer l'agent
        eval_metrics = self._evaluate_agent(env, agent, self.eval_episodes)

        # Combiner les métriques
        all_metrics = {**train_metrics, **eval_metrics}

        # Calculer un score global (moyenne pondérée des métriques normalisées)
        score = self._calculate_score(all_metrics)

        if self.verbose > 0:
            logger.info(f"Combinaison {index+1}/{total} - Score: {score:.4f}")
            if self.verbose > 1:
                for metric, value in all_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")

        return score, params, all_metrics

    def _train_agent(self, env, agent, params, n_episodes, max_steps):
        """
        Entraîne l'agent et collecte les métriques d'entraînement.

        Args:
            env: Environnement de trading
            agent: Agent RL à entraîner
            params: Paramètres d'hyperparamètres
            n_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximal d'étapes par épisode

        Returns:
            dict: Métriques d'entraînement
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            step = 0
            done = False

            # Gérer les séquences pour les agents GRU
            if hasattr(agent, "use_gru") and agent.use_gru:
                # Créer une séquence initiale
                sequence = np.array([state] * agent.sequence_length)
                # Assurer que la forme est (sequence_length, state_size) et non (sequence_length, 1, state_size)
                if len(sequence.shape) > 2:
                    sequence = sequence.reshape(agent.sequence_length, -1)
                current_state = sequence
            else:
                current_state = state

            while not done and (max_steps is None or step < max_steps):
                # Sélectionner une action
                action = agent.act(current_state)

                # Exécuter l'action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Enregistrer l'expérience
                if hasattr(agent, "use_gru") and agent.use_gru:
                    # Pour les agents GRU, on stocke la séquence complète
                    agent.remember(current_state, action, reward, next_state, done)

                    # Mettre à jour la séquence en supprimant le plus ancien état et ajoutant le nouveau
                    # Assurer que next_state a la bonne forme avant de l'empiler
                    next_state_reshaped = next_state.reshape(1, -1)
                    sequence = np.vstack([sequence[1:], next_state_reshaped])
                    current_state = sequence
                else:
                    agent.remember(current_state, action, reward, next_state, done)
                    current_state = next_state

                episode_reward += reward
                step += 1

                # Entraîner l'agent si assez d'expériences sont collectées
                if (
                    step % params.get("train_frequency", 1) == 0
                    and agent.memory.size() >= agent.batch_size
                ):
                    agent.train()

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

            if self.verbose > 1 and episode % 5 == 0:
                logger.info(
                    f"Épisode {episode}/{n_episodes}, Récompense: {episode_reward:.2f}"
                )

        # Métriques d'entraînement
        training_metrics = {
            "avg_reward": np.mean(episode_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "episodes_completed": n_episodes,
        }

        return training_metrics

    def _evaluate_agent(self, env, agent, eval_episodes):
        """
        Évalue l'agent après entraînement.

        Args:
            env: Environnement de trading
            agent: Agent RL entraîné
            eval_episodes: Nombre d'épisodes d'évaluation

        Returns:
            dict: Métriques d'évaluation
        """
        episode_rewards = []
        portfolio_values = []
        trades = []

        for episode in range(eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            portfolio_history = [env.portfolio_value]

            # Gérer les séquences pour les agents GRU
            if hasattr(agent, "use_gru") and agent.use_gru:
                # Créer une séquence initiale
                sequence = np.array([state] * agent.sequence_length)
                # Assurer que la forme est (sequence_length, state_size) et non (sequence_length, 1, state_size)
                if len(sequence.shape) > 2:
                    sequence = sequence.reshape(agent.sequence_length, -1)
                current_state = sequence
            else:
                current_state = state

            while not done:
                # Sélectionner une action (en mode évaluation)
                action = agent.act(current_state, evaluate=True)

                # Exécuter l'action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Collecter les métriques
                episode_reward += reward
                portfolio_history.append(env.portfolio_value)

                # Enregistrer le trade si effectué
                if "trade_executed" in info and info["trade_executed"]:
                    trades.append(info)

                # Mettre à jour l'état courant ou la séquence
                if hasattr(agent, "use_gru") and agent.use_gru:
                    # Mettre à jour la séquence en supprimant le plus ancien état et ajoutant le nouveau
                    # Assurer que next_state a la bonne forme avant de l'empiler
                    next_state_reshaped = next_state.reshape(1, -1)
                    sequence = np.vstack([sequence[1:], next_state_reshaped])
                    current_state = sequence
                else:
                    current_state = next_state

            episode_rewards.append(episode_reward)
            portfolio_values.append(portfolio_history)

        # Calculer les métriques d'évaluation
        eval_metrics = {
            "eval_avg_reward": np.mean(episode_rewards),
            "total_reward": np.sum(episode_rewards),
        }

        # Calculer les métriques financières
        if "sharpe_ratio" in self.metrics:
            sharpe = self._calculate_sharpe_ratio(portfolio_values)
            eval_metrics["sharpe_ratio"] = sharpe

        if "max_drawdown" in self.metrics:
            mdd = self._calculate_max_drawdown(portfolio_values)
            eval_metrics["max_drawdown"] = mdd

        if "win_rate" in self.metrics and trades:
            win_rate = self._calculate_win_rate(trades)
            eval_metrics["win_rate"] = win_rate

        return eval_metrics

    def _calculate_sharpe_ratio(self, portfolio_values_list):
        """
        Calcule le ratio de Sharpe moyen sur plusieurs épisodes.

        Args:
            portfolio_values_list: Liste des historiques de valeur du portefeuille

        Returns:
            float: Ratio de Sharpe moyen
        """
        sharpe_ratios = []

        for portfolio_values in portfolio_values_list:
            returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])

            # Calculer Sharpe si possible
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )  # Annualiser (252 jours de trading)
                sharpe_ratios.append(sharpe)

        return np.mean(sharpe_ratios) if sharpe_ratios else 0

    def _calculate_max_drawdown(self, portfolio_values_list):
        """
        Calcule le drawdown maximum moyen sur plusieurs épisodes.

        Args:
            portfolio_values_list: Liste des historiques de valeur du portefeuille

        Returns:
            float: Drawdown maximum moyen (valeur positive)
        """
        max_drawdowns = []

        for portfolio_values in portfolio_values_list:
            # Calculer le drawdown maximum
            peak = portfolio_values[0]
            max_dd = 0

            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            max_drawdowns.append(max_dd)

        # Retourner la valeur positive (plus c'est petit, mieux c'est)
        return np.mean(max_drawdowns)

    def _calculate_win_rate(self, trades):
        """
        Calcule le taux de réussite des trades.

        Args:
            trades: Liste des informations de trades

        Returns:
            float: Pourcentage de trades gagnants (0-1)
        """
        if not trades:
            return 0

        winning_trades = sum(1 for trade in trades if trade.get("profit", 0) > 0)
        return winning_trades / len(trades)

    def _calculate_score(self, metrics):
        """
        Calcule un score global à partir des métriques.

        Args:
            metrics: Dictionnaire des métriques

        Returns:
            float: Score global
        """
        # Poids des métriques pour le score final
        weights = {
            "eval_avg_reward": 0.3,
            "total_reward": 0.3,
            "sharpe_ratio": 0.2,
            "max_drawdown": -0.1,  # Négatif car on veut minimiser le drawdown
            "win_rate": 0.1,
        }

        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight

        return score

    def _update_best(self, score, params, metrics):
        """
        Met à jour les meilleurs paramètres si le score est meilleur.

        Args:
            score: Score de la configuration évaluée
            params: Paramètres évalués
            metrics: Métriques calculées
        """
        # Ajouter aux résultats
        result = {
            "params": params,
            "score": score,
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.results.append(result)

        # Mettre à jour le meilleur
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            if self.verbose > 0:
                logger.info(f"Nouveau meilleur score: {score:.4f} avec {params}")

    def _save_results(self):
        """
        Sauvegarde les résultats de l'optimisation.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sauvegarder les résultats au format JSON
        results_file = os.path.join(self.save_dir, f"hyperopt_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(
                {
                    "best_params": self.best_params,
                    "best_score": self.best_score,
                    "results": self.results,
                },
                f,
                indent=2,
            )

        # Créer un DataFrame pour analyse
        df_results = []
        for result in self.results:
            row = {**result["params"], "score": result["score"]}
            for metric, value in result["metrics"].items():
                row[metric] = value
            df_results.append(row)

        df = pd.DataFrame(df_results)

        # Sauvegarder le DataFrame au format CSV
        csv_file = os.path.join(self.save_dir, f"hyperopt_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)

        # Générer des visualisations
        self._generate_plots(df, timestamp)

        logger.info(f"Résultats sauvegardés dans {self.save_dir}")

    def _generate_plots(self, df, timestamp):
        """
        Génère des visualisations des résultats.

        Args:
            df: DataFrame contenant les résultats
            timestamp: Horodatage pour les noms de fichiers
        """
        # Tracer les distributions des scores
        plt.figure(figsize=(10, 6))
        plt.hist(df["score"], bins=20, alpha=0.7)
        plt.axvline(
            self.best_score,
            color="r",
            linestyle="--",
            label=f"Meilleur score: {self.best_score:.4f}",
        )
        plt.xlabel("Score")
        plt.ylabel("Fréquence")
        plt.title("Distribution des scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"score_distribution_{timestamp}.png"))

        # Tracer l'importance des hyperparamètres
        if len(df) > 5:  # Suffisamment de données pour l'analyse
            try:
                # Pour chaque hyperparamètre
                for param in self.param_grid.keys():
                    if param in df.columns and len(df[param].unique()) > 1:
                        plt.figure(figsize=(10, 6))

                        # Regrouper par paramètre et calculer le score moyen
                        param_scores = df.groupby(param)["score"].mean().reset_index()

                        # Tracer le paramètre vs score
                        plt.bar(
                            param_scores[param].astype(str),
                            param_scores["score"],
                            alpha=0.7,
                        )
                        plt.xlabel(param)
                        plt.ylabel("Score moyen")
                        plt.title(f"Impact de {param} sur le score")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                self.save_dir, f"param_impact_{param}_{timestamp}.png"
                            )
                        )
            except Exception as e:
                logger.warning(
                    f"Erreur lors de la génération des graphiques d'importance des paramètres: {e}"
                )

        # Fermer toutes les figures pour libérer la mémoire
        plt.close("all")


def optimize_sac_agent(
    train_data,
    param_grid=None,
    n_episodes=50,
    eval_episodes=10,
    save_dir=None,
    n_jobs=1,
):
    """
    Optimise les hyperparamètres d'un agent SAC pour un environnement de trading.

    Args:
        train_data: DataFrame contenant les données d'entraînement
        param_grid: Dictionnaire des hyperparamètres à optimiser (None = paramètres par défaut)
        n_episodes: Nombre d'épisodes d'entraînement par configuration
        eval_episodes: Nombre d'épisodes d'évaluation par configuration
        save_dir: Répertoire pour sauvegarder les résultats (None = utiliser la valeur par défaut)
        n_jobs: Nombre de processus parallèles

    Returns:
        dict: Meilleurs hyperparamètres trouvés
    """
    # Paramètres par défaut si non spécifiés
    if param_grid is None:
        param_grid = {
            "actor_learning_rate": [1e-4, 3e-4, 1e-3],
            "critic_learning_rate": [1e-4, 3e-4, 1e-3],
            "batch_size": [64, 128, 256],
            "hidden_size": [128, 256, 512],
            "entropy_regularization": [0.0, 0.001, 0.01],
            "grad_clip_value": [None, 0.5, 1.0],
        }
        
    # Utiliser INFO_RETOUR_DIR/sac_optimization par défaut
    if save_dir is None:
        save_dir = INFO_RETOUR_DIR / "sac_optimization"

    # Fonction pour créer l'environnement de trading
    def create_env():
        return TradingEnvironment(
            df=train_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=20,
            reward_function="sharpe",
            action_type="continuous",
        )

    # Créer l'optimiseur d'hyperparamètres
    optimizer = HyperparameterOptimizer(
        env_creator=create_env,
        agent_class=SACAgent,
        param_grid=param_grid,
        n_episodes=n_episodes,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        n_jobs=n_jobs,
    )

    # Effectuer la recherche par grille
    best_params, best_score = optimizer.grid_search()

    return best_params


def optimize_gru_sac_agent(
    train_data,
    param_grid=None,
    n_episodes=50,
    eval_episodes=10,
    save_dir=None,
    n_jobs=1,
):
    """
    Optimise les hyperparamètres d'un agent SAC avec couches GRU pour un environnement de trading.

    Args:
        train_data: DataFrame contenant les données d'entraînement
        param_grid: Dictionnaire des hyperparamètres à optimiser (None = paramètres par défaut)
        n_episodes: Nombre d'épisodes d'entraînement par configuration
        eval_episodes: Nombre d'épisodes d'évaluation par configuration
        save_dir: Répertoire pour sauvegarder les résultats (None = utiliser la valeur par défaut)
        n_jobs: Nombre de processus parallèles

    Returns:
        dict: Meilleurs hyperparamètres trouvés
    """
    # Paramètres par défaut si non spécifiés
    if param_grid is None:
        param_grid = {
            "actor_learning_rate": [1e-4, 3e-4, 1e-3],
            "critic_learning_rate": [1e-4, 3e-4, 1e-3],
            "batch_size": [64, 128],
            "hidden_size": [128, 256],
            "entropy_regularization": [0.0, 0.01],
            "grad_clip_value": [0.5, 1.0],
            "use_gru": [True],
            "sequence_length": [5, 10],
            "gru_units": [32, 64, 128],
        }
        
    # Utiliser INFO_RETOUR_DIR/gru_sac_optimization par défaut
    if save_dir is None:
        save_dir = INFO_RETOUR_DIR / "gru_sac_optimization"

    # Fonction pour créer l'environnement de trading
    def create_env():
        return TradingEnvironment(
            df=train_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=20,
            reward_function="sharpe",
            action_type="continuous",
        )

    # Créer l'optimiseur d'hyperparamètres
    optimizer = HyperparameterOptimizer(
        env_creator=create_env,
        agent_class=SACAgent,
        param_grid=param_grid,
        n_episodes=n_episodes,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        n_jobs=n_jobs,
    )

    # Effectuer la recherche par grille
    best_params, best_score = optimizer.grid_search()

    return best_params
