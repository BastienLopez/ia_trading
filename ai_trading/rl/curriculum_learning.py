import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logger = logging.getLogger("CurriculumLearning")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class CurriculumLearning:
    """
    Implémente une stratégie d'apprentissage progressif (curriculum learning) pour les agents de trading.

    Le curriculum learning permet d'augmenter progressivement la difficulté de l'environnement
    pendant l'entraînement, ce qui aide l'agent à apprendre des stratégies plus complexes de manière progressive.

    Caractéristiques :
    - Contrôle de la volatilité du marché
    - Augmentation progressive des frais de transaction
    - Réduction progressive de la fenêtre d'observation
    - Ajout progressif d'actifs (pour le trading multi-actifs)
    - Progression des fonctions de récompense
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        difficulty_increment: float = 0.1,
        success_threshold: float = 0.5,
        patience: int = 5,
        curriculum_type: str = "mixed",  # "volatility", "transaction_fee", "window_size", "reward", "mixed"
        agent_performance_fn: Optional[Callable] = None,
        env_params: Optional[Dict] = None,
    ):
        """
        Initialise le système d'apprentissage progressif.

        Args:
            df (pd.DataFrame): DataFrame contenant les données du marché
            initial_difficulty (float): Niveau de difficulté initial (0.0 à 1.0)
            max_difficulty (float): Niveau de difficulté maximal
            difficulty_increment (float): Incrément de difficulté après chaque réussite
            success_threshold (float): Seuil de performance pour considérer qu'un niveau est réussi
            patience (int): Nombre d'épisodes consécutifs pour considérer une réussite
            curriculum_type (str): Type de curriculum learning à appliquer
            agent_performance_fn (Callable): Fonction pour évaluer la performance de l'agent
            env_params (Dict): Paramètres de base de l'environnement
        """
        self.df = df
        self.initial_difficulty = initial_difficulty
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_increment = difficulty_increment
        self.success_threshold = success_threshold
        self.patience = patience
        self.curriculum_type = curriculum_type
        self.agent_performance_fn = agent_performance_fn

        # Paramètres par défaut de l'environnement si non spécifiés
        self.env_params = env_params or {
            "initial_balance": 10000.0,
            "transaction_fee": 0.001,
            "window_size": 20,
            "include_technical_indicators": True,
            "risk_management": True,
            "normalize_observation": True,
            "reward_function": "simple",
            "action_type": "discrete",
            "n_discrete_actions": 5,
        }

        # Historique des performances
        self.performance_history = []
        self.success_streak = 0

        # Niveaux de difficulté et leurs paramètres
        self.difficulty_levels = self._create_difficulty_levels()

        logger.info(
            f"Système de curriculum learning initialisé avec difficulté {initial_difficulty} "
            f"et type '{curriculum_type}'"
        )

    def _create_difficulty_levels(self) -> Dict[float, Dict]:
        """
        Crée les niveaux de difficulté avec leurs paramètres correspondants.

        Returns:
            Dict[float, Dict]: Dictionnaire des niveaux de difficulté avec leurs paramètres
        """
        levels = {}

        # Générer des niveaux de difficulté de initial_difficulty à max_difficulty
        current = self.initial_difficulty
        while current <= self.max_difficulty:
            level_params = self._generate_params_for_difficulty(current)
            levels[current] = level_params
            current = round(current + self.difficulty_increment, 2)

        return levels

    def _generate_params_for_difficulty(self, difficulty: float) -> Dict:
        """
        Génère les paramètres d'environnement pour un niveau de difficulté donné.

        Args:
            difficulty (float): Niveau de difficulté (0.0 à 1.0)

        Returns:
            Dict: Paramètres d'environnement adaptés au niveau de difficulté
        """
        # Copier les paramètres de base
        params = self.env_params.copy()

        # Ajuster les paramètres selon le type de curriculum et la difficulté
        if self.curriculum_type in ["volatility", "mixed"]:
            # Sélection des données basée sur la volatilité
            # Plus le niveau est élevé, plus les périodes volatiles sont incluses
            params["volatility_percentile"] = difficulty

        if self.curriculum_type in ["transaction_fee", "mixed"]:
            # Augmenter progressivement les frais de transaction
            base_fee = self.env_params.get("transaction_fee", 0.001)
            # Commencer avec des frais très bas, puis augmenter progressivement
            params["transaction_fee"] = base_fee * (0.1 + 0.9 * difficulty)

        if self.curriculum_type in ["window_size", "mixed"]:
            # Réduire progressivement la fenêtre d'observation
            base_window = self.env_params.get("window_size", 20)
            # Commencer avec une grande fenêtre, puis réduire
            params["window_size"] = int(base_window * (2.0 - difficulty))

        if self.curriculum_type in ["reward", "mixed"]:
            # Progression des fonctions de récompense
            reward_functions = ["simple", "transaction_penalty", "sharpe", "drawdown"]
            # Au début, récompense simple; à la fin, récompense avec drawdown
            reward_index = min(
                int(difficulty * len(reward_functions)), len(reward_functions) - 1
            )
            params["reward_function"] = reward_functions[reward_index]

        return params

    def create_environment(self) -> TradingEnvironment:
        """
        Crée un environnement de trading avec le niveau de difficulté actuel.

        Returns:
            TradingEnvironment: Environnement de trading configuré
        """
        # Obtenir les paramètres pour le niveau de difficulté actuel
        params = self.difficulty_levels[self.current_difficulty].copy()

        # Filtrer les données selon la volatilité si nécessaire
        if "volatility_percentile" in params:
            volatility_percentile = params.pop("volatility_percentile")
            filtered_df = self._filter_data_by_volatility(volatility_percentile)
        else:
            filtered_df = self.df

        # Créer l'environnement avec les paramètres adaptés
        env = TradingEnvironment(filtered_df, **params)

        logger.info(
            f"Environnement créé avec difficulté {self.current_difficulty} et paramètres: {params}"
        )
        return env

    def _filter_data_by_volatility(self, volatility_percentile: float) -> pd.DataFrame:
        """
        Filtre les données selon leur volatilité.

        Args:
            volatility_percentile (float): Percentile de volatilité (0.0 à 1.0)

        Returns:
            pd.DataFrame: Données filtrées
        """
        # Récupérer la taille de fenêtre requise par l'environnement
        window_size = self.env_params.get("window_size", 20)
        # Ajouter une marge de sécurité
        min_required_size = window_size * 3

        # Calculer la volatilité (écart-type des rendements) sur une fenêtre glissante
        returns = self.df["close"].pct_change().dropna()
        volatility = returns.rolling(window=20).std().dropna()

        # Déterminer le seuil de volatilité
        threshold = volatility.quantile(volatility_percentile)

        # Sélectionner les périodes où la volatilité est inférieure au seuil
        # Plus le percentile est bas, plus la volatilité est faible
        valid_indices = volatility[volatility <= threshold].index

        # Récupérer le dataframe filtré
        filtered_df = self.df.loc[valid_indices]

        # Assurez-vous que les données sont suffisantes pour l'environnement
        if len(filtered_df) < min_required_size:
            logger.warning(
                f"Filtrage par volatilité a réduit les données à {len(filtered_df)} lignes, "
                f"ce qui est insuffisant pour la taille de fenêtre {window_size}."
            )

            # Si trop peu de données, utiliser une stratégie adaptative
            if len(filtered_df) == 0:
                # Si aucune donnée ne correspond, prendre les moins volatiles
                sorted_volatility = volatility.sort_values()
                # Prendre au moins min_required_size points ou 30% des données, selon le max
                n_points = max(min_required_size, int(len(self.df) * 0.3))
                selected_indices = sorted_volatility.iloc[:n_points].index
                filtered_df = self.df.loc[selected_indices]
                logger.info(f"Utilisation des {n_points} points les moins volatiles.")
            elif len(filtered_df) < min_required_size:
                # Si quelques données existent mais pas assez, trouver la plus longue séquence
                # et compléter avec des données adjacentes si nécessaire
                filtered_df = self._get_longest_consecutive_segment(filtered_df)

                # Si toujours insuffisant, ajouter des points avant et après
                if len(filtered_df) < min_required_size and len(filtered_df) > 0:
                    # Déterminer la première et dernière date
                    first_date = filtered_df.index.min()
                    last_date = filtered_df.index.max()

                    # Nombre de points à ajouter
                    points_needed = min_required_size - len(filtered_df)

                    # Ajouter des points avant et après en quantité égale
                    before_indices = self.df[self.df.index < first_date].index[
                        -points_needed // 2 :
                    ]
                    after_indices = self.df[self.df.index > last_date].index[
                        : points_needed // 2
                    ]

                    # Combiner tous les indices
                    all_indices = (
                        before_indices.tolist()
                        + filtered_df.index.tolist()
                        + after_indices.tolist()
                    )
                    filtered_df = self.df.loc[all_indices]

                    logger.info(
                        f"Ajout de points adjacents pour atteindre {len(filtered_df)} points."
                    )

        # Vérification finale
        if len(filtered_df) < min_required_size:
            logger.warning(
                f"Impossible d'obtenir suffisamment de données. Utilisation des données originales."
            )
            # En dernier recours, utiliser les données originales avec une sélection aléatoire
            random_indices = np.random.choice(
                len(self.df), size=min(len(self.df), min_required_size), replace=False
            )
            filtered_df = self.df.iloc[sorted(random_indices)]

        logger.info(
            f"Données filtrées par volatilité: {len(filtered_df)} points (percentile {volatility_percentile})"
        )
        return filtered_df

    def _get_longest_consecutive_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sélectionne la plus longue séquence consécutive d'un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame avec des indices potentiellement non consécutifs

        Returns:
            pd.DataFrame: Plus longue séquence consécutive
        """
        if df.empty:
            return df

        # Vérifier si l'index est un DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            # Convertir les dates en positions numériques pour trouver les séquences consécutives
            # On utilise les positions entières plutôt que les timestamps directement
            indices = np.arange(len(df))
            df_reset = df.reset_index()
            date_col = df_reset.columns[0]  # Première colonne contient les dates

            # Grouper les données par différence constante pour identifier les séquences
            groups = []
            if len(indices) > 0:
                current_group = [0]  # Commencer avec la première position

                for i in range(1, len(indices)):
                    # Calculer la différence en secondes avec la date précédente
                    prev_date = df_reset.iloc[i - 1][date_col]
                    curr_date = df_reset.iloc[i][date_col]

                    # Vérifier si les dates sont consécutives (par exemple, différence d'un jour)
                    # Cela dépend de la fréquence attendue des données
                    time_diff = (curr_date - prev_date).total_seconds()
                    expected_diff = (
                        86400  # 1 jour en secondes, ajustez selon votre fréquence
                    )

                    if abs(time_diff - expected_diff) < 3600:  # Tolérance d'une heure
                        current_group.append(i)
                    else:
                        groups.append(current_group)
                        current_group = [i]

                groups.append(current_group)

            # Trouver le groupe le plus long
            if groups:
                longest_group = max(groups, key=len)
                # Obtenir les indices originaux correspondant au groupe le plus long
                original_indices = df.index[longest_group]
                return df.loc[original_indices]
            return df
        else:
            # Pour les index non-datetime, utiliser la méthode originale
            indices = df.index.tolist()
            sequences = []
            current_seq = [indices[0]]

            for i in range(1, len(indices)):
                # Vérifier si les indices sont consécutifs
                if isinstance(indices[i], (int, np.integer)) and isinstance(
                    indices[i - 1], (int, np.integer)
                ):
                    if indices[i] == indices[i - 1] + 1:
                        current_seq.append(indices[i])
                    else:
                        sequences.append(current_seq)
                        current_seq = [indices[i]]
                else:
                    # Si les indices ne sont pas des entiers, les traiter comme non consécutifs
                    sequences.append(current_seq)
                    current_seq = [indices[i]]

            # Ajouter la dernière séquence
            sequences.append(current_seq)

            # Trouver la plus longue séquence
            if sequences:
                longest_seq = max(sequences, key=len)
                return df.loc[longest_seq]
            return df

    def update_difficulty(self, agent_performance: float) -> bool:
        """
        Met à jour le niveau de difficulté en fonction de la performance de l'agent.

        Args:
            agent_performance (float): Mesure de performance de l'agent (0.0 à 1.0)

        Returns:
            bool: True si la difficulté a été augmentée, False sinon
        """
        self.performance_history.append(agent_performance)

        # Vérifier si l'agent a atteint le seuil de succès
        if agent_performance >= self.success_threshold:
            self.success_streak += 1
            logger.info(
                f"Succès {self.success_streak}/{self.patience} avec performance {agent_performance:.4f}"
            )
        else:
            self.success_streak = 0
            logger.info(
                f"Performance insuffisante: {agent_performance:.4f} < {self.success_threshold}"
            )

        # Si l'agent a réussi consécutivement le nombre de fois requis, augmenter la difficulté
        if self.success_streak >= self.patience:
            self.success_streak = 0

            # Calculer la nouvelle difficulté
            new_difficulty = min(
                self.current_difficulty + self.difficulty_increment, self.max_difficulty
            )
            # Arrondir la difficulté à 6 décimales pour éviter les erreurs de précision
            new_difficulty = round(new_difficulty, 6)

            # Si on a atteint la difficulté maximale, on reste à ce niveau
            if new_difficulty == self.current_difficulty:
                logger.info(
                    f"Difficulté maximale déjà atteinte: {self.current_difficulty}"
                )
                return False

            # Mettre à jour la difficulté
            self.current_difficulty = new_difficulty
            logger.info(f"Difficulté augmentée à {self.current_difficulty}")
            return True

        return False

    def get_current_params(self) -> Dict:
        """
        Retourne les paramètres actuels pour le niveau de difficulté courant.

        Returns:
            Dict: Paramètres de l'environnement pour le niveau actuel
        """
        return self.difficulty_levels[self.current_difficulty].copy()

    def reset(self):
        """
        Réinitialise le système de curriculum learning à son état initial.
        """
        self.current_difficulty = self.initial_difficulty
        self.performance_history = []
        self.success_streak = 0
        logger.info(
            f"Système de curriculum learning réinitialisé à la difficulté {self.initial_difficulty}"
        )


class CurriculumTrainer:
    """
    Classe pour entraîner un agent avec curriculum learning.
    """

    def __init__(
        self,
        agent: Any,
        curriculum: CurriculumLearning,
        episodes_per_level: int = 100,
        max_episodes: int = 2000,
        eval_every: int = 10,
    ):
        """
        Initialise l'entraîneur avec curriculum learning.

        Args:
            agent: Agent d'apprentissage par renforcement
            curriculum (CurriculumLearning): Système de curriculum learning
            episodes_per_level (int): Nombre maximum d'épisodes par niveau
            max_episodes (int): Nombre maximum d'épisodes au total
            eval_every (int): Fréquence d'évaluation (en épisodes)
        """
        self.agent = agent
        self.curriculum = curriculum
        self.episodes_per_level = episodes_per_level
        self.max_episodes = max_episodes
        self.eval_every = eval_every

        self.total_episodes = 0
        self.level_episodes = 0
        self.training_history = {
            "episode": [],
            "level": [],
            "reward": [],
            "portfolio_value": [],
            "performance": [],
        }

        logger.info(
            f"Entraîneur avec curriculum learning initialisé: "
            f"max_episodes={max_episodes}, episodes_per_level={episodes_per_level}"
        )

    def train(self, verbose: bool = True) -> Dict:
        """
        Entraîne l'agent avec curriculum learning.

        Args:
            verbose (bool): Afficher les logs détaillés

        Returns:
            Dict: Historique d'entraînement
        """
        # Commencer par le niveau de difficulté initial
        current_level = self.curriculum.current_difficulty
        env = self.curriculum.create_environment()

        logger.info(f"Démarrage de l'entraînement avec curriculum learning")
        logger.info(f"Niveau initial: {current_level}")

        while self.total_episodes < self.max_episodes:
            # Réinitialiser les compteurs pour le niveau
            self.level_episodes = 0
            level_rewards = []

            # Entraîner sur le niveau actuel
            for _ in range(self.episodes_per_level):
                if self.total_episodes >= self.max_episodes:
                    break

                # Entraîner un épisode
                episode_reward, portfolio_value = self._train_episode(env)
                level_rewards.append(episode_reward)

                # Enregistrer les métriques
                self.training_history["episode"].append(self.total_episodes)
                self.training_history["level"].append(current_level)
                self.training_history["reward"].append(episode_reward)
                self.training_history["portfolio_value"].append(portfolio_value)

                # Incrémenter les compteurs
                self.level_episodes += 1
                self.total_episodes += 1

                # Évaluer périodiquement et mettre à jour la difficulté si nécessaire
                if self.level_episodes % self.eval_every == 0:
                    performance = self._evaluate_agent(env)
                    self.training_history["performance"].append(performance)

                    # Mettre à jour la difficulté en fonction de la performance
                    difficulty_increased = self.curriculum.update_difficulty(
                        performance
                    )

                    if difficulty_increased:
                        # Si la difficulté a augmenté, créer un nouvel environnement
                        logger.info(
                            f"Niveau {current_level} terminé. Passage au niveau {self.curriculum.current_difficulty}"
                        )
                        current_level = self.curriculum.current_difficulty
                        env = self.curriculum.create_environment()
                        break

                # Afficher les progrès
                if verbose and self.total_episodes % 10 == 0:
                    logger.info(
                        f"Épisode {self.total_episodes}/{self.max_episodes} | "
                        f"Niveau {current_level} | "
                        f"Récompense: {episode_reward:.2f} | "
                        f"Portefeuille: {portfolio_value:.2f}"
                    )

            # Si on a terminé tous les épisodes du niveau sans augmenter la difficulté
            if self.level_episodes >= self.episodes_per_level:
                logger.info(
                    f"Nombre maximum d'épisodes atteint pour le niveau {current_level}"
                )

                # Faire une dernière évaluation et tenter d'augmenter la difficulté
                performance = self._evaluate_agent(env)
                difficulty_increased = self.curriculum.update_difficulty(performance)

                if difficulty_increased:
                    current_level = self.curriculum.current_difficulty
                    env = self.curriculum.create_environment()

        logger.info(f"Entraînement terminé après {self.total_episodes} épisodes")
        logger.info(f"Niveau final atteint: {current_level}")

        return self.training_history

    def _train_episode(self, env: TradingEnvironment) -> Tuple[float, float]:
        """
        Entraîne l'agent pour un épisode.

        Args:
            env (TradingEnvironment): Environnement de trading

        Returns:
            Tuple[float, float]: Récompense totale et valeur finale du portefeuille
        """
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action = self.agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Entraîner l'agent - s'assurer que update est appelé dans tous les cas
            self.agent.update(state, action, reward, next_state, done)

            # Si l'agent a une méthode remember et replay, les utiliser aussi
            if hasattr(self.agent, "remember"):
                # Pour les agents basés sur DQN
                self.agent.remember(state, action, reward, next_state, done)
                if hasattr(self.agent, "memory") and hasattr(self.agent, "batch_size"):
                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.replay()

            state = next_state
            total_reward += reward

        # Récupérer la valeur finale du portefeuille
        portfolio_value = env.get_portfolio_value()

        return total_reward, portfolio_value

    def _evaluate_agent(self, env: TradingEnvironment) -> float:
        """
        Évalue la performance de l'agent.

        Args:
            env (TradingEnvironment): Environnement de trading

        Returns:
            float: Score de performance normalisé entre 0 et 1
        """
        # Si une fonction d'évaluation personnalisée est fournie, l'utiliser
        if self.curriculum.agent_performance_fn is not None:
            return self.curriculum.agent_performance_fn(self.agent, env)

        # Évaluation par défaut: faire une moyenne sur plusieurs épisodes
        n_eval_episodes = 5
        returns = []
        portfolio_values = []

        for _ in range(n_eval_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_return = 0

            while not (done or truncated):
                # S'assurer que evaluate=True est toujours passé
                action = self.agent.get_action(state, evaluate=True)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                state = next_state

            returns.append(episode_return)
            portfolio_values.append(env.get_portfolio_value())

        # Calculer le rendement moyen et normaliser entre 0 et 1
        avg_return = np.mean(returns)
        avg_portfolio = np.mean(portfolio_values)

        # Calculer le ROI (Return On Investment)
        roi = (avg_portfolio / env.initial_balance) - 1

        # Normaliser le ROI en une performance entre 0 et 1
        # Un ROI de -0.2 (perte de 20%) donne une performance de 0
        # Un ROI de 0.5 (gain de 50%) donne une performance de 1
        normalized_performance = min(max((roi + 0.2) / 0.7, 0), 1)

        logger.info(
            f"Évaluation: ROI={roi:.4f}, Performance={normalized_performance:.4f}"
        )

        return normalized_performance


class GRUCurriculumLearning:
    """
    Implémentation du curriculum learning pour l'agent SAC avec couches GRU.
    Cette classe étend le concept de curriculum learning pour prendre en compte
    la nature séquentielle des données temporelles et l'utilisation des couches GRU.
    """

    def __init__(
        self,
        initial_difficulty=0.2,
        max_difficulty=1.0,
        difficulty_increment=0.1,
        success_threshold=0.7,
        evaluation_window=5,
        sequence_length=10,
        gru_units=128,
        hidden_size=256,
    ):
        """
        Initialise le système de curriculum learning adapté pour GRU.

        Args:
            initial_difficulty (float): Niveau de difficulté initial (0.0 à 1.0)
            max_difficulty (float): Niveau de difficulté maximal
            difficulty_increment (float): Incrément de difficulté après succès
            success_threshold (float): Seuil de réussite pour augmenter la difficulté
            evaluation_window (int): Nombre d'épisodes pour évaluer la performance
            sequence_length (int): Longueur des séquences pour GRU
            gru_units (int): Nombre d'unités dans les couches GRU
            hidden_size (int): Taille des couches cachées des réseaux
        """
        self.difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_increment = difficulty_increment
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        self.recent_performances = []
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        self.hidden_size = hidden_size

        logger.info(
            f"GRU Curriculum Learning initialisé: difficulté={self.difficulty}, "
            f"séquence={sequence_length}, unités GRU={gru_units}"
        )

    def create_environment(self, df, volatility_window=20, **kwargs):
        """
        Crée un environnement de trading avec une difficulté ajustée.

        Args:
            df (DataFrame): Données de marché complètes
            volatility_window (int): Fenêtre pour calculer la volatilité
            **kwargs: Arguments supplémentaires pour l'environnement

        Returns:
            TradingEnvironment: Environnement de trading adapté au niveau de difficulté
        """
        # Filtrer les données selon la difficulté (basée sur la volatilité)
        filtered_df = self._filter_data_by_volatility(df, volatility_window)

        # Créer l'environnement avec les données filtrées
        env = TradingEnvironment(
            df=filtered_df,
            window_size=kwargs.get("window_size", 20),
            initial_balance=kwargs.get("initial_balance", 10000.0),
            transaction_fee=kwargs.get("transaction_fee", 0.001),
            reward_function=kwargs.get("reward_function", "simple"),
            action_type=kwargs.get("action_type", "continuous"),
        )

        logger.info(
            f"Environnement créé avec difficulté {self.difficulty}: {len(filtered_df)} points de données"
        )
        return env

    def _filter_data_by_volatility(self, df, window=20):
        """
        Filtre les données selon la volatilité, en fonction du niveau de difficulté.

        Args:
            df (DataFrame): Données complètes
            window (int): Fenêtre pour calculer la volatilité

        Returns:
            DataFrame: Données filtrées selon la volatilité et le niveau de difficulté
        """
        # Calculer la volatilité sur la fenêtre spécifiée
        returns = df["close"].pct_change().dropna()
        volatility = returns.rolling(window=window).std().dropna()

        # Associer la volatilité aux données
        volatility_df = df.iloc[window:].copy()
        volatility_df["volatility"] = volatility.values

        # Déterminer les seuils de volatilité
        min_vol = volatility_df["volatility"].min()
        max_vol = volatility_df["volatility"].max()

        # Calculer le seuil de volatilité en fonction de la difficulté
        # Plus la difficulté est élevée, plus on inclut de données volatiles
        vol_threshold = min_vol + self.difficulty * (max_vol - min_vol)

        # Filtrer les données
        if self.difficulty < 1.0:
            filtered_df = volatility_df[volatility_df["volatility"] <= vol_threshold]
        else:
            # À difficulté maximale, utiliser toutes les données
            filtered_df = volatility_df

        # S'assurer qu'il y a suffisamment de données pour l'entraînement avec GRU
        min_required_size = 3 * self.sequence_length  # Au moins 3 séquences complètes

        if len(filtered_df) < min_required_size:
            logger.warning(
                f"Données insuffisantes après filtrage ({len(filtered_df)} points). "
                f"Adaptation de la stratégie de filtrage."
            )

            if len(filtered_df) == 0:
                # Si aucune donnée ne correspond aux critères, prendre les moins volatiles
                filtered_df = volatility_df.sort_values("volatility").head(
                    min_required_size
                )
            else:
                # Si quelques points mais pas assez, essayer de trouver des segments consécutifs
                # et ajouter des points adjacents si nécessaire
                if len(filtered_df) < min_required_size:
                    # Ajouter plus de points jusqu'à atteindre la taille minimale
                    additional_points = min_required_size - len(filtered_df)
                    # Trier le reste des données par volatilité et prendre les moins volatiles
                    remaining_df = volatility_df[
                        ~volatility_df.index.isin(filtered_df.index)
                    ]
                    remaining_df = remaining_df.sort_values("volatility")
                    filtered_df = pd.concat(
                        [filtered_df, remaining_df.head(additional_points)]
                    )
                    filtered_df = filtered_df.sort_index()  # Trier par date

        logger.info(
            f"Données filtrées: {len(filtered_df)} points avec seuil de volatilité {vol_threshold:.6f}"
        )
        return filtered_df

    def create_agent(self, env):
        """
        Crée un agent SAC avec couches GRU, adapté à l'environnement.

        Args:
            env: Environnement de trading

        Returns:
            SACAgent: Agent SAC avec GRU configuré
        """
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        action_bounds = (env.action_space.low[0], env.action_space.high[0])

        # Créer l'agent avec GRU activé
        agent = SACAgent(
            state_size=state_size,
            action_size=action_size,
            action_bounds=action_bounds,
            use_gru=True,
            sequence_length=self.sequence_length,
            gru_units=self.gru_units,
            hidden_size=self.hidden_size,
            grad_clip_value=1.0,
            entropy_regularization=0.001,
        )

        logger.info(
            f"Agent SAC avec GRU créé: state_size={state_size}, action_size={action_size}"
        )
        return agent

    def update_difficulty(self, performance):
        """
        Met à jour le niveau de difficulté basé sur les performances récentes.

        Args:
            performance (float): Score de performance de l'épisode

        Returns:
            bool: True si la difficulté a été augmentée
        """
        # Ajouter la performance à l'historique
        self.recent_performances.append(performance)

        # Limiter l'historique à la fenêtre d'évaluation
        if len(self.recent_performances) > self.evaluation_window:
            self.recent_performances.pop(0)

        # Si nous avons assez de données pour évaluer
        if len(self.recent_performances) >= self.evaluation_window:
            avg_performance = np.mean(self.recent_performances)

            # Si la performance moyenne dépasse le seuil et on n'a pas atteint la difficulté max
            if (
                avg_performance >= self.success_threshold
                and self.difficulty < self.max_difficulty
            ):
                # Augmenter la difficulté
                old_difficulty = self.difficulty
                self.difficulty = min(
                    self.difficulty + self.difficulty_increment, self.max_difficulty
                )
                # Arrondir la difficulté à 6 décimales pour éviter les erreurs de précision
                self.difficulty = round(self.difficulty, 6)

                # Réinitialiser l'historique des performances
                self.recent_performances = []

                logger.info(
                    f"Difficulté augmentée: {old_difficulty:.2f} -> {self.difficulty:.2f}"
                )
                return True

        return False

    def reset(self):
        """
        Réinitialise le système de curriculum learning.
        """
        self.recent_performances = []
        logger.info("Système de curriculum learning réinitialisé")


class GRUCurriculumTrainer:
    """
    Entraîneur pour les agents SAC avec GRU utilisant le curriculum learning.
    """

    def __init__(
        self,
        curriculum,
        data,
        episodes_per_level=20,
        max_episodes=200,
        eval_frequency=5,
        save_path="ai_trading/info_retour/models/rl/gru_sac",
    ):
        """
        Initialise l'entraîneur pour le curriculum learning avec GRU.

        Args:
            curriculum (GRUCurriculumLearning): Système de curriculum learning pour GRU
            data (DataFrame): Données de marché complètes
            episodes_per_level (int): Nombre d'épisodes minimum par niveau de difficulté
            max_episodes (int): Nombre maximum d'épisodes d'entraînement
            eval_frequency (int): Fréquence d'évaluation (en épisodes)
            save_path (str): Chemin pour sauvegarder les modèles
        """
        self.curriculum = curriculum
        self.data = data
        self.episodes_per_level = episodes_per_level
        self.max_episodes = max_episodes
        self.eval_frequency = eval_frequency
        self.save_path = save_path

        # Créer le dossier de sauvegarde s'il n'existe pas
        import os

        os.makedirs(save_path, exist_ok=True)

        logger.info(
            f"GRU Curriculum Trainer initialisé: max_episodes={max_episodes}, "
            f"episodes_per_level={episodes_per_level}"
        )

    def train(self, window_size=20, initial_balance=10000.0, transaction_fee=0.001):
        """
        Entraîne l'agent en utilisant le curriculum learning avec GRU.

        Args:
            window_size (int): Taille de la fenêtre d'observation
            initial_balance (float): Solde initial
            transaction_fee (float): Frais de transaction

        Returns:
            SACAgent: Agent entraîné
            dict: Historique d'entraînement
        """
        # Historique d'entraînement
        history = {
            "rewards": [],
            "difficulties": [],
            "profits": [],
            "transactions": [],
            "portfolio_values": [],
        }

        # Créer l'environnement initial
        env = self.curriculum.create_environment(
            self.data,
            window_size=window_size,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            action_type="continuous",
        )

        # Créer l'agent
        agent = self.curriculum.create_agent(env)

        # Variables de suivi
        total_episodes = 0
        episodes_at_current_level = 0
        best_eval_reward = -float("inf")

        logger.info(
            "Début de l'entraînement avec curriculum learning pour SAC avec GRU"
        )

        # Boucle d'entraînement
        while total_episodes < self.max_episodes:
            # Incrémenter les compteurs
            total_episodes += 1
            episodes_at_current_level += 1

            # Réinitialiser l'environnement
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False

            # Séquence d'états pour GRU
            state_sequence = [state] * agent.sequence_length

            while not (done or truncated):
                # Sélectionner une action basée sur la séquence d'états
                action = agent.act(np.array(state_sequence, dtype=np.float32))

                # Exécuter l'action dans l'environnement
                next_state, reward, done, truncated, info = env.step(action)

                # Mettre à jour la séquence d'états
                state_sequence.pop(0)
                state_sequence.append(next_state)

                # Stocker l'expérience dans le tampon de replay
                agent.remember(state, action, reward, next_state, done)

                # Entraîner l'agent si assez d'expériences
                if len(agent.sequence_buffer) > agent.batch_size * 3:
                    train_metrics = agent.train()

                # Mettre à jour l'état et accumuler la récompense
                state = next_state
                episode_reward += reward

            # Enregistrer les métriques de l'épisode
            history["rewards"].append(episode_reward)
            history["difficulties"].append(self.curriculum.difficulty)
            portfolio_value = env.portfolio_value()
            history["portfolio_values"].append(portfolio_value)
            history["profits"].append(portfolio_value - initial_balance)
            history["transactions"].append(env.transaction_count)

            logger.info(
                f"Épisode {total_episodes}, Niveau {self.curriculum.difficulty:.2f}, "
                f"Récompense: {episode_reward:.2f}, Profit: {portfolio_value - initial_balance:.2f}"
            )

            # Évaluation périodique
            if total_episodes % self.eval_frequency == 0:
                eval_reward = self._evaluate_agent(agent, env)

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    # Sauvegarder le meilleur modèle
                    agent.save(f"{self.save_path}/best_model")
                    logger.info(
                        f"Nouveau meilleur modèle sauvegardé avec récompense {best_eval_reward:.2f}"
                    )

                # Mettre à jour la difficulté si nécessaire
                if (
                    self.curriculum.update_difficulty(eval_reward)
                    or episodes_at_current_level >= self.episodes_per_level
                ):
                    # Recréer l'environnement avec la nouvelle difficulté
                    env = self.curriculum.create_environment(
                        self.data,
                        window_size=window_size,
                        initial_balance=initial_balance,
                        transaction_fee=transaction_fee,
                        action_type="continuous",
                    )
                    episodes_at_current_level = 0

                    # Sauvegarder un point de contrôle
                    agent.save(
                        f"{self.save_path}/checkpoint_difficulty_{self.curriculum.difficulty:.2f}"
                    )
                    logger.info(
                        f"Point de contrôle sauvegardé pour la difficulté {self.curriculum.difficulty:.2f}"
                    )

        # Entraînement terminé, sauvegarder le modèle final
        agent.save(f"{self.save_path}/final_model")
        logger.info(
            f"Entraînement terminé après {total_episodes} épisodes. Modèle final sauvegardé."
        )

        return agent, history

    def _evaluate_agent(self, agent, env, n_episodes=3):
        """
        Évalue l'agent sur plusieurs épisodes sans exploration.

        Args:
            agent: Agent à évaluer
            env: Environnement de trading
            n_episodes (int): Nombre d'épisodes d'évaluation

        Returns:
            float: Récompense moyenne sur les épisodes
        """
        total_rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False

            # Initialiser la séquence d'états pour GRU
            state_sequence = [state] * agent.sequence_length

            while not (done or truncated):
                # Sélectionner une action déterministe (sans exploration)
                action = agent.act(
                    np.array(state_sequence, dtype=np.float32), deterministic=True
                )

                # Exécuter l'action
                next_state, reward, done, truncated, _ = env.step(action)

                # Mettre à jour la séquence d'états
                state_sequence.pop(0)
                state_sequence.append(next_state)

                # Mettre à jour l'état et accumuler la récompense
                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        logger.info(
            f"Évaluation: récompense moyenne sur {n_episodes} épisodes: {avg_reward:.2f}"
        )
        return avg_reward
