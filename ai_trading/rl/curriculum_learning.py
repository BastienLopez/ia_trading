import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logger = logging.getLogger("CurriculumLearning")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        env_params: Optional[Dict] = None
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
            "n_discrete_actions": 5
        }
        
        # Historique des performances
        self.performance_history = []
        self.success_streak = 0
        
        # Niveaux de difficulté et leurs paramètres
        self.difficulty_levels = self._create_difficulty_levels()
        
        logger.info(f"Système de curriculum learning initialisé avec difficulté {initial_difficulty} "
                   f"et type '{curriculum_type}'")
    
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
            reward_index = min(int(difficulty * len(reward_functions)), len(reward_functions) - 1)
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
        
        logger.info(f"Environnement créé avec difficulté {self.current_difficulty} et paramètres: {params}")
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
        returns = self.df['close'].pct_change().dropna()
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
            logger.warning(f"Filtrage par volatilité a réduit les données à {len(filtered_df)} lignes, "
                          f"ce qui est insuffisant pour la taille de fenêtre {window_size}.")
            
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
                    before_indices = self.df[self.df.index < first_date].index[-points_needed//2:]
                    after_indices = self.df[self.df.index > last_date].index[:points_needed//2]
                    
                    # Combiner tous les indices
                    all_indices = before_indices.tolist() + filtered_df.index.tolist() + after_indices.tolist()
                    filtered_df = self.df.loc[all_indices]
                    
                    logger.info(f"Ajout de points adjacents pour atteindre {len(filtered_df)} points.")
        
        # Vérification finale
        if len(filtered_df) < min_required_size:
            logger.warning(f"Impossible d'obtenir suffisamment de données. Utilisation des données originales.")
            # En dernier recours, utiliser les données originales avec une sélection aléatoire
            random_indices = np.random.choice(len(self.df), size=min(len(self.df), min_required_size), replace=False)
            filtered_df = self.df.iloc[sorted(random_indices)]
        
        logger.info(f"Données filtrées par volatilité: {len(filtered_df)} points (percentile {volatility_percentile})")
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
                    prev_date = df_reset.iloc[i-1][date_col]
                    curr_date = df_reset.iloc[i][date_col]
                    
                    # Vérifier si les dates sont consécutives (par exemple, différence d'un jour)
                    # Cela dépend de la fréquence attendue des données
                    time_diff = (curr_date - prev_date).total_seconds()
                    expected_diff = 86400  # 1 jour en secondes, ajustez selon votre fréquence
                    
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
                if isinstance(indices[i], (int, np.integer)) and isinstance(indices[i-1], (int, np.integer)):
                    if indices[i] == indices[i-1] + 1:
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
            logger.info(f"Succès {self.success_streak}/{self.patience} avec performance {agent_performance:.4f}")
        else:
            self.success_streak = 0
            logger.info(f"Performance insuffisante: {agent_performance:.4f} < {self.success_threshold}")
        
        # Si l'agent a réussi consécutivement le nombre de fois requis, augmenter la difficulté
        if self.success_streak >= self.patience:
            self.success_streak = 0
            
            # Calculer la nouvelle difficulté
            new_difficulty = min(self.current_difficulty + self.difficulty_increment, self.max_difficulty)
            
            # Si on a atteint la difficulté maximale, on reste à ce niveau
            if new_difficulty == self.current_difficulty:
                logger.info(f"Difficulté maximale déjà atteinte: {self.current_difficulty}")
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
        logger.info(f"Système de curriculum learning réinitialisé à la difficulté {self.initial_difficulty}")


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
            "performance": []
        }
        
        logger.info(f"Entraîneur avec curriculum learning initialisé: "
                   f"max_episodes={max_episodes}, episodes_per_level={episodes_per_level}")
    
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
                    difficulty_increased = self.curriculum.update_difficulty(performance)
                    
                    if difficulty_increased:
                        # Si la difficulté a augmenté, créer un nouvel environnement
                        logger.info(f"Niveau {current_level} terminé. Passage au niveau {self.curriculum.current_difficulty}")
                        current_level = self.curriculum.current_difficulty
                        env = self.curriculum.create_environment()
                        break
                    
                # Afficher les progrès
                if verbose and self.total_episodes % 10 == 0:
                    logger.info(f"Épisode {self.total_episodes}/{self.max_episodes} | "
                               f"Niveau {current_level} | "
                               f"Récompense: {episode_reward:.2f} | "
                               f"Portefeuille: {portfolio_value:.2f}")
            
            # Si on a terminé tous les épisodes du niveau sans augmenter la difficulté
            if self.level_episodes >= self.episodes_per_level:
                logger.info(f"Nombre maximum d'épisodes atteint pour le niveau {current_level}")
                
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
            if hasattr(self.agent, 'remember'):
                # Pour les agents basés sur DQN
                self.agent.remember(state, action, reward, next_state, done)
                if hasattr(self.agent, 'memory') and hasattr(self.agent, 'batch_size'):
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
        
        logger.info(f"Évaluation: ROI={roi:.4f}, Performance={normalized_performance:.4f}")
        
        return normalized_performance 