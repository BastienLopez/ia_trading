class AdaptiveExploration:
    """Stratégie d'exploration adaptative pour l'agent RL."""
    
    def __init__(self, initial_epsilon=0.1, min_epsilon=0.01, decay=0.995):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.action_counts = {}  # Pour UCB
        self.total_steps = 0
        
    def should_explore(self, state, market_volatility=None):
        """
        Détermine si l'agent doit explorer en fonction des conditions actuelles.
        
        Args:
            state: État actuel de l'environnement
            market_volatility: Mesure de la volatilité du marché (optionnel)
            
        Returns:
            bool: True si l'agent doit explorer, False sinon
        """
        self.total_steps += 1
        
        # Adapter epsilon selon la volatilité du marché
        if market_volatility is not None:
            # Augmenter l'exploration dans les marchés plus volatils
            adjusted_epsilon = self.epsilon * (1 + market_volatility)
        else:
            adjusted_epsilon = self.epsilon
            
        # Décider d'explorer ou d'exploiter
        should_explore = np.random.random() < adjusted_epsilon
        
        # Décrémenter epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        
        return should_explore
    
    def get_ucb_action(self, state_str, q_values, c=2.0):
        """
        Sélectionne une action selon la stratégie Upper Confidence Bound.
        
        Args:
            state_str: Représentation de l'état sous forme de chaîne
            q_values: Valeurs Q pour chaque action
            c: Paramètre d'exploration UCB
            
        Returns:
            int: Action sélectionnée
        """
        if state_str not in self.action_counts:
            self.action_counts[state_str] = np.zeros(len(q_values))
            
        # Calculer les scores UCB
        exploration_term = np.sqrt(np.log(self.total_steps + 1) / (self.action_counts[state_str] + 1e-6))
        ucb_scores = q_values + c * exploration_term
        
        # Sélectionner l'action avec le score UCB le plus élevé
        action = np.argmax(ucb_scores)
        
        # Mettre à jour le compteur d'actions
        self.action_counts[state_str][action] += 1
        
        return action 