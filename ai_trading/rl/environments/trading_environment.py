    def step(self, action):
        """Exécute une action dans l'environnement."""
        # Convertir l'action en position
        position = self._action_to_position(action)
        
        # Gérer les ordres en attente
        self._process_pending_orders()
        
        # Exécuter l'action actuelle
        if position != self.current_position:
            # Créer un nouvel ordre
            order = {
                "position": position,
                "price": self.current_price,
                "delay": self.execution_delay,
                "timestamp": self.current_step
            }
            self.pending_orders.append(order)
        
        # Passer à l'étape suivante
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Obtenir l'état suivant
        next_state = self._get_state()
        
        # Calculer la récompense
        reward = self._calculate_reward()
        
        # Mettre à jour la position actuelle
        self.current_position = position
        
        return next_state, reward, done, False, {} 

    def _apply_slippage(self, price, action_value):
        """Applique le slippage au prix en fonction du modèle choisi."""
        if self.slippage_model == "fixed":
            slippage = self.slippage_value
        elif self.slippage_model == "dynamic":
            # Calculer la volatilité moyenne sur les 20 derniers pas
            volatility = self.df["volatility"].iloc[max(0, self.current_step - 20):self.current_step].mean()
            
            # Calculer le volume moyen sur les 20 derniers pas
            volume = self.df["volume"].iloc[max(0, self.current_step - 20):self.current_step].mean()
            current_volume = self.df["volume"].iloc[self.current_step]
            
            # Calculer le slippage dynamique
            slippage = self.slippage_value * (1 + volatility) * (current_volume / volume)
        else:
            slippage = 0.0
            
        # Appliquer le slippage
        if action_value > 0:  # Achat
            return price * (1 + slippage)
        elif action_value < 0:  # Vente
            return price * (1 - slippage)
        else:
            return price 

    def _process_pending_orders(self):
        """Traite les ordres en attente."""
        # Liste des ordres à supprimer
        orders_to_remove = []
        
        # Traiter chaque ordre en attente
        for order in self.pending_orders:
            # Vérifier si le délai est écoulé
            if self.current_step - order["timestamp"] >= order["delay"]:
                # Appliquer le slippage
                execution_price = self._apply_slippage(order["price"], order["position"])
                
                # Mettre à jour le solde
                if order["position"] > 0:  # Achat
                    self.balance -= execution_price * order["position"]
                elif order["position"] < 0:  # Vente
                    self.balance += execution_price * abs(order["position"])
                
                # Mettre à jour la position actuelle
                self.current_position = order["position"]
                
                # Marquer l'ordre pour suppression
                orders_to_remove.append(order)
        
        # Supprimer les ordres traités
        for order in orders_to_remove:
            self.pending_orders.remove(order) 

    def _update_position(self, action_value):
        """Met à jour la position en fonction de l'action."""
        # Calculer la nouvelle position
        new_position = self.current_position + action_value
        
        # Vérifier les limites de position
        if new_position > self.max_position:
            new_position = self.max_position
        elif new_position < -self.max_position:
            new_position = -self.max_position
            
        # Calculer la différence de position
        position_diff = new_position - self.current_position
        
        # Si la position a changé
        if position_diff != 0:
            # Calculer le prix d'exécution avec slippage
            execution_price = self._apply_slippage(self.current_price, position_diff)
            
            # Mettre à jour le solde
            if position_diff > 0:  # Achat
                self.balance -= execution_price * position_diff
            elif position_diff < 0:  # Vente
                self.balance += execution_price * abs(position_diff)
                
            # Mettre à jour la position actuelle
            self.current_position = new_position
            
        return position_diff 

    def _calculate_reward(self, position_diff):
        """Calcule la récompense pour l'action."""
        # Récompense de base basée sur le profit
        profit = self.current_price - self.last_price
        base_reward = profit * self.current_position
        
        # Pénalité pour les changements de position
        position_change_penalty = -abs(position_diff) * self.position_change_penalty
        
        # Pénalité pour les positions non fermées
        if self.current_position != 0:
            position_penalty = -abs(self.current_position) * self.position_penalty
        else:
            position_penalty = 0
            
        # Récompense totale
        total_reward = base_reward + position_change_penalty + position_penalty
        
        # Normaliser la récompense
        if self.reward_scaling:
            total_reward = total_reward / self.initial_balance
            
        return total_reward 

    def _get_state(self):
        """Retourne l'état actuel de l'environnement."""
        # Prix et indicateurs techniques
        state = [self.current_price]
        
        if self.use_rsi:
            state.append(self.rsi[-1])
            
        if self.use_macd:
            state.extend([self.macd[-1], self.macd_signal[-1]])
            
        if self.use_bollinger:
            state.extend([self.bollinger_upper[-1], self.bollinger_lower[-1]])
            
        # Position et solde
        state.extend([self.current_position, self.balance])
        
        # Convertir en numpy array
        state = np.array(state, dtype=np.float32)
        
        # Normaliser si nécessaire
        if self.state_normalization:
            state = (state - self.state_mean) / (self.state_std + 1e-8)
            
        return state 