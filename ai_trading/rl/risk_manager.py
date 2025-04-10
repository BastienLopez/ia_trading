import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Gestionnaire de risque pour limiter les pertes et protéger les gains.
    """
    
    def __init__(self, max_drawdown=0.15, take_profit=0.3, stop_loss=0.1, trailing_stop=0.05):
        """
        Initialise le gestionnaire de risque.
        
        Args:
            max_drawdown (float): Drawdown maximum autorisé (ex: 0.15 = 15%)
            take_profit (float): Seuil de prise de profit (ex: 0.3 = 30%)
            stop_loss (float): Seuil de stop loss (ex: 0.1 = 10%)
            trailing_stop (float): Stop suiveur (ex: 0.05 = 5%)
        """
        self.max_drawdown = max_drawdown
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.highest_value = 0
        self.trailing_threshold = 0
        
        logger.info(f"Gestionnaire de risque initialisé avec max_drawdown={max_drawdown}, "
                   f"take_profit={take_profit}, stop_loss={stop_loss}, trailing_stop={trailing_stop}")
    
    def should_limit_position(self, portfolio_history, current_position=0):
        """
        Détermine si une position doit être limitée en fonction de l'historique du portefeuille.
        
        Args:
            portfolio_history (list): Historique des valeurs du portefeuille
            current_position (float): Position actuelle (quantité détenue)
            
        Returns:
            bool: True si la position doit être limitée
        """
        if len(portfolio_history) < 2:
            return False
        
        current_value = portfolio_history[-1]
        initial_value = portfolio_history[0]
        
        # Mettre à jour la valeur la plus élevée
        if current_value > self.highest_value:
            self.highest_value = current_value
            self.trailing_threshold = current_value * (1 - self.trailing_stop)
        
        # Calculer le drawdown actuel
        drawdown = (self.highest_value - current_value) / self.highest_value if self.highest_value > 0 else 0
        
        # Vérifier les conditions de gestion du risque
        if drawdown >= self.max_drawdown:
            logger.warning(f"Drawdown maximum atteint: {drawdown:.2%}")
            return True
        
        if current_position > 0:
            # Vérifier le stop loss
            position_change = (current_value - initial_value) / initial_value
            if position_change <= -self.stop_loss:
                logger.warning(f"Stop loss déclenché: {position_change:.2%}")
                return True
            
            # Vérifier le trailing stop
            if current_value < self.trailing_threshold:
                logger.warning(f"Trailing stop déclenché: {current_value:.2f} < {self.trailing_threshold:.2f}")
                return True
            
            # Vérifier le take profit
            if position_change >= self.take_profit:
                logger.info(f"Take profit atteint: {position_change:.2%}")
                return True
        
        return False
    
    def adjust_action(self, action, current_position):
        """
        Ajuste l'action en fonction des contraintes de risque.
        
        Args:
            action (int/float): Action originale
            current_position (float): Position actuelle
            
        Returns:
            int/float: Action ajustée
        """
        # Si nous avons une position et que nous devons la limiter, vendre
        if current_position > 0:
            if isinstance(action, int):  # Action discrète
                return 2  # Vendre (supposant que 2 est l'action de vente)
            else:  # Action continue
                return -1.0  # Vendre tout
        
        # Si nous n'avons pas de position et que le risque est élevé, ne pas acheter
        if current_position == 0:
            if isinstance(action, int):  # Action discrète
                return 0  # Hold
            else:  # Action continue
                return 0.0  # Ne rien faire
        
        return action 