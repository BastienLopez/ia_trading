import numpy as np
import pandas as pd
import logging

# Configuration du logger
logger = logging.getLogger("AdvancedRewards")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AdvancedRewardCalculator:
    """
    Calculateur de récompenses avancées pour l'apprentissage par renforcement en trading.
    Implémente plusieurs types de récompenses:
    - Ratio de Sharpe
    - Pénalisation des transactions fréquentes
    - Récompense basée sur le drawdown
    """
    
    def __init__(self, 
                 risk_free_rate=0.0,
                 window_size=30,
                 transaction_cost=0.001,
                 frequency_penalty=0.0005,
                 max_drawdown_penalty=2.0):
        """
        Initialise le calculateur de récompenses avancées.
        
        Args:
            risk_free_rate (float): Taux sans risque pour le calcul du ratio de Sharpe (annualisé)
            window_size (int): Taille de la fenêtre pour les calculs (Sharpe, drawdown)
            transaction_cost (float): Coût de transaction fixe (% du montant de la transaction)
            frequency_penalty (float): Pénalité pour les transactions fréquentes
            max_drawdown_penalty (float): Facteur multiplicateur de pénalité pour le drawdown
        """
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.frequency_penalty = frequency_penalty
        self.max_drawdown_penalty = max_drawdown_penalty
        
        # Historique des valeurs de portefeuille pour le calcul du ratio de Sharpe
        self.portfolio_values = []
        # Historique des actions prises pour pénaliser les transactions fréquentes
        self.action_history = []
        
        logger.info(f"Calculateur de récompenses avancées initialisé: "
                   f"taux sans risque={risk_free_rate:.4f}, "
                   f"fenêtre={window_size}, "
                   f"coût transaction={transaction_cost:.4f}, "
                   f"pénalité fréquence={frequency_penalty:.4f}, "
                   f"pénalité drawdown={max_drawdown_penalty:.2f}")
    
    def calculate_sharpe_ratio(self, portfolio_value):
        """
        Calcule une récompense basée sur le ratio de Sharpe pour la période actuelle.
        
        Args:
            portfolio_value (float): Valeur actuelle du portefeuille
            
        Returns:
            float: Récompense basée sur le ratio de Sharpe
        """
        # Ajouter la valeur de portefeuille à l'historique
        self.portfolio_values.append(portfolio_value)
        
        # Si nous n'avons pas assez de données pour calculer le ratio de Sharpe, retourner 0
        if len(self.portfolio_values) < 2:
            return 0
        
        # Limiter la taille de l'historique à la taille de la fenêtre
        if len(self.portfolio_values) > self.window_size:
            self.portfolio_values = self.portfolio_values[-self.window_size:]
        
        # Calculer les rendements quotidiens
        portfolio_array = np.array(self.portfolio_values)
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        
        # Si la variance est nulle, retourner 0 (éviter division par zéro)
        if np.std(returns) == 0:
            return 0
        
        # Convertir le taux sans risque annuel en taux quotidien
        daily_risk_free = (1 + self.risk_free_rate) ** (1/252) - 1
        
        # Calculer le ratio de Sharpe quotidien non annualisé
        excess_returns = returns - daily_risk_free
        sharpe = np.mean(excess_returns) / np.std(returns)
        
        # Mise à l'échelle pour avoir une récompense appropriée
        # Multiplier par 100 pour avoir des valeurs plus grandes que 1 pour un bon Sharpe
        reward = sharpe * 100
        
        logger.debug(f"Sharpe ratio: {sharpe:.4f}, récompense: {reward:.4f}")
        return reward
    
    def calculate_transaction_cost_penalty(self, action, position_size):
        """
        Calcule la pénalité due aux coûts de transaction.
        
        Args:
            action (int): Action prise (0=hold, 1=buy, 2=sell)
            position_size (float): Taille de la position (en valeur monétaire)
            
        Returns:
            float: Pénalité due aux coûts de transaction
        """
        # Ajouter l'action à l'historique
        self.action_history.append(action)
        
        # Limiter la taille de l'historique
        if len(self.action_history) > self.window_size:
            self.action_history = self.action_history[-self.window_size:]
        
        # Si l'action est "hold" (0), pas de coût de transaction
        if action == 0:
            return 0
        
        # Calculer le coût de transaction fixe
        transaction_cost = position_size * self.transaction_cost
        
        # Calculer la pénalité pour les transactions fréquentes
        if len(self.action_history) > 1:
            # Compter le nombre de changements d'action dans la fenêtre
            action_changes = sum(1 for i in range(1, len(self.action_history)) 
                               if self.action_history[i] != self.action_history[i-1] 
                               and self.action_history[i] != 0)
            
            # Pénalité exponentielle si transactions trop fréquentes
            frequency_penalty = action_changes * self.frequency_penalty * position_size
        else:
            frequency_penalty = 0
        
        total_penalty = transaction_cost + frequency_penalty
        logger.debug(f"Coût transaction: {transaction_cost:.4f}, "
                    f"Pénalité fréquence: {frequency_penalty:.4f}, "
                    f"Total: {total_penalty:.4f}")
        
        return total_penalty
    
    def calculate_drawdown_penalty(self, portfolio_values=None):
        """
        Calcule une pénalité basée sur le drawdown maximum.
        Plus le drawdown est important, plus la pénalité est élevée.
        
        Args:
            portfolio_values (list, optional): Liste des valeurs de portefeuille.
                                              Si None, utilise l'historique interne.
            
        Returns:
            float: Pénalité basée sur le drawdown
        """
        if portfolio_values is None:
            portfolio_values = self.portfolio_values
        
        # Si nous n'avons pas assez de données, retourner 0
        if len(portfolio_values) < 2:
            return 0
        
        # Calculer le drawdown
        portfolio_array = np.array(portfolio_values)
        max_so_far = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - max_so_far) / max_so_far
        
        # Calculer le drawdown maximum (en valeur absolue)
        max_drawdown = abs(np.min(drawdown))
        
        # Calculer la pénalité: plus le drawdown est grand, plus la pénalité est élevée
        # La pénalité est proportionnelle au carré du drawdown maximum
        penalty = max_drawdown ** 2 * self.max_drawdown_penalty
        
        logger.debug(f"Drawdown maximum: {max_drawdown:.4f}, pénalité: {penalty:.4f}")
        return penalty
    
    def calculate_combined_reward(self, portfolio_value, action, position_size):
        """
        Calcule une récompense combinée intégrant tous les mécanismes.
        
        Args:
            portfolio_value (float): Valeur actuelle du portefeuille
            action (int): Action prise (0=hold, 1=buy, 2=sell)
            position_size (float): Taille de la position (en valeur monétaire)
            
        Returns:
            float: Récompense combinée
        """
        # Calculer les différentes composantes de la récompense
        sharpe_reward = self.calculate_sharpe_ratio(portfolio_value)
        transaction_penalty = self.calculate_transaction_cost_penalty(action, position_size)
        drawdown_penalty = self.calculate_drawdown_penalty()
        
        # Combiner les récompenses (sharpe positif, pénalités négatives)
        combined_reward = sharpe_reward - transaction_penalty - drawdown_penalty
        
        logger.debug(f"Récompense Sharpe: {sharpe_reward:.4f}, "
                    f"Pénalité transaction: {transaction_penalty:.4f}, "
                    f"Pénalité drawdown: {drawdown_penalty:.4f}, "
                    f"Récompense totale: {combined_reward:.4f}")
        
        return combined_reward
    
    def reset(self):
        """
        Réinitialise l'état du calculateur de récompenses.
        À appeler au début de chaque épisode.
        """
        self.portfolio_values = []
        self.action_history = []
        logger.debug("Réinitialisé le calculateur de récompenses")


class SharpeRatioReward:
    """
    Classe simplifiée pour le calcul de récompense basée uniquement sur le ratio de Sharpe.
    """
    
    def __init__(self, risk_free_rate=0.0, window_size=30, annualize=False):
        """
        Initialise le calculateur de récompense basée sur le ratio de Sharpe.
        
        Args:
            risk_free_rate (float): Taux sans risque annualisé
            window_size (int): Taille de la fenêtre pour le calcul
            annualize (bool): Si True, annualise le ratio de Sharpe
        """
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.annualize = annualize
        self.returns = []
        
    def calculate(self, current_return):
        """
        Calcule la récompense basée sur le ratio de Sharpe.
        
        Args:
            current_return (float): Rendement actuel
            
        Returns:
            float: Récompense basée sur le ratio de Sharpe
        """
        # Ajouter le rendement à l'historique
        self.returns.append(current_return)
        
        # Limiter la taille de l'historique
        if len(self.returns) > self.window_size:
            self.returns = self.returns[-self.window_size:]
        
        # Si nous n'avons pas assez de données, retourner le rendement actuel
        if len(self.returns) < 2:
            return current_return
        
        # Calculer le ratio de Sharpe
        returns_array = np.array(self.returns)
        excess_returns = returns_array - (self.risk_free_rate / 252)  # Taux quotidien
        sharpe = np.mean(excess_returns) / np.std(returns_array)
        
        # Annualiser si nécessaire
        if self.annualize:
            sharpe = sharpe * np.sqrt(252)  # Environ 252 jours de trading par an
        
        # Mettre à l'échelle pour une récompense significative
        reward = sharpe * 10
        
        return reward
    
    def reset(self):
        """
        Réinitialise l'historique des rendements.
        """
        self.returns = []


class TransactionCostReward:
    """
    Classe simplifiée pour le calcul de récompense intégrant les coûts de transaction.
    """
    
    def __init__(self, base_cost=0.001, frequency_penalty=0.0005, window_size=20):
        """
        Initialise le calculateur de récompense avec coûts de transaction.
        
        Args:
            base_cost (float): Coût de base par transaction (%)
            frequency_penalty (float): Pénalité pour transactions fréquentes
            window_size (int): Taille de la fenêtre pour le calcul de fréquence
        """
        self.base_cost = base_cost
        self.frequency_penalty = frequency_penalty
        self.window_size = window_size
        self.actions = []
        
    def calculate(self, pnl, action, position_value=1.0):
        """
        Calcule la récompense intégrant les coûts de transaction.
        
        Args:
            pnl (float): Profit and Loss actuel
            action (int): Action prise (0=hold, 1=buy, 2=sell)
            position_value (float): Valeur de la position
            
        Returns:
            float: Récompense nette après coûts de transaction
        """
        # Ajouter l'action à l'historique
        self.actions.append(action)
        
        # Limiter la taille de l'historique
        if len(self.actions) > self.window_size:
            self.actions = self.actions[-self.window_size:]
        
        # Pas de coût si on ne fait rien (hold)
        if action == 0:
            return pnl
        
        # Coût de base
        cost = position_value * self.base_cost
        
        # Pénalité pour transactions fréquentes
        if len(self.actions) > 1:
            # Nombre de changements d'action non-hold
            changes = sum(1 for i in range(1, len(self.actions)) 
                         if self.actions[i] != 0 and self.actions[i-1] != 0 
                         and self.actions[i] != self.actions[i-1])
            
            frequency_cost = changes * self.frequency_penalty * position_value
        else:
            frequency_cost = 0
        
        # Récompense nette
        net_reward = pnl - cost - frequency_cost
        
        return net_reward
    
    def reset(self):
        """
        Réinitialise l'historique des actions.
        """
        self.actions = []


class DrawdownReward:
    """
    Classe simplifiée pour le calcul de récompense basée sur le drawdown.
    """
    
    def __init__(self, penalty_factor=2.0, window_size=30):
        """
        Initialise le calculateur de récompense basée sur le drawdown.
        
        Args:
            penalty_factor (float): Facteur de pénalité pour le drawdown
            window_size (int): Taille de la fenêtre pour le calcul
        """
        self.penalty_factor = penalty_factor
        self.window_size = window_size
        self.portfolio_values = []
        
    def calculate(self, pnl, portfolio_value):
        """
        Calcule la récompense intégrant une pénalité pour le drawdown.
        
        Args:
            pnl (float): Profit and Loss actuel
            portfolio_value (float): Valeur actuelle du portefeuille
            
        Returns:
            float: Récompense ajustée avec pénalité de drawdown
        """
        # Ajouter la valeur à l'historique
        self.portfolio_values.append(portfolio_value)
        
        # Limiter la taille de l'historique
        if len(self.portfolio_values) > self.window_size:
            self.portfolio_values = self.portfolio_values[-self.window_size:]
        
        # Si nous n'avons pas assez de données, retourner le PnL
        if len(self.portfolio_values) < 2:
            return pnl
        
        # Calculer le drawdown
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Pénalité proportionnelle au carré du drawdown
        penalty = max_drawdown ** 2 * self.penalty_factor
        
        # Récompense ajustée
        adjusted_reward = pnl - penalty
        
        return adjusted_reward
    
    def reset(self):
        """
        Réinitialise l'historique des valeurs de portefeuille.
        """
        self.portfolio_values = [] 