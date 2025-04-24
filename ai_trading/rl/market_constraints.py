import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MarketConstraints:
    """Gère les contraintes de marché réalistes pour l'environnement de trading."""
    
    def __init__(
        self,
        slippage_model: str = "dynamic",
        base_slippage: float = 0.001,
        execution_delay: int = 0,
        market_impact_factor: float = 0.1,
        orderbook_depth: Optional[Dict] = None
    ):
        """
        Initialise les contraintes de marché.
        
        Args:
            slippage_model (str): Type de modèle de slippage ("fixed", "dynamic", "orderbook")
            base_slippage (float): Slippage de base pour le modèle fixe
            execution_delay (int): Délai d'exécution en pas de temps
            market_impact_factor (float): Facteur d'impact marché
            orderbook_depth (Dict): Profondeur du carnet d'ordres par symbole
        """
        self.slippage_model = slippage_model
        self.base_slippage = base_slippage
        self.execution_delay = execution_delay
        self.market_impact_factor = market_impact_factor
        self.orderbook_depth = orderbook_depth or {}
        
    def calculate_slippage(
        self,
        symbol: str,
        action_value: float,
        volume: float,
        volatility: float,
        avg_volume: float
    ) -> float:
        """
        Calcule le slippage pour une transaction.
        
        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            volume (float): Volume de la transaction
            volatility (float): Volatilité actuelle
            avg_volume (float): Volume moyen sur la période
            
        Returns:
            float: Slippage calculé
        """
        # Volume nul = pas de slippage
        if volume == 0:
            return 0.0
            
        if self.slippage_model == "fixed":
            return self.base_slippage
            
        elif self.slippage_model == "dynamic":
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            slippage = self.base_slippage * (1 + volatility) * volume_ratio
            return min(slippage, 0.01)  # Limite à 1%
            
        elif self.slippage_model == "orderbook":
            if symbol in self.orderbook_depth:
                depth = self.orderbook_depth[symbol]
                volume_ratio = volume / depth['total_volume'] if depth['total_volume'] > 0 else 1
                spread_impact = depth['spread_pct'] / 100
                # Ajout d'un petit facteur pour garantir que le slippage est strictement supérieur au minimum
                return min(spread_impact * volume_ratio * 1.01, 0.01)
            return self.base_slippage
            
        return 0.0
        
    def calculate_market_impact(
        self,
        symbol: str,
        action_value: float,
        volume: float,
        price: float,
        avg_volume: float
    ) -> Tuple[float, int]:
        """
        Calcule l'impact sur le marché d'une transaction.
        
        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            volume (float): Volume de la transaction
            price (float): Prix actuel
            avg_volume (float): Volume moyen sur la période
            
        Returns:
            Tuple[float, int]: (Impact sur le prix en %, temps de récupération en pas)
        """
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
        
        # Impact immédiat (en pourcentage)
        immediate_impact = self.market_impact_factor * volume_ratio * abs(action_value)
        
        # Temps de récupération (en pas de temps)
        recovery_time = int(np.ceil(60 * (volume_ratio ** 2)))
        
        if symbol in self.orderbook_depth:
            depth = self.orderbook_depth[symbol]
            depth_ratio = volume / depth['total_volume'] if depth['total_volume'] > 0 else 1
            immediate_impact *= (1 + depth_ratio)
            
        return min(immediate_impact, 0.05), min(recovery_time, 100)  # Limite à 5% et 100 pas
        
    def calculate_execution_delay(
        self,
        symbol: str,
        action_value: float,
        volume: float,
        avg_volume: float
    ) -> int:
        """
        Calcule le délai d'exécution pour une transaction.
        
        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            volume (float): Volume de la transaction
            avg_volume (float): Volume moyen sur la période
            
        Returns:
            int: Délai d'exécution en pas de temps
        """
        # Volume nul = pas de délai
        if volume == 0 or abs(action_value) < 1e-6:
            return 0
            
        if self.execution_delay == 0:
            return 0
            
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        base_delay = self.execution_delay
        
        # Ajustement en fonction du volume
        volume_factor = np.log1p(volume_ratio)
        
        # Ajustement en fonction de la profondeur du carnet
        if symbol in self.orderbook_depth:
            depth = self.orderbook_depth[symbol]
            market_load = depth.get('volume_imbalance', 0)
            load_factor = np.exp(abs(market_load)) - 1
        else:
            load_factor = 0
            
        total_delay = int(base_delay * (1 + volume_factor) * (1 + load_factor))
        return min(total_delay, 10)  # Limite à 10 pas
        
    def update_orderbook_depth(self, symbol: str, depth_data: Dict):
        """
        Met à jour les données de profondeur du carnet d'ordres.
        
        Args:
            symbol (str): Symbole de l'actif
            depth_data (Dict): Nouvelles données de profondeur
        """
        self.orderbook_depth[symbol] = depth_data 