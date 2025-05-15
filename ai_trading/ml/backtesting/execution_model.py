"""
Module de modélisation d'exécution d'ordres pour backtesting.

Ce module fournit des classes pour simuler l'exécution réaliste d'ordres
en tenant compte de facteurs tels que la liquidité du marché, le slippage,
et les délais d'exécution.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Types d'ordres supportés par le modèle d'exécution."""
    MARKET = "MARKET"         # Ordre au marché
    LIMIT = "LIMIT"           # Ordre à cours limité
    STOP = "STOP"             # Ordre stop
    STOP_LIMIT = "STOP_LIMIT" # Ordre stop à cours limité

class OrderSide(Enum):
    """Côté de l'ordre (achat ou vente)."""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Statuts possibles d'un ordre."""
    PENDING = "PENDING"       # Ordre en attente
    PARTIAL = "PARTIAL"       # Partiellement exécuté
    FILLED = "FILLED"         # Totalement exécuté
    CANCELLED = "CANCELLED"   # Annulé
    REJECTED = "REJECTED"     # Rejeté

@dataclass
class Order:
    """Classe représentant un ordre de trading."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None      # Prix pour les ordres limit
    stop_price: Optional[float] = None # Prix pour les ordres stop
    timestamp: pd.Timestamp = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        """Initialisation post-création."""
        if self.metadata is None:
            self.metadata = {}
        
        # Générer un timestamp si non fourni
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()
            
    def is_filled(self) -> bool:
        """Vérifie si l'ordre est entièrement exécuté."""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """Vérifie si l'ordre est actif (en attente ou partiellement exécuté)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]
    
    def cancel(self) -> bool:
        """Annule l'ordre s'il est encore actif."""
        if self.is_active():
            self.status = OrderStatus.CANCELLED
            return True
        return False
    
    def update_fill(self, fill_qty: float, fill_price: float, fees: float = 0.0):
        """
        Met à jour l'ordre avec une exécution partielle ou complète.
        
        Args:
            fill_qty: Quantité exécutée
            fill_price: Prix d'exécution
            fees: Frais associés à cette exécution
        """
        if self.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FILLED]:
            logger.warning(f"Tentative de mise à jour d'un ordre {self.status.value}: {self.id}")
            return False
        
        # Vérifier que la quantité n'excède pas la quantité restante
        remaining_qty = self.quantity - self.filled_quantity
        if fill_qty > remaining_qty:
            fill_qty = remaining_qty
            
        # Mettre à jour la quantité exécutée et le prix moyen
        if self.filled_quantity > 0:
            # Calculer le nouveau prix moyen pondéré
            self.avg_fill_price = (self.avg_fill_price * self.filled_quantity + 
                                  fill_price * fill_qty) / (self.filled_quantity + fill_qty)
        else:
            self.avg_fill_price = fill_price
            
        self.filled_quantity += fill_qty
        self.fees += fees
        
        # Mettre à jour le statut
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_quantity = self.quantity  # S'assurer de ne pas dépasser
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
            
        return True


class LiquidityProfile:
    """
    Modèle de profil de liquidité pour estimer le slippage et les coûts d'impact marché.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le profil de liquidité.
        
        Args:
            config: Configuration pour le profil de liquidité
        """
        self.config = config or {}
        
        # Paramètres par défaut
        self.default_params = {
            "base_spread": 0.0005,  # 0.05% (5 points de base)
            "volatility_factor": 1.5,  # Multiplicateur pour la volatilité
            "volume_profile": {
                "low_liquidity_threshold": 0.2,  # 20% du volume quotidien moyen
                "high_liquidity_threshold": 1.0,  # 100% du volume quotidien moyen
            },
            "market_impact": {
                "fixed": 0.0001,  # Impact fixe de 1 point de base
                "variable": 0.15,  # Coefficient d'impact variable basé sur la taille
            }
        }
        
        # Fusionner avec la configuration fournie
        for key, default_value in self.default_params.items():
            if key in self.config:
                if isinstance(default_value, dict) and isinstance(self.config[key], dict):
                    for sub_key, sub_value in default_value.items():
                        if sub_key not in self.config[key]:
                            self.config[key][sub_key] = sub_value
                else:
                    continue  # Conserver la valeur fournie
            else:
                self.config[key] = default_value
                
    def calculate_slippage(self, 
                          order: Order, 
                          market_data: pd.DataFrame, 
                          liquidity_data: Dict = None) -> float:
        """
        Calcule le slippage pour un ordre donné en fonction des conditions du marché.
        
        Args:
            order: L'ordre à exécuter
            market_data: Données de marché (OHLCV)
            liquidity_data: Données supplémentaires sur la liquidité (optionnel)
            
        Returns:
            Slippage estimé en pourcentage du prix
        """
        # Données par défaut si non fournies
        if liquidity_data is None:
            liquidity_data = {}
            
        # Extraire les données nécessaires
        current_price = market_data['close'].iloc[-1]
        recent_volatility = self._calculate_volatility(market_data)
        recent_volume = market_data['volume'].iloc[-20:].mean()
        
        # Récupérer le volume moyen quotidien s'il est disponible
        avg_daily_volume = liquidity_data.get('avg_daily_volume', recent_volume * 24)
        
        # Calculer la liquidité relative
        order_volume_ratio = (order.quantity * current_price) / avg_daily_volume
        
        # Calculer le spread de base ajusté à la volatilité
        base_spread = self.config["base_spread"] * (1 + recent_volatility * self.config["volatility_factor"])
        
        # Calculer l'impact de marché
        fixed_impact = self.config["market_impact"]["fixed"]
        variable_impact = self.config["market_impact"]["variable"] * order_volume_ratio
        
        # Calculer le slippage final
        slippage = base_spread + fixed_impact + variable_impact
        
        # Ajuster le slippage en fonction de la liquidité relative
        if order_volume_ratio < self.config["volume_profile"]["low_liquidity_threshold"]:
            # Bonne liquidité
            liquidity_factor = 1.0
        elif order_volume_ratio > self.config["volume_profile"]["high_liquidity_threshold"]:
            # Faible liquidité
            liquidity_factor = 2.5
        else:
            # Liquidité moyenne, interpolation linéaire
            range_ratio = (order_volume_ratio - self.config["volume_profile"]["low_liquidity_threshold"]) / \
                        (self.config["volume_profile"]["high_liquidity_threshold"] - 
                         self.config["volume_profile"]["low_liquidity_threshold"])
            liquidity_factor = 1.0 + 1.5 * range_ratio
            
        slippage *= liquidity_factor
        
        # Le slippage est toujours positif pour les ordres au marché
        # Pour les achats, le prix est plus élevé; pour les ventes, le prix est plus bas
        return slippage
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """
        Calcule la volatilité récente du marché.
        
        Args:
            market_data: Données de marché (OHLCV)
            
        Returns:
            Volatilité estimée
        """
        # Utiliser par exemple l'écart-type des rendements sur les 20 dernières périodes
        returns = market_data['close'].pct_change().dropna()
        if len(returns) < 20:
            return 0.01  # Valeur par défaut si pas assez de données
            
        return returns.iloc[-20:].std()


class ExecutionModel:
    """
    Modèle d'exécution des ordres qui simule les conditions réelles du marché.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le modèle d'exécution.
        
        Args:
            config: Configuration pour le modèle
        """
        self.config = config or {}
        
        # Paramètres par défaut
        self.default_params = {
            "latency": {
                "mean": 0.5,  # Latence moyenne en secondes
                "std": 0.2,   # Écart-type de la latence
                "min": 0.1,   # Latence minimale
                "max": 2.0    # Latence maximale
            },
            "partial_fills": {
                "enabled": True,  # Activer les exécutions partielles
                "probability": 0.3,  # Probabilité d'une exécution partielle
                "min_ratio": 0.1,  # Ratio minimal d'exécution (10%)
                "max_ratio": 0.9   # Ratio maximal d'exécution (90%)
            },
            "limit_order": {
                "fill_probability_at_price": 0.7,  # Probabilité d'exécution à prix limité
                "better_price_probability": 0.2    # Probabilité d'exécution à meilleur prix
            }
        }
        
        # Fusionner avec la configuration fournie
        for key, default_value in self.default_params.items():
            if key in self.config:
                if isinstance(default_value, dict) and isinstance(self.config[key], dict):
                    for sub_key, sub_value in default_value.items():
                        if sub_key not in self.config[key]:
                            self.config[key][sub_key] = sub_value
                else:
                    continue  # Conserver la valeur fournie
            else:
                self.config[key] = default_value
                
        # Initialiser le profil de liquidité
        self.liquidity_profile = LiquidityProfile(self.config.get("liquidity", None))
        
    def execute_order(self, 
                     order: Order, 
                     market_data: pd.DataFrame, 
                     liquidity_data: Dict = None) -> Order:
        """
        Exécute un ordre en simulant les conditions du marché.
        
        Args:
            order: L'ordre à exécuter
            market_data: Données de marché pour la simulation
            liquidity_data: Données supplémentaires sur la liquidité
            
        Returns:
            L'ordre mis à jour après exécution
        """
        if market_data.empty:
            logger.warning(f"Données de marché vides pour {order.symbol}. Ordre rejeté.")
            order.status = OrderStatus.REJECTED
            return order
            
        if not order.is_active():
            logger.warning(f"Ordre {order.id} n'est pas actif: {order.status.value}")
            return order
            
        # Simuler la latence si activée
        if self.config["latency"]["enabled"]:
            latency = self._simulate_latency()
            # En backtesting, on peut juste noter la latence sans vraiment attendre
            order.metadata["latency"] = latency
            
        # Exécuter selon le type d'ordre
        if order.type == OrderType.MARKET:
            self._execute_market_order(order, market_data, liquidity_data)
        elif order.type == OrderType.LIMIT:
            self._execute_limit_order(order, market_data, liquidity_data)
        elif order.type == OrderType.STOP:
            self._execute_stop_order(order, market_data, liquidity_data)
        elif order.type == OrderType.STOP_LIMIT:
            self._execute_stop_limit_order(order, market_data, liquidity_data)
        else:
            logger.warning(f"Type d'ordre non supporté: {order.type.value}")
            order.status = OrderStatus.REJECTED
            
        return order
        
    def _simulate_latency(self) -> float:
        """
        Simule la latence d'exécution.
        
        Returns:
            Latence simulée en secondes
        """
        params = self.config["latency"]
        latency = np.random.normal(params["mean"], params["std"])
        
        # Borner la latence
        latency = max(params["min"], min(params["max"], latency))
        return latency
        
    def _execute_market_order(self, 
                             order: Order, 
                             market_data: pd.DataFrame, 
                             liquidity_data: Dict = None) -> Order:
        """
        Exécute un ordre au marché.
        
        Args:
            order: L'ordre à exécuter
            market_data: Données de marché
            liquidity_data: Données de liquidité
            
        Returns:
            L'ordre après exécution
        """
        current_price = market_data['close'].iloc[-1]
        
        # Calculer le slippage
        slippage_pct = self.liquidity_profile.calculate_slippage(order, market_data, liquidity_data)
        
        # Appliquer le slippage au prix
        if order.side == OrderSide.BUY:
            # Pour un achat, le prix est plus élevé (slippage positif)
            execution_price = current_price * (1 + slippage_pct)
        else:
            # Pour une vente, le prix est plus bas (slippage négatif)
            execution_price = current_price * (1 - slippage_pct)
            
        # Déterminer si l'ordre est exécuté partiellement
        if self.config["partial_fills"]["enabled"] and np.random.random() < self.config["partial_fills"]["probability"]:
            # Exécution partielle
            params = self.config["partial_fills"]
            partial_ratio = np.random.uniform(params["min_ratio"], params["max_ratio"])
            fill_quantity = order.quantity * partial_ratio
        else:
            # Exécution complète
            fill_quantity = order.quantity
            
        # Calculer les frais (à intégrer avec le modèle de coûts de transaction)
        fees = 0.0  # À remplacer par le calcul réel des frais
        
        # Mettre à jour l'ordre
        order.update_fill(fill_quantity, execution_price, fees)
        order.slippage = slippage_pct
        
        return order
        
    def _execute_limit_order(self, 
                            order: Order, 
                            market_data: pd.DataFrame, 
                            liquidity_data: Dict = None) -> None:
        """
        Exécute un ordre à cours limité.
        
        Args:
            order: L'ordre à exécuter
            market_data: Données de marché
            liquidity_data: Données de liquidité
        """
        if order.price is None:
            logger.warning(f"Prix non spécifié pour l'ordre limite {order.id}")
            order.status = OrderStatus.REJECTED
            return
            
        current_price = market_data['close'].iloc[-1]
        
        # Vérifier si le prix limite est atteint
        is_executable = False
        if order.side == OrderSide.BUY and current_price <= order.price:
            is_executable = True
        elif order.side == OrderSide.SELL and current_price >= order.price:
            is_executable = True
            
        if not is_executable:
            # L'ordre reste en attente
            return
            
        # Déterminer si l'ordre est exécuté au prix limite ou à un meilleur prix
        params = self.config["limit_order"]
        if np.random.random() < params["better_price_probability"]:
            # Exécution à un meilleur prix
            if order.side == OrderSide.BUY:
                # Pour un achat, un meilleur prix est inférieur au prix limite
                execution_price = order.price * (1 - np.random.uniform(0, 0.001))
            else:
                # Pour une vente, un meilleur prix est supérieur au prix limite
                execution_price = order.price * (1 + np.random.uniform(0, 0.001))
        else:
            # Exécution au prix limite
            execution_price = order.price
            
        # Déterminer si l'ordre est exécuté partiellement
        if self.config["partial_fills"]["enabled"] and np.random.random() < self.config["partial_fills"]["probability"]:
            # Exécution partielle
            params = self.config["partial_fills"]
            partial_ratio = np.random.uniform(params["min_ratio"], params["max_ratio"])
            fill_quantity = order.quantity * partial_ratio
        else:
            # Exécution complète
            fill_quantity = order.quantity
            
        # Calculer les frais (à intégrer avec le modèle de coûts de transaction)
        fees = 0.0  # À remplacer par le calcul réel des frais
        
        # Mettre à jour l'ordre
        order.update_fill(fill_quantity, execution_price, fees)
        
    def _execute_stop_order(self, 
                           order: Order, 
                           market_data: pd.DataFrame, 
                           liquidity_data: Dict = None) -> None:
        """
        Exécute un ordre stop.
        
        Args:
            order: L'ordre à exécuter
            market_data: Données de marché
            liquidity_data: Données de liquidité
        """
        if order.stop_price is None:
            logger.warning(f"Prix stop non spécifié pour l'ordre stop {order.id}")
            order.status = OrderStatus.REJECTED
            return
            
        current_price = market_data['close'].iloc[-1]
        
        # Vérifier si le prix stop est atteint
        is_triggered = False
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            is_triggered = True
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            is_triggered = True
            
        if not is_triggered:
            # L'ordre reste en attente
            return
            
        # Une fois déclenché, exécuter comme un ordre au marché
        # Convertir en ordre au marché
        order.type = OrderType.MARKET
        self._execute_market_order(order, market_data, liquidity_data)
        
    def _execute_stop_limit_order(self, 
                                 order: Order, 
                                 market_data: pd.DataFrame, 
                                 liquidity_data: Dict = None) -> None:
        """
        Exécute un ordre stop à cours limité.
        
        Args:
            order: L'ordre à exécuter
            market_data: Données de marché
            liquidity_data: Données de liquidité
        """
        if order.stop_price is None or order.price is None:
            logger.warning(f"Prix stop ou limite non spécifié pour l'ordre stop-limite {order.id}")
            order.status = OrderStatus.REJECTED
            return
            
        current_price = market_data['close'].iloc[-1]
        
        # Vérifier si le prix stop est atteint
        is_triggered = False
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            is_triggered = True
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            is_triggered = True
            
        if not is_triggered:
            # L'ordre reste en attente
            return
            
        # Une fois déclenché, exécuter comme un ordre à cours limité
        # Convertir en ordre à cours limité
        order.type = OrderType.LIMIT
        self._execute_limit_order(order, market_data, liquidity_data)

# Alias pour faciliter la compatibilité avec les tests
OrderExecutionModel = ExecutionModel 