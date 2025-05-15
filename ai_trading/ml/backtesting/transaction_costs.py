"""
Module pour modéliser les coûts de transaction dans un environnement de trading.

Ce module permet de simuler de façon réaliste les différents coûts associés
aux transactions, incluant les frais fixes, variables, impôts et slippage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

from ai_trading.ml.backtesting.execution_model import Order, OrderSide

# Configuration du logging
logger = logging.getLogger(__name__)

class FeeType(Enum):
    """Types de frais pris en charge par le modèle."""
    FIXED = "FIXED"               # Frais fixes par transaction
    PERCENTAGE = "PERCENTAGE"     # Pourcentage de la valeur
    TIERED = "TIERED"             # Frais par paliers
    COMBINED = "COMBINED"         # Combinaison de frais fixes et variables
    CUSTOM = "CUSTOM"             # Calcul personnalisé

class TransactionCostModel:
    """
    Modèle qui calcule tous les coûts associés aux transactions.
    Inclut les frais d'exchange, taxes, slippage et spread.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le modèle de coûts de transaction.
        
        Args:
            config: Configuration des différents types de coûts
        """
        self.config = config or {}
        
        # Paramètres par défaut
        self.default_params = {
            "exchange_fees": {
                "type": FeeType.COMBINED.value,
                "maker": 0.0010,  # 0.10% pour les ordres maker (limit)
                "taker": 0.0015,  # 0.15% pour les ordres taker (market)
                "fixed": 0.0,     # Frais fixes par transaction
                "min_fee": 0.0    # Frais minimum par transaction
            },
            "regulatory_fees": {
                "enabled": False,
                "rate": 0.0001    # 0.01% de frais réglementaires
            },
            "taxes": {
                "enabled": False,
                "rate": 0.0       # Taux d'imposition
            },
            "spread": {
                "model": "fixed",  # Modèle de spread (fixed, percentage, historical)
                "value": 0.0005,   # 0.05% spread fixe ou pourcentage
            },
            "slippage": {
                "model": "volume_based",  # Modèle de slippage
                "fixed": 0.0005,          # Slippage fixe (0.05%)
                "volume_factor": 0.1,     # Facteur d'impact de volume
                "volatility_factor": 0.5  # Facteur d'impact de volatilité
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
    
    def calculate_total_cost(self, 
                           order: Order, 
                           market_data: pd.DataFrame, 
                           liquidity_data: Dict = None) -> Dict[str, float]:
        """
        Calcule le coût total d'une transaction.
        
        Args:
            order: L'ordre à évaluer
            market_data: Données de marché
            liquidity_data: Données de liquidité optionnelles
            
        Returns:
            Dictionnaire avec la ventilation des coûts par catégorie
        """
        if not order.is_filled():
            logger.warning(f"Impossible de calculer les coûts pour un ordre non exécuté: {order.id}")
            return {"total": 0.0}
            
        # Calculer la valeur de la transaction
        transaction_value = order.filled_quantity * order.avg_fill_price
        
        # Calculer les différents coûts
        exchange_fees = self._calculate_exchange_fees(order, transaction_value)
        regulatory_fees = self._calculate_regulatory_fees(transaction_value)
        tax_costs = self._calculate_taxes(transaction_value)
        spread_cost = self._calculate_spread_cost(order, market_data)
        
        # Compiler tous les coûts
        costs = {
            "exchange_fees": exchange_fees,
            "regulatory_fees": regulatory_fees,
            "taxes": tax_costs,
            "spread": spread_cost,
            "slippage": order.slippage * transaction_value if hasattr(order, "slippage") else 0.0,
            "total": exchange_fees + regulatory_fees + tax_costs + spread_cost
        }
        
        # Si le slippage est défini, l'ajouter au total
        if hasattr(order, "slippage"):
            costs["total"] += costs["slippage"]
            
        return costs
    
    def _calculate_exchange_fees(self, order: Order, transaction_value: float) -> float:
        """
        Calcule les frais d'exchange basés sur le type d'ordre et la configuration.
        
        Args:
            order: L'ordre concerné
            transaction_value: Valeur totale de la transaction
            
        Returns:
            Montant des frais d'exchange
        """
        fee_config = self.config["exchange_fees"]
        fee_type = FeeType(fee_config["type"])
        
        # Déterminer si l'ordre est maker ou taker
        is_taker = order.type.value in ["MARKET", "STOP"]
        rate = fee_config["taker"] if is_taker else fee_config["maker"]
        
        if fee_type == FeeType.FIXED:
            return fee_config["fixed"]
            
        elif fee_type == FeeType.PERCENTAGE:
            return transaction_value * rate
            
        elif fee_type == FeeType.COMBINED:
            percentage_fee = transaction_value * rate
            fixed_fee = fee_config["fixed"]
            total_fee = percentage_fee + fixed_fee
            
            # Appliquer le frais minimum si défini
            if "min_fee" in fee_config and fee_config["min_fee"] > 0:
                total_fee = max(total_fee, fee_config["min_fee"])
                
            return total_fee
            
        elif fee_type == FeeType.TIERED:
            # Modèle par paliers (à implémenter selon les besoins)
            logger.warning("Modèle de frais par paliers non implémenté")
            return transaction_value * rate
            
        elif fee_type == FeeType.CUSTOM:
            # Appliquer une fonction personnalisée si définie
            if "custom_function" in fee_config:
                try:
                    return fee_config["custom_function"](order, transaction_value)
                except Exception as e:
                    logger.error(f"Erreur dans la fonction personnalisée de calcul de frais: {e}")
                    
            return transaction_value * rate
        
        else:
            logger.warning(f"Type de frais non reconnu: {fee_type}")
            return 0.0
    
    def _calculate_regulatory_fees(self, transaction_value: float) -> float:
        """
        Calcule les frais réglementaires.
        
        Args:
            transaction_value: Valeur totale de la transaction
            
        Returns:
            Montant des frais réglementaires
        """
        reg_config = self.config["regulatory_fees"]
        
        if reg_config["enabled"]:
            return transaction_value * reg_config["rate"]
        else:
            return 0.0
    
    def _calculate_taxes(self, transaction_value: float) -> float:
        """
        Calcule les taxes applicables.
        
        Args:
            transaction_value: Valeur totale de la transaction
            
        Returns:
            Montant des taxes
        """
        tax_config = self.config["taxes"]
        
        if tax_config["enabled"]:
            return transaction_value * tax_config["rate"]
        else:
            return 0.0
    
    def _calculate_spread_cost(self, order: Order, market_data: pd.DataFrame) -> float:
        """
        Calcule le coût lié au spread bid-ask.
        
        Args:
            order: L'ordre concerné
            market_data: Données de marché
            
        Returns:
            Coût lié au spread
        """
        spread_config = self.config["spread"]
        
        # Les ordres limit n'ont généralement pas de coût de spread
        if order.type.value in ["LIMIT", "STOP_LIMIT"]:
            return 0.0
            
        # Pour les ordres market, calculer le spread selon le modèle
        spread_model = spread_config["model"]
        transaction_value = order.filled_quantity * order.avg_fill_price
        
        if spread_model == "fixed":
            # Spread fixe en valeur absolue
            return spread_config["value"]
            
        elif spread_model == "percentage":
            # Spread en pourcentage de la transaction
            return transaction_value * spread_config["value"]
            
        elif spread_model == "historical":
            # Utiliser des données historiques de spread si disponibles
            if "historical_spread" in market_data.columns:
                last_spread = market_data["historical_spread"].iloc[-1]
                return transaction_value * last_spread
            else:
                logger.warning("Données historiques de spread non disponibles")
                return transaction_value * spread_config["value"]  # Fallback
        
        else:
            logger.warning(f"Modèle de spread non reconnu: {spread_model}")
            return 0.0


class AdvancedTransactionCostModel(TransactionCostModel):
    """
    Modèle avancé de coûts de transaction qui ajoute des fonctionnalités
    supplémentaires comme l'analyse d'impact de marché et la modélisation
    de la profondeur du livre d'ordres.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le modèle avancé de coûts de transaction.
        
        Args:
            config: Configuration pour le modèle
        """
        super().__init__(config)
        
        # Paramètres supplémentaires pour le modèle avancé
        self.advanced_params = {
            "market_impact": {
                "enabled": True,
                "model": "square_root",   # Modèle d'impact (linear, square_root)
                "factor": 0.1             # Facteur d'impact
            },
            "order_book": {
                "simulation": "realistic",  # Mode de simulation (simple, realistic)
                "depth_factor": 1.0,        # Facteur de profondeur du carnet
                "recovery_rate": 0.8        # Taux de rétablissement après impact
            },
            "temporal": {
                "intraday_pattern": None,  # Modèle de variation intra-journalière des coûts
                "volatility_adjustment": True  # Ajustement basé sur la volatilité
            }
        }
        
        # Fusionner avec la configuration existante
        for key, value in self.advanced_params.items():
            if key not in self.config:
                self.config[key] = value
    
    def calculate_total_cost(self, 
                           order: Order, 
                           market_data: pd.DataFrame, 
                           liquidity_data: Dict = None,
                           order_book: Dict = None) -> Dict[str, float]:
        """
        Calcule le coût total avec prise en compte des impacts avancés.
        
        Args:
            order: L'ordre à évaluer
            market_data: Données de marché
            liquidity_data: Données de liquidité
            order_book: Données du carnet d'ordres
            
        Returns:
            Dictionnaire avec la ventilation des coûts par catégorie
        """
        # Calculer les coûts de base
        base_costs = super().calculate_total_cost(order, market_data, liquidity_data)
        
        # Ajouter les coûts d'impact de marché si activés
        if self.config["market_impact"]["enabled"]:
            market_impact = self._calculate_market_impact(order, market_data, liquidity_data, order_book)
            base_costs["market_impact"] = market_impact
            base_costs["total"] += market_impact
        
        return base_costs
    
    def _calculate_market_impact(self, 
                               order: Order, 
                               market_data: pd.DataFrame, 
                               liquidity_data: Dict = None,
                               order_book: Dict = None) -> float:
        """
        Calcule l'impact de marché d'une transaction importante.
        
        Args:
            order: L'ordre concerné
            market_data: Données de marché
            liquidity_data: Données de liquidité
            order_book: Données du carnet d'ordres
            
        Returns:
            Coût d'impact de marché
        """
        if not order.is_filled():
            return 0.0
            
        # Récupérer les paramètres de configuration
        impact_config = self.config["market_impact"]
        impact_model = impact_config["model"]
        impact_factor = impact_config["factor"]
        
        # Récupérer le volume et le prix
        transaction_value = order.filled_quantity * order.avg_fill_price
        
        # Obtenir le volume moyen quotidien
        if liquidity_data and "avg_daily_volume" in liquidity_data:
            daily_volume = liquidity_data["avg_daily_volume"]
        else:
            # Estimer à partir des données disponibles
            daily_volume = market_data["volume"].mean() * 24
            
        # Calculer le ratio de participation
        participation_rate = (order.filled_quantity * order.avg_fill_price) / daily_volume
        
        # Appliquer le modèle d'impact
        if impact_model == "linear":
            # Impact proportionnel au ratio de participation
            impact = transaction_value * impact_factor * participation_rate
            
        elif impact_model == "square_root":
            # Modèle en racine carrée (plus réaliste)
            impact = transaction_value * impact_factor * np.sqrt(participation_rate)
            
        else:
            logger.warning(f"Modèle d'impact non reconnu: {impact_model}")
            impact = 0.0
            
        return impact


class ExchangeSpecificCostModel(TransactionCostModel):
    """
    Modèle de coûts spécifique à un exchange particulier, avec des
    structures de frais prédéfinies pour les principaux exchanges crypto.
    """
    
    # Frais connus des exchanges populaires
    EXCHANGE_FEE_STRUCTURES = {
        "binance": {
            "type": FeeType.COMBINED.value,
            "maker": 0.0010,  # 0.10%
            "taker": 0.0010,  # 0.10%
            "fixed": 0.0,
            "min_fee": 0.0,
            "tiered": True,
            "withdrawal_fee": {}  # Frais de retrait par actif
        },
        "coinbase_pro": {
            "type": FeeType.COMBINED.value,
            "maker": 0.0050,  # 0.50%
            "taker": 0.0050,  # 0.50%
            "fixed": 0.0,
            "min_fee": 0.0,
            "tiered": True,
            "withdrawal_fee": {}
        },
        "bitfinex": {
            "type": FeeType.COMBINED.value,
            "maker": 0.0010,  # 0.10%
            "taker": 0.0020,  # 0.20%
            "fixed": 0.0,
            "min_fee": 0.0,
            "tiered": True,
            "withdrawal_fee": {}
        },
        "kraken": {
            "type": FeeType.COMBINED.value,
            "maker": 0.0016,  # 0.16%
            "taker": 0.0026,  # 0.26%
            "fixed": 0.0,
            "min_fee": 0.0,
            "tiered": True,
            "withdrawal_fee": {}
        },
        "ftx": {
            "type": FeeType.COMBINED.value,
            "maker": 0.0002,  # 0.02%
            "taker": 0.0007,  # 0.07%
            "fixed": 0.0,
            "min_fee": 0.0,
            "tiered": True,
            "withdrawal_fee": {}
        },
        "default": {
            "type": FeeType.COMBINED.value,
            "maker": 0.0020,  # 0.20% 
            "taker": 0.0020,  # 0.20%
            "fixed": 0.0,
            "min_fee": 0.0,
            "tiered": False,
            "withdrawal_fee": {}
        }
    }
    
    def __init__(self, exchange_name: str, config: Dict = None):
        """
        Initialise le modèle de coûts spécifique à un exchange.
        
        Args:
            exchange_name: Nom de l'exchange (binance, coinbase_pro, etc.)
            config: Configuration supplémentaire
        """
        # Récupérer la structure de frais pour cet exchange
        exchange_name = exchange_name.lower()
        fee_structure = self.EXCHANGE_FEE_STRUCTURES.get(
            exchange_name, 
            self.EXCHANGE_FEE_STRUCTURES["default"]
        )
        
        # Créer la configuration de base
        base_config = {
            "exchange_fees": fee_structure,
            # Autres paramètres par défaut
            "regulatory_fees": {
                "enabled": False,
                "rate": 0.0
            },
            "taxes": {
                "enabled": False,
                "rate": 0.0
            },
            "spread": {
                "model": "percentage",
                "value": 0.0005
            },
            "slippage": {
                "model": "volume_based",
                "fixed": 0.0005,
                "volume_factor": 0.1,
                "volatility_factor": 0.5
            }
        }
        
        # Fusionner avec la configuration fournie
        if config:
            for key, value in config.items():
                if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                    base_config[key].update(value)
                else:
                    base_config[key] = value
        
        # Initialiser avec la configuration finale
        super().__init__(base_config)
        
        # Stocker le nom de l'exchange
        self.exchange_name = exchange_name
    
    def calculate_withdrawal_fee(self, asset: str, amount: float) -> float:
        """
        Calcule les frais de retrait pour un actif spécifique.
        
        Args:
            asset: L'actif concerné (BTC, ETH, etc.)
            amount: Montant du retrait
            
        Returns:
            Frais de retrait
        """
        withdrawal_fees = self.config["exchange_fees"]["withdrawal_fee"]
        
        if asset in withdrawal_fees:
            fee = withdrawal_fees[asset]
            
            # Vérifier si le frais est fixe ou pourcentage
            if isinstance(fee, float) or isinstance(fee, int):
                return fee  # Frais fixe
            elif isinstance(fee, dict) and "type" in fee:
                if fee["type"] == "fixed":
                    return fee["value"]
                elif fee["type"] == "percentage":
                    return amount * fee["value"]
        
        # Frais par défaut
        return 0.0  # Aucun frais par défaut 