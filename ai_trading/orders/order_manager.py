"""
Gestionnaire d'ordres professionnels avec limites dynamiques pour le système de trading.

Ce module fournit la classe OrderManager qui gère:
- Création et suivi des ordres
- Exécution des ordres en fonction des conditions de marché
- Limites dynamiques basées sur la liquidité et la volatilité
- Gestion des ordres conditionnels (OCO, trailing stop)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

from ai_trading.orders.order_types import (
    OCOOrder,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    create_oco_order,
    create_order,
)
from ai_trading.risk.advanced_risk_manager import AdvancedRiskManager

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Gestionnaire d'ordres professionnels avec limites dynamiques.

    Gère la création, le suivi et l'exécution des ordres avec des fonctionnalités avancées:
    - Limites dynamiques basées sur la liquidité et la volatilité
    - Analyse du carnet d'ordres et détection des niveaux de support/résistance
    - Optimisation des coûts d'exécution et réduction du slippage
    - Stratégies de placement d'ordres avancées
    """

    def __init__(self, risk_manager=None, config=None):
        """
        Initialise le gestionnaire d'ordres.

        Args:
            risk_manager (AdvancedRiskManager, optional): Gestionnaire de risques
            config (dict, optional): Configuration du gestionnaire d'ordres
                - max_orders_per_symbol (int): Nombre maximum d'ordres par symbole
                - max_active_orders (int): Nombre maximum d'ordres actifs total
                - default_time_in_force (str): Durée de validité par défaut
                - default_slippage (float): Slippage par défaut pour les ordres au marché
                - smart_routing (bool): Activer le routage intelligent des ordres
                - adaptive_limits (bool): Activer les limites adaptatives
                - position_sizing_method (str): Méthode de dimensionnement des positions
                - order_update_interval (int): Intervalle de mise à jour des ordres
        """
        self.config = config or {}

        # Gestionnaire de risques (peut être None)
        self.risk_manager = risk_manager

        # Paramètres de configuration
        self.max_orders_per_symbol = self.config.get("max_orders_per_symbol", 20)
        self.max_active_orders = self.config.get("max_active_orders", 100)
        self.default_time_in_force = self.config.get(
            "default_time_in_force", TimeInForce.GTC
        )
        self.default_slippage = self.config.get("default_slippage", 0.001)
        self.smart_routing = self.config.get("smart_routing", True)
        self.adaptive_limits = self.config.get("adaptive_limits", True)
        self.position_sizing_method = self.config.get(
            "position_sizing_method", "risk_based"
        )
        self.order_update_interval = self.config.get("order_update_interval", 1)

        # Collections pour stocker les ordres
        self.orders: Dict[str, Order] = {}  # Tous les ordres par ID
        self.orders_by_symbol: Dict[str, List[str]] = {}  # Ordres par symbole
        self.active_orders: List[str] = []  # Ordres actifs
        self.oco_orders: Dict[str, OCOOrder] = {}  # Ordres OCO

        # Contexte de marché
        self.market_data: Dict[str, Dict] = {}  # Données de marché par symbole
        self.orderbook_depth: Dict[str, Dict] = {}  # Profondeur du carnet d'ordres
        self.volatility_metrics: Dict[str, Dict] = {}  # Métriques de volatilité
        self.support_resistance_levels: Dict[str, List[float]] = (
            {}
        )  # Niveaux de support/résistance

        # Exécution simulée
        self.execution_data: Dict[str, List[Dict]] = {}  # Données d'exécution (fills)

        logger.info("Gestionnaire d'ordres professionnels initialisé")

    def create_order(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        order_type: Union[str, OrderType],
        quantity: float = None,
        price: float = None,
        stop_price: float = None,
        capital_amount: float = None,
        risk_amount: float = None,
        **kwargs,
    ) -> Order:
        """
        Crée un nouvel ordre avec les paramètres spécifiés.

        Args:
            symbol (str): Symbole de l'actif
            side (Union[str, OrderSide]): Côté de l'ordre (achat ou vente)
            order_type (Union[str, OrderType]): Type d'ordre
            quantity (float, optional): Quantité de l'actif
            price (float, optional): Prix limite
            stop_price (float, optional): Prix stop
            capital_amount (float, optional): Montant de capital à utiliser
            risk_amount (float, optional): Montant de risque à prendre
            **kwargs: Paramètres supplémentaires

        Returns:
            Order: L'ordre créé
        """
        # Vérifier les limites d'ordres
        if self._check_order_limits(symbol):
            logger.warning(f"Limite d'ordres atteinte pour {symbol}")
            return None

        # Déterminer la quantité si non spécifiée
        if quantity is None:
            quantity = self._calculate_position_size(
                symbol, side, price, stop_price, capital_amount, risk_amount
            )

        # Convertir les strings en enums si nécessaire
        if isinstance(side, str):
            side = OrderSide(side)

        if isinstance(order_type, str):
            order_type = OrderType(order_type)

        # Créer l'ordre
        order = create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            **kwargs,
        )

        # Ajouter l'ordre aux collections
        self.orders[order.order_id] = order

        if symbol not in self.orders_by_symbol:
            self.orders_by_symbol[symbol] = []
        self.orders_by_symbol[symbol].append(order.order_id)

        # Ajouter aux ordres actifs si applicable
        if order.status in (OrderStatus.PENDING, OrderStatus.ACTIVE):
            self.active_orders.append(order.order_id)

        logger.info(
            f"Ordre créé: {order.order_id} - {symbol} {side.value} {order_type.value} {quantity}"
        )
        return order

    def create_oco_order(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: float,
        limit_price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None,
        **kwargs,
    ) -> OCOOrder:
        """
        Crée un ordre OCO (One-Cancels-the-Other).

        Args:
            symbol (str): Symbole de l'actif
            side (Union[str, OrderSide]): Côté de l'ordre (achat ou vente)
            quantity (float): Quantité
            limit_price (float): Prix limite pour l'ordre limite
            stop_price (float): Prix stop pour l'ordre stop
            stop_limit_price (float, optional): Prix limite pour l'ordre stop-limite
            **kwargs: Paramètres supplémentaires

        Returns:
            OCOOrder: L'ordre OCO créé
        """
        # Vérifier les limites d'ordres
        if self._check_order_limits(symbol, count=2):  # OCO = 2 ordres
            logger.warning(f"Limite d'ordres atteinte pour {symbol}")
            return None

        # Créer l'ordre OCO
        oco_order = create_oco_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
            **kwargs,
        )

        # Ajouter les ordres individuels aux collections
        self.orders[oco_order.limit_order.order_id] = oco_order.limit_order
        self.orders[oco_order.stop_order.order_id] = oco_order.stop_order

        if symbol not in self.orders_by_symbol:
            self.orders_by_symbol[symbol] = []
        self.orders_by_symbol[symbol].append(oco_order.limit_order.order_id)
        self.orders_by_symbol[symbol].append(oco_order.stop_order.order_id)

        # Ajouter aux ordres actifs
        self.active_orders.append(oco_order.limit_order.order_id)
        self.active_orders.append(oco_order.stop_order.order_id)

        # Ajouter aux ordres OCO
        self.oco_orders[oco_order.oco_id] = oco_order

        logger.info(
            f"Ordre OCO créé: {oco_order.oco_id} - {symbol} {side} ({limit_price}/{stop_price})"
        )
        return oco_order

    def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre.

        Args:
            order_id (str): ID de l'ordre à annuler

        Returns:
            bool: True si l'annulation a réussi, False sinon
        """
        if order_id not in self.orders:
            logger.warning(f"Ordre {order_id} non trouvé")
            return False

        order = self.orders[order_id]
        success = order.cancel()

        if success:
            # Supprimer des ordres actifs
            if order_id in self.active_orders:
                self.active_orders.remove(order_id)

            # Vérifier si c'est un ordre OCO et annuler l'autre si nécessaire
            if "oco_id" in order.tags:
                oco_id = order.tags["oco_id"]
                if oco_id in self.oco_orders:
                    self.oco_orders[oco_id].on_order_filled(order_id)

            logger.info(f"Ordre {order_id} annulé")

        return success

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Annule tous les ordres, éventuellement pour un symbole spécifique.

        Args:
            symbol (str, optional): Symbole pour lequel annuler les ordres

        Returns:
            int: Nombre d'ordres annulés
        """
        count = 0
        orders_to_cancel = []

        if symbol:
            # Annuler tous les ordres pour un symbole spécifique
            if symbol in self.orders_by_symbol:
                orders_to_cancel = self.orders_by_symbol[symbol].copy()
        else:
            # Annuler tous les ordres actifs
            orders_to_cancel = self.active_orders.copy()

        for order_id in orders_to_cancel:
            if self.cancel_order(order_id):
                count += 1

        logger.info(f"{count} ordres annulés" + (f" pour {symbol}" if symbol else ""))
        return count

    def process_orders(self, market_data: Dict[str, Dict]) -> List[Dict]:
        """
        Traite tous les ordres actifs en fonction des données de marché actuelles.

        Args:
            market_data (Dict[str, Dict]): Données de marché par symbole
                Format: {
                    'symbol1': {'price': 100.0, 'timestamp': '2023-01-01T12:00:00', ...},
                    'symbol2': {'price': 200.0, 'timestamp': '2023-01-01T12:00:00', ...}
                }

        Returns:
            List[Dict]: Liste des exécutions effectuées
        """
        # Mettre à jour les données de marché
        self.market_data.update(market_data)

        # Liste des exécutions
        executions = []

        # Traiter les ordres actifs pour chaque symbole
        for symbol, data in market_data.items():
            if symbol not in self.orders_by_symbol:
                continue

            current_price = data.get("price")
            if current_price is None:
                continue

            # Liste des ordres à mettre à jour ou exécuter
            for order_id in self.orders_by_symbol[symbol]:
                order = self.orders[order_id]

                # Ignorer les ordres qui ne sont pas actifs
                if order.status not in (
                    OrderStatus.PENDING,
                    OrderStatus.ACTIVE,
                    OrderStatus.PARTIALLY_FILLED,
                ):
                    continue

                # Activer les ordres en attente
                if order.status == OrderStatus.PENDING:
                    order.update_status(OrderStatus.ACTIVE)

                # Mettre à jour les ordres trailing stop
                if order.order_type == OrderType.TRAILING_STOP:
                    order.update_stop_price(current_price)

                # Vérifier si l'ordre peut être exécuté
                if order.can_execute(current_price):
                    # Exécuter l'ordre (totalement ou partiellement)
                    execution = self._execute_order(order_id, current_price, data)
                    if execution:
                        executions.append(execution)

        return executions

    def _execute_order(
        self, order_id: str, price: float, market_data: Dict
    ) -> Optional[Dict]:
        """
        Exécute un ordre au prix spécifié.

        Args:
            order_id (str): ID de l'ordre à exécuter
            price (float): Prix d'exécution
            market_data (Dict): Données de marché

        Returns:
            Dict: Détails de l'exécution, ou None si l'ordre n'a pas été exécuté
        """
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]

        # Vérifier si l'ordre peut être exécuté
        if not order.can_execute(price):
            return None

        # Calculer la quantité à exécuter (pour les ordres partiels)
        remaining_quantity = order.get_remaining_quantity()
        if remaining_quantity <= 0:
            return None

        # Dans une implémentation réelle, on tiendrait compte de la liquidité disponible
        execution_quantity = remaining_quantity

        # Appliquer le slippage pour les ordres au marché
        execution_price = price
        if order.order_type == OrderType.MARKET:
            execution_price = self._apply_slippage(price, order.side)

        # Ajouter l'exécution à l'ordre
        timestamp = datetime.now()
        if "timestamp" in market_data:
            try:
                timestamp = datetime.fromisoformat(market_data["timestamp"])
            except (ValueError, TypeError):
                pass

        order.add_fill(execution_price, execution_quantity, timestamp)

        # Si l'ordre est complètement exécuté, le retirer des ordres actifs
        if order.status == OrderStatus.FILLED:
            if order_id in self.active_orders:
                self.active_orders.remove(order_id)

            # Vérifier si c'est un ordre OCO et annuler l'autre
            if "oco_id" in order.tags:
                oco_id = order.tags["oco_id"]
                if oco_id in self.oco_orders:
                    self.oco_orders[oco_id].on_order_filled(order_id)

        # Créer le rapport d'exécution
        execution = {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "price": execution_price,
            "quantity": execution_quantity,
            "timestamp": timestamp.isoformat(),
            "order_type": order.order_type.value,
            "status": order.status.value,
        }

        logger.info(
            f"Ordre exécuté: {order_id} - {execution_quantity} @ {execution_price}"
        )
        return execution

    def _calculate_position_size(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        price: Optional[float],
        stop_price: Optional[float],
        capital_amount: Optional[float],
        risk_amount: Optional[float],
    ) -> float:
        """
        Calcule la taille de position optimale en fonction de différents paramètres.

        Args:
            symbol (str): Symbole de l'actif
            side (Union[str, OrderSide]): Côté de l'ordre
            price (float, optional): Prix d'entrée
            stop_price (float, optional): Prix de stop-loss
            capital_amount (float, optional): Montant de capital à utiliser
            risk_amount (float, optional): Montant de risque à prendre

        Returns:
            float: Quantité à trader
        """
        # Utiliser le gestionnaire de risques avancé si disponible
        if (
            self.risk_manager
            and isinstance(self.risk_manager, AdvancedRiskManager)
            and price
            and stop_price
        ):
            # Si on a accès aux données historiques via le risk_manager
            if hasattr(self.risk_manager, "calculate_position_size"):
                risk_per_trade = self.risk_manager.config.get("risk_per_trade", 0.02)
                capital = (
                    capital_amount or 10000.0
                )  # Valeur par défaut si non spécifiée

                # Calcul basé sur le risque
                position_size = self.risk_manager.calculate_position_size(
                    capital=capital, entry_price=price, stop_loss_price=stop_price
                )

                logger.debug(
                    f"Taille de position calculée via risk_manager: {position_size}"
                )
                return position_size

        # Méthode de calcul par défaut si le gestionnaire de risques n'est pas disponible
        if self.position_sizing_method == "risk_based" and price and stop_price:
            # Calcul basé sur le risque (% du capital)
            risk_pct = 0.02  # 2% du capital par défaut
            capital = capital_amount or 10000.0  # Valeur par défaut

            if risk_amount:
                risk_money = risk_amount
            else:
                risk_money = capital * risk_pct

            # Calculer le risque par unité
            if side == OrderSide.BUY or side == "buy":
                risk_per_unit = price - stop_price
            else:  # SELL
                risk_per_unit = stop_price - price

            # Éviter la division par zéro
            if abs(risk_per_unit) < 1e-6:
                risk_per_unit = 0.01 * price  # 1% du prix

            # Calculer la quantité
            quantity = risk_money / abs(risk_per_unit)

            logger.debug(f"Taille de position basée sur le risque: {quantity}")
            return quantity

        elif self.position_sizing_method == "equal_weight":
            # Méthode de pondération égale (allocation fixe du capital)
            capital = capital_amount or 10000.0
            allocation_pct = 0.1  # 10% du capital par défaut

            if price:
                quantity = (capital * allocation_pct) / price
                logger.debug(f"Taille de position à pondération égale: {quantity}")
                return quantity

        # Méthode par défaut : quantité fixe
        default_quantity = 1.0
        logger.debug(f"Taille de position par défaut: {default_quantity}")
        return default_quantity

    def _check_order_limits(self, symbol: str, count: int = 1) -> bool:
        """
        Vérifie si les limites d'ordres sont atteintes.

        Args:
            symbol (str): Symbole de l'actif
            count (int): Nombre d'ordres à ajouter

        Returns:
            bool: True si les limites sont atteintes, False sinon
        """
        # Vérifier la limite globale
        if len(self.active_orders) + count > self.max_active_orders:
            return True

        # Vérifier la limite par symbole
        if symbol in self.orders_by_symbol:
            active_for_symbol = sum(
                1
                for order_id in self.orders_by_symbol[symbol]
                if order_id in self.active_orders
            )
            if active_for_symbol + count > self.max_orders_per_symbol:
                return True

        return False

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        Applique le slippage au prix en fonction du côté de l'ordre.

        Args:
            price (float): Prix original
            side (OrderSide): Côté de l'ordre

        Returns:
            float: Prix avec slippage
        """
        slippage = self.default_slippage

        if side == OrderSide.BUY:
            return price * (1 + slippage)
        else:  # SELL
            return price * (1 - slippage)

    def update_market_data(self, market_data: Dict[str, Dict]):
        """
        Met à jour les données de marché.

        Args:
            market_data (Dict[str, Dict]): Données de marché par symbole
        """
        self.market_data.update(market_data)

    def update_orderbook_depth(self, symbol: str, depth_data: Dict):
        """
        Met à jour les données de profondeur du carnet d'ordres.

        Args:
            symbol (str): Symbole de l'actif
            depth_data (Dict): Données de profondeur
        """
        self.orderbook_depth[symbol] = depth_data

    def update_volatility_metrics(self, symbol: str, metrics: Dict):
        """
        Met à jour les métriques de volatilité.

        Args:
            symbol (str): Symbole de l'actif
            metrics (Dict): Métriques de volatilité
        """
        self.volatility_metrics[symbol] = metrics

    def update_support_resistance_levels(self, symbol: str, levels: List[float]):
        """
        Met à jour les niveaux de support et résistance.

        Args:
            symbol (str): Symbole de l'actif
            levels (List[float]): Niveaux de support et résistance
        """
        self.support_resistance_levels[symbol] = levels

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Récupère tous les ordres actifs, éventuellement pour un symbole spécifique.

        Args:
            symbol (str, optional): Symbole spécifique

        Returns:
            List[Order]: Liste des ordres actifs
        """
        if symbol:
            if symbol not in self.orders_by_symbol:
                return []

            return [
                self.orders[order_id]
                for order_id in self.orders_by_symbol[symbol]
                if order_id in self.active_orders
            ]
        else:
            return [self.orders[order_id] for order_id in self.active_orders]

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Récupère un ordre par son ID.

        Args:
            order_id (str): ID de l'ordre

        Returns:
            Order: L'ordre correspondant, ou None s'il n'existe pas
        """
        return self.orders.get(order_id)
