"""
Module d'intégration du système d'ordres professionnels dans l'environnement de trading RL.

Ce module fournit les classes et fonctions nécessaires pour:
- Intégrer le gestionnaire d'ordres avancé dans l'environnement RL
- Convertir les actions de l'agent en ordres professionnels
- Gérer l'exécution des ordres en fonction des contraintes de marché
"""

import logging
from typing import Dict, List, Optional

from ai_trading.orders.order_manager import OrderManager
from ai_trading.orders.order_types import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class OrderExecutionEnv:
    """
    Couche d'intégration entre l'environnement de trading RL et le système d'ordres professionnels.

    Cette classe:
    - Traduit les actions de l'agent en ordres professionnels
    - Gère l'exécution des ordres avec des contraintes de marché réalistes
    - Fournit des informations d'exécution à l'environnement de trading
    """

    def __init__(self, risk_manager=None, config=None):
        """
        Initialise l'environnement d'exécution des ordres.

        Args:
            risk_manager (AdvancedRiskManager, optional): Gestionnaire de risques
            config (dict, optional): Configuration de l'environnement
                - use_limit_orders (bool): Utiliser des ordres limites au lieu d'ordres au marché
                - use_stop_orders (bool): Utiliser des ordres stop pour les sorties
                - use_oco_orders (bool): Utiliser des ordres OCO pour take-profit/stop-loss
                - use_trailing_stops (bool): Utiliser des trailing stops pour protéger les gains
                - limit_order_offset (float): Décalage des ordres limites (en %)
                - dynamic_limit_orders (bool): Ajuster dynamiquement les ordres limites
                - slippage_model (str): Modèle de slippage ('fixed', 'dynamic')
                - max_slippage (float): Slippage maximum
                - min_execution_size (float): Taille minimale d'exécution
                - execution_delay (int): Délai d'exécution en périodes
        """
        self.config = config or {}
        self.risk_manager = risk_manager

        # Configuration des types d'ordres
        self.use_limit_orders = self.config.get("use_limit_orders", True)
        self.use_stop_orders = self.config.get("use_stop_orders", True)
        self.use_oco_orders = self.config.get("use_oco_orders", True)
        self.use_trailing_stops = self.config.get("use_trailing_stops", False)

        # Paramètres des ordres limites
        self.limit_order_offset = self.config.get("limit_order_offset", 0.001)  # 0.1%
        self.dynamic_limit_orders = self.config.get("dynamic_limit_orders", True)

        # Paramètres d'exécution
        self.slippage_model = self.config.get("slippage_model", "dynamic")
        self.max_slippage = self.config.get("max_slippage", 0.01)  # 1%
        self.min_execution_size = self.config.get("min_execution_size", 0.001)
        self.execution_delay = self.config.get("execution_delay", 0)

        # Initialisation du gestionnaire d'ordres
        self.order_manager = OrderManager(risk_manager=risk_manager, config=config)

        # État interne
        self.current_orders: Dict[str, List[str]] = {}  # Ordres par symbole
        self.position_status: Dict[str, Dict] = {}  # État des positions par symbole
        self.execution_queue: List[Dict] = []  # File d'exécution des ordres
        self.execution_history: List[Dict] = []  # Historique des exécutions

        logger.info("Environnement d'exécution des ordres initialisé")

    def process_action(self, symbol: str, action_value: float, data: Dict) -> Dict:
        """
        Traite une action de l'agent et la traduit en ordre(s).

        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            data (Dict): Données de marché pour le symbole

        Returns:
            Dict: Résultat de l'exécution avec les clés:
                - 'executed': Bool indiquant si l'ordre a été exécuté
                - 'price': Prix d'exécution
                - 'quantity': Quantité exécutée
                - 'orders': Liste des ordres créés
        """
        # Initialiser le résultat
        result = {"executed": False, "price": None, "quantity": 0, "orders": []}

        # Ignore les micro-actions
        if -0.05 <= action_value <= 0.05:
            return result

        # Déterminer le prix actuel
        current_price = data.get("close", data.get("price"))
        if current_price is None:
            logger.warning(f"Pas de prix disponible pour {symbol}")
            return result

        # Déterminer le côté de l'ordre
        side = OrderSide.BUY if action_value > 0 else OrderSide.SELL

        # Déterminer la taille de la position
        position_size = abs(action_value)

        # Déterminer le montant de capital à utiliser (pour le calcul de position)
        capital_amount = data.get("portfolio_value", 10000.0) * position_size

        # Créer l'ordre approprié selon la configuration
        order_id = None

        if self.use_limit_orders:
            # Calculer le prix limite
            offset = (
                self._calculate_dynamic_offset(symbol, data)
                if self.dynamic_limit_orders
                else self.limit_order_offset
            )

            if side == OrderSide.BUY:
                limit_price = current_price * (1 - offset)
            else:  # SELL
                limit_price = current_price * (1 + offset)

            # Créer un ordre limite
            order = self.order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                price=limit_price,
                capital_amount=capital_amount,
                risk_amount=None,
            )

            if order:
                order_id = order.order_id
                result["orders"].append(order_id)

                logger.info(
                    f"Ordre limite créé: {symbol} {side.value} @ {limit_price:.4f}"
                )
        else:
            # Créer un ordre au marché
            order = self.order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                capital_amount=capital_amount,
                risk_amount=None,
            )

            if order:
                order_id = order.order_id
                result["orders"].append(order_id)

                logger.info(f"Ordre marché créé: {symbol} {side.value}")

        # Si on utilise des ordres stop, créer les ordres stop-loss/take-profit
        if order and self.use_stop_orders:
            self._create_stop_orders(
                symbol, side, current_price, order.quantity, result
            )

        # Mettre à jour l'état des positions
        if order_id:
            if symbol not in self.current_orders:
                self.current_orders[symbol] = []
            self.current_orders[symbol].append(order_id)

        # Traiter immédiatement l'exécution si c'est un ordre au marché sans délai
        if order and order.order_type == OrderType.MARKET and self.execution_delay == 0:
            execution = self.order_manager._execute_order(order_id, current_price, data)
            if execution:
                result["executed"] = True
                result["price"] = execution["price"]
                result["quantity"] = execution["quantity"]
                self.execution_history.append(execution)

        return result

    def _create_stop_orders(
        self,
        symbol: str,
        entry_side: OrderSide,
        current_price: float,
        quantity: float,
        result: Dict,
    ) -> None:
        """
        Crée les ordres stop-loss et take-profit pour une position.

        Args:
            symbol (str): Symbole de l'actif
            entry_side (OrderSide): Côté de l'ordre d'entrée
            current_price (float): Prix actuel
            quantity (float): Quantité de l'ordre d'entrée
            result (Dict): Dictionnaire de résultat à mettre à jour
        """
        # Déterminer le côté des ordres de sortie (inverse de l'entrée)
        exit_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY

        # Déterminer les prix stop-loss et take-profit
        if entry_side == OrderSide.BUY:
            # Pour un achat: stop-loss en-dessous, take-profit au-dessus
            stop_loss_price = current_price * 0.95  # -5%
            take_profit_price = current_price * 1.15  # +15%
        else:
            # Pour une vente: stop-loss au-dessus, take-profit en-dessous
            stop_loss_price = current_price * 1.05  # +5%
            take_profit_price = current_price * 0.85  # -15%

        # Utiliser les calculs du gestionnaire de risques si disponible
        if self.risk_manager and hasattr(self.risk_manager, "calculate_atr_stop_loss"):
            # Récupérer les données historiques si elles sont accessibles
            if hasattr(self.risk_manager, "_get_historical_data"):
                historical_data = self.risk_manager._get_historical_data(symbol)
                if historical_data is not None:
                    # Calculer le stop-loss basé sur l'ATR
                    stop_loss_price = self.risk_manager.calculate_atr_stop_loss(
                        historical_data,
                        direction="long" if entry_side == OrderSide.BUY else "short",
                        current_price=current_price,
                    )

                    # Calculer le take-profit basé sur l'ATR
                    take_profit_price = self.risk_manager.calculate_atr_take_profit(
                        historical_data,
                        direction="long" if entry_side == OrderSide.BUY else "short",
                        current_price=current_price,
                    )

        # Créer les ordres selon la configuration
        if self.use_oco_orders:
            # Créer un ordre OCO (One-Cancels-the-Other)
            oco_order = self.order_manager.create_oco_order(
                symbol=symbol,
                side=exit_side,
                quantity=quantity,
                limit_price=take_profit_price,
                stop_price=stop_loss_price,
            )

            if oco_order:
                result["orders"].append(oco_order.limit_order.order_id)
                result["orders"].append(oco_order.stop_order.order_id)

                logger.info(
                    f"Ordre OCO créé: {symbol} {exit_side.value} "
                    f"(TP: {take_profit_price:.4f} / SL: {stop_loss_price:.4f})"
                )
        else:
            # Créer des ordres stop-loss et take-profit séparés
            stop_loss = self.order_manager.create_order(
                symbol=symbol,
                side=exit_side,
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=stop_loss_price,
            )

            take_profit = self.order_manager.create_order(
                symbol=symbol,
                side=exit_side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=take_profit_price,
            )

            if stop_loss:
                result["orders"].append(stop_loss.order_id)
                logger.info(
                    f"Stop-loss créé: {symbol} {exit_side.value} @ {stop_loss_price:.4f}"
                )

            if take_profit:
                result["orders"].append(take_profit.order_id)
                logger.info(
                    f"Take-profit créé: {symbol} {exit_side.value} @ {take_profit_price:.4f}"
                )

    def _calculate_dynamic_offset(self, symbol: str, data: Dict) -> float:
        """
        Calcule un décalage dynamique pour les ordres limites basé sur la volatilité.

        Args:
            symbol (str): Symbole de l'actif
            data (Dict): Données de marché pour le symbole

        Returns:
            float: Décalage à appliquer au prix
        """
        # Valeur de base
        base_offset = self.limit_order_offset

        # Facteur de volatilité si disponible
        if "volatility" in data:
            volatility = data["volatility"]
            volatility_factor = 1.0 + (volatility * 5.0)  # Ajustement arbitraire
            offset = base_offset * volatility_factor
            return min(offset, self.max_slippage)  # Limiter à max_slippage

        # Facteur basé sur le spread si disponible
        if "spread" in data:
            spread = data["spread"]
            spread_factor = 1.0 + (
                spread / 0.001
            )  # Normaliser par rapport à un spread de base de 0.1%
            offset = base_offset * spread_factor
            return min(offset, self.max_slippage)

        return base_offset

    def process_market_update(self, market_data: Dict[str, Dict]) -> List[Dict]:
        """
        Traite une mise à jour des données de marché et exécute les ordres éligibles.

        Args:
            market_data (Dict[str, Dict]): Données de marché par symbole

        Returns:
            List[Dict]: Liste des exécutions réalisées
        """
        # Mettre à jour les données de marché dans le gestionnaire d'ordres
        self.order_manager.update_market_data(market_data)

        # Traiter les ordres
        executions = self.order_manager.process_orders(market_data)

        # Ajouter à l'historique des exécutions
        self.execution_history.extend(executions)

        # Mettre à jour l'état des positions
        for execution in executions:
            symbol = execution["symbol"]
            side = execution["side"]
            quantity = execution["quantity"]
            price = execution["price"]

            # Mettre à jour la position
            if symbol not in self.position_status:
                self.position_status[symbol] = {"quantity": 0, "avg_price": 0}

            current_position = self.position_status[symbol]

            if side == "buy":
                # Ajouter à la position
                new_quantity = current_position["quantity"] + quantity
                avg_price = (
                    (
                        current_position["quantity"] * current_position["avg_price"]
                        + quantity * price
                    )
                    / new_quantity
                    if new_quantity > 0
                    else 0
                )

                current_position["quantity"] = new_quantity
                current_position["avg_price"] = avg_price
            else:  # sell
                # Soustraire de la position
                current_position["quantity"] -= quantity
                # Si la position devient négative, c'est une position short
                if current_position["quantity"] < 0:
                    current_position["avg_price"] = price
                elif current_position["quantity"] == 0:
                    current_position["avg_price"] = 0

        return executions

    def close_all_positions(
        self, symbol: Optional[str] = None, market_data: Dict[str, Dict] = None
    ):
        """
        Ferme toutes les positions ouvertes, éventuellement pour un symbole spécifique.

        Args:
            symbol (str, optional): Symbole spécifique pour lequel fermer les positions
            market_data (Dict[str, Dict], optional): Données de marché actuelles

        Returns:
            List[Dict]: Liste des exécutions réalisées
        """
        # Annuler tous les ordres en attente
        self.order_manager.cancel_all_orders(symbol)

        # Créer des ordres de marché pour fermer les positions
        executions = []

        symbols_to_close = [symbol] if symbol else list(self.position_status.keys())

        for sym in symbols_to_close:
            if sym not in self.position_status:
                continue

            position = self.position_status[sym]
            quantity = position["quantity"]

            if abs(quantity) < self.min_execution_size:
                continue

            # Déterminer le côté de l'ordre
            side = OrderSide.SELL if quantity > 0 else OrderSide.BUY

            # Créer un ordre au marché
            current_price = None
            if market_data and sym in market_data:
                current_price = market_data[sym].get(
                    "close", market_data[sym].get("price")
                )

            order = self.order_manager.create_order(
                symbol=sym,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(quantity),
            )

            if order and current_price:
                # Exécuter immédiatement
                execution = self.order_manager._execute_order(
                    order.order_id, current_price, market_data.get(sym, {})
                )
                if execution:
                    executions.append(execution)

                    # Mettre à jour l'état de la position
                    self.position_status[sym] = {"quantity": 0, "avg_price": 0}

        return executions

    def get_position(self, symbol: str) -> Dict:
        """
        Récupère l'état de la position pour un symbole.

        Args:
            symbol (str): Symbole de l'actif

        Returns:
            Dict: État de la position avec les clés 'quantity' et 'avg_price'
        """
        if symbol not in self.position_status:
            return {"quantity": 0, "avg_price": 0}

        return self.position_status[symbol]

    def get_all_positions(self) -> Dict[str, Dict]:
        """
        Récupère l'état de toutes les positions.

        Returns:
            Dict[str, Dict]: État de toutes les positions
        """
        return self.position_status

    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Récupère tous les ordres actifs, éventuellement pour un symbole spécifique.

        Args:
            symbol (str, optional): Symbole spécifique

        Returns:
            List[Order]: Liste des ordres actifs
        """
        return self.order_manager.get_active_orders(symbol)

    def get_execution_history(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère l'historique des exécutions, éventuellement pour un symbole spécifique.

        Args:
            symbol (str, optional): Symbole spécifique

        Returns:
            List[Dict]: Historique des exécutions
        """
        if symbol:
            return [ex for ex in self.execution_history if ex["symbol"] == symbol]
        else:
            return self.execution_history


class DynamicLimitOrderStrategy:
    """
    Stratégie pour les ordres limites dynamiques qui s'ajustent en fonction
    des conditions de marché.
    """

    def __init__(self, config=None):
        """
        Initialise la stratégie d'ordres limites dynamiques.

        Args:
            config (dict, optional): Configuration de la stratégie
                - base_offset (float): Décalage de base pour les ordres limites
                - volatility_factor (float): Facteur de volatilité pour l'ajustement
                - spread_factor (float): Facteur de spread pour l'ajustement
                - volume_factor (float): Facteur de volume pour l'ajustement
                - momentum_factor (float): Facteur de momentum pour l'ajustement
                - max_offset (float): Décalage maximum pour les ordres limites
        """
        self.config = config or {}

        # Paramètres de base
        self.base_offset = self.config.get("base_offset", 0.001)  # 0.1%
        self.volatility_factor = self.config.get("volatility_factor", 5.0)
        self.spread_factor = self.config.get("spread_factor", 1.0)
        self.volume_factor = self.config.get("volume_factor", 0.5)
        self.momentum_factor = self.config.get("momentum_factor", 2.0)
        self.max_offset = self.config.get("max_offset", 0.01)  # 1%

        # État interne
        self.market_conditions = {}  # Conditions de marché par symbole

    def calculate_limit_price(
        self, symbol: str, side: OrderSide, current_price: float, market_data: Dict
    ) -> float:
        """
        Calcule le prix limite optimal pour un ordre.

        Args:
            symbol (str): Symbole de l'actif
            side (OrderSide): Côté de l'ordre (achat ou vente)
            current_price (float): Prix actuel
            market_data (Dict): Données de marché

        Returns:
            float: Prix limite optimal
        """
        # Calculer le décalage
        offset = self._calculate_dynamic_offset(symbol, market_data)

        # Calculer le prix limite
        if side == OrderSide.BUY:
            return current_price * (1 - offset)
        else:  # SELL
            return current_price * (1 + offset)

    def _calculate_dynamic_offset(self, symbol: str, market_data: Dict) -> float:
        """
        Calcule un décalage dynamique basé sur les conditions de marché.

        Args:
            symbol (str): Symbole de l'actif
            market_data (Dict): Données de marché

        Returns:
            float: Décalage à appliquer au prix
        """
        # Décalage de base
        offset = self.base_offset

        # Facteur de volatilité
        if "volatility" in market_data:
            volatility = market_data["volatility"]
            offset *= 1.0 + (volatility * self.volatility_factor)

        # Facteur de spread
        if "spread" in market_data:
            spread = market_data["spread"]
            offset *= 1.0 + (spread * self.spread_factor)

        # Facteur de volume
        if "volume" in market_data and "avg_volume" in market_data:
            volume = market_data["volume"]
            avg_volume = market_data["avg_volume"]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            offset *= 1.0 + ((volume_ratio - 1) * self.volume_factor)

        # Facteur de momentum
        if "momentum" in market_data:
            momentum = market_data["momentum"]
            offset *= 1.0 + (abs(momentum) * self.momentum_factor)

        # Limiter à l'offset maximum
        return min(offset, self.max_offset)

    def update_order_prices(
        self, order_manager: OrderManager, symbol: str, market_data: Dict
    ) -> int:
        """
        Met à jour les prix des ordres limites actifs en fonction des nouvelles conditions.

        Args:
            order_manager (OrderManager): Gestionnaire d'ordres
            symbol (str): Symbole de l'actif
            market_data (Dict): Données de marché actuelles

        Returns:
            int: Nombre d'ordres mis à jour
        """
        active_orders = order_manager.get_active_orders(symbol)
        current_price = market_data.get("close", market_data.get("price"))

        if not current_price:
            return 0

        count = 0
        for order in active_orders:
            # Ne mettre à jour que les ordres limites
            if order.order_type != OrderType.LIMIT:
                continue

            # Calculer le nouveau prix limite
            new_price = self.calculate_limit_price(
                symbol, order.side, current_price, market_data
            )

            # Si le nouveau prix est meilleur pour le trader, mettre à jour l'ordre
            if (order.side == OrderSide.BUY and new_price < order.price) or (
                order.side == OrderSide.SELL and new_price > order.price
            ):

                # Annuler l'ordre existant
                order_manager.cancel_order(order.order_id)

                # Créer un nouvel ordre avec le nouveau prix
                new_order = order_manager.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.get_remaining_quantity(),
                    price=new_price,
                )

                if new_order:
                    count += 1

        return count
