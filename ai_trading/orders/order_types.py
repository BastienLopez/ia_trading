"""
Définition des types d'ordres professionnels pour le système de trading.

Ce module fournit les classes et fonctions pour:
- Ordres limites (Limit orders)
- Ordres au marché (Market orders)
- Ordres stop (Stop orders)
- Ordres stop-limite (Stop-limit orders)
- Ordres trailing stop (Trailing stop orders)
- Ordres iceberg (Iceberg orders)
- Ordres OCO (One-Cancels-the-Other)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Côté de l'ordre (achat ou vente)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Type d'ordre."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    OCO = "one_cancels_other"


class OrderStatus(Enum):
    """Statut de l'ordre."""

    PENDING = "pending"  # En attente de traitement
    ACTIVE = "active"  # Actif et en attente d'exécution
    PARTIALLY_FILLED = "partially_filled"  # Partiellement exécuté
    FILLED = "filled"  # Entièrement exécuté
    CANCELLED = "cancelled"  # Annulé
    REJECTED = "rejected"  # Rejeté
    EXPIRED = "expired"  # Expiré (après time_in_force)


class TimeInForce(Enum):
    """Durée de validité de l'ordre."""

    GTC = "good_till_cancelled"  # Valable jusqu'à annulation
    IOC = "immediate_or_cancel"  # Exécution immédiate ou annulation
    FOK = "fill_or_kill"  # Exécution totale ou annulation
    GTD = "good_till_date"  # Valable jusqu'à une date précise
    DAY = "day"  # Valable pour la journée


@dataclass
class Order:
    """Classe de base pour tous les types d'ordres."""

    # Paramètres obligatoires
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    # Paramètres optionnels
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    expiry_date: Optional[datetime] = None

    # Métadonnées générées automatiquement
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Suivi de l'exécution
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)

    # Paramètres avancés
    post_only: bool = False
    hidden: bool = False
    iceberg_quantity: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validation après initialisation."""
        if self.quantity <= 0:
            raise ValueError("La quantité doit être positive")

        if (
            self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
            and self.price is None
        ):
            raise ValueError(
                f"Le prix est requis pour les ordres de type {self.order_type.value}"
            )

        if (
            self.order_type
            in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP)
            and self.stop_price is None
        ):
            raise ValueError(
                f"Le prix stop est requis pour les ordres de type {self.order_type.value}"
            )

        if self.order_type == OrderType.ICEBERG and self.iceberg_quantity is None:
            raise ValueError(
                "La quantité iceberg est requise pour les ordres de type iceberg"
            )

    def update_status(self, new_status: OrderStatus):
        """Met à jour le statut de l'ordre."""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now()
        logger.info(
            f"Ordre {self.order_id} - Statut modifié: {old_status.value} -> {new_status.value}"
        )

    def add_fill(self, price: float, quantity: float, timestamp: datetime = None):
        """
        Ajoute une exécution partielle à l'ordre.

        Args:
            price (float): Prix d'exécution
            quantity (float): Quantité exécutée
            timestamp (datetime, optional): Horodatage de l'exécution
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Créer l'enregistrement d'exécution
        fill = {"price": price, "quantity": quantity, "timestamp": timestamp}

        # Ajouter à la liste des exécutions
        self.fills.append(fill)

        # Mettre à jour les statistiques
        self.filled_quantity += quantity

        # Calculer le nouveau prix moyen d'exécution
        total_value = sum(f["price"] * f["quantity"] for f in self.fills)
        self.average_fill_price = (
            total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        )

        # Mettre à jour le statut
        if self.filled_quantity >= self.quantity:
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)

        logger.info(
            f"Ordre {self.order_id} - Exécution partielle: {quantity} à {price}"
        )

    def cancel(self):
        """Annule l'ordre s'il n'est pas déjà complètement exécuté."""
        if self.status in (
            OrderStatus.PENDING,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        ):
            self.update_status(OrderStatus.CANCELLED)
            return True
        return False

    def can_execute(self, current_price: float) -> bool:
        """
        Vérifie si l'ordre peut être exécuté au prix actuel.

        Args:
            current_price (float): Prix actuel du marché

        Returns:
            bool: True si l'ordre peut être exécuté, False sinon
        """
        if self.status not in (
            OrderStatus.PENDING,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        ):
            return False

        # Ordre au marché
        if self.order_type == OrderType.MARKET:
            return True

        # Ordre limite (achat: prix <= prix limite, vente: prix >= prix limite)
        if self.order_type == OrderType.LIMIT:
            if self.side == OrderSide.BUY and current_price <= self.price:
                return True
            elif self.side == OrderSide.SELL and current_price >= self.price:
                return True

        # Ordre stop (achat: prix >= prix stop, vente: prix <= prix stop)
        if self.order_type == OrderType.STOP:
            if self.side == OrderSide.BUY and current_price >= self.stop_price:
                return True
            elif self.side == OrderSide.SELL and current_price <= self.stop_price:
                return True

        # Ordre stop-limite (vérifie d'abord le déclenchement du stop, puis les conditions de limite)
        if self.order_type == OrderType.STOP_LIMIT:
            # Vérifie si le stop est déclenché
            if (self.side == OrderSide.BUY and current_price >= self.stop_price) or (
                self.side == OrderSide.SELL and current_price <= self.stop_price
            ):
                # Si le stop est déclenché, vérifie les conditions de limite
                if self.side == OrderSide.BUY and current_price <= self.price:
                    return True
                elif self.side == OrderSide.SELL and current_price >= self.price:
                    return True

        return False

    def get_remaining_quantity(self) -> float:
        """Retourne la quantité restante à exécuter."""
        return max(0.0, self.quantity - self.filled_quantity)

    def to_dict(self) -> Dict:
        """Convertit l'ordre en dictionnaire pour sérialisation."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "post_only": self.post_only,
            "hidden": self.hidden,
            "iceberg_quantity": self.iceberg_quantity,
            "tags": self.tags,
        }


@dataclass
class MarketOrder(Order):
    """Ordre au marché - exécution immédiate au meilleur prix disponible."""

    def __post_init__(self):
        """Initialisation spécifique aux ordres market."""
        self.order_type = OrderType.MARKET
        super().__post_init__()


@dataclass
class LimitOrder(Order):
    """Ordre limite - exécution uniquement à un prix spécifié ou meilleur."""

    def __post_init__(self):
        """Initialisation spécifique aux ordres limit."""
        self.order_type = OrderType.LIMIT
        super().__post_init__()


@dataclass
class StopOrder(Order):
    """
    Ordre stop - devient un ordre au marché lorsque le prix atteint le niveau stop.
    Utilisé pour limiter les pertes ou protéger les gains.
    """

    def __post_init__(self):
        """Initialisation spécifique aux ordres stop."""
        self.order_type = OrderType.STOP
        super().__post_init__()


@dataclass
class StopLimitOrder(Order):
    """
    Ordre stop-limite - devient un ordre limite lorsque le prix atteint le niveau stop.
    Combine les caractéristiques des ordres stop et limite.
    """

    def __post_init__(self):
        """Initialisation spécifique aux ordres stop-limit."""
        self.order_type = OrderType.STOP_LIMIT
        super().__post_init__()


@dataclass
class TrailingStopOrder(Order):
    """
    Ordre trailing stop - le prix stop s'ajuste dynamiquement en fonction du mouvement
    du prix en faveur du trader.
    """

    # Distance ou pourcentage de trailing
    trail_value: float = 0.0
    is_trail_percent: bool = False

    # Prix de référence le plus favorable atteint
    reference_price: Optional[float] = None

    def __post_init__(self):
        """Initialisation spécifique aux ordres trailing stop."""
        self.order_type = OrderType.TRAILING_STOP
        super().__post_init__()

        if self.trail_value <= 0:
            raise ValueError("La valeur de trailing doit être positive")

    def update_stop_price(self, current_price: float):
        """
        Met à jour le prix stop en fonction du mouvement du prix.

        Args:
            current_price (float): Prix actuel du marché
        """
        # Initialiser le prix de référence si nécessaire
        if self.reference_price is None:
            self.reference_price = current_price
            self.stop_price = self._calculate_stop_price(current_price)
            return

        # Pour les ordres d'achat, le prix de référence est le plus bas atteint
        if self.side == OrderSide.BUY:
            if current_price < self.reference_price:
                self.reference_price = current_price
                self.stop_price = self._calculate_stop_price(current_price)
                logger.debug(
                    f"Ordre trailing stop {self.order_id} - Nouveau prix stop: {self.stop_price}"
                )

        # Pour les ordres de vente, le prix de référence est le plus haut atteint
        else:  # SELL
            if current_price > self.reference_price:
                self.reference_price = current_price
                self.stop_price = self._calculate_stop_price(current_price)
                logger.debug(
                    f"Ordre trailing stop {self.order_id} - Nouveau prix stop: {self.stop_price}"
                )

    def _calculate_stop_price(self, base_price: float) -> float:
        """
        Calcule le prix stop en fonction du prix de base et de la valeur de trailing.

        Args:
            base_price (float): Prix de base pour le calcul

        Returns:
            float: Nouveau prix stop
        """
        trail_amount = self.trail_value
        if self.is_trail_percent:
            trail_amount = base_price * (self.trail_value / 100.0)

        if self.side == OrderSide.BUY:
            return base_price + trail_amount
        else:  # SELL
            return base_price - trail_amount


@dataclass
class IcebergOrder(LimitOrder):
    """
    Ordre iceberg - affiche seulement une partie de la quantité totale sur le marché.
    Utile pour les transactions de gros volume pour minimiser l'impact sur le marché.
    """

    def __post_init__(self):
        """Initialisation spécifique aux ordres iceberg."""
        super().__post_init__()
        self.order_type = OrderType.ICEBERG

        if self.iceberg_quantity is None or self.iceberg_quantity <= 0:
            raise ValueError("La quantité iceberg doit être positive")

        if self.iceberg_quantity > self.quantity:
            raise ValueError(
                "La quantité iceberg ne peut pas dépasser la quantité totale"
            )


@dataclass
class OCOOrder:
    """
    Ordre OCO (One-Cancels-the-Other) - combine un ordre limite et un ordre stop.
    Lorsqu'un des ordres est exécuté, l'autre est automatiquement annulé.
    """

    # Les deux ordres associés
    limit_order: LimitOrder
    stop_order: Union[StopOrder, StopLimitOrder]

    # Métadonnées
    oco_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validation après initialisation."""
        if self.limit_order.symbol != self.stop_order.symbol:
            raise ValueError("Les deux ordres doivent concerner le même symbole")

        if self.limit_order.side != self.stop_order.side:
            raise ValueError(
                "Les deux ordres doivent avoir le même côté (achat ou vente)"
            )

        # Ajouter un tag pour identifier les ordres comme partie d'un OCO
        self.limit_order.tags["oco_id"] = self.oco_id
        self.stop_order.tags["oco_id"] = self.oco_id

    def is_active(self) -> bool:
        """Vérifie si l'ordre OCO est toujours actif (aucun des deux ordres n'a été exécuté)."""
        limit_active = self.limit_order.status in (
            OrderStatus.PENDING,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        )
        stop_active = self.stop_order.status in (
            OrderStatus.PENDING,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        )
        return limit_active and stop_active

    def cancel(self) -> bool:
        """Annule les deux ordres."""
        limit_cancelled = self.limit_order.cancel()
        stop_cancelled = self.stop_order.cancel()
        return limit_cancelled or stop_cancelled

    def on_order_filled(self, filled_order_id: str):
        """
        Gère l'exécution d'un des ordres (annule l'autre).

        Args:
            filled_order_id (str): ID de l'ordre exécuté
        """
        if filled_order_id == self.limit_order.order_id:
            self.stop_order.cancel()
            logger.info(f"OCO {self.oco_id} - Ordre limite exécuté, ordre stop annulé")
        elif filled_order_id == self.stop_order.order_id:
            self.limit_order.cancel()
            logger.info(f"OCO {self.oco_id} - Ordre stop exécuté, ordre limite annulé")

    def to_dict(self) -> Dict:
        """Convertit l'ordre OCO en dictionnaire pour sérialisation."""
        return {
            "oco_id": self.oco_id,
            "limit_order": self.limit_order.to_dict(),
            "stop_order": self.stop_order.to_dict(),
            "created_at": self.created_at.isoformat(),
        }


# Fonction utilitaire pour créer un ordre basé sur ses paramètres
def create_order(
    symbol: str,
    side: Union[str, OrderSide],
    order_type: Union[str, OrderType],
    quantity: float,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    time_in_force: Union[str, TimeInForce] = TimeInForce.GTC,
    **kwargs,
) -> Order:
    """
    Crée un ordre du type spécifié.

    Args:
        symbol (str): Symbole de l'actif
        side (Union[str, OrderSide]): Côté de l'ordre (achat ou vente)
        order_type (Union[str, OrderType]): Type d'ordre
        quantity (float): Quantité
        price (float, optional): Prix limite
        stop_price (float, optional): Prix stop
        time_in_force (Union[str, TimeInForce], optional): Durée de validité
        **kwargs: Paramètres supplémentaires

    Returns:
        Order: Instance de l'ordre créé
    """
    # Convertir les enums à partir des strings si nécessaire
    if isinstance(side, str):
        side = OrderSide(side)

    if isinstance(order_type, str):
        order_type = OrderType(order_type)

    if isinstance(time_in_force, str):
        time_in_force = TimeInForce(time_in_force)

    # Créer l'ordre en fonction du type
    if order_type == OrderType.MARKET:
        return MarketOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            time_in_force=time_in_force,
            **kwargs,
        )
    elif order_type == OrderType.LIMIT:
        return LimitOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            **kwargs,
        )
    elif order_type == OrderType.STOP:
        return StopOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs,
        )
    elif order_type == OrderType.STOP_LIMIT:
        return StopLimitOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs,
        )
    elif order_type == OrderType.TRAILING_STOP:
        return TrailingStopOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            time_in_force=time_in_force,
            **kwargs,
        )
    elif order_type == OrderType.ICEBERG:
        return IcebergOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            **kwargs,
        )
    else:
        raise ValueError(f"Type d'ordre non supporté: {order_type}")


# Fonction utilitaire pour créer un ordre OCO
def create_oco_order(
    symbol: str,
    side: Union[str, OrderSide],
    quantity: float,
    limit_price: float,
    stop_price: float,
    stop_limit_price: Optional[float] = None,
    time_in_force: Union[str, TimeInForce] = TimeInForce.GTC,
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
        time_in_force (Union[str, TimeInForce], optional): Durée de validité
        **kwargs: Paramètres supplémentaires

    Returns:
        OCOOrder: Instance de l'ordre OCO créé
    """
    # Créer l'ordre limite
    limit_order = create_order(
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=limit_price,
        time_in_force=time_in_force,
        **kwargs,
    )

    # Créer l'ordre stop ou stop-limite
    if stop_limit_price is not None:
        stop_order = create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LIMIT,
            quantity=quantity,
            price=stop_limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs,
        )
    else:
        stop_order = create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs,
        )

    # Créer et retourner l'ordre OCO
    return OCOOrder(limit_order=limit_order, stop_order=stop_order)
