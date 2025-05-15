"""
Moteur de backtesting avancé pour évaluer des stratégies de trading
avec un haut niveau de réalisme.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import uuid
from copy import deepcopy

from ai_trading.ml.backtesting.execution_model import ExecutionModel, Order, OrderType, OrderSide, OrderStatus
from ai_trading.ml.backtesting.transaction_costs import TransactionCostModel

# Configuration du logging
logger = logging.getLogger(__name__)

class PositionType(Enum):
    """Types de positions possibles."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

@dataclass
class Position:
    """Classe représentant une position de trading."""
    symbol: str
    type: PositionType = PositionType.FLAT
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_time: datetime = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_open(self) -> bool:
        """Vérifie si la position est ouverte."""
        return self.type != PositionType.FLAT and self.quantity > 0

@dataclass
class Trade:
    """Classe représentant un trade (séquence d'ordres liés)."""
    id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    entry_type: PositionType
    quantity: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration: Optional[timedelta] = None
    orders: List[Order] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_open(self) -> bool:
        """Vérifie si le trade est toujours ouvert."""
        return self.exit_time is None
    
    @property
    def profit_factor(self) -> float:
        """Calcule le ratio profit/perte pour ce trade."""
        if self.pnl <= 0:
            return 0.0
        return self.pnl / (self.fees or 1.0)

@dataclass
class BacktestResult:
    """Résultat complet d'un backtest."""
    trades: List[Trade]
    positions: Dict[str, Position]
    equity_curve: pd.Series
    drawdowns: pd.Series
    statistics: Dict
    orders: List[Order]
    market_data: pd.DataFrame
    config: Dict
    metadata: Dict = field(default_factory=dict)
    
    @property
    def total_pnl(self) -> float:
        """Calcule le P&L total du backtest."""
        return sum(trade.pnl for trade in self.trades)
    
    @property
    def win_rate(self) -> float:
        """Calcule le taux de trades gagnants."""
        if not self.trades:
            return 0.0
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        return winning_trades / len(self.trades)
    
    @property
    def max_drawdown(self) -> float:
        """Retourne le drawdown maximum."""
        return self.statistics.get('max_drawdown', 0.0)

class BacktestEngine:
    """
    Moteur de backtesting avancé avec simulation réaliste des exécutions
    d'ordres, gestion du slippage et des coûts de transaction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le moteur de backtesting.
        
        Args:
            config: Configuration du backtest
        """
        self.config = config or {}
        
        # Paramètres par défaut
        self.default_params = {
            "initial_capital": 100000.0,  # Capital initial
            "position_sizing": {
                "type": "fixed",          # Type de sizing (fixed, percent, risk)
                "value": 1000.0,          # Valeur pour le sizing
                "max_position_size": 0.2,  # Taille max de position (% du capital)
                "max_positions": 5         # Nombre max de positions simultanées
            },
            "risk_management": {
                "stop_loss": {
                    "enabled": True,
                    "type": "percent",     # percent ou fixed
                    "value": 0.02          # 2% de perte max par position
                },
                "take_profit": {
                    "enabled": False,
                    "type": "percent",
                    "value": 0.05          # 5% de profit cible
                },
                "trailing_stop": {
                    "enabled": False,
                    "type": "percent",
                    "value": 0.015,        # 1.5% de trailing
                    "activation": 0.01     # Activation à 1% de profit
                }
            },
            "execution": {
                "latency": {
                    "enabled": True,
                    "mean": 0.5,           # Latence moyenne en secondes
                    "std": 0.2             # Écart-type de la latence
                },
                "partial_fills": {
                    "enabled": True,
                    "probability": 0.3     # Probabilité d'exécution partielle
                }
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
                
        # Initialiser les modèles d'exécution et de coûts
        self.execution_model = ExecutionModel(self.config.get("execution", None))
        self.transaction_cost_model = TransactionCostModel(self.config.get("transaction_costs", None))
        
        # État du backtest
        self.reset()
        
    def reset(self):
        """Réinitialise l'état du backtest."""
        self.current_time = None
        self.capital = self.config["initial_capital"]
        self.equity = self.config["initial_capital"]
        self.positions = {}  # Positions par symbole
        self.trades = []     # Historique des trades
        self.orders = []     # Historique des ordres
        self.pending_orders = []  # Ordres en attente d'exécution
        
        # Séries pour les métriques
        self.equity_curve = pd.Series()
        self.drawdowns = pd.Series()
        self.cash_flow = pd.Series()
        
        # Données de marché
        self.market_data = {}
        
    def run(self, 
           market_data: Dict[str, pd.DataFrame], 
           strategy_fn: Callable, 
           strategy_params: Dict = None) -> BacktestResult:
        """
        Exécute le backtest avec la stratégie fournie.
        
        Args:
            market_data: Dictionnaire des données de marché par symbole
            strategy_fn: Fonction de stratégie qui génère des signaux
            strategy_params: Paramètres pour la stratégie
            
        Returns:
            Résultats du backtest
        """
        self.reset()
        
        # Valider les données d'entrée
        if not market_data:
            raise ValueError("Aucune donnée de marché fournie pour le backtest")
            
        # Stocker les données de marché
        self.market_data = {k: df.copy() for k, df in market_data.items()}
        
        # Déterminer les dates communes à toutes les données
        common_dates = self._get_common_dates(market_data)
        if not common_dates:
            raise ValueError("Aucune date commune trouvée dans les données de marché")
            
        # Initialiser les positions pour chaque symbole
        for symbol in market_data.keys():
            self.positions[symbol] = Position(symbol=symbol)
            
        # Boucle principale du backtest
        for i, date in enumerate(common_dates):
            self.current_time = date
            
            # Récupérer les données actuelles pour tous les symboles
            current_data = {symbol: self._get_current_slice(df, date) 
                          for symbol, df in market_data.items()}
            
            # Exécuter les ordres en attente
            self._process_pending_orders(current_data)
            
            # Mettre à jour les positions et évaluer les stops
            self._update_positions(current_data)
            
            # Appeler la fonction de stratégie pour générer des signaux
            if strategy_fn:
                strategy_signals = strategy_fn(
                    current_time=date,
                    market_data=current_data,
                    positions=self.positions,
                    capital=self.capital,
                    equity=self.equity,
                    params=strategy_params
                )
                
                # Traiter les signaux de la stratégie
                self._process_strategy_signals(strategy_signals, current_data)
            
            # Mettre à jour les métriques
            self._update_metrics(date)
            
        # Calculer les statistiques finales
        statistics = self._calculate_statistics()
        
        # Créer et retourner le résultat du backtest
        return BacktestResult(
            trades=self.trades,
            positions=self.positions,
            equity_curve=self.equity_curve,
            drawdowns=self.drawdowns,
            statistics=statistics,
            orders=self.orders,
            market_data=pd.concat({k: pd.DataFrame(v) for k, v in self.market_data.items()}, axis=1),
            config=self.config
        ) 
        
    def _get_common_dates(self, market_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """
        Trouve les dates communes à tous les DataFrames de données de marché.
        
        Args:
            market_data: Dictionnaire des données de marché par symbole
            
        Returns:
            Liste des dates communes, triées par ordre chronologique
        """
        # Extraire les index (dates) de chaque DataFrame
        all_dates = [set(df.index) for df in market_data.values()]
        
        # Trouver l'intersection
        if not all_dates:
            return []
            
        common_dates = set.intersection(*all_dates)
        
        # Trier les dates
        return sorted(list(common_dates))
    
    def _get_current_slice(self, df: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """
        Récupère une tranche de données jusqu'à la date spécifiée.
        
        Args:
            df: DataFrame des données de marché
            date: Date limite (incluse)
            
        Returns:
            DataFrame contenant les données jusqu'à la date spécifiée
        """
        # Sélectionner les données jusqu'à la date courante (incluse)
        return df.loc[:date].copy()
    
    def _process_pending_orders(self, current_data: Dict[str, pd.DataFrame]) -> None:
        """
        Traite les ordres en attente d'exécution.
        
        Args:
            current_data: Données de marché actuelles par symbole
        """
        # Si aucun ordre en attente, ne rien faire
        if not self.pending_orders:
            return
            
        # Traiter chaque ordre en attente
        remaining_orders = []
        for order in self.pending_orders:
            # Vérifier si le symbole de l'ordre existe dans les données actuelles
            if order.symbol not in current_data:
                logger.warning(f"Symbole {order.symbol} non trouvé dans les données de marché. Ordre ignoré.")
                order.status = OrderStatus.REJECTED
                self.orders.append(order)
                continue
                
            # Récupérer les données actuelles pour ce symbole
            symbol_data = current_data[order.symbol]
            
            # Exécuter l'ordre avec le modèle d'exécution
            executed_order = self.execution_model.execute_order(
                order, 
                symbol_data,
                {} # Données de liquidité (à améliorer plus tard)
            )
            
            # Calculer les coûts de transaction
            if executed_order.is_filled() or executed_order.status == OrderStatus.PARTIAL:
                costs = self.transaction_cost_model.calculate_total_cost(
                    executed_order,
                    symbol_data
                )
                executed_order.fees = costs.get("total", 0.0)
                
            # Si l'ordre est toujours actif, le conserver dans la liste des ordres en attente
            if executed_order.is_active():
                remaining_orders.append(executed_order)
            else:
                # Sinon, l'ajouter à l'historique
                self.orders.append(executed_order)
                
                # Mettre à jour la position correspondante
                self._update_position_from_order(executed_order)
                
        # Mettre à jour la liste des ordres en attente
        self.pending_orders = remaining_orders
        
    def _update_position_from_order(self, order: Order) -> None:
        """
        Met à jour une position en fonction d'un ordre exécuté.
        
        Args:
            order: Ordre exécuté
        """
        if not order.is_filled() and order.status != OrderStatus.PARTIAL:
            return
            
        symbol = order.symbol
        
        # Récupérer ou créer la position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            
        position = self.positions[symbol]
        
        # Quantité exécutée
        fill_qty = order.filled_quantity
        fill_price = order.avg_fill_price
        
        # Mise à jour en fonction du type d'ordre
        if order.side == OrderSide.BUY:
            # Si la position était à plat, il s'agit d'une nouvelle position longue
            if position.type == PositionType.FLAT:
                position.type = PositionType.LONG
                position.avg_entry_price = fill_price
                position.entry_time = order.timestamp
                position.quantity = fill_qty
            # Si c'était déjà une position longue, mettre à jour le prix moyen
            elif position.type == PositionType.LONG:
                # Calculer le nouveau prix moyen pondéré
                position.avg_entry_price = (position.avg_entry_price * position.quantity + 
                                          fill_price * fill_qty) / (position.quantity + fill_qty)
                position.quantity += fill_qty
            # Si c'était une position courte, réduire la position
            elif position.type == PositionType.SHORT:
                # Si l'achat est supérieur à la position courte, inverser la position
                if fill_qty > position.quantity:
                    # Calculer le P&L sur la portion qui ferme la position courte
                    close_qty = position.quantity
                    remaining_qty = fill_qty - close_qty
                    
                    # P&L réalisé: différence entre prix d'entrée et de sortie multipliée par la quantité
                    pnl = (position.avg_entry_price - fill_price) * close_qty
                    position.realized_pnl += pnl
                    
                    # Créer un nouveau trade fermé
                    self._record_closed_trade(
                        symbol, PositionType.SHORT, position.entry_time, order.timestamp,
                        position.avg_entry_price, fill_price, close_qty, pnl, position.fees + order.fees
                    )
                    
                    # Ouvrir une nouvelle position longue avec le reste
                    position.type = PositionType.LONG
                    position.avg_entry_price = fill_price
                    position.entry_time = order.timestamp
                    position.quantity = remaining_qty
                    position.fees = order.fees
                else:
                    # Réduire la position courte
                    pnl = (position.avg_entry_price - fill_price) * fill_qty
                    position.realized_pnl += pnl
                    position.quantity -= fill_qty
                    position.fees += order.fees
                    
                    # Si la position est totalement fermée
                    if position.quantity <= 0:
                        # Créer un trade fermé
                        self._record_closed_trade(
                            symbol, PositionType.SHORT, position.entry_time, order.timestamp,
                            position.avg_entry_price, fill_price, fill_qty, pnl, position.fees
                        )
                        
                        # Réinitialiser la position
                        position.type = PositionType.FLAT
                        position.quantity = 0
                        position.avg_entry_price = 0
                        position.entry_time = None
                        position.fees = 0
                        
        elif order.side == OrderSide.SELL:
            # Logique similaire pour les ventes
            if position.type == PositionType.FLAT:
                position.type = PositionType.SHORT
                position.avg_entry_price = fill_price
                position.entry_time = order.timestamp
                position.quantity = fill_qty
            elif position.type == PositionType.SHORT:
                position.avg_entry_price = (position.avg_entry_price * position.quantity + 
                                         fill_price * fill_qty) / (position.quantity + fill_qty)
                position.quantity += fill_qty
            elif position.type == PositionType.LONG:
                if fill_qty > position.quantity:
                    close_qty = position.quantity
                    remaining_qty = fill_qty - close_qty
                    
                    pnl = (fill_price - position.avg_entry_price) * close_qty
                    position.realized_pnl += pnl
                    
                    self._record_closed_trade(
                        symbol, PositionType.LONG, position.entry_time, order.timestamp,
                        position.avg_entry_price, fill_price, close_qty, pnl, position.fees + order.fees
                    )
                    
                    position.type = PositionType.SHORT
                    position.avg_entry_price = fill_price
                    position.entry_time = order.timestamp
                    position.quantity = remaining_qty
                    position.fees = order.fees
                else:
                    pnl = (fill_price - position.avg_entry_price) * fill_qty
                    position.realized_pnl += pnl
                    position.quantity -= fill_qty
                    position.fees += order.fees
                    
                    if position.quantity <= 0:
                        self._record_closed_trade(
                            symbol, PositionType.LONG, position.entry_time, order.timestamp,
                            position.avg_entry_price, fill_price, fill_qty, pnl, position.fees
                        )
                        
                        position.type = PositionType.FLAT
                        position.quantity = 0
                        position.avg_entry_price = 0
                        position.entry_time = None
                        position.fees = 0
        
        # Mettre à jour le capital disponible
        transaction_value = fill_qty * fill_price
        fees = order.fees
        
        if order.side == OrderSide.BUY:
            self.capital -= (transaction_value + fees)
        else:
            self.capital += (transaction_value - fees)
            
    def _record_closed_trade(self, 
                           symbol: str, 
                           trade_type: PositionType, 
                           entry_time: datetime, 
                           exit_time: datetime,
                           entry_price: float, 
                           exit_price: float, 
                           quantity: float, 
                           pnl: float,
                           fees: float) -> None:
        """
        Enregistre un trade fermé dans l'historique.
        
        Args:
            symbol: Symbole du trade
            trade_type: Type de trade (LONG ou SHORT)
            entry_time: Date d'entrée
            exit_time: Date de sortie
            entry_price: Prix d'entrée
            exit_price: Prix de sortie
            quantity: Quantité
            pnl: Profit et perte réalisé
            fees: Frais de transaction
        """
        trade_id = str(uuid.uuid4())
        
        # Calculer le pourcentage de P&L
        if entry_price > 0:
            if trade_type == PositionType.LONG:
                pnl_pct = (exit_price / entry_price) - 1.0
            else:  # SHORT
                pnl_pct = 1.0 - (exit_price / entry_price)
        else:
            pnl_pct = 0.0
            
        # Calculer la durée du trade
        duration = exit_time - entry_time
        
        # Créer l'objet Trade
        trade = Trade(
            id=trade_id,
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_type=trade_type,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=fees,
            duration=duration
        )
        
        # Ajouter à l'historique
        self.trades.append(trade)
        
    def _update_positions(self, current_data: Dict[str, pd.DataFrame]) -> None:
        """
        Met à jour toutes les positions avec les données de marché actuelles.
        
        Args:
            current_data: Données de marché actuelles par symbole
        """
        for symbol, position in self.positions.items():
            if not position.is_open or symbol not in current_data:
                continue
                
            # Récupérer le prix actuel
            current_price = current_data[symbol]['close'].iloc[-1]
            
            # Calculer le P&L non réalisé
            if position.type == PositionType.LONG:
                position.unrealized_pnl = (current_price - position.avg_entry_price) * position.quantity
            else:  # SHORT
                position.unrealized_pnl = (position.avg_entry_price - current_price) * position.quantity
                
            # Vérifier les stop loss et take profit
            self._check_risk_management(symbol, position, current_price)
            
    def _check_risk_management(self, symbol: str, position: Position, current_price: float) -> None:
        """
        Vérifie et applique les règles de gestion des risques (stop loss, take profit).
        
        Args:
            symbol: Symbole de la position
            position: Position à vérifier
            current_price: Prix actuel du marché
        """
        if not position.is_open:
            return
            
        risk_config = self.config["risk_management"]
        
        # Vérifier le stop loss
        if risk_config["stop_loss"]["enabled"]:
            stop_value = risk_config["stop_loss"]["value"]
            stop_type = risk_config["stop_loss"]["type"]
            
            stop_triggered = False
            
            if position.type == PositionType.LONG:
                if stop_type == "percent":
                    stop_price = position.avg_entry_price * (1 - stop_value)
                    if current_price <= stop_price:
                        stop_triggered = True
                else:  # fixed
                    if position.unrealized_pnl <= -stop_value:
                        stop_triggered = True
            else:  # SHORT
                if stop_type == "percent":
                    stop_price = position.avg_entry_price * (1 + stop_value)
                    if current_price >= stop_price:
                        stop_triggered = True
                else:  # fixed
                    if position.unrealized_pnl <= -stop_value:
                        stop_triggered = True
                        
            if stop_triggered:
                # Créer un ordre de fermeture au marché
                order_side = OrderSide.BUY if position.type == PositionType.SHORT else OrderSide.SELL
                stop_order = Order(
                    id=f"stop_{uuid.uuid4()}",
                    symbol=symbol,
                    side=order_side,
                    type=OrderType.MARKET,
                    quantity=position.quantity,
                    timestamp=self.current_time,
                    metadata={"reason": "stop_loss"}
                )
                
                # Ajouter l'ordre à la liste des ordres en attente
                self.pending_orders.append(stop_order)
                
        # Vérifier le take profit
        if risk_config["take_profit"]["enabled"]:
            tp_value = risk_config["take_profit"]["value"]
            tp_type = risk_config["take_profit"]["type"]
            
            tp_triggered = False
            
            if position.type == PositionType.LONG:
                if tp_type == "percent":
                    tp_price = position.avg_entry_price * (1 + tp_value)
                    if current_price >= tp_price:
                        tp_triggered = True
                else:  # fixed
                    if position.unrealized_pnl >= tp_value:
                        tp_triggered = True
            else:  # SHORT
                if tp_type == "percent":
                    tp_price = position.avg_entry_price * (1 - tp_value)
                    if current_price <= tp_price:
                        tp_triggered = True
                else:  # fixed
                    if position.unrealized_pnl >= tp_value:
                        tp_triggered = True
                        
            if tp_triggered:
                # Créer un ordre de fermeture au marché
                order_side = OrderSide.BUY if position.type == PositionType.SHORT else OrderSide.SELL
                tp_order = Order(
                    id=f"tp_{uuid.uuid4()}",
                    symbol=symbol,
                    side=order_side,
                    type=OrderType.MARKET,
                    quantity=position.quantity,
                    timestamp=self.current_time,
                    metadata={"reason": "take_profit"}
                )
                
                # Ajouter l'ordre à la liste des ordres en attente
                self.pending_orders.append(tp_order)
                
        # Vérifier le trailing stop si activé et si la position est en profit
        if risk_config["trailing_stop"]["enabled"]:
            ts_value = risk_config["trailing_stop"]["value"]
            ts_activation = risk_config["trailing_stop"]["activation"]
            ts_type = risk_config.get("trailing_stop", {}).get("type", "percent")
            
            # Vérifier si le trailing stop est déjà activé ou si la position est assez en profit
            is_activated = position.metadata.get("trailing_stop_activated", False)
            
            if position.type == PositionType.LONG:
                profit_pct = (current_price / position.avg_entry_price) - 1
                
                if is_activated or profit_pct >= ts_activation:
                    # Activer le trailing stop
                    if not is_activated:
                        position.metadata["trailing_stop_activated"] = True
                        position.metadata["trailing_stop_level"] = current_price * (1 - ts_value)
                        
                    # Mettre à jour le niveau du trailing stop si le prix augmente
                    new_stop_level = current_price * (1 - ts_value)
                    if new_stop_level > position.metadata.get("trailing_stop_level", 0):
                        position.metadata["trailing_stop_level"] = new_stop_level
                        
                    # Vérifier si le prix a traversé le niveau du trailing stop
                    if current_price <= position.metadata.get("trailing_stop_level", 0):
                        # Créer un ordre de fermeture au marché
                        ts_order = Order(
                            id=f"ts_{uuid.uuid4()}",
                            symbol=symbol,
                            side=OrderSide.SELL,
                            type=OrderType.MARKET,
                            quantity=position.quantity,
                            timestamp=self.current_time,
                            metadata={"reason": "trailing_stop"}
                        )
                        
                        # Ajouter l'ordre à la liste des ordres en attente
                        self.pending_orders.append(ts_order)
            else:  # SHORT
                profit_pct = 1 - (current_price / position.avg_entry_price)
                
                if is_activated or profit_pct >= ts_activation:
                    # Activer le trailing stop
                    if not is_activated:
                        position.metadata["trailing_stop_activated"] = True
                        position.metadata["trailing_stop_level"] = current_price * (1 + ts_value)
                        
                    # Mettre à jour le niveau du trailing stop si le prix diminue
                    new_stop_level = current_price * (1 + ts_value)
                    if new_stop_level < position.metadata.get("trailing_stop_level", float('inf')):
                        position.metadata["trailing_stop_level"] = new_stop_level
                        
                    # Vérifier si le prix a traversé le niveau du trailing stop
                    if current_price >= position.metadata.get("trailing_stop_level", float('inf')):
                        # Créer un ordre de fermeture au marché
                        ts_order = Order(
                            id=f"ts_{uuid.uuid4()}",
                            symbol=symbol,
                            side=OrderSide.BUY,
                            type=OrderType.MARKET,
                            quantity=position.quantity,
                            timestamp=self.current_time,
                            metadata={"reason": "trailing_stop"}
                        )
                        
                        # Ajouter l'ordre à la liste des ordres en attente
                        self.pending_orders.append(ts_order)
                        
    def _process_strategy_signals(self, signals: List[Dict], current_data: Dict[str, pd.DataFrame]) -> None:
        """
        Traite les signaux générés par la stratégie.
        
        Args:
            signals: Liste des signaux à traiter
            current_data: Données de marché actuelles par symbole
        """
        if not signals:
            return
            
        for signal in signals:
            # Vérifier les champs obligatoires
            required_fields = ['type', 'symbol', 'quantity']
            if not all(field in signal for field in required_fields):
                logger.warning(f"Signal invalide, champs manquants: {signal}")
                continue
                
            signal_type = signal['type']
            symbol = signal['symbol']
            quantity = float(signal['quantity'])
            
            # Vérifier que le symbole existe dans les données
            if symbol not in current_data:
                logger.warning(f"Symbole {symbol} non trouvé dans les données de marché.")
                continue
                
            # Créer l'ordre en fonction du type de signal
            if signal_type == 'BUY':
                order_side = OrderSide.BUY
            elif signal_type == 'SELL':
                order_side = OrderSide.SELL
            else:
                logger.warning(f"Type de signal non reconnu: {signal_type}")
                continue
                
            # Déterminer le type d'ordre
            order_type = signal.get('order_type', OrderType.MARKET)
            
            # Créer l'ordre
            order = Order(
                id=f"{signal_type.lower()}_{uuid.uuid4()}",
                symbol=symbol,
                side=order_side,
                type=order_type,
                quantity=quantity,
                price=signal.get('price', None),
                stop_price=signal.get('stop_price', None),
                timestamp=self.current_time,
                metadata=signal.get('metadata', {})
            )
            
            # Ajouter les stop-loss et take-profit à l'ordre
            if 'stop_loss' in signal:
                order.metadata['stop_loss'] = signal['stop_loss']
            if 'take_profit' in signal:
                order.metadata['take_profit'] = signal['take_profit']
                
            # Ajouter l'ordre à la file d'attente
            self.pending_orders.append(order)
            
    def _update_metrics(self, date: datetime) -> None:
        """
        Met à jour les métriques du backtest.
        
        Args:
            date: Date courante du backtest
        """
        # Calculer l'équité totale (capital + valeur des positions)
        total_equity = self.capital
        
        for position in self.positions.values():
            if position.is_open:
                total_equity += position.unrealized_pnl
                
        # Mettre à jour la courbe d'équité
        self.equity_curve[date] = total_equity
        
        # Calculer et mettre à jour le drawdown
        peak = self.equity_curve.cummax()
        self.drawdowns[date] = (self.equity_curve[date] - peak[date]) / peak[date] if peak[date] > 0 else 0.0
        
        # Mettre à jour l'équité courante
        self.equity = total_equity
        
    def _calculate_statistics(self) -> Dict:
        """
        Calcule les statistiques de performance du backtest.
        
        Returns:
            Dictionnaire contenant les statistiques
        """
        stats = {}
        
        # Si pas de trades ou pas d'équité, retourner des statistiques vides
        if not self.trades or self.equity_curve.empty:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0
            }
            
        # Nombre total de trades
        stats["total_trades"] = len(self.trades)
        
        # Nombre de trades gagnants et perdants
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        stats["winning_trades"] = len(winning_trades)
        stats["losing_trades"] = len(losing_trades)
        
        # Taux de réussite
        stats["win_rate"] = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        # Profit total
        stats["total_profit"] = sum(t.pnl for t in winning_trades)
        stats["total_loss"] = sum(t.pnl for t in losing_trades)
        stats["net_profit"] = stats["total_profit"] + stats["total_loss"]
        
        # Profit factor
        stats["profit_factor"] = abs(stats["total_profit"] / stats["total_loss"]) if stats["total_loss"] < 0 else 0.0
        
        # Profits moyens
        stats["avg_profit"] = stats["total_profit"] / len(winning_trades) if winning_trades else 0.0
        stats["avg_loss"] = stats["total_loss"] / len(losing_trades) if losing_trades else 0.0
        stats["avg_trade"] = stats["net_profit"] / len(self.trades) if self.trades else 0.0
        
        # Rendement et drawdown
        initial_equity = self.config["initial_capital"]
        final_equity = self.equity_curve.iloc[-1]
        
        stats["total_return"] = (final_equity / initial_equity) - 1.0
        stats["max_drawdown"] = self.drawdowns.min() if not self.drawdowns.empty else 0.0
        
        # Calculer le ratio de Sharpe
        if len(self.equity_curve) > 1:
            # Calculer les rendements quotidiens
            returns = self.equity_curve.pct_change().dropna()
            
            # Calculer le Sharpe Ratio (supposons un taux sans risque de 0)
            risk_free_rate = 0.0
            excess_returns = returns - risk_free_rate
            
            if len(excess_returns) > 0 and excess_returns.std() > 0:
                # Supposons une année de 252 jours de trading
                stats["sharpe_ratio"] = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            else:
                stats["sharpe_ratio"] = 0.0
                
            # Volatilité annualisée
            stats["volatility"] = returns.std() * np.sqrt(252)
            
            # Rendement annualisé
            if len(self.equity_curve) > 252:
                years = len(self.equity_curve) / 252
                stats["annualized_return"] = (final_equity / initial_equity) ** (1 / years) - 1
            else:
                stats["annualized_return"] = stats["total_return"]
        else:
            stats["sharpe_ratio"] = 0.0
            stats["volatility"] = 0.0
            stats["annualized_return"] = 0.0
            
        # Durée moyenne des trades
        if self.trades:
            trade_durations = [t.duration.total_seconds() / 3600 for t in self.trades if t.duration]  # en heures
            stats["avg_trade_duration"] = np.mean(trade_durations) if trade_durations else 0.0
            
        # Ratio rendement/drawdown (MAR)
        stats["mar_ratio"] = abs(stats["annualized_return"] / stats["max_drawdown"]) if stats["max_drawdown"] != 0 else 0.0
            
        return stats 

# Classes nécessaires pour les tests

class SlippageModel:
    """
    Modèle de slippage pour simuler les écarts entre prix théoriques et exécution.
    """
    
    def __init__(self, slippage_factor=0.001, volume_impact=False):
        """
        Initialise le modèle de slippage.
        
        Args:
            slippage_factor: Facteur de slippage (ex: 0.001 pour 0.1%)
            volume_impact: Si True, considère l'impact du volume sur le slippage
        """
        self.slippage_factor = slippage_factor
        self.volume_impact = volume_impact
    
    def calculate_slippage(self, order, current_time, market_data):
        """
        Calcule le slippage pour un ordre donné.
        
        Args:
            order: Dictionnaire contenant les détails de l'ordre
            current_time: Horodatage actuel
            market_data: DataFrame contenant les données de marché
        
        Returns:
            Montant du slippage en unités de prix
        """
        # Slippage de base
        base_slippage = market_data['close'].iloc[-1] * self.slippage_factor
        
        # Si l'impact du volume est activé, ajuster le slippage
        if self.volume_impact and 'quantity' in order:
            # Formule simplifiée: plus le volume est important, plus le slippage est élevé
            quantity = abs(order['quantity'])
            volume_factor = min(1.0, quantity / (market_data['volume'].iloc[-1] * 0.01))
            return base_slippage * (1 + volume_factor)
        
        return base_slippage


class TransactionCostModel:
    """
    Modèle de coûts de transaction pour simuler les frais et commissions.
    """
    
    def __init__(self, commission_rate=0.001, fixed_commission=0.0):
        """
        Initialise le modèle de coûts de transaction.
        
        Args:
            commission_rate: Taux de commission en pourcentage
            fixed_commission: Commission fixe par ordre
        """
        self.commission_rate = commission_rate
        self.fixed_commission = fixed_commission
    
    def calculate_commission(self, order, executed_price, executed_quantity):
        """
        Calcule la commission pour une exécution donnée.
        
        Args:
            order: Ordre exécuté (peut être None)
            executed_price: Prix d'exécution
            executed_quantity: Quantité exécutée
        
        Returns:
            Montant de la commission
        """
        # Calculer le montant de la transaction
        transaction_value = executed_price * executed_quantity
        
        # Calculer la commission
        commission = transaction_value * self.commission_rate + self.fixed_commission
        
        return commission


class StressScenario:
    """
    Scénario de stress pour tester la robustesse des stratégies.
    """
    
    def __init__(self, name, price_shock, volatility_multiplier, volume_multiplier, duration, start_date):
        """
        Initialise un scénario de stress.
        
        Args:
            name: Nom du scénario
            price_shock: Choc de prix (ex: -0.3 pour -30%)
            volatility_multiplier: Multiplicateur de volatilité
            volume_multiplier: Multiplicateur de volume
            duration: Durée du scénario (timedelta)
            start_date: Date de début du scénario
        """
        self.name = name
        self.price_shock = price_shock
        self.volatility_multiplier = volatility_multiplier
        self.volume_multiplier = volume_multiplier
        self.duration = duration
        self.start_date = start_date
        
    def apply_to_data(self, market_data):
        """
        Applique le scénario de stress aux données de marché.
        
        Args:
            market_data: Dictionnaire de DataFrames contenant les données par symbole
        
        Returns:
            Dictionnaire de DataFrames modifiés
        """
        # Créer une copie des données pour ne pas modifier l'original
        stressed_data = {symbol: df.copy() for symbol, df in market_data.items()}
        
        # Définir la période de stress
        end_date = self.start_date + self.duration
        
        # Appliquer le scénario à chaque symbole
        for symbol, df in stressed_data.items():
            # Filtrer les données dans la période de stress
            stress_mask = (df.index >= self.start_date) & (df.index <= end_date)
            
            if not any(stress_mask):
                continue
                
            # Appliquer le choc de prix
            if self.price_shock != 0:
                shock_factor = 1 + self.price_shock
                df.loc[stress_mask, 'open'] *= shock_factor
                df.loc[stress_mask, 'high'] *= shock_factor
                df.loc[stress_mask, 'low'] *= shock_factor
                df.loc[stress_mask, 'close'] *= shock_factor
            
            # Augmenter la volatilité
            if self.volatility_multiplier != 1:
                # Calculer l'écart moyen high-low et l'amplifier
                avg_range = df['high'] - df['low']
                additional_range = avg_range * (self.volatility_multiplier - 1) / 2
                
                df.loc[stress_mask, 'high'] += additional_range[stress_mask]
                df.loc[stress_mask, 'low'] -= additional_range[stress_mask]
            
            # Modifier le volume
            if self.volume_multiplier != 1:
                df.loc[stress_mask, 'volume'] *= self.volume_multiplier
        
        return stressed_data


class OrderExecutionModel:
    """
    Modèle d'exécution d'ordres pour simuler le comportement réel des ordres.
    """
    
    def __init__(self, slippage_model=None, transaction_cost_model=None):
        """
        Initialise le modèle d'exécution d'ordres.
        
        Args:
            slippage_model: Modèle de slippage à utiliser
            transaction_cost_model: Modèle de coûts de transaction à utiliser
        """
        self.slippage_model = slippage_model or SlippageModel()
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
    
    def execute_order(self, order, current_time, market_data):
        """
        Exécute un ordre sur la base des conditions de marché actuelles.
        
        Args:
            order: Dictionnaire contenant les détails de l'ordre
            current_time: Horodatage actuel
            market_data: DataFrame contenant les données de marché
        
        Returns:
            Dictionnaire avec les détails de l'exécution ou None si non exécuté
        """
        if market_data.empty:
            return None
        
        # Traiter différents types d'ordres
        if order['type'] == 'market':
            return self._execute_market_order(order, current_time, market_data)
        elif order['type'] == 'limit':
            return self._execute_limit_order(order, current_time, market_data)
        else:
            # Types d'ordres non supportés
            return None
    
    def _execute_market_order(self, order, current_time, market_data):
        """Exécute un ordre au marché."""
        close_price = market_data['close'].iloc[-1]
        
        # Calculer le slippage
        slippage = self.slippage_model.calculate_slippage(order, current_time, market_data)
        
        # Appliquer le slippage selon le sens de l'ordre
        if order['action'] == 'buy':
            executed_price = close_price + slippage
        else:
            executed_price = close_price - slippage
        
        # Calculer la commission
        commission = self.transaction_cost_model.calculate_commission(
            order, executed_price, order['quantity']
        )
        
        # Créer le résultat d'exécution
        return {
            'execution_time': current_time,
            'executed_price': executed_price,
            'executed_quantity': order['quantity'],
            'commission': commission,
            'slippage': slippage
        }
    
    def _execute_limit_order(self, order, current_time, market_data):
        """Exécute un ordre limite si les conditions sont remplies."""
        high_price = market_data['high'].iloc[-1]
        low_price = market_data['low'].iloc[-1]
        
        # Vérifier si l'ordre limite est déclenché
        if (order['action'] == 'buy' and order['price'] >= low_price) or \
           (order['action'] == 'sell' and order['price'] <= high_price):
            
            # Calculer la commission
            commission = self.transaction_cost_model.calculate_commission(
                order, order['price'], order['quantity']
            )
            
            # Créer le résultat d'exécution
            return {
                'execution_time': current_time,
                'executed_price': order['price'],
                'executed_quantity': order['quantity'],
                'commission': commission,
                'slippage': 0.0  # Pas de slippage pour les ordres limites
            }
        
        # L'ordre limite n'est pas déclenché
        return None 