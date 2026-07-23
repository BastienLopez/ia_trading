"""Ledger FIFO auditable pour les fills d'un backtest long-only."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClosedTrade:
    entry_time: datetime
    exit_time: datetime
    quantity: float
    entry_price: float
    exit_price: float
    entry_fee: float
    exit_fee: float
    pnl_net: float
    return_pct: float
    duration_seconds: float
    exit_reason: str
    outcome: str


class FifoTradeLedger:
    """Associe chaque vente aux lots achetés les plus anciens, sans PnL fictif."""

    def __init__(self) -> None:
        self._lots: deque[dict] = deque()
        self.closed: list[ClosedTrade] = []

    def buy(self, timestamp, quantity: float, price: float, fee: float) -> None:
        if quantity <= 0 or price <= 0 or fee < 0:
            raise ValueError("Fill d'achat invalide")
        self._lots.append({"time": pd.Timestamp(timestamp), "quantity": float(quantity), "price": float(price), "fee": float(fee)})

    def sell(self, timestamp, quantity: float, price: float, fee: float, reason: str) -> list[ClosedTrade]:
        if quantity <= 0 or price <= 0 or fee < 0:
            raise ValueError("Fill de vente invalide")
        remaining = float(quantity)
        closed: list[ClosedTrade] = []
        while remaining > 1e-12 and self._lots:
            lot = self._lots[0]
            matched = min(remaining, lot["quantity"])
            entry_fee = lot["fee"] * matched / lot["quantity"]
            exit_fee = fee * matched / quantity
            cost = matched * lot["price"] + entry_fee
            proceeds = matched * price - exit_fee
            pnl_net = proceeds - cost
            trade = ClosedTrade(
                entry_time=lot["time"], exit_time=pd.Timestamp(timestamp), quantity=matched,
                entry_price=lot["price"], exit_price=float(price), entry_fee=entry_fee,
                exit_fee=exit_fee, pnl_net=pnl_net, return_pct=(proceeds / cost - 1.0),
                duration_seconds=(pd.Timestamp(timestamp) - lot["time"]).total_seconds(), exit_reason=reason,
                outcome="win" if pnl_net > 0 else "loss" if pnl_net < 0 else "break_even",
            )
            closed.append(trade)
            self.closed.append(trade)
            lot["quantity"] -= matched
            lot["fee"] -= entry_fee
            remaining -= matched
            if lot["quantity"] <= 1e-12:
                self._lots.popleft()
        if remaining > 1e-9:
            raise ValueError("Vente supérieure à la position FIFO")
        return closed

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(trade) for trade in self.closed])

    def open_lots_dataframe(self) -> pd.DataFrame:
        """Expose les lots FIFO encore ouverts pour l'audit de fin de run."""
        return pd.DataFrame(list(self._lots), columns=["time", "quantity", "price", "fee"])

    def metrics(self, min_closed_trades: int = 30) -> dict:
        pnls = np.asarray([trade.pnl_net for trade in self.closed], dtype=float)
        if not len(pnls):
            return {"closed_trade_count": 0, "win_rate": None, "profit_factor": None, "average_win": None, "average_loss": None, "expectancy": None, "minimum_trade_count_met": False}
        wins, losses = pnls[pnls > 0], pnls[pnls < 0]
        gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
        return {
            "closed_trade_count": int(len(pnls)), "win_rate": float(len(wins) / len(pnls)),
            "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
            "average_win": float(wins.mean()) if len(wins) else 0.0,
            "average_loss": float(losses.mean()) if len(losses) else 0.0,
            "expectancy": float(pnls.mean()), "minimum_trade_count_met": bool(len(pnls) >= min_closed_trades),
        }
