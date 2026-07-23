import pandas as pd

from ai_trading.rl.trade_ledger import FifoTradeLedger


def test_fifo_ledger_calculates_net_pnl_and_win_rate():
    ledger = FifoTradeLedger()
    ledger.buy(pd.Timestamp("2026-01-01"), 2, 100, 2)
    ledger.sell(pd.Timestamp("2026-01-02"), 1, 120, 1, "take_profit")
    ledger.sell(pd.Timestamp("2026-01-03"), 1, 90, 1, "stop_loss")
    trades = ledger.dataframe()
    assert len(trades) == 2
    assert trades.iloc[0]["pnl_net"] == 18
    assert trades.iloc[1]["pnl_net"] == -12
    assert trades["outcome"].tolist() == ["win", "loss"]
    assert ledger.metrics(min_closed_trades=2)["win_rate"] == 0.5


def test_fifo_ledger_exposes_remaining_open_lots():
    ledger = FifoTradeLedger()
    ledger.buy(pd.Timestamp("2026-01-01"), 2, 100, 2)
    ledger.sell(pd.Timestamp("2026-01-02"), 0.5, 120, 0.5, "partial_take_profit")
    open_lots = ledger.open_lots_dataframe()
    assert len(open_lots) == 1
    assert open_lots.iloc[0]["quantity"] == 1.5
