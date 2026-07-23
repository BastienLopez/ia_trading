import numpy as np
import pandas as pd

from ai_trading.dashboard.trade_analysis import TradeAnalyzer
from ai_trading.dashboard.visualization_3d import Visualizer3D


def test_dashboard_3d_and_post_trade_outputs_are_populated_and_serializable(tmp_path):
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    market = pd.DataFrame(
        {
            "close": np.linspace(100, 115, 30),
            "open": np.linspace(99, 114, 30),
            "high": np.linspace(101, 116, 30),
            "low": np.linspace(98, 113, 30),
            "rsi": np.tile([35.0, 45.0, 55.0, 65.0, 75.0], 6),
            "macd": np.repeat([-2.0, -1.0, 1.0, 2.0, 3.0, 4.0], 5),
            "volatility": np.linspace(0.1, 0.3, 30),
            "momentum": np.sin(np.linspace(0, 3, 30)),
            "atr": np.linspace(1, 2, 30),
        },
        index=index,
    )
    trades = pd.DataFrame(
        {
            "entry_time": index[:12],
            "exit_time": index[1:13],
            "symbol": np.tile(["BTC", "ETH", "SOL"], 4),
            "direction": np.tile(["buy", "sell"], 6),
            "entry_price": np.linspace(100, 120, 12),
            "exit_price": np.linspace(101, 119, 12),
            "quantity": np.linspace(0.1, 1.2, 12),
            "profit": np.array([-8, -3, 2, 5, -1, 4, 9, -2, 6, 3, -4, 7]),
            "status": "closed",
            "entry_volatility": np.linspace(0.1, 0.3, 12),
        }
    )
    visualizer = Visualizer3D()
    figures = [
        visualizer.create_multi_indicator_surface(market, "rsi", "macd", "close"),
        visualizer.create_portfolio_trajectory(market, n_components=3),
        visualizer.create_trade_clusters_3d(trades, n_clusters=3),
        TradeAnalyzer().create_performance_summary(trades),
        TradeAnalyzer().create_win_loss_analysis(trades),
        TradeAnalyzer().create_trade_attribution(trades, market.reset_index(names="timestamp")),
    ]

    for number, figure in enumerate(figures):
        assert len(figure.data) > 0
        path = tmp_path / f"dashboard_{number}.html"
        figure.write_html(path)
        assert path.stat().st_size > 10_000
