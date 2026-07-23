from ai_trading.strategies.arbitrage.statistical_arbitrage import StatisticalArbitrageStrategy


def test_statistical_arbitrage_detects_profitable_pair_and_labels_paper_execution():
    strategy = StatisticalArbitrageStrategy(
        pairs=[("BTC", "ETH")], entry_threshold=2.0, transaction_fee=0.001
    )
    strategy.pair_models["BTC_ETH"] = {
        "asset1": "BTC",
        "asset2": "ETH",
        "alpha": 0.0,
        "beta": 1.0,
        "std_dev": 1.0,
        "half_life": 4.0,
        "pvalue": 0.001,
    }
    opportunities = strategy.find_opportunities(
        {"BTC": {"close": 100.0}, "ETH": {"close": 104.0}}
    )

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity["direction"] == "short_long"
    assert strategy.calculate_profit_after_fees(opportunity) > strategy.min_profit_threshold
    execution = strategy.execute_arbitrage(opportunity)
    assert execution["status"] == "executed"
    assert execution["execution_details"]["simulated"] is True
