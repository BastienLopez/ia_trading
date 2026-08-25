"""Tests du garde-fou anti-cherry-picking P3 seul/P3+P4."""

from ai_trading.scripts.compare_p3_p4_validation import compare


def _metrics(total_return, buy_hold, profit_factor=1.2, expectancy=2.0):
    return {
        "strategy": {"total_return": total_return},
        "buy_and_hold": {"total_return": buy_hold},
        "closed_trade_metrics": {"profit_factor": profit_factor, "expectancy": expectancy},
    }


def _manifest(p4):
    return {
        "evaluation_mode": "validation", "transaction_fee": 0.001,
        "validation_seeds": [73, 211, 997],
        "candidate_ids": ["candidate"], "p4": {"enabled": p4},
        "assets": ["BTC/USDT"], "timeframe": "1h", "candles": {"train": 400},
        "initial_balance": 10000.0, "allow_short_cli": True, "short_limits_cli": {"max_short_exposure": 0.25},
    }


def test_comparison_uses_only_preregistered_windows_and_rejects_non_compliant_promotion():
    protocol = {
        "comparison_id": "test", "evaluation_mode": "validation", "windows": [4, 5],
        "validation_seeds": [73, 211, 997], "transaction_fee": 0.001, "candidate_id": "candidate",
        "promotion_rule": {
            "minimum_p4_better_windows": 2, "require_positive_median_delta": True,
            "require_p4_above_buy_and_hold_median": True,
            "require_positive_profit_factor_and_expectancy_each_window": True,
        },
    }
    p3_rows = [{"window": 4, "validation_candidates": [{"candidate_id": "candidate", "metrics": _metrics(.01, .0)}]},
               {"window": 5, "validation_candidates": [{"candidate_id": "candidate", "metrics": _metrics(.02, .0)}]}]
    p4_rows = [{"window": 4, "validation_candidates": [{"candidate_id": "candidate", "metrics": _metrics(.03, .0)}]},
               {"window": 5, "validation_candidates": [{"candidate_id": "candidate", "metrics": _metrics(.01, .0, .9, -1)}]}]

    result = compare(protocol, (_manifest(False), p3_rows), (_manifest(True), p4_rows))

    assert result["valid"] is True
    assert result["promotion"] is False
    assert any("profit factor ou expectancy" in reason for reason in result["promotion_rejection_reasons"])


def test_comparison_refuses_windows_not_in_the_preregistered_protocol():
    protocol = {
        "comparison_id": "test", "evaluation_mode": "validation", "windows": [4],
        "validation_seeds": [73, 211, 997], "transaction_fee": 0.001, "candidate_id": "candidate",
        "promotion_rule": {},
    }
    rows = [{"window": 6, "validation_candidates": [{"candidate_id": "candidate", "metrics": _metrics(.01, .0)}]}]
    result = compare(protocol, (_manifest(False), rows), (_manifest(True), rows))
    assert result == {"protocol": "test", "valid": False,
                      "errors": ["fenêtres exécutées différentes du protocole pré-enregistré"], "promotion": False}
