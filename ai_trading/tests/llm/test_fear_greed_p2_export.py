import pandas as pd
import pytest

from ai_trading.scripts.export_fear_greed_p2_observations import observations_from_payload


def test_fear_greed_payload_is_converted_to_real_p2_contract_for_each_asset():
    payload = {"data": [{"timestamp": "1713139200", "value": "25"}, {"timestamp": "1713225600", "value": "75"}]}
    observations = observations_from_payload(payload, ["BTC/USDT", "ETH/USDT"], "1d")
    assert set(observations.columns) >= {"timestamp", "asset", "timeframe", "sentiment_score", "quality", "source"}
    assert set(observations["asset"]) == {"BTC", "ETH"}
    assert observations["sentiment_score"].tolist() == [-0.5, 0.5, -0.5, 0.5]
    assert observations["source"].nunique() == 1


def test_fear_greed_rejects_intraday_interpolation():
    with pytest.raises(ValueError, match="quotidien"):
        observations_from_payload({"data": [{"timestamp": "1713139200", "value": "50"}]}, ["BTC/USDT"], "1h")
