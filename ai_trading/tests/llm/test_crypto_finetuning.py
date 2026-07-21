import pandas as pd
import pytest
from unittest.mock import patch

from ai_trading.llm.sentiment_analysis.crypto_finetuning import (
    chronological_split,
    dataset_fingerprint,
    load_crypto_sentiment_dataset,
    load_dlt_sentiment_news,
    validate_training_dataset,
)


def test_crypto_dataset_is_normalized_deduplicated_and_temporally_split(tmp_path):
    source = tmp_path / "crypto.csv"
    pd.DataFrame(
        {
            "title": ["BTC rises", "BTC rises", "ETH falls"] * 4,
            "body": ["demand", "demand", "liquidations"] * 4,
            "global_sentiment_label": ["positive", "positive", "negative"] * 4,
            "published_at": pd.date_range("2025-01-01", periods=12, freq="h"),
        }
    ).to_csv(source, index=False)

    dataset = load_crypto_sentiment_dataset([source])
    train, validation = chronological_split(pd.concat([dataset] * 5, ignore_index=True))

    assert set(dataset["label_name"]) == {"positive", "negative"}
    assert dataset_fingerprint(dataset) == dataset_fingerprint(dataset.copy())
    assert train["published_at"].max() <= validation["published_at"].min()


def test_dlt_loader_uses_documented_schema_and_label_mapping():
    source = pd.DataFrame(
        {
            "text": ["flat", "bearish", "bullish", "bullish"],
            "market_direction": [0, 1, 2, 2],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03"],
        }
    )
    with patch("datasets.load_dataset") as load_dataset:
        load_dataset.return_value.to_pandas.return_value = source
        dataset = load_dlt_sentiment_news()

    assert dataset["label_name"].tolist() == ["neutral", "negative", "positive"]
    assert dataset["label"].tolist() == [1, 0, 2]


def test_dataset_validation_rejects_missing_class_or_insufficient_sample():
    dataset = pd.DataFrame({"label_name": ["positive"] * 1_000})
    with pytest.raises(ValueError, match="Classes absentes"):
        validate_training_dataset(dataset)
