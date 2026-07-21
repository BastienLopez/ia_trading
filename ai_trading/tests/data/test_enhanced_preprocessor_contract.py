import unittest

import numpy as np
import pandas as pd

from ai_trading.utils.enhanced_preprocessor import (
    EnhancedMarketDataPreprocessor,
    EnhancedTextDataPreprocessor,
)


class EnhancedPreprocessorContractTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = EnhancedMarketDataPreprocessor(scaling="minmax")

    def test_cleaning_never_backfills_a_future_value(self):
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        raw = pd.DataFrame({"close": [np.nan, 10.0, 20.0], "volume": [1.0, 2.0, 3.0]}, index=index)

        cleaned = self.preprocessor.clean_market_data(raw)

        self.assertEqual(cleaned.index.min(), index[1])
        self.assertNotIn(index[0], cleaned.index)
        self.assertEqual(cleaned.loc[index[1], "close"], 10.0)

    def test_split_normalization_fits_only_the_train_period(self):
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        raw = pd.DataFrame({"close": [1.0, 2.0, 100.0, 200.0], "source": ["fixture"] * 4}, index=index)
        train, validation, test = raw.iloc[:2], raw.iloc[2:3], raw.iloc[3:]

        normalized_train, normalized_validation, normalized_test = self.preprocessor.normalize_splits(
            train, validation, test
        )

        self.assertEqual(normalized_train["close"].tolist(), [0.0, 1.0])
        self.assertGreater(normalized_validation["close"].iloc[0], 1.0)
        self.assertGreater(normalized_test["close"].iloc[0], normalized_validation["close"].iloc[0])
        self.assertEqual(normalized_validation["source"].iloc[0], "fixture")

    def test_full_market_pipeline_outputs_finite_features_without_optional_nlp_dependencies(self):
        index = pd.date_range("2024-01-01", periods=80, freq="D")
        close = np.linspace(100.0, 180.0, len(index))
        raw = pd.DataFrame(
            {
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": np.linspace(1_000.0, 2_000.0, len(index)),
                "market_cap": np.linspace(1_000_000.0, 2_000_000.0, len(index)),
            },
            index=index,
        )

        processed = self.preprocessor.preprocess_market_data(raw)

        self.assertIsNotNone(processed)
        self.assertGreater(len(processed), 0)
        self.assertIn("rsi_14", processed.columns)
        self.assertTrue(np.isfinite(processed.select_dtypes(include=[np.number]).to_numpy()).all())

    def test_text_preprocessing_remains_available_without_network_downloads(self):
        preprocessor = EnhancedTextDataPreprocessor(language="english")

        frame = preprocessor.preprocess_social_data(
            [{"text": "Bitcoin moves today", "created_at": "2024-01-01"}]
        )

        self.assertEqual(frame.loc[0, "clean_text"], "bitcoin moves today")
        self.assertIsInstance(frame.loc[0, "tokens"], list)


if __name__ == "__main__":
    unittest.main()
