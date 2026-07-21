import unittest

import pandas as pd

from ai_trading.data.quality import MarketDataQualityError, validate_ohlcv_frame
from ai_trading.validation.temporal_cross_validator import TemporalCrossValidator


def valid_frame(rows=12):
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "open": [100 + index for index in range(rows)],
            "high": [101 + index for index in range(rows)],
            "low": [99 + index for index in range(rows)],
            "close": [100.5 + index for index in range(rows)],
            "volume": [10 + index for index in range(rows)],
        },
        index=index,
    )


class MarketQualityTests(unittest.TestCase):
    def test_valid_ohlcv_is_accepted(self):
        validate_ohlcv_frame(valid_frame())

    def test_duplicate_or_incoherent_ohlcv_is_rejected(self):
        duplicate = valid_frame()
        duplicate.index = list(duplicate.index[:-1]) + [duplicate.index[-2]]
        with self.assertRaises(MarketDataQualityError):
            validate_ohlcv_frame(duplicate)

        incoherent = valid_frame()
        incoherent.loc[incoherent.index[0], "high"] = 1
        with self.assertRaises(MarketDataQualityError):
            validate_ohlcv_frame(incoherent)


class TemporalValidationTests(unittest.TestCase):
    def test_walk_forward_splits_progress_without_overlap_or_leakage(self):
        frame = valid_frame(rows=100)
        splits = list(TemporalCrossValidator(n_splits=3, train_ratio=0.6, gap_ratio=0.2).split(frame))
        self.assertEqual(len(splits), 3)
        starts = []
        for train_indices, test_indices in splits:
            self.assertLess(max(train_indices), min(test_indices))
            self.assertGreater(min(test_indices) - max(train_indices), 1)
            self.assertEqual(set(train_indices).intersection(test_indices), set())
            starts.append(min(test_indices))
        self.assertEqual(starts, sorted(set(starts)))

    def test_invalid_split_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            TemporalCrossValidator(n_splits=0)
        with self.assertRaises(ValueError):
            TemporalCrossValidator(train_ratio=0.8, gap_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
