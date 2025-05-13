import pytest
import torch

from ai_trading.rl.models.cnn_model import HybridCNNAttention, PriceGraphCNN


@pytest.fixture
def sample_data():
    batch_size = 32
    seq_len = 100
    input_channels = 5  # OHLCV
    return torch.randn(batch_size, input_channels, seq_len)


@pytest.fixture
def cnn_model():
    return PriceGraphCNN(
        input_channels=5,
        output_dim=3,
        kernel_sizes=[3, 5, 7],
        n_filters=[64, 128, 256],
        dropout=0.2,
    )


@pytest.fixture
def hybrid_model():
    return HybridCNNAttention(
        input_channels=5,
        hidden_dim=128,
        output_dim=3,
        cnn_kernel_sizes=[3, 5, 7],
        cnn_filters=[64, 128, 256],
        num_heads=4,
        dropout=0.2,
    )
