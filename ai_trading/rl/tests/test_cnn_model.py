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


def test_cnn_initialization(cnn_model):
    assert isinstance(cnn_model, PriceGraphCNN)
    assert cnn_model.input_channels == 5
    assert cnn_model.output_dim == 3
    assert len(cnn_model.conv_layers) == 3


def test_cnn_forward(cnn_model, sample_data):
    output, attention_weights = cnn_model(sample_data)

    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, torch.Tensor)
    assert output.shape == (32, 3)  # (batch_size, output_dim)
    assert attention_weights.shape == (32, 100, 1)  # (batch_size, seq_len, 1)


def test_cnn_gradient_flow(cnn_model, sample_data):
    output, _ = cnn_model(sample_data)
    loss = output.sum()
    loss.backward()

    for name, param in cnn_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


def test_hybrid_initialization(hybrid_model):
    assert isinstance(hybrid_model, HybridCNNAttention)
    assert isinstance(hybrid_model.cnn, PriceGraphCNN)
    assert hybrid_model.output_layer.out_features == 3


def test_hybrid_forward(hybrid_model, sample_data):
    output, attention_weights = hybrid_model(sample_data)

    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, dict)
    assert output.shape == (32, 3)  # (batch_size, output_dim)

    # Vérifier les poids d'attention
    assert "cnn" in attention_weights
    assert "temporal" in attention_weights
    assert attention_weights["cnn"].shape == (32, 100, 1)  # (batch_size, seq_len, 1)
    assert attention_weights["temporal"].shape == (
        32,
        4,
        1,
        1,
    )  # (batch_size, num_heads, seq_len, seq_len)


def test_hybrid_with_mask(hybrid_model, sample_data):
    mask = torch.ones(32, 1, 100)  # (batch_size, 1, seq_len)
    mask[:, :, 50:] = 0  # Masquer la seconde moitié

    output, attention_weights = hybrid_model(sample_data, mask)

    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, dict)
    assert output.shape == (32, 3)

    # Vérifier que le masque a été appliqué
    temporal_attention = attention_weights["temporal"]
    assert torch.all(temporal_attention[:, :, :, 50:] == 0)


def test_hybrid_gradient_flow(hybrid_model, sample_data):
    output, _ = hybrid_model(sample_data)
    loss = output.sum()
    loss.backward()

    for name, param in hybrid_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


def test_different_sequence_lengths(hybrid_model):
    for seq_len in [50, 100, 150]:
        data = torch.randn(32, 5, seq_len)
        output, attention_weights = hybrid_model(data)
        assert output.shape == (32, 3)
        assert attention_weights["cnn"].shape == (32, seq_len, 1)
