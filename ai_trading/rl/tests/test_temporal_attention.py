import pytest
import torch

from ai_trading.rl.models.temporal_attention import (
    TemporalAttention,
    TemporalAttentionModel,
)


@pytest.fixture
def sample_data():
    batch_size = 32
    seq_len = 50
    input_dim = 64
    return torch.randn(batch_size, seq_len, input_dim)


@pytest.fixture
def attention_model():
    return TemporalAttention(
        input_dim=64, hidden_dim=128, num_heads=4, dropout=0.1, max_len=100
    )


@pytest.fixture
def full_model():
    return TemporalAttentionModel(
        input_dim=64,
        hidden_dim=128,
        output_dim=3,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_len=100,
    )


def test_temporal_attention_initialization(attention_model):
    assert isinstance(attention_model, TemporalAttention)
    assert attention_model.input_dim == 64
    assert attention_model.hidden_dim == 128
    assert attention_model.num_heads == 4
    assert attention_model.head_dim == 32


def test_temporal_attention_forward(attention_model, sample_data):
    output, weights = attention_model(sample_data)

    assert isinstance(output, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert output.shape == (32, 50, 128)  # (batch_size, seq_len, hidden_dim)
    assert weights.shape == (32, 4, 50, 50)  # (batch_size, num_heads, seq_len, seq_len)


def test_temporal_attention_with_mask(attention_model, sample_data):
    mask = torch.ones(32, 50, 50)
    mask[:, :, 25:] = 0  # Masquer la seconde moitié de la séquence

    output, weights = attention_model(sample_data, mask)

    assert isinstance(output, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert output.shape == (32, 50, 128)
    assert weights.shape == (32, 4, 50, 50)

    # Vérifier que les poids d'attention sont nuls pour les positions masquées
    assert torch.all(weights[:, :, :, 25:] == 0)


def test_temporal_attention_model_initialization(full_model):
    assert isinstance(full_model, TemporalAttentionModel)
    assert len(full_model.attention_layers) == 2
    assert len(full_model.ff_layers) == 2
    assert full_model.output_layer.out_features == 3


def test_temporal_attention_model_forward(full_model, sample_data):
    output, attention_weights = full_model(sample_data)

    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, list)
    assert output.shape == (32, 3)  # (batch_size, output_dim)
    assert len(attention_weights) == 2  # Nombre de couches d'attention

    for weights in attention_weights:
        assert weights.shape == (32, 4, 50, 50)


def test_temporal_attention_model_gradient_flow(full_model, sample_data):
    # Vérifier que les gradients peuvent être calculés
    output, _ = full_model(sample_data)
    loss = output.sum()
    loss.backward()

    # Vérifier que les gradients sont non nuls
    for name, param in full_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


def test_temporal_attention_model_different_sequence_lengths(full_model):
    # Tester avec différentes longueurs de séquence
    for seq_len in [10, 25, 50, 75]:
        data = torch.randn(32, seq_len, 64)
        output, _ = full_model(data)
        assert output.shape == (32, 3)
