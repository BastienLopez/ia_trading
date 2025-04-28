import pytest
import torch

from ai_trading.rl.models.temporal_transformer import (
    FinancialTemporalTransformer,
    PositionalEncoding,
    TemporalTransformerBlock,
)


@pytest.fixture
def sample_data():
    batch_size = 32
    seq_len = 100
    input_dim = 5  # OHLCV
    return torch.randn(batch_size, seq_len, input_dim)


@pytest.fixture
def transformer_model():
    return FinancialTemporalTransformer(
        input_dim=5,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=1000,
        output_dim=1,
    )


def test_positional_encoding():
    d_model = 512
    max_len = 1000
    pe = PositionalEncoding(d_model, max_len)

    # Tester avec différentes longueurs de séquence
    for seq_len in [50, 100, 200]:
        x = torch.randn(seq_len, 1, d_model)
        output = pe(x)
        assert output.shape == (seq_len, 1, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_temporal_transformer_block():
    d_model = 512
    nhead = 8
    batch_size = 32
    seq_len = 100

    block = TemporalTransformerBlock(d_model, nhead)
    x = torch.randn(batch_size, seq_len, d_model)

    output = block(x)
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_transformer_initialization(transformer_model):
    assert isinstance(transformer_model, FinancialTemporalTransformer)
    assert len(transformer_model.transformer_blocks) == 6
    assert transformer_model.input_projection.in_features == 5
    assert transformer_model.input_projection.out_features == 512


def test_transformer_forward(transformer_model, sample_data):
    output, attention_weights = transformer_model(sample_data)

    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, list)
    assert output.shape == (32, 1)  # (batch_size, output_dim)
    assert len(attention_weights) == 6  # num_layers

    # Vérifier les dimensions des poids d'attention
    for weights in attention_weights:
        assert weights.shape == (32, 100, 512)  # (batch_size, seq_len, d_model)


def test_transformer_with_mask(transformer_model, sample_data):
    # Créer un masque pour les 50 dernières positions
    mask = torch.ones(32, 100)  # (batch_size, seq_len)
    mask[:, 50:] = 0

    output, attention_weights = transformer_model(
        sample_data, src_key_padding_mask=mask
    )

    assert output.shape == (32, 1)
    assert len(attention_weights) == 6


def test_transformer_different_sequence_lengths(transformer_model):
    for seq_len in [50, 100, 150]:
        data = torch.randn(32, seq_len, 5)
        output, attention_weights = transformer_model(data)
        assert output.shape == (32, 1)
        assert attention_weights[0].shape == (32, seq_len, 512)


def test_transformer_gradient_flow(transformer_model, sample_data):
    # Vérifier que les gradients peuvent être calculés
    output, _ = transformer_model(sample_data)
    loss = output.sum()
    loss.backward()

    # Vérifier que les gradients ne sont pas nuls
    for name, param in transformer_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.all(param.grad == 0)
