import pytest
import torch

from ai_trading.rl.models.temporal_transformer import FinancialTemporalTransformer
from ai_trading.rl.trading_system import RLTradingSystem


@pytest.fixture
def trading_system():
    return RLTradingSystem()


@pytest.fixture
def sample_training_data():
    # Générer des données d'entraînement synthétiques
    n_samples = 100
    seq_len = 50
    input_dim = 5  # OHLCV
    data = torch.randn(n_samples, seq_len, input_dim)
    targets = torch.randn(n_samples, 1)  # Prix futur
    return data, targets


def test_create_transformer(trading_system):
    trading_system.create_transformer(input_dim=5, d_model=512, nhead=8, num_layers=6)
    assert trading_system._transformer is not None
    assert isinstance(trading_system._transformer, FinancialTemporalTransformer)


def test_train_transformer(trading_system, sample_training_data):
    data, targets = sample_training_data

    # Créer le transformer
    trading_system.create_transformer(input_dim=5, d_model=512, nhead=8, num_layers=6)

    # Entraîner le transformer
    trading_system.train_transformer(
        data=data,
        targets=targets,
        epochs=2,  # Petit nombre d'époques pour les tests
        batch_size=32,
        learning_rate=0.001,
    )

    # Faire une prédiction pour vérifier que le modèle est en mode évaluation
    with torch.no_grad():
        predictions, _ = trading_system.predict_with_transformer(data[:1])
    assert not trading_system._transformer.training


def test_predict_with_transformer(trading_system, sample_training_data):
    data, _ = sample_training_data

    # Créer et entraîner le transformer
    trading_system.create_transformer(input_dim=5, d_model=512, nhead=8, num_layers=6)

    # Faire des prédictions
    predictions, attention_weights = trading_system.predict_with_transformer(
        data[:10]
    )  # Utiliser un petit batch

    assert isinstance(predictions, torch.Tensor)
    assert isinstance(attention_weights, list)
    assert predictions.shape == (10, 1)  # (batch_size, output_dim)
    assert len(attention_weights) == 6  # num_layers


def test_load_transformer(trading_system, tmp_path):
    # Créer un modèle et le sauvegarder
    trading_system.create_transformer(input_dim=5, d_model=512, nhead=8, num_layers=6)

    save_path = tmp_path / "transformer.pth"
    torch.save(trading_system._transformer.state_dict(), save_path)

    # Créer un nouveau système et charger le modèle
    new_system = RLTradingSystem()
    new_system.create_transformer(input_dim=5, d_model=512, nhead=8, num_layers=6)
    new_system.load_transformer(str(save_path))

    # Vérifier que les poids sont les mêmes
    for p1, p2 in zip(
        trading_system._transformer.parameters(), new_system._transformer.parameters()
    ):
        assert torch.allclose(p1, p2)


def test_transformer_integration(trading_system, sample_training_data):
    data, targets = sample_training_data

    # Créer et entraîner le transformer
    trading_system.create_transformer(input_dim=5, d_model=512, nhead=8, num_layers=6)

    trading_system.train_transformer(
        data=data, targets=targets, epochs=2, batch_size=32
    )

    # Faire des prédictions
    predictions, _ = trading_system.predict_with_transformer(data[:10])

    # Vérifier que les prédictions sont raisonnables
    assert not torch.isnan(predictions).any()
    assert not torch.isinf(predictions).any()
    assert predictions.shape == (10, 1)
