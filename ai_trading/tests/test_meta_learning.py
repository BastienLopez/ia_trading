"""
Tests pour le meta-learning et transfer learning inter-marchés.

Ces tests vérifient:
1. Le bon fonctionnement de MAML pour l'adaptation rapide
2. Le bon fonctionnement du transfer learning entre marchés
3. Le générateur de tâches pour le meta-learning
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from ai_trading.rl.models.market_task_generator import MarketTaskGenerator
from ai_trading.rl.models.meta_learning import MAML
from ai_trading.rl.models.transfer_learning import (
    DomainAdaptation,
    MarketTransferLearning,
)


# Fonction utilitaire pour créer des données synthétiques
def create_synthetic_market_data(n_markets=3, n_points=200):
    """Crée des données synthétiques pour les tests."""
    markets = [f"Market{i}" for i in range(n_markets)]
    market_data = {}

    for i, market in enumerate(markets):
        # Paramètres spécifiques au marché
        volatility = 0.01 + (i * 0.005)
        trend = 0.0002 * (i - n_markets / 2)
        start_price = 100 * (1 + 0.2 * i)

        # Générer prix
        prices = [start_price]
        for _ in range(n_points - 1):
            price_change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)

        # Créer DataFrame
        df = pd.DataFrame()
        df["close"] = prices
        df["open"] = df["close"].shift(1) * (1 + np.random.normal(0, 0.002, n_points))
        df["high"] = df[["open", "close"]].max(axis=1) * (
            1 + np.abs(np.random.normal(0, 0.003, n_points))
        )
        df["low"] = df[["open", "close"]].min(axis=1) * (
            1 - np.abs(np.random.normal(0, 0.003, n_points))
        )
        df["volume"] = np.random.gamma(shape=2.0, scale=1000, size=n_points) * (
            1 + 0.1 * i
        )

        # Remplacer NaN
        df.fillna(method="bfill", inplace=True)

        # Indicateurs
        df["rsi"] = 50 + 15 * np.sin(np.linspace(0, 10 + i, n_points))
        df["volatility"] = np.abs(
            df["close"].pct_change().rolling(window=20).std().fillna(0)
        )

        market_data[market] = df

    return market_data


# Modèle simple pour les tests
class SimpleModel(nn.Module):
    """Modèle simple pour les tests."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # Gérer les entrées 3D (batch, seq_len, features)
        if x.dim() == 3:
            batch_size, seq_len, n_features = x.size()
            # Prendre la dernière timestep
            x = x[:, -1, :]

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class TestMarketTaskGenerator:
    """Tests pour le générateur de tâches."""

    @pytest.fixture
    def market_data(self):
        """Données de marché pour les tests."""
        return create_synthetic_market_data(n_markets=3, n_points=200)

    def test_init(self, market_data):
        """Test de l'initialisation du générateur de tâches."""
        task_generator = MarketTaskGenerator(
            market_data=market_data,
            target_column="close",
            sequence_length=10,
            normalization="local",
        )

        # Vérifier que les données sont préparées
        assert len(task_generator.prepared_data) == len(market_data)

        # Vérifier que chaque marché a des données
        for market in market_data.keys():
            assert market in task_generator.prepared_data
            assert "features" in task_generator.prepared_data[market]
            assert "target" in task_generator.prepared_data[market]
            assert "length" in task_generator.prepared_data[market]

    def test_generate_task(self, market_data):
        """Test de la génération d'une tâche."""
        task_generator = MarketTaskGenerator(
            market_data=market_data,
            target_column="close",
            sequence_length=10,
            normalization="local",
        )

        # Générer une tâche
        support_x, support_y, query_x, query_y = task_generator.generate_task(
            market_name="Market0", support_size=5, query_size=5
        )

        # Vérifier les dimensions
        assert support_x.size() == (5, 10, len(market_data["Market0"].columns) - 1)
        assert support_y.size() == (5, 1)
        assert query_x.size() == (5, 10, len(market_data["Market0"].columns) - 1)
        assert query_y.size() == (5, 1)

        # Vérifier le type
        assert isinstance(support_x, torch.Tensor)
        assert isinstance(support_y, torch.Tensor)
        assert isinstance(query_x, torch.Tensor)
        assert isinstance(query_y, torch.Tensor)

    def test_generate_multi_market_batch(self, market_data):
        """Test de la génération d'un batch multi-marchés."""
        task_generator = MarketTaskGenerator(
            market_data=market_data,
            target_column="close",
            sequence_length=10,
            normalization="local",
        )

        # Générer un batch
        batch = task_generator.generate_multi_market_batch(
            batch_size=3, support_size=5, query_size=5
        )

        # Vérifier le nombre de tâches
        assert len(batch) == 3

        # Vérifier le format de chaque tâche
        for task in batch:
            assert len(task) == 4  # (support_x, support_y, query_x, query_y)
            support_x, support_y, query_x, query_y = task

            assert support_x.size(0) == 5  # support_size
            assert query_x.size(0) == 5  # query_size

    def test_create_market_dataloader(self, market_data):
        """Test de la création d'un DataLoader."""
        task_generator = MarketTaskGenerator(
            market_data=market_data,
            target_column="close",
            sequence_length=10,
            normalization="local",
        )

        # Créer un DataLoader
        train_loader, test_loader = task_generator.create_market_dataloader(
            market_name="Market0", batch_size=4, test_size=0.2
        )

        # Vérifier que les loaders sont créés
        assert train_loader is not None
        assert test_loader is not None

        # Vérifier les dimensions des données
        for x, y in train_loader:
            assert x.size()[1:] == (10, len(market_data["Market0"].columns) - 1)
            assert y.size()[1:] == (1,)
            break


class TestMAML:
    """Tests pour l'implémentation de MAML."""

    @pytest.fixture
    def model(self):
        """Modèle simple pour les tests."""
        return SimpleModel(input_dim=5)

    @pytest.fixture
    def tasks(self):
        """Tâches synthétiques pour les tests."""
        n_tasks = 3
        support_size = 5
        query_size = 5
        seq_len = 10
        features = 5

        tasks = []
        for _ in range(n_tasks):
            # Créer des données synthétiques
            support_x = torch.randn(support_size, seq_len, features)
            support_y = torch.randn(support_size, 1)
            query_x = torch.randn(query_size, seq_len, features)
            query_y = torch.randn(query_size, 1)

            tasks.append((support_x, support_y, query_x, query_y))

        return tasks

    def test_init(self, model):
        """Test de l'initialisation de MAML."""
        maml = MAML(model=model, inner_lr=0.01, outer_lr=0.001, inner_steps=3)

        # Vérifier les attributs
        assert maml.inner_lr == 0.01
        assert maml.outer_lr == 0.001
        assert maml.inner_steps == 3
        assert maml.first_order is False
        assert isinstance(maml.meta_optimizer, torch.optim.Optimizer)

    def test_adapt(self, model):
        """Test de l'adaptation du modèle."""
        maml = MAML(model=model, inner_lr=0.01, outer_lr=0.001, inner_steps=3)

        # Données de support
        support_x = torch.randn(5, 10, 5)
        support_y = torch.randn(5, 1)

        # Adapter le modèle
        adapted_model = maml.adapt(support_x, support_y)

        # Vérifier que le modèle est différent
        for p1, p2 in zip(model.parameters(), adapted_model.parameters()):
            assert not torch.allclose(p1, p2)

    def test_meta_train_step(self, model, tasks):
        """Test d'une étape d'entraînement méta."""
        maml = MAML(model=model, inner_lr=0.01, outer_lr=0.001, inner_steps=3)

        # Traiter un batch de tâches
        batch_loss = maml._process_task_batch(tasks)

        # Vérifier que la perte est calculée
        assert isinstance(batch_loss, float)
        assert batch_loss > 0


class TestTransferLearning:
    """Tests pour le transfer learning."""

    @pytest.fixture
    def base_model(self):
        """Modèle de base pour les tests."""
        return SimpleModel(input_dim=5)

    def test_init(self, base_model):
        """Test de l'initialisation du transfer learning."""
        transfer = MarketTransferLearning(
            base_model=base_model,
            fine_tune_layers=["fc2", "fc3"],
            learning_rate=0.0005,
            feature_mapping=True,
        )

        # Vérifier les attributs
        assert transfer.fine_tune_layers == ["fc2", "fc3"]
        assert transfer.learning_rate == 0.0005
        assert transfer.feature_mapping is True
        assert transfer.feature_mapper is not None
        assert isinstance(transfer.optimizer, torch.optim.Optimizer)

    def test_freeze_layers(self, base_model):
        """Test du gel des couches."""
        transfer = MarketTransferLearning(
            base_model=base_model, fine_tune_layers=["fc2", "fc3"], learning_rate=0.0005
        )

        # Vérifier que les couches appropriées sont gelées/dégelées
        for name, param in transfer.base_model.named_parameters():
            if any(layer in name for layer in ["fc2", "fc3"]):
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False

    def test_predict(self, base_model):
        """Test de la prédiction."""
        transfer = MarketTransferLearning(base_model=base_model, feature_mapping=True)

        # Données d'entrée
        x = torch.randn(5, 5)

        # Faire une prédiction
        with torch.no_grad():
            y_pred = transfer.predict(x)

        # Vérifier la dimension de sortie
        assert y_pred.size() == (5, 1)


class TestDomainAdaptation:
    """Tests pour l'adaptation de domaine."""

    @pytest.fixture
    def source_model(self):
        """Modèle source pour les tests."""
        return SimpleModel(input_dim=5)

    def test_init(self, source_model):
        """Test de l'initialisation de l'adaptation de domaine."""
        domain_adapt = DomainAdaptation(
            source_model=source_model, adaptation_type="dann", lambda_param=0.1
        )

        # Vérifier les attributs
        assert domain_adapt.adaptation_type == "dann"
        assert domain_adapt.lambda_param == 0.1
        assert domain_adapt.domain_discriminator is not None
        assert isinstance(domain_adapt.model_optimizer, torch.optim.Optimizer)
        assert isinstance(domain_adapt.discriminator_optimizer, torch.optim.Optimizer)

    def test_coral_loss(self, source_model):
        """Test de la perte CORAL."""
        domain_adapt = DomainAdaptation(
            source_model=source_model, adaptation_type="coral", lambda_param=0.1
        )

        # Données source et cible
        source = torch.randn(10, 5)
        target = torch.randn(10, 5)

        # Calculer la perte CORAL
        loss = domain_adapt._coral_loss(source, target)

        # Vérifier que la perte est calculée
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_mmd_loss(self, source_model):
        """Test de la perte MMD."""
        domain_adapt = DomainAdaptation(
            source_model=source_model, adaptation_type="mmd", lambda_param=0.1
        )

        # Données source et cible
        source = torch.randn(10, 5)
        target = torch.randn(10, 5)

        # Calculer la perte MMD
        loss = domain_adapt._mmd_loss(source, target)

        # Vérifier que la perte est calculée
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_train_step(self, source_model):
        """Test d'une étape d'entraînement."""
        domain_adapt = DomainAdaptation(
            source_model=source_model, adaptation_type="dann", lambda_param=0.1
        )

        # Données d'entraînement
        source_data = torch.randn(10, 5)
        source_labels = torch.randn(10, 1)
        target_data = torch.randn(10, 5)

        # Effectuer une étape d'entraînement
        metrics = domain_adapt.train_step(
            source_data=source_data,
            source_labels=source_labels,
            target_data=target_data,
        )

        # Vérifier les métriques
        assert "task_loss" in metrics
        assert "domain_loss" in metrics
        assert "total_loss" in metrics
        assert metrics["total_loss"] > 0
