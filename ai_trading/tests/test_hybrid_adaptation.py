"""
Tests pour le modèle hybride combinant meta-learning et adaptation de domaine.

Ces tests vérifient:
1. L'initialisation correcte du modèle hybride
2. Les fonctions d'adaptation inter-marchés
3. L'entraînement combiné meta-domain
"""

import pytest
import torch

from ai_trading.rl.models.hybrid_market_adaptation import HybridMarketAdaptation
from ai_trading.tests.test_meta_learning import (
    SimpleModel,
    create_synthetic_market_data,
)


class TestHybridMarketAdaptation:
    """Tests pour le modèle d'adaptation hybride."""

    @pytest.fixture
    def model(self):
        """Modèle simple pour les tests."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return SimpleModel(input_dim=5).to(device)

    @pytest.fixture
    def market_data(self):
        """Données de marché synthétiques pour les tests."""
        return create_synthetic_market_data(n_markets=3, n_points=200)

    def test_init(self, model):
        """Test de l'initialisation du modèle hybride."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hybrid = HybridMarketAdaptation(
            model=model,
            meta_inner_lr=0.01,
            meta_outer_lr=0.001,
            meta_inner_steps=3,
            domain_lambda=0.2,
            adaptation_type="dann",
            device=device,
        )

        # Vérifier que les composants sont correctement initialisés
        assert hybrid.maml is not None
        assert hybrid.domain_adaptation is not None
        assert hybrid.maml.inner_lr == 0.01
        assert hybrid.maml.outer_lr == 0.001
        assert hybrid.maml.inner_steps == 3
        assert hybrid.domain_adaptation.lambda_param == 0.2
        assert hybrid.domain_adaptation.adaptation_type == "dann"

    def test_adapt_to_market(self, model):
        """Test de l'adaptation à un nouveau marché."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hybrid = HybridMarketAdaptation(
            model=model,
            meta_inner_lr=0.01,
            meta_outer_lr=0.001,
            meta_inner_steps=3,
            domain_lambda=0.2,
            adaptation_type="coral",  # Utiliser CORAL pour les tests (plus léger)
            device=device,
        )

        # Créer des données synthétiques pour les marchés source et cible
        batch_size = 10
        seq_len = 5
        features = 5

        # S'assurer que tous les tenseurs sont sur le même device
        source_x = torch.randn(batch_size, seq_len, features).to(device)
        source_y = torch.randn(batch_size, 1).to(device)
        target_x = torch.randn(batch_size, seq_len, features).to(device)
        target_y = torch.randn(batch_size, 1).to(device)

        # Adapter le modèle
        adapted_model = hybrid.adapt_to_market(
            source_market_data=(source_x, source_y),
            target_market_data=(target_x, target_y),
            adaptation_steps=4,  # Réduire pour les tests
            target_support_size=5,
        )

        # Vérifier que le modèle adapté est différent du modèle original
        for p1, p2 in zip(model.parameters(), adapted_model.parameters()):
            assert not torch.allclose(p1, p2)

        # Vérifier que le modèle adapté peut faire des prédictions
        with torch.no_grad():
            predictions = adapted_model(target_x)
            assert predictions.shape == (batch_size, 1)

    def test_meta_domain_train(self, model):
        """Test d'entraînement meta-domain."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hybrid = HybridMarketAdaptation(
            model=model,
            meta_inner_lr=0.01,
            meta_outer_lr=0.001,
            meta_inner_steps=2,  # Réduire pour les tests
            domain_lambda=0.2,
            adaptation_type="coral",  # Utiliser CORAL pour les tests (plus léger)
            device=device,
        )

        # Créer une fonction simple de génération de tâches
        def task_generator(support_size, query_size):
            # Créer des données synthétiques
            seq_len = 5
            features = 5

            # S'assurer que tous les tenseurs sont sur le même device
            support_x = torch.randn(support_size, seq_len, features).to(device)
            support_y = torch.randn(support_size, 1).to(device)
            query_x = torch.randn(query_size, seq_len, features).to(device)
            query_y = torch.randn(query_size, 1).to(device)

            return support_x, support_y, query_x, query_y

        # Entraîner le modèle hybride (version simplifiée pour les tests)
        history = hybrid.meta_domain_train(
            task_generator=task_generator,
            num_tasks=4,
            num_epochs=2,
            support_size=5,
            query_size=5,
            domain_weight=0.3,
            batch_size=2,
        )

        # Vérifier que l'historique contient les métriques attendues
        assert "meta_loss" in history
        assert "domain_loss" in history
        assert "total_loss" in history
        assert len(history["meta_loss"]) == 2  # 2 epochs

        # Vérifier que les pertes sont des nombres positifs
        for key in history:
            for value in history[key]:
                assert isinstance(value, float)
                assert value >= 0

    def test_save_load(self, model, tmp_path):
        """Test des fonctions de sauvegarde et chargement."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hybrid = HybridMarketAdaptation(
            model=model,
            meta_inner_lr=0.01,
            meta_outer_lr=0.001,
            meta_inner_steps=3,
            domain_lambda=0.2,
            adaptation_type="dann",
            device=device,
        )

        # Créer un chemin temporaire pour le modèle
        model_path = tmp_path / "hybrid_model_test.pt"

        # Modifier certains paramètres pour tester le chargement
        initial_state = {
            "inner_lr": hybrid.maml.inner_lr,
            "lambda_param": hybrid.domain_adaptation.lambda_param,
        }

        # Sauvegarder le modèle
        hybrid.save(str(model_path))

        # Modifier les paramètres
        hybrid.maml.inner_lr = 0.05
        hybrid.domain_adaptation.lambda_param = 0.5

        # Recharger le modèle
        hybrid.load(str(model_path))

        # Vérifier que les paramètres sont restaurés
        assert hybrid.maml.inner_lr == initial_state["inner_lr"]
        assert hybrid.domain_adaptation.lambda_param == initial_state["lambda_param"]
