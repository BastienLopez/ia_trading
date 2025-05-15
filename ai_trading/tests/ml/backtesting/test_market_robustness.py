import numpy as np
import pandas as pd
import pytest

from ai_trading.validation.market_robustness import (
    MarketRegimeClassifier,
    RobustnessEvaluator,
)


class TestMarketRegimeClassifier:
    @pytest.fixture
    def sample_data(self):
        """Crée un ensemble de données de test."""
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        np.random.seed(42)

        # Créer différents régimes
        prices = []
        volatility_factor = 1.0

        for i in range(200):
            # Changer la volatilité tous les 50 jours
            if i % 50 == 0:
                volatility_factor = np.random.choice([0.5, 1.0, 2.0, 3.0])

            # Générer un prix avec une volatilité variable
            if i == 0:
                prices.append(100)
            else:
                change = np.random.normal(0, 0.01 * volatility_factor)
                prices.append(prices[-1] * (1 + change))

        df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "high": [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                "low": [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                "volume": [
                    np.random.randint(1000, 10000) * (1 + (i % 50) / 100)
                    for i in range(200)
                ],
            }
        )

        df.set_index("date", inplace=True)
        return df

    def test_extract_features(self, sample_data):
        """Teste l'extraction des caractéristiques."""
        classifier = MarketRegimeClassifier()
        features = classifier.extract_features(sample_data)

        # Vérifier que les caractéristiques existent
        assert "volatility" in features.columns
        assert "momentum" in features.columns
        assert "rel_volume" in features.columns
        assert "trend" in features.columns

        # Vérifier que les NaN ont été supprimés
        assert not features.isnull().any().any()

        # Vérifier que les valeurs sont dans les plages attendues
        assert features["volatility"].min() >= 0
        assert features["momentum"].min() >= 0
        assert features["momentum"].max() <= 100

    def test_fit_predict(self, sample_data):
        """Teste l'entraînement et la prédiction des régimes."""
        classifier = MarketRegimeClassifier(n_regimes=3, n_init=10)

        # Entraîner le classifieur
        classifier.fit(sample_data)
        assert classifier.is_fitted

        # Prédire les régimes
        regimes = classifier.predict(sample_data)

        # Vérifier les dimensions de la sortie
        features = classifier.extract_features(sample_data)
        assert len(regimes) == len(features)

        # Vérifier que les régimes sont dans l'intervalle correct
        assert np.min(regimes) >= 0
        assert np.max(regimes) < 3

    def test_get_regime_characteristics(self, sample_data):
        """Teste le calcul des caractéristiques des régimes."""
        classifier = MarketRegimeClassifier(n_regimes=2, n_init=10)
        classifier.fit(sample_data)

        regime_stats = classifier.get_regime_characteristics(sample_data)

        # Vérifier que tous les régimes sont représentés
        assert len(regime_stats) == 2

        # Vérifier que chaque régime a les caractéristiques attendues
        for regime, stats in regime_stats.items():
            assert "volatility" in stats
            assert "momentum" in stats
            assert "volume" in stats
            assert "trend" in stats
            assert "size" in stats
            assert stats["size"] > 0


class TestRobustnessEvaluator:
    @pytest.fixture
    def sample_data_and_returns(self):
        """Crée un ensemble de données et de rendements simulés."""
        # Utiliser la même logique que pour le test du classifieur
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        np.random.seed(42)

        prices = []
        volatility_factor = 1.0

        for i in range(200):
            # Changer la volatilité tous les 50 jours
            if i % 50 == 0:
                volatility_factor = np.random.choice([0.5, 1.0, 2.0, 3.0])

            # Générer un prix avec une volatilité variable
            if i == 0:
                prices.append(100)
            else:
                change = np.random.normal(0, 0.01 * volatility_factor)
                prices.append(prices[-1] * (1 + change))

        df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "high": [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                "low": [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                "volume": [
                    np.random.randint(1000, 10000) * (1 + (i % 50) / 100)
                    for i in range(200)
                ],
            }
        )

        df.set_index("date", inplace=True)

        # Créer une série de rendements simulés
        returns = pd.Series(np.random.normal(0.001, 0.01, len(dates)), index=dates)

        return df, returns

    def test_evaluate(self, sample_data_and_returns):
        """Teste l'évaluation de performance dans différents régimes."""
        df, returns = sample_data_and_returns

        # Définir des métriques de test
        def mean_return(x):
            return x.mean()

        def volatility(x):
            return x.std()

        metrics = [mean_return, volatility]

        # Créer et entraîner un classifieur
        classifier = MarketRegimeClassifier(n_regimes=2, n_init=10)
        classifier.fit(df)

        # Créer l'évaluateur et exécuter l'évaluation
        evaluator = RobustnessEvaluator(classifier)
        results = evaluator.evaluate(df, returns, metrics)

        # Vérifier les résultats
        assert len(results) > 0
        for regime, regime_metrics in results.items():
            assert "mean_return" in regime_metrics
            assert "volatility" in regime_metrics

    def test_stress_test(self, sample_data_and_returns):
        """Teste les tests de stress."""
        df, _ = sample_data_and_returns

        # Définir une fonction de stratégie simple pour le test
        def simple_strategy(data):
            """Stratégie simple qui retourne des rendements aléatoires."""
            dates = data.index
            return pd.Series(np.random.normal(0.001, 0.01, len(dates)), index=dates)

        # Définir des métriques de test
        def mean_return(x):
            return x.mean()

        def volatility(x):
            return x.std()

        metrics = [mean_return, volatility]

        # Définir des scénarios de stress
        scenarios = [
            {"close": 0.9, "volume": 0.8},  # Baisse des prix et du volume
            {"close": 1.1, "volume": 1.2},  # Hausse des prix et du volume
        ]

        # Créer l'évaluateur avec un classifieur
        classifier = MarketRegimeClassifier(n_regimes=2, n_init=10)
        evaluator = RobustnessEvaluator(classifier)

        # Exécuter le test de stress
        stress_results = evaluator.stress_test(df, simple_strategy, scenarios, metrics)

        # Vérifier les résultats - tenir compte du scénario de base inclus
        assert len(stress_results) == len(scenarios) + 1  # +1 pour le scénario de base
        assert "base" in stress_results

        for scenario_name, results in stress_results.items():
            assert results is not None
            assert "mean_return" in results
            assert "volatility" in results
