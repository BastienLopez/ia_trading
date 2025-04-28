import numpy as np
import pandas as pd
import pytest

from ai_trading.validation.statistical_tests import StatisticalTestSuite


class TestStatisticalTestSuite:
    @pytest.fixture
    def sample_returns(self):
        """Crée des rendements simulés pour les tests."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="D")

        # Créer des rendements avec une moyenne positive
        returns = pd.Series(np.random.normal(0.001, 0.01, len(dates)), index=dates)
        return returns

    @pytest.fixture
    def benchmark_returns(self):
        """Crée des rendements de benchmark simulés."""
        np.random.seed(123)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="D")

        # Créer des rendements de benchmark avec une moyenne légèrement inférieure
        returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
        return returns

    def test_sharpe_ratio(self, sample_returns):
        """Teste le calcul du ratio de Sharpe."""
        sharpe = StatisticalTestSuite.sharpe_ratio(sample_returns)

        # Vérifier que le ratio est calculé correctement
        expected_sharpe = (
            np.sqrt(252) * sample_returns.mean() / sample_returns.std(ddof=1)
        )
        assert np.isclose(sharpe, expected_sharpe)

        # Vérifier que les paramètres optionnels fonctionnent
        sharpe_with_rf = StatisticalTestSuite.sharpe_ratio(
            sample_returns, risk_free_rate=0.02
        )
        assert sharpe_with_rf != sharpe

    def test_sortino_ratio(self, sample_returns):
        """Teste le calcul du ratio de Sortino."""
        sortino = StatisticalTestSuite.sortino_ratio(sample_returns)

        # Vérifier que le ratio est un nombre fini et positif (pour des rendements positifs)
        assert np.isfinite(sortino)

        # Créer des rendements sans volatilité à la baisse
        positive_returns = pd.Series([0.01] * 100)
        inf_sortino = StatisticalTestSuite.sortino_ratio(positive_returns)
        assert np.isinf(inf_sortino)

    def test_maximum_drawdown(self, sample_returns):
        """Teste le calcul du drawdown maximum."""
        max_dd = StatisticalTestSuite.maximum_drawdown(sample_returns)

        # Vérifier que le drawdown est négatif (ou zéro dans des cas extrêmes)
        assert max_dd <= 0

        # Test avec une série qui monte uniquement
        rising_returns = pd.Series([0.01] * 100)
        rising_dd = StatisticalTestSuite.maximum_drawdown(rising_returns)
        assert rising_dd == 0

    def test_calmar_ratio(self, sample_returns):
        """Teste le calcul du ratio de Calmar."""
        calmar = StatisticalTestSuite.calmar_ratio(sample_returns)

        # Vérifier que le ratio est calculé correctement
        max_dd = StatisticalTestSuite.maximum_drawdown(sample_returns)
        if max_dd != 0:
            expected_calmar = -(252 * sample_returns.mean()) / max_dd
            assert np.isclose(calmar, expected_calmar)

    def test_t_test_returns(self, sample_returns):
        """Teste le test de t pour les rendements."""
        t_test_results = StatisticalTestSuite.t_test_returns(sample_returns)

        # Vérifier que le résultat contient les attributs attendus
        assert "statistic" in t_test_results
        assert "p_value" in t_test_results
        assert "is_significant" in t_test_results

        # Vérifier la cohérence entre p-value et is_significant
        assert t_test_results["is_significant"] == (t_test_results["p_value"] < 0.05)

    def test_jarque_bera_test(self, sample_returns):
        """Teste le test de Jarque-Bera."""
        jb_results = StatisticalTestSuite.jarque_bera_test(sample_returns)

        # Vérifier que le résultat contient les attributs attendus
        assert "statistic" in jb_results
        assert "p_value" in jb_results
        assert "is_normal" in jb_results

        # Vérifier la cohérence entre p-value et is_normal
        assert jb_results["is_normal"] == (jb_results["p_value"] > 0.05)

    def test_ljung_box_test(self, sample_returns):
        """Teste le test de Ljung-Box."""
        lb_results = StatisticalTestSuite.ljung_box_test(sample_returns)

        # Vérifier que le résultat contient les attributs attendus
        assert "statistic" in lb_results
        assert "p_value" in lb_results
        assert "is_independent" in lb_results

        # Vérifier la cohérence entre p-value et is_independent
        assert lb_results["is_independent"] == (lb_results["p_value"] > 0.05)

    def test_white_reality_check(self, sample_returns, benchmark_returns):
        """Teste le White Reality Check."""
        wrc_results = StatisticalTestSuite.white_reality_check(
            benchmark_returns, sample_returns, n_bootstrap=100
        )

        # Vérifier que le résultat contient les attributs attendus
        assert "statistic" in wrc_results
        assert "p_value" in wrc_results
        assert "outperforms_benchmark" in wrc_results

        # Vérifier la cohérence entre p-value et outperforms_benchmark
        assert wrc_results["outperforms_benchmark"] == (wrc_results["p_value"] < 0.05)

    def test_stability_test(self, sample_returns):
        """Teste le test de stabilité."""
        stability_results = StatisticalTestSuite.stability_test(
            sample_returns, window_size=50
        )

        # Vérifier que le résultat contient les attributs attendus
        assert "sharpe_mean" in stability_results
        assert "sharpe_std" in stability_results
        assert "sharpe_min" in stability_results
        assert "sharpe_max" in stability_results
        assert "sharpe_stability" in stability_results

        # Vérifier que les valeurs sont cohérentes
        assert (
            stability_results["sharpe_min"]
            <= stability_results["sharpe_mean"]
            <= stability_results["sharpe_max"]
        )

    def test_run_all_tests(self, sample_returns, benchmark_returns):
        """Teste l'exécution de tous les tests en une seule fois."""
        # Test sans benchmark
        all_results = StatisticalTestSuite.run_all_tests(sample_returns)

        # Vérifier que les résultats contiennent toutes les catégories attendues
        assert "basic_metrics" in all_results
        assert "t_test" in all_results
        assert "jarque_bera" in all_results
        assert "ljung_box" in all_results
        assert "stability" in all_results
        assert "white_reality_check" not in all_results

        # Test avec benchmark
        all_results_with_benchmark = StatisticalTestSuite.run_all_tests(
            sample_returns, benchmark_returns
        )

        # Vérifier que le white reality check est inclus avec un benchmark
        assert "white_reality_check" in all_results_with_benchmark
