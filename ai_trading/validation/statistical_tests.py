import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional, Union, Callable
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from scipy.stats import chi2

# Tentative d'importation optionnelle de arch
try:
    from arch.bootstrap import IIDBootstrap
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

class StatisticalTestSuite:
    """
    Suite de tests statistiques pour évaluer la performance des stratégies de trading.
    """
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
        """
        Calcule le ratio de Sharpe.
        
        Args:
            returns (pd.Series): Rendements quotidiens
            risk_free_rate (float): Taux sans risque annualisé
            annualization (int): Facteur d'annualisation (252 jours de trading)
            
        Returns:
            float: Ratio de Sharpe annualisé
        """
        excess_returns = returns - risk_free_rate / annualization
        return np.sqrt(annualization) * excess_returns.mean() / excess_returns.std(ddof=1)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
        """
        Calcule le ratio de Sortino (pénalise uniquement la volatilité négative).
        
        Args:
            returns (pd.Series): Rendements quotidiens
            risk_free_rate (float): Taux sans risque annualisé
            annualization (int): Facteur d'annualisation
            
        Returns:
            float: Ratio de Sortino annualisé
        """
        excess_returns = returns - risk_free_rate / annualization
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
            
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        
        if downside_deviation == 0:
            return np.inf
            
        return np.sqrt(annualization) * excess_returns.mean() / downside_deviation
    
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> float:
        """
        Calcule le drawdown maximum.
        
        Args:
            returns (pd.Series): Rendements quotidiens
            
        Returns:
            float: Drawdown maximum (valeur négative)
        """
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
        """
        Calcule le ratio de Calmar (rendement moyen / drawdown maximal).
        
        Args:
            returns (pd.Series): Rendements quotidiens
            risk_free_rate (float): Taux sans risque annualisé
            annualization (int): Facteur d'annualisation
            
        Returns:
            float: Ratio de Calmar
        """
        excess_returns = returns - risk_free_rate / annualization
        annual_return = annualization * excess_returns.mean()
        max_dd = StatisticalTestSuite.maximum_drawdown(returns)
        
        if max_dd == 0:
            return np.inf
            
        return -annual_return / max_dd
    
    @staticmethod
    def t_test_returns(returns: pd.Series) -> Dict[str, float]:
        """
        Effectue un test t pour déterminer si les rendements sont significativement positifs.
        
        Args:
            returns (pd.Series): Rendements quotidiens
            
        Returns:
            Dict[str, float]: Résultats du test t (statistique et p-value)
        """
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    
    @staticmethod
    def jarque_bera_test(returns: pd.Series) -> Dict[str, float]:
        """
        Test de Jarque-Bera pour la normalité des rendements.
        
        Args:
            returns (pd.Series): Rendements quotidiens
            
        Returns:
            Dict[str, float]: Résultats du test (statistique et p-value)
        """
        jb_stat, p_value = stats.jarque_bera(returns.dropna())
        return {
            'statistic': jb_stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    @staticmethod
    def ljung_box_test(returns: pd.Series, lags: int = 10) -> Dict[str, float]:
        """
        Test de Ljung-Box pour l'autocorrélation des rendements.
        
        Args:
            returns (pd.Series): Rendements quotidiens
            lags (int): Nombre de lags à tester
            
        Returns:
            Dict[str, float]: Résultats du test (statistique et p-value)
        """
        # Calculer les autocorrélations jusqu'au lag spécifié
        acf_values = acf(returns.dropna(), nlags=lags, fft=True)[1:]  # Exclure le lag 0
        n = len(returns.dropna())
        
        # Calculer la statistique de Ljung-Box
        lb_stat = n * (n + 2) * np.sum([(acf_val**2) / (n - k) for k, acf_val in enumerate(acf_values, 1)])
        
        # Calculer la p-value
        p_value = 1 - chi2.cdf(lb_stat, lags)
        
        return {
            'statistic': float(lb_stat),
            'p_value': float(p_value),
            'is_independent': p_value > 0.05
        }
    
    @staticmethod
    def white_reality_check(benchmark_returns: pd.Series, strategy_returns: pd.Series, 
                           n_bootstrap: int = 1000) -> Dict[str, float]:
        """
        Test de White Reality Check pour déterminer si la stratégie est significativement 
        meilleure que le benchmark.
        
        Args:
            benchmark_returns (pd.Series): Rendements du benchmark
            strategy_returns (pd.Series): Rendements de la stratégie
            n_bootstrap (int): Nombre d'échantillons bootstrap
            
        Returns:
            Dict[str, float]: Résultats du test (statistique et p-value)
        """
        # Aligner les séries
        aligned_data = pd.concat([benchmark_returns, strategy_returns], axis=1).dropna()
        benchmark = aligned_data.iloc[:, 0]
        strategy = aligned_data.iloc[:, 1]
        
        # Différence de performance
        perf_diff = strategy - benchmark
        mean_diff = perf_diff.mean()
        
        # Bootstrap - Utiliser arch si disponible, sinon l'implémentation manuelle
        if HAS_ARCH:
            try:
                bs = IIDBootstrap(perf_diff.values)
                
                def statistic(x):
                    return x.mean()
                
                # Simuler la distribution sous H0 (pas de surperformance)
                simulated_diffs = []
                for _ in range(n_bootstrap):
                    # Utilisation compatible avec différentes versions de arch
                    sim_res = bs.bootstrap(1, statistic)
                    if isinstance(sim_res, tuple):
                        simulated_diff = sim_res[0][0]
                    else:
                        simulated_diff = sim_res.mean()
                    simulated_diffs.append(simulated_diff)
            except Exception:
                # En cas d'erreur avec arch, utiliser l'implémentation manuelle
                simulated_diffs = StatisticalTestSuite._manual_bootstrap(perf_diff, n_bootstrap)
        else:
            # Implémentation manuelle du bootstrap
            simulated_diffs = StatisticalTestSuite._manual_bootstrap(perf_diff, n_bootstrap)
        
        # Calculer la p-value empirique
        p_value = np.mean(np.array(simulated_diffs) > mean_diff)
        
        return {
            'statistic': mean_diff,
            'p_value': p_value,
            'outperforms_benchmark': p_value < 0.05
        }
    
    @staticmethod
    def _manual_bootstrap(series: pd.Series, n_bootstrap: int = 1000) -> List[float]:
        """
        Implémentation manuelle du bootstrap pour les séries temporelles.
        
        Args:
            series (pd.Series): Série temporelle à bootstrapper
            n_bootstrap (int): Nombre d'échantillons bootstrap
            
        Returns:
            List[float]: Liste des statistiques bootstrappées
        """
        n_samples = len(series)
        results = []
        
        for _ in range(n_bootstrap):
            # Échantillonnage avec remplacement
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = series.iloc[bootstrap_indices]
            results.append(bootstrap_sample.mean())
            
        return results

    @staticmethod
    def stability_test(returns: pd.Series, window_size: int = 63) -> Dict[str, float]:
        """
        Teste la stabilité des performances à travers le temps.
        
        Args:
            returns (pd.Series): Rendements quotidiens
            window_size (int): Taille de la fenêtre glissante (par défaut 3 mois de trading)
            
        Returns:
            Dict[str, float]: Mesures de stabilité
        """
        rolling_sharpe = returns.rolling(window=window_size).apply(
            lambda x: StatisticalTestSuite.sharpe_ratio(x, annualization=window_size), raw=False
        )
        
        return {
            'sharpe_mean': rolling_sharpe.mean(),
            'sharpe_std': rolling_sharpe.std(),
            'sharpe_min': rolling_sharpe.min(),
            'sharpe_max': rolling_sharpe.max(),
            'sharpe_stability': rolling_sharpe.std() / abs(rolling_sharpe.mean()) if rolling_sharpe.mean() != 0 else np.inf
        }
    
    @staticmethod
    def run_all_tests(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Dict[str, float]]:
        """
        Exécute tous les tests statistiques sur les rendements.
        
        Args:
            returns (pd.Series): Rendements quotidiens de la stratégie
            benchmark_returns (pd.Series, optional): Rendements quotidiens du benchmark
            
        Returns:
            Dict[str, Dict[str, float]]: Résultats de tous les tests
        """
        results = {
            'basic_metrics': {
                'sharpe_ratio': StatisticalTestSuite.sharpe_ratio(returns),
                'sortino_ratio': StatisticalTestSuite.sortino_ratio(returns),
                'max_drawdown': StatisticalTestSuite.maximum_drawdown(returns),
                'calmar_ratio': StatisticalTestSuite.calmar_ratio(returns),
                'annualized_return': returns.mean() * 252,
                'annualized_volatility': returns.std() * np.sqrt(252)
            },
            't_test': StatisticalTestSuite.t_test_returns(returns),
            'jarque_bera': StatisticalTestSuite.jarque_bera_test(returns),
            'ljung_box': StatisticalTestSuite.ljung_box_test(returns),
            'stability': StatisticalTestSuite.stability_test(returns)
        }
        
        if benchmark_returns is not None:
            results['white_reality_check'] = StatisticalTestSuite.white_reality_check(
                benchmark_returns, returns
            )
            
        return results 