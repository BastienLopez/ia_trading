"""
Module pour corriger le biais de survivance dans les backtests.

Ce module contient des classes et fonctions pour:
1. Corriger le biais de survivance dans les données historiques
2. Implémenter une validation croisée spécifique aux séries temporelles
3. Effectuer une analyse de bootstrap pour évaluer la robustesse des stratégies
4. Réaliser des tests statistiques rigoureux sur les résultats des backtests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
from enum import Enum
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.stats.api as sms
import scipy.stats as stats

# Configuration du logging
logger = logging.getLogger(__name__)

class MarketRegime(str, Enum):
    """Régimes de marché identifiables pour l'analyse de robustesse."""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    BEAR_MARKET = "bear_market"
    BULL_MARKET = "bull_market"
    RECOVERY = "recovery"
    CRISIS = "crisis"

class MarketRegimeDetector:
    """
    Classe pour détecter le régime de marché actuel basé sur différentes métriques.
    """
    
    def __init__(self, 
                 volatility_threshold: float = 0.015,
                 trend_threshold: float = 0.1,
                 lookback_period: int = 90,
                 config: Dict = None):
        """
        Initialise le détecteur de régime de marché.
        
        Args:
            volatility_threshold: Seuil de volatilité pour considérer un marché comme hautement volatil
            trend_threshold: Seuil de performance pour identifier les marchés haussiers/baissiers
            lookback_period: Période d'observation pour calculer les indicateurs (en jours)
            config: Configuration supplémentaire (optionnel)
        """
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.lookback_period = lookback_period
        
        # Initialiser la configuration par défaut
        self.config = {
            "window_size": 60,
            "trend_threshold": trend_threshold,
            "volatility_threshold": volatility_threshold
        }
        
        # Mettre à jour avec la configuration personnalisée si fournie
        if config:
            self.config.update(config)
        
    def detect_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """
        Détecte le régime de marché actuel à partir des données de prix.
        
        Args:
            price_data: DataFrame avec au moins une colonne 'close' et un index de dates
            
        Returns:
            Le régime de marché détecté
        """
        if len(price_data) < self.lookback_period:
            logger.warning(f"Données insuffisantes: {len(price_data)} points vs {self.lookback_period} requis")
            return MarketRegime.NORMAL
        
        # Calcul des rendements
        returns = price_data['close'].pct_change().dropna()
        
        # Limiter à la période d'observation
        recent_returns = returns.iloc[-self.lookback_period:]
        
        # Calculer la volatilité (écart-type annualisé)
        volatility = recent_returns.std() * np.sqrt(252)
        
        # Calculer la performance cumulée
        cumulative_return = (1 + recent_returns).prod() - 1
        
        # Détecter les crises (forte baisse et forte volatilité)
        drawdown = self._calculate_max_drawdown(recent_returns)
        
        # Logique de détection de régime
        if drawdown < -0.2 and volatility > 1.5 * self.volatility_threshold:
            return MarketRegime.CRISIS
        
        if volatility > self.volatility_threshold:
            if cumulative_return < -self.trend_threshold:
                return MarketRegime.BEAR_MARKET
            elif cumulative_return > self.trend_threshold:
                return MarketRegime.BULL_MARKET
            else:
                return MarketRegime.HIGH_VOLATILITY
                
        if cumulative_return < -self.trend_threshold:
            return MarketRegime.BEAR_MARKET
        elif cumulative_return > self.trend_threshold:
            return MarketRegime.BULL_MARKET
        
        # Détecter les phases de reprise
        if drawdown < -0.1 and cumulative_return > 0:
            return MarketRegime.RECOVERY
            
        return MarketRegime.NORMAL
    
    def detect_regimes(self, market_data, asset_symbol):
        """
        Détecte les régimes de marché sur une période de temps.
        
        Args:
            market_data: Dictionnaire de DataFrames avec données de marché
            asset_symbol: Symbole de l'actif à analyser
            
        Returns:
            DataFrame avec les régimes détectés
        """
        if asset_symbol not in market_data:
            logger.error(f"Actif {asset_symbol} non trouvé dans les données")
            return {}
        
        asset_data = market_data[asset_symbol].copy()
        
        # Utiliser une fenêtre glissante pour détecter les régimes
        window_size = self.config["window_size"]
        
        # Initialiser les colonnes de résultats
        asset_data['regime'] = None
        asset_data['trend'] = np.nan
        asset_data['volatility'] = np.nan
        
        if len(asset_data) <= window_size:
            logger.warning(f"Données insuffisantes pour {asset_symbol}: {len(asset_data)} points vs {window_size} requis")
            return {asset_symbol: asset_data}
        
        # Calculer rendements
        asset_data['returns'] = asset_data['close'].pct_change()
        
        # Pour chaque fenêtre, détecter le régime
        for i in range(window_size, len(asset_data)):
            window_data = asset_data.iloc[i-window_size:i]
            
            # Détecter le régime
            regime = self.detect_regime(window_data)
            
            # Stocker le résultat
            asset_data.loc[asset_data.index[i-1], 'regime'] = regime.value
            
            # Calculer tendance et volatilité
            returns_window = window_data['returns'].dropna()
            asset_data.loc[asset_data.index[i-1], 'trend'] = (1 + returns_window).prod() - 1
            asset_data.loc[asset_data.index[i-1], 'volatility'] = returns_window.std() * np.sqrt(252)
        
        # Retirer la colonne des rendements
        asset_data = asset_data.drop(columns=['returns'])
        
        return {asset_symbol: asset_data}
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calcule le drawdown maximum pour une série de rendements.
        
        Args:
            returns: Série de rendements
            
        Returns:
            Le drawdown maximum (valeur négative)
        """
        # Calculer la valeur cumulée
        cumulative = (1 + returns).cumprod()
        
        # Calculer le drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative / rolling_max) - 1
        
        return drawdown.min()

class SurvivorshipBiasCorrector:
    """
    Classe pour corriger le biais de survivance dans les données historiques.
    """
    
    def __init__(self, delisting_database: Optional[pd.DataFrame] = None, config: Dict = None):
        """
        Initialise le correcteur de biais de survivance.
        
        Args:
            delisting_database: DataFrame contenant l'historique des actifs disparus
                                avec colonnes ['symbol', 'delisting_date', 'reason', 'last_price']
            config: Configuration personnalisée pour la correction (optionnel)
        """
        self.delisting_database = delisting_database
        
        # Configuration par défaut
        self.config = {
            "delisted_assets_impact": -0.8,  # Impact par défaut pour les actifs en faillite
            "min_history_ratio": 0.5,  # Ratio minimum d'historique pour inclure un actif
            "decay_period": 30,  # Période de déclin avant la radiation (en jours)
            "volume_decay": 0.7  # Déclin de volume avant radiation (en pourcentage)
        }
        
        # Mettre à jour avec la configuration personnalisée si fournie
        if config:
            self.config.update(config)
        
    def load_delisting_database(self, filepath: str) -> None:
        """
        Charge une base de données d'actifs disparus depuis un fichier.
        
        Args:
            filepath: Chemin vers le fichier CSV ou Parquet
        """
        if filepath.endswith('.csv'):
            self.delisting_database = pd.read_csv(filepath, parse_dates=['delisting_date'])
        elif filepath.endswith('.parquet'):
            self.delisting_database = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Format de fichier non supporté: {filepath}")
        
        logger.info(f"Base de données de délisting chargée: {len(self.delisting_database)} entrées")
        
    def correct_dataset(self, market_data: Dict[str, pd.DataFrame], delisted_info: Dict = None,
                        start_date: pd.Timestamp = None, end_date: pd.Timestamp = None) -> Dict[str, pd.DataFrame]:
        """
        Corrige un ensemble de données en réintégrant les actifs disparus.
        
        Args:
            market_data: Dictionnaire {symbol: DataFrame} des données de marché
            delisted_info: Dictionnaire avec infos sur les actifs disparus (optionnel)
            start_date: Date de début de la période d'analyse (optionnel)
            end_date: Date de fin de la période d'analyse (optionnel)
            
        Returns:
            Dictionnaire avec les données corrigées
        """
        # Utiliser la base de données chargée ou les infos fournies
        if delisted_info is None and self.delisting_database is None:
            logger.warning("Pas d'informations sur les actifs disparus disponibles, retour des données originales")
            return market_data.copy()
        
        # Créer une copie des données originales
        corrected_data = market_data.copy()
        
        # Si des dates ne sont pas spécifiées, les déduire des données
        if start_date is None or end_date is None:
            all_dates = []
            for df in market_data.values():
                all_dates.extend(df.index.tolist())
            
            all_dates = pd.DatetimeIndex(sorted(set(all_dates)))
            if len(all_dates) == 0:
                return corrected_data
            
            if start_date is None:
                start_date = all_dates[0]
            if end_date is None:
                end_date = all_dates[-1]
        
        # Si des infos de délisting sont fournies, les utiliser
        if delisted_info:
            for symbol, info in delisted_info.items():
                delisting_date = info["delisting_date"]
                price_impact = info.get("price_impact", self.config["delisted_assets_impact"])
                
                # Si l'actif est déjà dans les données et a été délisted pendant la période d'analyse
                if symbol in corrected_data and start_date <= delisting_date <= end_date:
                    # Obtenir les dernières données connues
                    asset_data = corrected_data[symbol]
                    last_known_date = asset_data.index.max()
                    
                    # Si les données s'arrêtent avant la fin de la période, prolonger avec l'impact
                    if last_known_date < end_date:
                        # Créer une série de dates pour la période manquante
                        missing_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), 
                                                     end=end_date, freq='D')
                        
                        # Créer un DataFrame pour les données manquantes
                        last_price = asset_data['close'].iloc[-1]
                        new_price = last_price * (1 + price_impact)
                        
                        # Calculer les prix pour la période manquante
                        missing_data = pd.DataFrame(index=missing_dates)
                        missing_data['close'] = new_price
                        missing_data['open'] = new_price
                        missing_data['high'] = new_price * 1.001
                        missing_data['low'] = new_price * 0.999
                        missing_data['volume'] = 0  # Volume nul après délisting
                        
                        # Combiner avec les données existantes
                        corrected_data[symbol] = pd.concat([asset_data, missing_data])
        
        # Sinon, utiliser la base de données de délisting
        elif self.delisting_database is not None:
            # Filtrer pour la période concernée
            relevant_delistings = self.delisting_database[
                (self.delisting_database['delisting_date'] >= start_date) & 
                (self.delisting_database['delisting_date'] <= end_date)
            ]
            
            for _, delisting in relevant_delistings.iterrows():
                symbol = delisting['symbol']
                delisting_date = delisting['delisting_date']
                last_price = delisting.get('last_price', 0)
                reason = delisting.get('reason', 'unknown')
                
                # Déterminer l'impact de prix selon la raison
                if reason == 'bankruptcy':
                    price_impact = -0.9  # -90% pour faillite
                elif reason == 'merger':
                    price_impact = 0.1   # +10% pour fusion/acquisition
                else:
                    price_impact = self.config["delisted_assets_impact"]
                
                # Créer des données synthétiques si l'actif n'est pas présent ou s'arrête trop tôt
                if symbol not in corrected_data:
                    synthetic_data = self._create_synthetic_data(delisting, start_date, end_date)
                    if synthetic_data is not None:
                        corrected_data[symbol] = synthetic_data
                elif corrected_data[symbol].index.max() < end_date:
                    # Prolonger les données existantes
                    last_known_date = corrected_data[symbol].index.max()
                    missing_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), 
                                                end=end_date, freq='D')
                    
                    last_price = corrected_data[symbol]['close'].iloc[-1]
                    new_price = last_price * (1 + price_impact)
                    
                    missing_data = pd.DataFrame(index=missing_dates)
                    missing_data['close'] = new_price
                    missing_data['open'] = new_price
                    missing_data['high'] = new_price * 1.001
                    missing_data['low'] = new_price * 0.999
                    missing_data['volume'] = 0
                    
                    corrected_data[symbol] = pd.concat([corrected_data[symbol], missing_data])
        
        return corrected_data
    
    def _create_synthetic_data(self, delisting: pd.Series, start_date: pd.Timestamp, 
                              end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Crée des données synthétiques pour un actif délisté.
        
        Args:
            delisting: Série contenant les informations de délisting
            start_date: Date de début des données synthétiques
            end_date: Date de fin des données synthétiques
            
        Returns:
            DataFrame des données synthétiques ou None si impossible à créer
        """
        symbol = delisting['symbol']
        delisting_date = delisting['delisting_date']
        last_price = delisting.get('last_price', 0)
        
        # Si nous n'avons pas d'information de prix, essayer de trouver un proxy
        if last_price == 0 or pd.isna(last_price):
            logger.warning(f"Prix manquant pour {symbol}, utilisation d'un prix proxy")
            last_price = 100.0  # Prix par défaut
        
        # Créer un DataFrame avec des NaN après la date de délisting
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        synthetic_data = pd.DataFrame(index=date_range)
        
        # Fixer les valeurs jusqu'à la date de délisting
        valid_dates = date_range[date_range <= delisting_date]
        
        if len(valid_dates) == 0:
            logger.warning(f"Pas de dates valides pour {symbol}, délisting hors période")
            return None
        
        # Créer une série de prix avec tendance baissière avant le délisting
        decay_period = min(30, len(valid_dates))  # 30 jours ou moins
        
        prices = np.ones(len(valid_dates)) * last_price
        if decay_period > 0:
            # Créer une baisse progressive sur les derniers jours
            decay_start = len(valid_dates) - decay_period
            decay_factor = np.linspace(1.0, 0.7, decay_period)  # 30% de baisse
            prices[decay_start:] *= decay_factor
        
        # Créer des données OHLCV synthétiques
        synthetic_data['open'] = prices
        synthetic_data['high'] = prices * 1.02  # +2% d'écart
        synthetic_data['low'] = prices * 0.98   # -2% d'écart
        synthetic_data['close'] = prices
        
        # Volume décroissant vers le délisting
        base_volume = 1000.0
        volumes = np.ones(len(valid_dates)) * base_volume
        if decay_period > 0:
            volume_decay = np.linspace(1.0, 0.3, decay_period)  # 70% de baisse de volume
            volumes[len(valid_dates) - decay_period:] *= volume_decay
        
        synthetic_data['volume'] = volumes
        
        # Après la date de délisting, marquer toutes les valeurs comme NaN
        synthetic_data.loc[date_range > delisting_date, :] = np.nan
        
        return synthetic_data 

class TimeSeriesCV:
    """
    Validation croisée adaptée aux séries temporelles pour l'évaluation des stratégies.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0, test_size: int = None, config: Dict = None):
        """
        Initialise la validation croisée pour séries temporelles.
        
        Args:
            n_splits: Nombre de divisions (splits)
            gap: Nombre de points à ignorer entre train et test
            test_size: Taille fixe de l'ensemble de test (en points)
            config: Configuration personnalisée (optionnel)
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        
        # Configuration par défaut
        self.config = {
            "n_splits": n_splits,
            "gap": gap,
            "test_size": test_size,
            "validation_ratio": 0.2,
            "min_history": 60
        }
        
        # Mettre à jour avec la configuration personnalisée si fournie
        if config:
            self.config.update(config)
            # Mettre à jour les attributs si spécifiés dans la config
            if "n_splits" in config:
                self.n_splits = config["n_splits"]
            if "gap" in config:
                self.gap = config["gap"]
            if "test_size" in config:
                self.test_size = config["test_size"]

    def split(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Divise les données en ensembles d'entraînement et de test adaptés aux séries temporelles.
        
        Args:
            data: DataFrame ou dictionnaire de DataFrames à diviser
            
        Returns:
            Liste de tuples (train_data, test_data)
        """
        # Convertir en dictionnaire si nécessaire
        if isinstance(data, pd.DataFrame):
            data_dict = {'main': data}
        else:
            data_dict = data
        
        # Trouver les dates communes à tous les actifs
        common_dates = self._find_common_dates(data_dict)
        
        # Utiliser TimeSeriesSplit de scikit-learn
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap, test_size=self.test_size)
        
        # Liste pour stocker les résultats
        splits = []
        
        # Créer les indices basés sur les dates communes
        indices = np.arange(len(common_dates))
        
        for train_idx, test_idx in tscv.split(indices):
            # Récupérer les dates correspondantes
            train_dates = common_dates[train_idx]
            test_dates = common_dates[test_idx]
            
            # Créer les sous-ensembles de données
            train_data = {}
            test_data = {}
            
            for symbol, df in data_dict.items():
                # Filtrer par dates
                train_data[symbol] = df.loc[df.index.isin(train_dates)].copy()
                test_data[symbol] = df.loc[df.index.isin(test_dates)].copy()
            
            # Si input était un dataframe, retourner des dataframes
            if len(data_dict) == 1 and 'main' in data_dict:
                splits.append((train_data['main'], test_data['main']))
            else:
                splits.append((train_data, test_data))
        
        return splits
    
    def _find_common_dates(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """
        Trouve les dates communes à tous les DataFrames.
        
        Args:
            data_dict: Dictionnaire de DataFrames
            
        Returns:
            Index des dates communes
        """
        # Liste pour stocker tous les ensembles de dates
        all_dates_sets = []
        
        # Récupérer les dates de chaque DataFrame
        for df in data_dict.values():
            dates_set = set(df.index.date)
            all_dates_sets.append(dates_set)
        
        # Trouver l'intersection
        if all_dates_sets:
            common_dates_set = all_dates_sets[0]
            for dates_set in all_dates_sets[1:]:
                common_dates_set &= dates_set
            
            # Convertir en liste, trier et reconvertir en DatetimeIndex
            common_dates_list = sorted(common_dates_set)
            common_dates = pd.DatetimeIndex([pd.Timestamp(d) for d in common_dates_list])
            
            return common_dates
        else:
            return pd.DatetimeIndex([]) 

class BootstrapAnalyzer:
    """
    Analyse la robustesse des stratégies par bootstrap.
    """
    
    def __init__(self, n_iterations: int = 1000, sample_size: float = 0.8, 
                 confidence_level: float = 0.95):
        """
        Initialise l'analyseur bootstrap.
        
        Args:
            n_iterations: Nombre d'itérations bootstrap
            sample_size: Taille relative de l'échantillon (entre 0 et 1)
            confidence_level: Niveau de confiance pour les intervalles (entre 0 et 1)
        """
        self.n_iterations = n_iterations
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        
    def analyze(self, returns: pd.Series, metric_funcs: Dict[str, Callable] = None) -> Dict[str, Dict]:
        """
        Analyse la robustesse des performances d'une stratégie par bootstrap.
        
        Args:
            returns: Série de rendements de la stratégie
            metric_funcs: Dictionnaire de fonctions métriques {nom: fonction}
            
        Returns:
            Dictionnaire avec résultats de l'analyse bootstrap
        """
        # Si aucune fonction métrique n'est fournie, utiliser les métriques par défaut
        if metric_funcs is None:
            metric_funcs = {
                'mean_return': lambda x: x.mean() * 252,  # Rendement annualisé
                'sharpe_ratio': lambda x: x.mean() / x.std() * np.sqrt(252),  # Sharpe annualisé
                'sortino_ratio': self._sortino_ratio,
                'max_drawdown': self._max_drawdown
            }
        
        # Résultats pour chaque métrique
        bootstrap_results = {}
        
        # Effectuer le bootstrap
        for metric_name, metric_func in metric_funcs.items():
            # Valeur originale
            original_value = metric_func(returns)
            
            # Distribution bootstrap
            bootstrap_values = []
            
            for _ in range(self.n_iterations):
                # Échantillonnage aléatoire avec remise
                sample_size = int(len(returns) * self.sample_size)
                bootstrap_sample = returns.sample(n=sample_size, replace=True)
                
                # Calcul de la métrique sur l'échantillon
                bootstrap_value = metric_func(bootstrap_sample)
                bootstrap_values.append(bootstrap_value)
            
            # Calcul des statistiques
            bootstrap_values = np.array(bootstrap_values)
            mean_value = np.mean(bootstrap_values)
            std_value = np.std(bootstrap_values)
            
            # Intervalles de confiance
            alpha = 1 - self.confidence_level
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_values, lower_percentile)
            ci_upper = np.percentile(bootstrap_values, upper_percentile)
            
            # Stocker les résultats
            bootstrap_results[metric_name] = {
                'original': original_value,
                'bootstrap_mean': mean_value,
                'bootstrap_std': std_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': np.mean(bootstrap_values <= 0) if original_value > 0 else np.mean(bootstrap_values >= 0)
            }
        
        return bootstrap_results
    
    def _sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calcule le ratio de Sortino (annualisé).
        
        Args:
            returns: Série de rendements
            
        Returns:
            Ratio de Sortino
        """
        # Calculer le rendement annualisé
        annual_return = returns.mean() * 252
        
        # Calculer la semi-variance négative
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')  # Pas de rendements négatifs
            
        downside_deviation = np.sqrt(np.mean(negative_returns**2) * 252)
        
        if downside_deviation == 0:
            return float('inf')  # Éviter division par zéro
            
        return annual_return / downside_deviation
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """
        Calcule le drawdown maximum.
        
        Args:
            returns: Série de rendements
            
        Returns:
            Drawdown maximum (valeur négative)
        """
        # Calculer la valeur cumulée
        cum_returns = (1 + returns).cumprod()
        
        # Calculer le drawdown
        peak = cum_returns.cummax()
        drawdown = (cum_returns / peak) - 1
        
        return drawdown.min()

class StatisticalTester:
    """
    Effectue des tests statistiques rigoureux sur les résultats des backtests.
    """
    
    def __init__(self, alpha: float = 0.05, config: Dict = None):
        """
        Initialise le testeur statistique.
        
        Args:
            alpha: Seuil de significativité (par défaut 0.05)
            config: Configuration personnalisée (optionnel)
        """
        self.alpha = alpha
        self.test_results = {}
        
        # Configuration par défaut
        self.config = {
            "alpha": alpha,
            "tests": ["mean_return", "sharpe_ratio", "normality", "autocorrelation"],
            "autocorrelation_lags": 10,
            "min_samples": 30
        }
        
        # Mettre à jour avec la configuration personnalisée si fournie
        if config:
            self.config.update(config)
            # Mettre à jour alpha si spécifié dans la config
            if "alpha" in config:
                self.alpha = config["alpha"]
    
    def test_strategy(self, backtest_results):
        """
        Effectue une série de tests sur les résultats d'un backtest.
        
        Args:
            backtest_results: Résultats du backtest (doit contenir au moins une série de rendements)
            
        Returns:
            Résultats des tests statistiques
        """
        # Vérifier que les résultats contiennent une série de rendements
        if not hasattr(backtest_results, 'returns') or backtest_results.returns is None:
            raise ValueError("Les résultats de backtest doivent contenir une série de rendements")
        
        # Série de rendements
        returns = backtest_results.returns
        
        # Effectuer les tests
        self.test_mean_return(returns)
        self.test_sharpe_ratio(returns)
        self.test_normality(returns)
        self.test_autocorrelation(returns)
        
        # Résumer les résultats
        summary = self.summarize_tests()
        
        return summary
        
    def test_mean_return(self, returns: pd.Series) -> Dict:
        """
        Teste si le rendement moyen est significativement différent de zéro.
        
        Args:
            returns: Série de rendements
            
        Returns:
            Résultats du test
        """
        # Test t de Student
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Calculer le rendement annualisé
        annual_return = returns.mean() * 252
        
        # Intervalle de confiance
        ci = stats.t.interval(1 - self.alpha, len(returns) - 1, 
                              loc=returns.mean(), scale=stats.sem(returns))
        
        # Résultats
        results = {
            'statistic': t_stat,
            'p_value': p_value,
            'annual_return': annual_return,
            'ci_lower': ci[0] * 252,  # Annualisé
            'ci_upper': ci[1] * 252,  # Annualisé
            'significant': p_value < self.alpha
        }
        
        self.test_results['mean_return'] = results
        return results
    
    def test_sharpe_ratio(self, returns: pd.Series) -> Dict:
        """
        Teste si le ratio de Sharpe est significativement différent de zéro.
        
        Args:
            returns: Série de rendements
            
        Returns:
            Résultats du test
        """
        # Calculer le ratio de Sharpe
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        # Nombre d'observations
        n = len(returns)
        
        # Calculer l'erreur standard du ratio de Sharpe
        # Formule de Lo (2002)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
        
        # Calculer la statistique Z
        z_stat = sharpe / se_sharpe
        
        # P-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Intervalle de confiance
        ci_lower = sharpe - stats.norm.ppf(1 - self.alpha / 2) * se_sharpe
        ci_upper = sharpe + stats.norm.ppf(1 - self.alpha / 2) * se_sharpe
        
        # Résultats
        results = {
            'sharpe_ratio': sharpe,
            'statistic': z_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < self.alpha
        }
        
        self.test_results['sharpe_ratio'] = results
        return results
    
    def test_normality(self, returns: pd.Series) -> Dict:
        """
        Teste la normalité des rendements.
        
        Args:
            returns: Série de rendements
            
        Returns:
            Résultats du test
        """
        # Test de Jarque-Bera
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Test de Shapiro-Wilk
        sw_stat, sw_pvalue = stats.shapiro(returns)
        
        # Test d'Anderson-Darling
        ad_result = stats.anderson(returns, dist='norm')
        ad_stat = ad_result.statistic
        ad_critical_values = ad_result.critical_values
        ad_significance_levels = [15.0, 10.0, 5.0, 2.5, 1.0]
        
        # Déterminer si on rejette la normalité avec Anderson-Darling
        ad_reject = ad_stat > ad_critical_values[2]  # Niveau 5%
        
        # Statistiques descriptives
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Résultats
        results = {
            'jarque_bera': {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'reject_normality': jb_pvalue < self.alpha
            },
            'shapiro_wilk': {
                'statistic': sw_stat,
                'p_value': sw_pvalue,
                'reject_normality': sw_pvalue < self.alpha
            },
            'anderson_darling': {
                'statistic': ad_stat,
                'critical_values': dict(zip(ad_significance_levels, ad_critical_values)),
                'reject_normality': ad_reject
            },
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_conclusion': "Non-normale" if (jb_pvalue < self.alpha or sw_pvalue < self.alpha or ad_reject) else "Normale"
        }
        
        self.test_results['normality'] = results
        return results
    
    def test_autocorrelation(self, returns: pd.Series, lags: int = 10) -> Dict:
        """
        Teste l'autocorrélation des rendements.
        
        Args:
            returns: Série de rendements
            lags: Nombre de retards à tester
            
        Returns:
            Résultats du test
        """
        # Test de Ljung-Box
        lb_stat, lb_pvalue = stats.acorr_ljungbox(returns, lags=[lags])
        
        # Autocorrélations
        acf, acf_confint = sms.acf(returns, nlags=lags, alpha=self.alpha, fft=True)
        
        # Créer un dictionnaire des autocorrélations par lag
        acf_results = {}
        for lag in range(len(acf)):
            acf_results[lag] = {
                'acf': acf[lag],
                'ci_lower': acf_confint[lag][0],
                'ci_upper': acf_confint[lag][1],
                'significant': (acf[lag] < acf_confint[lag][0]) or (acf[lag] > acf_confint[lag][1])
            }
        
        # Résultats
        results = {
            'ljung_box': {
                'statistic': lb_stat[0],
                'p_value': lb_pvalue[0],
                'reject_no_autocorrelation': lb_pvalue[0] < self.alpha
            },
            'autocorrelations': acf_results,
            'autocorrelation_conclusion': "Présente" if lb_pvalue[0] < self.alpha else "Absente"
        }
        
        self.test_results['autocorrelation'] = results
        return results
    
    def summarize_tests(self, custom_results=None):
        """
        Résume tous les tests effectués.
        
        Args:
            custom_results: Résultats personnalisés à résumer (optionnel)
            
        Returns:
            Dictionnaire résumant les tests
        """
        # Utiliser les résultats personnalisés si fournis, sinon les résultats internes
        results = custom_results if custom_results is not None else self.test_results
        
        if not results:
            return {"message": "Aucun test n'a été effectué"}
        
        summary = {}
        
        # Résumé des rendements
        if 'mean_return' in results:
            summary['mean_return'] = {
                'annual_return': results['mean_return'].get('annual_return', 'N/A'),
                'significant': results['mean_return'].get('significant', False),
                'p_value': results['mean_return'].get('p_value', 1.0)
            }
        
        # Résumé du ratio de Sharpe
        if 'sharpe_ratio' in results:
            summary['sharpe_ratio'] = {
                'value': results['sharpe_ratio'].get('sharpe_ratio', 'N/A'),
                'significant': results['sharpe_ratio'].get('significant', False),
                'p_value': results['sharpe_ratio'].get('p_value', 1.0)
            }
        
        # Résumé de la normalité
        if 'normality' in results:
            summary['normality'] = {
                'conclusion': results['normality'].get('normality_conclusion', 'Inconnue'),
                'skewness': results['normality'].get('skewness', 0),
                'kurtosis': results['normality'].get('kurtosis', 0)
            }
        
        # Résumé de l'autocorrélation
        if 'autocorrelation' in results:
            summary['autocorrelation'] = {
                'conclusion': results['autocorrelation'].get('autocorrelation_conclusion', 'Inconnue'),
                'p_value': results['autocorrelation'].get('ljung_box', {}).get('p_value', 1.0)
            }
        
        return summary 