import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.trading_system import RLTradingSystem

# Configuration du logger
logger = logging.getLogger("MultiAssetTrading")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class MultiAssetTradingSystem:
    """
    Système de trading multi-actifs qui gère plusieurs cryptomonnaies et actifs traditionnels.
    """

    def __init__(
        self,
        crypto_assets: List[str] = ["BTC", "ETH"],
        traditional_assets: List[str] = ["XAU/USD", "AAPL", "NVDA"],
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.2,
    ):
        """
        Initialise le système de trading multi-actifs.

        Args:
            crypto_assets (List[str]): Liste des cryptomonnaies à trader
            traditional_assets (List[str]): Liste des actifs traditionnels à trader
            initial_balance (float): Balance initiale
            risk_per_trade (float): Risque maximum par trade (en %)
            max_position_size (float): Taille maximum de position par actif (en %)
        """
        self.crypto_assets = crypto_assets
        self.traditional_assets = traditional_assets
        self.assets = crypto_assets + traditional_assets
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        
        # Seuil de volatilité pour le rééquilibrage dynamique
        self.volatility_threshold = 0.15  # 15% est un seuil typique

        # Initialiser les systèmes de trading individuels
        self.trading_systems: Dict[str, RLTradingSystem] = {}
        self.data_integrator = RLDataIntegrator()

        # Initialiser les balances et positions
        self.balance = initial_balance
        self.positions: Dict[str, float] = {asset: 0.0 for asset in self.assets}
        self.prices: Dict[str, float] = {asset: 0.0 for asset in self.assets}
        
        # Initialiser les poids optimaux (allocation par défaut équilibrée)
        self.optimal_weights = {asset: 1.0 / len(self.assets) for asset in self.assets}
        self.weights = self.optimal_weights.copy()

        logger.info(
            f"Système de trading multi-actifs initialisé avec {len(self.assets)} actifs"
        )

    def collect_market_data(
        self, start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Collecte les données de marché pour tous les actifs.

        Args:
            start_date (str): Date de début
            end_date (str): Date de fin

        Returns:
            Dict[str, pd.DataFrame]: Dictionnaire des données de marché par actif
        """
        market_data = {}

        # Collecter les données pour les cryptomonnaies
        for asset in self.crypto_assets:
            try:
                data = self.data_integrator.collect_market_data(
                    symbol=asset,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                )
                market_data[asset] = data
                logger.info(f"Données collectées pour {asset}: {len(data)} points")
            except Exception as e:
                logger.error(
                    f"Erreur lors de la collecte des données pour {asset}: {e}"
                )

        # Collecter les données pour les actifs traditionnels
        for asset in self.traditional_assets:
            try:
                # TODO: Implémenter la collecte pour les actifs traditionnels
                # Pour l'instant, on génère des données synthétiques
                data = self.data_integrator._generate_synthetic_market_data(
                    start_date=start_date, end_date=end_date, interval="1d"
                )
                market_data[asset] = data
                logger.info(
                    f"Données synthétiques générées pour {asset}: {len(data)} points"
                )
            except Exception as e:
                logger.error(
                    f"Erreur lors de la collecte des données pour {asset}: {e}"
                )

        return market_data

    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calcule les métriques du portefeuille.

        Returns:
            Dict[str, float]: Métriques du portefeuille
        """
        total_value = self.balance
        for asset, position in self.positions.items():
            total_value += position * self.prices[asset]

        metrics = {
            "total_value": total_value,
            "return": (total_value - self.initial_balance) / self.initial_balance,
            "positions": self.positions.copy(),
            "prices": self.prices.copy(),
        }

        return metrics

    def calculate_simple_allocation(self):
        """
        Calcule une allocation simple du portefeuille en répartissant le capital de manière égale
        entre tous les actifs.

        Returns:
            dict: Allocation du capital pour chaque actif
        """
        total_assets = len(self.crypto_assets) + len(self.traditional_assets)
        if total_assets == 0:
            return {}

        # Calculer la part égale pour chaque actif
        equal_share = 1.0 / total_assets

        # Créer le dictionnaire d'allocation
        allocation = {}

        # Allouer aux crypto-monnaies
        for asset in self.crypto_assets:
            allocation[asset] = equal_share

        # Allouer aux actifs traditionnels
        for asset in self.traditional_assets:
            allocation[asset] = equal_share

        return allocation

    def set_custom_allocation(self, asset_weights: Dict[str, float]) -> None:
        """
        Définit une allocation personnalisée pour les actifs.

        Args:
            asset_weights (Dict[str, float]): Dictionnaire des poids pour chaque actif
                Exemple: {"BTC": 0.4, "ETH": 0.3, "XAU/USD": 0.3}
        """
        # Vérifier que tous les actifs existent
        for asset in asset_weights.keys():
            if asset not in self.assets:
                raise ValueError(f"Actif {asset} non reconnu")

        # Vérifier que la somme des poids est égale à 1
        total_weight = sum(asset_weights.values())
        if abs(total_weight - 1.0) > 1e-10:
            raise ValueError("La somme des poids doit être égale à 1")

        # Mettre à jour l'allocation
        self.custom_allocation = asset_weights
        logger.info(f"Allocation personnalisée définie: {asset_weights}")

    def get_current_allocation(self) -> Dict[str, float]:
        """
        Retourne l'allocation actuelle du portefeuille.

        Returns:
            Dict[str, float]: Allocation actuelle
        """
        if hasattr(self, "custom_allocation"):
            return self.custom_allocation
        return self.calculate_simple_allocation()

    def calculate_volatility_based_allocation(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calcule une allocation basée sur la volatilité inverse des actifs.
        Les actifs moins volatils reçoivent une plus grande allocation.

        Args:
            market_data (Dict[str, pd.DataFrame]): Données de marché par actif

        Returns:
            Dict[str, float]: Allocation basée sur la volatilité
        """
        volatilities = {}
        for asset, data in market_data.items():
            # Calculer la volatilité (écart-type des rendements)
            returns = data["close"].pct_change().dropna()
            volatility = returns.std()
            volatilities[asset] = volatility

        # Calculer les poids inversement proportionnels à la volatilité
        total_inverse_vol = sum(1 / v for v in volatilities.values())
        allocation = {
            asset: (1 / vol) / total_inverse_vol for asset, vol in volatilities.items()
        }

        logger.info(f"Allocation basée sur la volatilité: {allocation}")
        return allocation

    def calculate_correlation_based_allocation(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calcule une allocation basée sur la corrélation entre les actifs.
        Les actifs moins corrélés reçoivent une plus grande allocation.

        Args:
            market_data (Dict[str, pd.DataFrame]): Données de marché par actif

        Returns:
            Dict[str, float]: Allocation basée sur la corrélation
        """
        # Créer une matrice de rendements
        returns_df = pd.DataFrame()
        for asset, data in market_data.items():
            returns_df[asset] = data["close"].pct_change()

        # Calculer la matrice de corrélation
        corr_matrix = returns_df.corr()

        # Calculer la somme des corrélations pour chaque actif
        total_corr = corr_matrix.sum(axis=1)

        # Calculer les poids inversement proportionnels à la somme des corrélations
        total_inverse_corr = sum(1 / corr for corr in total_corr)
        allocation = {
            asset: (1 / corr) / total_inverse_corr for asset, corr in total_corr.items()
        }

        logger.info(f"Allocation basée sur la corrélation: {allocation}")
        return allocation

    def calculate_risk_parity_allocation(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calcule une allocation de parité de risque qui combine volatilité et corrélation.
        Cette méthode vise à égaliser la contribution au risque de chaque actif.

        Args:
            market_data (Dict[str, pd.DataFrame]): Données de marché par actif

        Returns:
            Dict[str, float]: Allocation de parité de risque
        """
        # Créer une matrice de rendements
        returns_df = pd.DataFrame()
        for asset, data in market_data.items():
            returns_df[asset] = data["close"].pct_change()

        # Calculer la matrice de covariance
        cov_matrix = returns_df.cov()

        # Calculer la volatilité de chaque actif
        volatilities = np.sqrt(np.diag(cov_matrix))

        # Calculer les poids de parité de risque
        inverse_vol = 1 / volatilities
        total_inverse_vol = sum(inverse_vol)
        allocation = {
            asset: vol / total_inverse_vol
            for asset, vol in zip(returns_df.columns, inverse_vol)
        }

        logger.info(f"Allocation de parité de risque: {allocation}")
        return allocation

    def calculate_adaptive_allocation(
        self, market_data: Dict[str, pd.DataFrame], weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calcule une allocation adaptative qui combine plusieurs stratégies.

        Args:
            market_data (Dict[str, pd.DataFrame]): Données de marché par actif
            weights (Dict[str, float]): Poids pour chaque stratégie
                Par défaut: {
                    'volatility': 0.4,
                    'correlation': 0.3,
                    'risk_parity': 0.3
                }

        Returns:
            Dict[str, float]: Allocation finale combinée
        """
        if weights is None:
            weights = {"volatility": 0.4, "correlation": 0.3, "risk_parity": 0.3}

        # Calculer les allocations pour chaque stratégie
        vol_allocation = self.calculate_volatility_based_allocation(market_data)
        corr_allocation = self.calculate_correlation_based_allocation(market_data)
        risk_allocation = self.calculate_risk_parity_allocation(market_data)

        # Combiner les allocations
        final_allocation = {}
        for asset in self.assets:
            final_allocation[asset] = (
                vol_allocation[asset] * weights["volatility"]
                + corr_allocation[asset] * weights["correlation"]
                + risk_allocation[asset] * weights["risk_parity"]
            )

        # Normaliser pour que la somme soit égale à 1
        total = sum(final_allocation.values())
        final_allocation = {
            asset: weight / total for asset, weight in final_allocation.items()
        }

        logger.info(f"Allocation adaptative finale: {final_allocation}")
        return final_allocation

    def calculate_rebalancing_threshold(
        self, current_allocation: Dict[str, float], target_allocation: Dict[str, float]
    ) -> float:
        """
        Calcule le seuil de rééquilibrage en fonction de l'écart entre l'allocation actuelle et cible.

        Args:
            current_allocation (Dict[str, float]): Allocation actuelle
            target_allocation (Dict[str, float]): Allocation cible

        Returns:
            float: Seuil de rééquilibrage
        """
        # Calculer l'écart maximum
        max_deviation = max(
            abs(current_allocation[asset] - target_allocation[asset])
            for asset in self.assets
        )

        # Ajuster le seuil en fonction de la volatilité du marché
        market_volatility = self.calculate_market_volatility()
        base_threshold = 0.05  # 5% de déviation par défaut

        # Réduire le seuil en période de haute volatilité
        if market_volatility > 0.02:  # 2% de volatilité
            base_threshold *= 0.8

        return base_threshold

    def calculate_market_volatility(
        self, market_data: Dict[str, pd.DataFrame] = None
    ) -> float:
        """
        Calcule la volatilité moyenne du marché.

        Args:
            market_data: Données de marché pour le calcul de la volatilité
        """
        volatilities = []
        data_to_use = market_data if market_data is not None else self.prices

        for asset in self.assets:
            if asset in data_to_use:
                # Traiter différemment selon le type de données
                if isinstance(data_to_use[asset], pd.DataFrame):
                    # Si c'est un DataFrame, utiliser la colonne 'close'
                    if 'close' in data_to_use[asset].columns:
                        prices = data_to_use[asset]["close"].values
                        if len(prices) > 1:
                            returns = np.diff(prices) / prices[:-1]
                            volatility = np.std(returns)
                            volatilities.append(volatility)
                elif isinstance(data_to_use[asset], (list, np.ndarray)):
                    # Si c'est une liste ou un array
                    prices = data_to_use[asset]
                    if len(prices) > 1:
                        returns = np.diff(prices) / prices[:-1]
                        volatility = np.std(returns)
                        volatilities.append(volatility)
                else:
                    # Si c'est une valeur unique (float), on ne peut pas calculer la volatilité
                    # et on la saute
                    continue

        # Retourner la volatilité moyenne ou 0 s'il n'y a pas assez de données
        return float(np.mean(volatilities)) if volatilities else 0.0

    def needs_rebalancing(
        self, current_allocation: Dict[str, float], target_allocation: Dict[str, float]
    ) -> bool:
        """
        Détermine si le portefeuille a besoin d'être rééquilibré.

        Args:
            current_allocation (Dict[str, float]): Allocation actuelle
            target_allocation (Dict[str, float]): Allocation cible

        Returns:
            bool: True si le rééquilibrage est nécessaire
        """
        threshold = self.calculate_rebalancing_threshold(
            current_allocation, target_allocation
        )
        
        # Calculer la déviation maximale
        max_deviation = max(
            abs(current_allocation[asset] - target_allocation[asset])
            for asset in self.assets
        )
        
        # Si un actif a une déviation très élevée (>15%), rééquilibrer immédiatement
        if max_deviation > 0.15:
            return True
            
        # Calculer la somme totale des déviations
        total_deviation = sum(
            abs(current_allocation[asset] - target_allocation[asset])
            for asset in self.assets
        )
        
        # Si la somme des déviations est significative (>20%), rééquilibrer
        if total_deviation > 0.20:
            return True

        # Vérifier si l'écart dépasse le seuil pour au moins un actif
        for asset in self.assets:
            deviation = abs(current_allocation[asset] - target_allocation[asset])
            if deviation > threshold:
                return True

        return False

    def rebalance_portfolio(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Rééquilibre le portefeuille en fonction des conditions du marché.

        Args:
            market_data: Données de marché pour le rééquilibrage
        """
        current_weights = self.calculate_dynamic_weights(market_data)
        self.weights = current_weights
        
        # Calculer une nouvelle allocation basée sur les conditions de marché actuelles
        new_allocation = self.calculate_adaptive_allocation(market_data)
        
        # Mettre à jour l'allocation personnalisée
        if hasattr(self, "custom_allocation"):
            self.custom_allocation = new_allocation
        else:
            self.set_custom_allocation(new_allocation)
            
        return new_allocation

    def calculate_dynamic_weights(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calcule les poids dynamiques en fonction de la volatilité et de la corrélation du marché.

        Args:
            market_data: Données de marché pour le calcul des poids
        """
        market_volatility = self.calculate_market_volatility(market_data)
        market_correlation = self.calculate_market_correlation(market_data)
        
        # Initialiser le dictionnaire des poids
        result = {}
        
        # Poids pour la volatilité - plus élevé pendant les périodes volatiles
        # Augmenter le poids de volatilité lorsque la volatilité du marché est élevée
        volatility_ratio = market_volatility / self.volatility_threshold
        volatility_weight = min(0.7, max(0.41, volatility_ratio * 0.5))
        
        # Poids pour la corrélation - plus élevé pendant les périodes stables
        # Inverse par rapport à la volatilité : quand la volatilité est faible, 
        # le marché est plus stable et on peut se fier davantage aux corrélations
        correlation_weight = min(0.6, max(0.2, 0.8 - volatility_weight))
        
        # Ajuster pour que les poids volatilité + corrélation ne dépassent pas 0.8
        total_weight = volatility_weight + correlation_weight
        if total_weight > 0.8:
            scaling_factor = 0.8 / total_weight
            volatility_weight *= scaling_factor
            correlation_weight *= scaling_factor
        
        result["volatility"] = volatility_weight
        result["correlation"] = correlation_weight
        
        # Répartir le reste des poids entre les actifs
        remaining_weight = 1.0 - (volatility_weight + correlation_weight)
        asset_weights = {}
        
        if market_volatility > self.volatility_threshold:
            # En période de forte volatilité, distribution égale entre les actifs
            for asset in self.assets:
                asset_weights[asset] = remaining_weight / len(self.assets)
        else:
            # En période de faible volatilité, utiliser les poids optimaux ou une distribution égale
            if hasattr(self, "optimal_weights") and self.optimal_weights:
                # Normaliser les poids optimaux pour qu'ils somment à remaining_weight
                total = sum(self.optimal_weights.values())
                for asset, weight in self.optimal_weights.items():
                    asset_weights[asset] = (weight / total) * remaining_weight
            else:
                # Si pas de poids optimaux, distribution égale
                for asset in self.assets:
                    asset_weights[asset] = remaining_weight / len(self.assets)
        
        # Combiner les deux dictionnaires
        result.update(asset_weights)
        
        return result

    def calculate_market_trend(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calcule la tendance moyenne du marché basée sur les moyennes mobiles.

        Args:
            market_data: Données de marché pour le calcul de la tendance

        Returns:
            float: Tendance du marché (-1 à 1)
        """
        if not market_data:
            return 0.0

        trends = []
        for asset, data in market_data.items():
            try:
                if (
                    not isinstance(data, pd.DataFrame)
                    or "close" not in data.columns
                    or len(data) < 20
                ):
                    continue

                # Calculer la moyenne mobile sur 20 périodes
                ma20 = data["close"].rolling(window=20).mean()
                if ma20.isna().all():
                    continue

                # Calculer la tendance basée sur la pente de la MA20
                latest_ma = ma20.iloc[-1]
                prev_ma = ma20.iloc[-20] if len(ma20) >= 20 else ma20.iloc[0]

                if latest_ma > 0 and prev_ma > 0:  # Éviter la division par zéro
                    trend = (latest_ma - prev_ma) / prev_ma
                    # Normaliser entre -1 et 1
                    trend = max(min(trend, 1), -1)
                    trends.append(trend)

            except Exception as e:
                logger.error(
                    f"Erreur lors du calcul de la tendance pour {asset}: {str(e)}"
                )
                continue

        return float(np.mean(trends)) if trends else 0.0

    def calculate_market_correlation(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Calcule la corrélation moyenne entre les actifs du marché.

        Args:
            market_data: Données de marché pour le calcul des corrélations

        Returns:
            float: Corrélation moyenne du marché
        """
        if not market_data or len(market_data) < 2:
            return 0.0

        # Créer un DataFrame avec les rendements de tous les actifs
        returns_df = pd.DataFrame()
        for asset, data in market_data.items():
            if isinstance(data, pd.DataFrame) and "close" in data.columns:
                returns_df[asset] = data["close"].pct_change()

        if returns_df.empty or len(returns_df.columns) < 2:
            return 0.0

        # Calculer la matrice de corrélation
        correlation_matrix = returns_df.corr()

        # Exclure la diagonale (corrélation avec soi-même)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlations = correlation_matrix.where(mask).stack()

        # Retourner la moyenne des corrélations
        return correlations.mean() if not correlations.empty else 0.0

    def update_positions(self, actions: Dict[str, float]) -> None:
        """
        Met à jour les positions selon les actions décidées.

        Args:
            actions (Dict[str, float]): Actions pour chaque actif (-1 à 1)
        """
        for asset, action in actions.items():
            if asset not in self.assets:
                logger.warning(f"Actif {asset} non reconnu")
                continue

            try:
                # Vérifier si nous avons un prix valide
                price = self.prices.get(asset, 0)
                if price <= 0:
                    logger.warning(f"Prix invalide pour {asset}: {price}")
                    continue

                # Calculer la taille de la position
                position_size = action * self.max_position_size * self.balance
                current_position = self.positions.get(asset, 0.0)

                # Calculer la différence à trader
                diff = position_size - (current_position * price)

                # Mettre à jour la balance et la position
                self.balance -= diff
                self.positions[asset] = position_size / price

            except Exception as e:
                logger.error(
                    f"Erreur lors de la mise à jour de la position pour {asset}: {str(e)}"
                )
                continue

    def train(self, market_data: Dict[str, pd.DataFrame], epochs: int = 100) -> None:
        """
        Entraîne les systèmes de trading pour chaque actif.

        Args:
            market_data (Dict[str, pd.DataFrame]): Données de marché par actif
            epochs (int): Nombre d'époques d'entraînement
        """
        if not market_data:
            logger.error("Aucune donnée de marché fournie pour l'entraînement")
            return

        for asset, data in market_data.items():
            try:
                if not isinstance(data, pd.DataFrame) or data.empty:
                    logger.warning(
                        f"Données invalides pour {asset}, passage à l'actif suivant"
                    )
                    continue

                # Vérifier les colonnes requises
                required_columns = ["open", "high", "low", "close", "volume"]
                if not all(col in data.columns for col in required_columns):
                    logger.warning(
                        f"Données incomplètes pour {asset}, passage à l'actif suivant"
                    )
                    continue

                # Créer un système de trading pour l'actif s'il n'existe pas
                if asset not in self.trading_systems:
                    config = {
                        "initial_balance": self.initial_balance / len(self.assets),
                        "risk_per_trade": self.risk_per_trade,
                        "max_position_size": self.max_position_size,
                    }
                    self.trading_systems[asset] = RLTradingSystem(config=config)
                    logger.info(f"Système de trading créé pour {asset}")

                # Entraîner le système
                logger.info(f"Début de l'entraînement pour {asset}")
                self.trading_systems[asset].train(data, epochs=epochs)
                logger.info(f"Entraînement terminé pour {asset}")

            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement pour {asset}: {str(e)}")
                continue

    def adjust_positions_for_correlation(
        self, actions: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Ajuste les positions en fonction des corrélations entre actifs pour améliorer la diversification.

        Args:
            actions (Dict[str, float]): Actions proposées pour chaque actif
            market_data (Dict[str, pd.DataFrame]): Données de marché pour le calcul des corrélations

        Returns:
            Dict[str, float]: Actions ajustées
        """
        # Calculer la matrice de corrélation
        returns_df = pd.DataFrame()
        for asset, data in market_data.items():
            if isinstance(data, pd.DataFrame) and "close" in data.columns:
                returns_df[asset] = data["close"].pct_change()

        if returns_df.empty or len(returns_df.columns) < 2:
            return actions

        correlation_matrix = returns_df.corr()
        adjusted_actions = actions.copy()

        # Paramètres de diversification
        max_correlation = 0.7  # Corrélation maximale acceptable
        min_weight = -0.5  # Poids minimum par actif
        max_weight = 0.5  # Poids maximum par actif

        # Ajuster les positions pour les paires fortement corrélées
        for asset1 in self.assets:
            for asset2 in self.assets:
                if asset1 >= asset2:
                    continue

                correlation = correlation_matrix.loc[asset1, asset2]
                if abs(correlation) > max_correlation:
                    # Réduire les positions similaires
                    if correlation > 0 and actions[asset1] * actions[asset2] > 0:
                        # Réduire la position la plus importante
                        if abs(actions[asset1]) > abs(actions[asset2]):
                            adjusted_actions[asset1] *= 0.8
                        else:
                            adjusted_actions[asset2] *= 0.8
                    # Augmenter les positions opposées
                    elif correlation < 0 and actions[asset1] * actions[asset2] < 0:
                        # Augmenter la position la plus faible
                        if abs(actions[asset1]) < abs(actions[asset2]):
                            adjusted_actions[asset1] *= 1.2
                        else:
                            adjusted_actions[asset2] *= 1.2

        # Normaliser les positions ajustées
        for asset in adjusted_actions:
            adjusted_actions[asset] = max(
                min(adjusted_actions[asset], max_weight), min_weight
            )

        return adjusted_actions

    def predict_actions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Prédit les actions pour chaque actif.

        Args:
            market_data (Dict[str, pd.DataFrame]): Données de marché par actif

        Returns:
            Dict[str, float]: Actions prédites pour chaque actif (-1 à 1)
        """
        if not market_data:
            logger.error("Aucune donnée de marché fournie pour la prédiction")
            return {asset: 0.0 for asset in self.assets}

        actions = {}
        for asset, data in market_data.items():
            try:
                if asset not in self.trading_systems:
                    logger.warning(f"Aucun système de trading trouvé pour {asset}")
                    actions[asset] = 0.0
                    continue

                if not isinstance(data, pd.DataFrame) or data.empty:
                    logger.warning(f"Données invalides pour {asset}")
                    actions[asset] = 0.0
                    continue

                if "close" not in data.columns:
                    logger.warning(f"Données de prix manquantes pour {asset}")
                    actions[asset] = 0.0
                    continue

                action = self.trading_systems[asset].predict_action(data.iloc[-1:])
                actions[asset] = action
                logger.info(f"Action prédite pour {asset}: {action:.4f}")

            except Exception as e:
                logger.error(f"Erreur lors de la prédiction pour {asset}: {str(e)}")
                actions[asset] = 0.0

        # Appliquer les contraintes de corrélation
        adjusted_actions = self.adjust_positions_for_correlation(actions, market_data)

        return adjusted_actions
