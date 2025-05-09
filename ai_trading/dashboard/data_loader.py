"""
Module de chargement des données pour le dashboard interactif.

Ce module permet de charger, transformer et préparer les données à afficher
dans le dashboard d'analyse post-trade.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from ai_trading.risk.risk_metrics import (
    calculate_drawdown,
    calculate_es,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
)
from ai_trading.dashboard.memory_optimizer import (
    CACHE_DIR,
    clean_cache,
    disk_cache,
    downscale_for_visualization,
    optimize_dataframe
)


class DataLoader:
    """
    Classe pour charger et préparer les données pour le dashboard.

    Cette classe centralise le chargement des données de transactions,
    de portefeuilles et de performances, et les prépare pour l'affichage.
    """

    def __init__(self, data_path: str = None):
        """
        Initialise le chargeur de données.

        Args:
            data_path: Chemin vers le dossier contenant les données
        """
        self.data_path = data_path or os.path.join("data", "dashboard")
        self._ensure_data_path_exists()

        # Cache pour les données
        self._cache = {}

        # Nettoyer le cache au démarrage
        clean_cache(CACHE_DIR, max_age_hours=72)

    def _ensure_data_path_exists(self):
        """Assure que le chemin de données existe."""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)

    def load_portfolio_history(self, portfolio_id: str = "main") -> pd.DataFrame:
        """
        Charge l'historique des valeurs du portefeuille.

        Args:
            portfolio_id: Identifiant du portefeuille

        Returns:
            DataFrame avec l'historique des valeurs du portefeuille
        """
        # Clé de cache
        cache_key = f"portfolio_history_{portfolio_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tenter de charger les données existantes
        filepath = os.path.join(self.data_path, f"{portfolio_id}_history.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=["date"])
        else:
            # Générer des données factices pour démonstration
            df = self._generate_portfolio_history()
            df.to_csv(filepath, index=False)

        # Mettre en cache
        self._cache[cache_key] = df
        return df

    def _generate_portfolio_history(self, days: int = 180) -> pd.DataFrame:
        """
        Génère des données de portefeuille factices pour démonstration.

        Args:
            days: Nombre de jours d'historique à générer

        Returns:
            DataFrame avec l'historique simulé
        """
        np.random.seed(42)

        # Générer les dates
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Générer la valeur du portefeuille avec une tendance haussière et des fluctuations
        initial_value = 10000
        daily_returns = np.random.normal(
            0.001, 0.015, len(dates)
        )  # moyenne +0.1%, écart-type 1.5%
        daily_returns[0] = 0  # Premier jour sans rendement

        # Ajouter une tendance et quelques événements marquants
        trend = np.linspace(0, 0.002 * len(dates), len(dates))  # Tendance haussière
        daily_returns += trend

        # Simuler quelques crashs/rallyes
        crash_idx = int(len(dates) * 0.3)
        rally_idx = int(len(dates) * 0.7)
        daily_returns[crash_idx : crash_idx + 5] = np.random.normal(-0.03, 0.01, 5)
        daily_returns[rally_idx : rally_idx + 5] = np.random.normal(0.025, 0.01, 5)

        # Calculer les valeurs cumulées
        cumulative_returns = (1 + daily_returns).cumprod()
        portfolio_values = initial_value * cumulative_returns

        # Créer le DataFrame
        df = pd.DataFrame(
            {
                "date": dates,
                "valeur": portfolio_values,
                "rendement_quotidien": daily_returns,
                "rendement_cumulatif": cumulative_returns - 1,
            }
        )

        return df

    def load_transactions(self, portfolio_id: str = "main") -> pd.DataFrame:
        """
        Charge les transactions du portefeuille.

        Args:
            portfolio_id: Identifiant du portefeuille

        Returns:
            DataFrame avec les transactions
        """
        # Clé de cache
        cache_key = f"transactions_{portfolio_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tenter de charger les données existantes
        filepath = os.path.join(self.data_path, f"{portfolio_id}_transactions.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=["date"])
        else:
            # Générer des données factices pour démonstration
            df = self._generate_transactions()
            df.to_csv(filepath, index=False)

        # Mettre en cache
        self._cache[cache_key] = df
        return df

    def _generate_transactions(self, n_transactions: int = 50) -> pd.DataFrame:
        """
        Génère des transactions factices pour démonstration.

        Args:
            n_transactions: Nombre de transactions à générer

        Returns:
            DataFrame avec les transactions simulées
        """
        np.random.seed(42)

        # Définir les actifs possibles
        assets = [
            "BTC",
            "ETH",
            "XRP",
            "ADA",
            "SOL",
            "BNB",
            "DOGE",
            "DOT",
            "MATIC",
            "LINK",
        ]
        transaction_types = ["achat", "vente"]

        # Générer les dates (sur 6 mois)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        dates = [
            start_date + timedelta(days=np.random.randint(0, 180))
            for _ in range(n_transactions)
        ]
        dates.sort()  # Trier par ordre chronologique

        # Générer les transactions
        transactions = []
        for i, date in enumerate(dates):
            asset = np.random.choice(assets)
            tx_type = np.random.choice(transaction_types)

            # Prix simulé
            price = None
            if asset == "BTC":
                price = np.random.uniform(25000, 65000)
            elif asset == "ETH":
                price = np.random.uniform(1500, 4000)
            else:
                price = np.random.uniform(0.1, 500)

            # Quantité simulée
            quantity = (
                np.random.uniform(0.01, 5)
                if asset in ["BTC", "ETH"]
                else np.random.uniform(10, 1000)
            )

            # Montant total
            montant = price * quantity

            # Identifiant de transaction
            tx_id = f"TX{i+1:04d}"

            # Signal d'origine
            signals = [
                "momentum",
                "mean_reversion",
                "pattern_breakout",
                "ml_prediction",
                "portfolio_rebalance",
            ]
            signal = np.random.choice(signals)

            # Performance relative (pour les ventes)
            perf_relative = np.random.uniform(-0.2, 0.4) if tx_type == "vente" else None

            transactions.append(
                {
                    "id": tx_id,
                    "date": date,
                    "actif": asset,
                    "type": tx_type,
                    "prix": price,
                    "quantite": quantity,
                    "montant": montant,
                    "signal": signal,
                    "perf_relative": perf_relative,
                }
            )

        return pd.DataFrame(transactions)

    def load_portfolio_allocation(
        self, date: Optional[str] = None, portfolio_id: str = "main"
    ) -> pd.DataFrame:
        """
        Charge l'allocation d'actifs du portefeuille à une date donnée.

        Args:
            date: Date pour laquelle récupérer l'allocation (latest si None)
            portfolio_id: Identifiant du portefeuille

        Returns:
            DataFrame avec l'allocation d'actifs
        """
        # Clé de cache
        cache_key = f"allocation_{portfolio_id}_{date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tenter de charger les données existantes
        filepath = os.path.join(self.data_path, f"{portfolio_id}_allocations.csv")
        if os.path.exists(filepath):
            all_allocations = pd.read_csv(filepath, parse_dates=["date"])

            # Filtrer par date si spécifiée
            if date:
                df = all_allocations[all_allocations["date"] == pd.to_datetime(date)]
            else:
                # Prendre la plus récente
                df = all_allocations.sort_values("date").iloc[-1:]
        else:
            # Générer des données factices pour démonstration
            all_allocations = self._generate_allocations()
            all_allocations.to_csv(filepath, index=False)

            # Filtrer de la même manière
            if date:
                df = all_allocations[all_allocations["date"] == pd.to_datetime(date)]
            else:
                df = all_allocations.sort_values("date").iloc[-1:]

        # Mettre en cache
        self._cache[cache_key] = df
        return df

    def _generate_allocations(self, n_snapshots: int = 20) -> pd.DataFrame:
        """
        Génère des allocations factices pour démonstration.

        Args:
            n_snapshots: Nombre de snapshots d'allocation à générer

        Returns:
            DataFrame avec les allocations simulées
        """
        np.random.seed(42)

        # Définir les actifs possibles
        assets = [
            "BTC",
            "ETH",
            "XRP",
            "ADA",
            "SOL",
            "BNB",
            "DOGE",
            "DOT",
            "MATIC",
            "LINK",
            "Cash",
        ]

        # Générer les dates (sur 6 mois, à intervalle régulier)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, periods=n_snapshots)

        # Générer les allocations
        allocations = []

        # Base weights qui évoluent lentement au fil du temps
        base_weights = np.array(
            [0.3, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        )

        for i, date in enumerate(dates):
            # Faire évoluer les allocations avec le temps
            noise = np.random.normal(0, 0.02, len(assets))
            weights = (
                base_weights + noise + i * 0.002 * (np.random.random(len(assets)) - 0.5)
            )

            # Normaliser pour assurer que la somme est 1
            weights = np.maximum(weights, 0)  # Pas de poids négatifs
            weights = weights / weights.sum()

            # Créer la ligne du DataFrame
            row = {"date": date}
            for asset, weight in zip(assets, weights):
                row[asset] = weight

            allocations.append(row)

        return pd.DataFrame(allocations)

    def load_risk_metrics(self, portfolio_id: str = "main") -> pd.DataFrame:
        """
        Charge les métriques de risque du portefeuille.

        Args:
            portfolio_id: Identifiant du portefeuille

        Returns:
            DataFrame avec les métriques de risque
        """
        # Calculer les métriques à partir de l'historique du portefeuille
        portfolio_history = self.load_portfolio_history(portfolio_id)
        returns = portfolio_history["rendement_quotidien"].values

        # Calculer les métriques principales
        var_95 = calculate_var(returns, confidence=0.95, method="historical")
        var_99 = calculate_var(returns, confidence=0.99, method="historical")
        es_95 = calculate_es(returns, confidence=0.95, method="historical")
        max_dd, dd_duration, recovery_time = calculate_drawdown(returns)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.01)
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.01)

        # Créer un DataFrame récapitulatif
        df = pd.DataFrame(
            [
                {
                    "metric": "VaR (95%)",
                    "value": var_95,
                    "description": "Perte potentielle maximale avec 95% de confiance",
                },
                {
                    "metric": "VaR (99%)",
                    "value": var_99,
                    "description": "Perte potentielle maximale avec 99% de confiance",
                },
                {
                    "metric": "Expected Shortfall (95%)",
                    "value": es_95,
                    "description": "Perte moyenne au-delà de la VaR 95%",
                },
                {
                    "metric": "Drawdown Maximum",
                    "value": abs(max_dd),
                    "description": "Plus grande perte du pic au creux",
                },
                {
                    "metric": "Durée Drawdown (jours)",
                    "value": dd_duration,
                    "description": "Durée du drawdown maximum en jours",
                },
                {
                    "metric": "Temps de Récupération (jours)",
                    "value": recovery_time,
                    "description": "Temps pour récupérer après le drawdown maximum",
                },
                {
                    "metric": "Ratio de Sharpe",
                    "value": sharpe,
                    "description": "Rendement excédentaire par unité de risque",
                },
                {
                    "metric": "Ratio de Sortino",
                    "value": sortino,
                    "description": "Rendement excédentaire par unité de risque négatif",
                },
            ]
        )

        return df

    def load_factor_exposures(self, portfolio_id: str = "main") -> pd.DataFrame:
        """
        Charge les expositions aux facteurs du portefeuille.

        Args:
            portfolio_id: Identifiant du portefeuille

        Returns:
            DataFrame avec les expositions aux facteurs
        """
        # Clé de cache
        cache_key = f"factor_exposures_{portfolio_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tenter de charger les données existantes
        filepath = os.path.join(self.data_path, f"{portfolio_id}_factor_exposures.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=["date"])
        else:
            # Générer des données factices pour démonstration
            df = self._generate_factor_exposures()
            df.to_csv(filepath, index=False)

        # Mettre en cache
        self._cache[cache_key] = df
        return df

    def _generate_factor_exposures(self, n_snapshots: int = 20) -> pd.DataFrame:
        """
        Génère des expositions aux facteurs factices pour démonstration.

        Args:
            n_snapshots: Nombre de snapshots d'exposition à générer

        Returns:
            DataFrame avec les expositions simulées
        """
        np.random.seed(42)

        # Définir les facteurs
        factors = ["Market", "Size", "Value", "Momentum", "Volatility", "Quality"]

        # Générer les dates (sur 6 mois, à intervalle régulier)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, periods=n_snapshots)

        # Générer les expositions
        exposures = []

        # Base exposures qui évoluent lentement au fil du temps
        base_exposures = np.array([1.0, 0.2, -0.1, 0.4, -0.3, 0.1])

        for i, date in enumerate(dates):
            # Faire évoluer les expositions avec le temps
            noise = np.random.normal(0, 0.05, len(factors))
            current_exposures = (
                base_exposures
                + noise
                + i * 0.003 * (np.random.random(len(factors)) - 0.5)
            )

            # Créer la ligne du DataFrame
            row = {"date": date}
            for factor, exposure in zip(factors, current_exposures):
                row[factor] = exposure

            exposures.append(row)

        return pd.DataFrame(exposures)

    def clear_cache(self):
        """Vide le cache de données."""
        self._cache = {}

    @disk_cache(CACHE_DIR, expiry_hours=24)
    def get_market_data(self, portfolio_id: str = "main") -> pd.DataFrame:
        """
        Récupère les données de marché avec les indicateurs techniques.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            
        Returns:
            DataFrame avec les données de marché et indicateurs techniques
        """
        # Clé de cache
        cache_key = f"market_data_{portfolio_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Tenter de charger les données existantes
        try:
            filename = f"market_data_{portfolio_id}.csv"
            data = self._load_data(filename)
            
            # Optimiser le DataFrame pour réduire la consommation mémoire
            data = optimize_dataframe(data)
            
            # Mettre en cache
            self._cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"Erreur lors du chargement des données de marché: {e}")
            # En cas d'erreur, générer des données synthétiques
            data = self._generate_synthetic_market_data()
            
            # Optimiser et mettre en cache
            data = optimize_dataframe(data)
            self._cache[cache_key] = data
            return data

    @disk_cache(CACHE_DIR, expiry_hours=24)
    def get_trades_data(self, portfolio_id: str = "main") -> pd.DataFrame:
        """
        Récupère les données de transactions.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            
        Returns:
            DataFrame avec les données de transactions
        """
        # Clé de cache
        cache_key = f"trades_data_{portfolio_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Tenter de charger les données existantes
        try:
            filename = f"trades_data_{portfolio_id}.csv"
            data = self._load_data(filename)
            
            # Optimiser le DataFrame pour réduire la consommation mémoire
            data = optimize_dataframe(data)
            
            # Mettre en cache
            self._cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"Erreur lors du chargement des données de transactions: {e}")
            # En cas d'erreur, générer des données synthétiques
            data = self._generate_synthetic_trade_data()
            
            # Optimiser et mettre en cache
            data = optimize_dataframe(data)
            self._cache[cache_key] = data
            return data
    
    def get_market_data_for_visualization(self, portfolio_id: str = "main", max_points: int = 5000) -> pd.DataFrame:
        """
        Récupère les données de marché optimisées pour la visualisation.
        Cette fonction réduit le nombre de points pour les visualisations complexes.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            max_points: Nombre maximum de points à conserver
            
        Returns:
            DataFrame échantillonné pour la visualisation
        """
        try:
            # Charger les données de marché
            data = self.get_market_data(portfolio_id)
            
            # Vérifier que les données ne sont pas vides
            if data is None or data.empty:
                # Si vide, générer des données synthétiques
                data = self._generate_synthetic_market_data()
                print("Utilisation de données de marché synthétiques pour la visualisation")
            
            # Échantillonner les données pour la visualisation
            result = downscale_for_visualization(data, max_points)
            
            # Vérifier que l'échantillonnage a bien fonctionné et que les données ne sont pas vides
            if result is None or result.empty:
                print("Échec de l'échantillonnage des données, utilisation des données complètes")
                return data
            
            return result
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour visualisation: {str(e)}")
            # En cas d'erreur, retourner des données synthétiques
            return self._generate_synthetic_market_data()
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """
        Sauvegarde des données dans un fichier.
        
        Args:
            data: DataFrame à sauvegarder
            filename: Nom du fichier
        """
        file_path = os.path.join(self.data_path, filename)
        
        # Déterminer le format de fichier
        if filename.endswith(".csv"):
            data.to_csv(file_path, index=False)
        elif filename.endswith(".parquet"):
            data.to_parquet(file_path, index=False)
        elif filename.endswith(".pkl") or filename.endswith(".pickle"):
            data.to_pickle(file_path)
        else:
            # Par défaut, sauvegarder en CSV
            if not filename.endswith(".csv"):
                filename += ".csv"
            file_path = os.path.join(self.data_path, filename)
            data.to_csv(file_path, index=False)

    def _load_data(self, filename: str) -> pd.DataFrame:
        """
        Charge un fichier de données.
        
        Args:
            filename: Nom du fichier à charger
            
        Returns:
            DataFrame contenant les données
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            # Si le fichier n'existe pas, essayer de générer des données synthétiques
            return self._generate_synthetic_data(filename)
            
        # Déterminer le format du fichier
        if filename.endswith(".csv"):
            return pd.read_csv(file_path, parse_dates=True)
        elif filename.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif filename.endswith(".pkl") or filename.endswith(".pickle"):
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {filename}")

    def _generate_synthetic_data(self, filename: str) -> pd.DataFrame:
        """
        Génère des données synthétiques si le fichier demandé n'existe pas.
        
        Args:
            filename: Nom du fichier demandé
            
        Returns:
            DataFrame contenant des données synthétiques
        """
        # Générer des données en fonction du type de fichier demandé
        if "market" in filename:
            return self._generate_synthetic_market_data()
        elif "trade" in filename:
            return self._generate_synthetic_trade_data()
        else:
            # Données par défaut si type inconnu
            return pd.DataFrame({
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="D"),
                "value": np.random.randn(100).cumsum()
            })

    def _generate_synthetic_market_data(self) -> pd.DataFrame:
        """
        Génère des données de marché synthétiques.
        
        Returns:
            DataFrame contenant des données de marché synthétiques
        """
        # Paramètres
        start_date = datetime.now() - timedelta(days=365)
        periods = 365
        
        # Dates et prix de base
        dates = pd.date_range(start=start_date, periods=periods, freq="D")
        base_price = 100
        
        # Simuler un mouvement de prix
        np.random.seed(42)  # Pour reproductibilité
        returns = np.random.normal(0.0005, 0.01, periods).cumsum()
        prices = base_price * (1 + returns)
        
        # Créer le DataFrame
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices * (1 + np.random.normal(0, 0.003, periods)),
            "high": prices * (1 + np.random.uniform(0.001, 0.02, periods)),
            "low": prices * (1 - np.random.uniform(0.001, 0.02, periods)),
            "close": prices,
            "volume": np.random.randint(1000, 100000, periods)
        })
        
        # Ajouter des indicateurs techniques
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["rsi"] = self._calculate_rsi(df["close"])
        df["macd"] = self._calculate_macd(df["close"])
        df["volatility"] = df["close"].rolling(window=20).std() / df["close"] * 100
        
        # Remplir les valeurs manquantes
        df.fillna(method="bfill", inplace=True)
        
        return df

    def _generate_synthetic_trade_data(self) -> pd.DataFrame:
        """
        Génère des données de transactions synthétiques.
        
        Returns:
            DataFrame contenant des données de transactions synthétiques
        """
        # Paramètres
        start_date = datetime.now() - timedelta(days=365)
        num_trades = 100
        
        # Générer des dates d'entrée aléatoires
        np.random.seed(42)
        entry_days = np.random.randint(0, 364, num_trades)
        entry_times = [start_date + timedelta(days=int(day)) for day in entry_days]
        
        # Durées de détention
        holding_periods = np.random.randint(1, 30, num_trades)
        exit_times = [entry_time + timedelta(days=int(period)) 
                     for entry_time, period in zip(entry_times, holding_periods)]
        
        # Prix d'entrée et de sortie
        base_price = 100
        entry_prices = base_price * (1 + np.random.normal(0, 0.1, num_trades))
        exit_prices = entry_prices * (1 + np.random.normal(0.01, 0.2, num_trades))
        
        # Types de transactions (achat/vente)
        types = np.random.choice(["achat", "vente"], num_trades)
        
        # Actifs
        symbols = np.random.choice(["BTC", "ETH", "SOL", "ADA", "DOT"], num_trades)
        
        # Montants
        quantities = np.random.uniform(0.1, 10, num_trades)
        
        # Profits
        profits = quantities * (exit_prices - entry_prices)
        # Pour les ventes, inverser le signe
        profits = [p if t == "achat" else -p for p, t in zip(profits, types)]
        
        # Créer le DataFrame
        df = pd.DataFrame({
            "entry_time": entry_times,
            "exit_time": exit_times,
            "symbol": symbols,
            "type": types,
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "quantity": quantities,
            "profit": profits,
            "profit_pct": [(exit_p / entry_p) - 1 if t == "achat" else (entry_p / exit_p) - 1
                         for entry_p, exit_p, t in zip(entry_prices, exit_prices, types)]
        })
        
        # Trier par date d'entrée
        df.sort_values("entry_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calcule l'indicateur RSI.
        
        Args:
            prices: Série de prix
            window: Fenêtre de calcul
            
        Returns:
            Série contenant les valeurs RSI
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """
        Calcule l'indicateur MACD.
        
        Args:
            prices: Série de prix
            fast: Période rapide
            slow: Période lente
            
        Returns:
            Série contenant les valeurs MACD
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        return ema_fast - ema_slow
