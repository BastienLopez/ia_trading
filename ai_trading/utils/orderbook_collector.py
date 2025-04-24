import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderBookCollector:
    """Collecte et analyse les données du carnet d'ordres et du flux d'ordres pour plusieurs actifs."""
    
    def __init__(self, symbols_config: List[Dict] = None):
        """
        Initialise le collecteur de données du carnet d'ordres.
        
        Args:
            symbols_config (List[Dict]): Liste de configurations pour chaque symbole
                Exemple: [
                    {'exchange': 'binance', 'symbol': 'BTC/USDT', 'limit': 100},
                    {'exchange': 'binance', 'symbol': 'ETH/USDT', 'limit': 100},
                    {'exchange': 'kraken', 'symbol': 'BTC/USD', 'limit': 100}
                ]
        """
        if symbols_config is None:
            symbols_config = [{'exchange': 'binance', 'symbol': 'BTC/USDT', 'limit': 100}]
            
        self.exchanges = {}
        self.symbols_config = symbols_config
        
        # Initialisation des connexions aux exchanges
        for config in symbols_config:
            exchange_id = config['exchange']
            if exchange_id not in self.exchanges:
                try:
                    self.exchanges[exchange_id] = getattr(ccxt, exchange_id)()
                    logger.info(f"Exchange {exchange_id} initialisé avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors de l'initialisation de {exchange_id}: {e}")
                    
    def fetch_orderbook(self, exchange_id: str, symbol: str, limit: int = 100) -> Dict:
        """
        Récupère le carnet d'ordres pour un symbole spécifique.
        
        Args:
            exchange_id (str): ID de l'exchange
            symbol (str): Paire de trading
            limit (int): Nombre de niveaux de prix
            
        Returns:
            Dict: Carnet d'ordres
        """
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ValueError(f"Exchange {exchange_id} non initialisé")
            return exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres pour {symbol} sur {exchange_id}: {e}")
            return None
            
    def calculate_orderbook_features(self, orderbook: Dict) -> Dict[str, float]:
        """
        Calcule les caractéristiques du carnet d'ordres.
        
        Args:
            orderbook (Dict): Carnet d'ordres
            
        Returns:
            Dict[str, float]: Caractéristiques calculées
        """
        if not orderbook:
            return {}
            
        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])
        
        bid_prices, bid_volumes = bids[:, 0], bids[:, 1]
        ask_prices, ask_volumes = asks[:, 0], asks[:, 1]
        
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        spread = ask_prices[0] - bid_prices[0]
        spread_pct = (spread / mid_price) * 100
        
        # Calcul des volumes cumulés
        cum_bid_volume = np.cumsum(bid_volumes)
        cum_ask_volume = np.cumsum(ask_volumes)
        
        # Calcul de l'imbalance
        total_bid_volume = cum_bid_volume[-1]
        total_ask_volume = cum_ask_volume[-1]
        volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        
        # Calcul de la profondeur à différents niveaux
        depth_levels = [5, 10, 20]
        depth_features = {}
        
        for level in depth_levels:
            if len(bid_prices) >= level and len(ask_prices) >= level:
                depth_range_bids = bid_prices[0] - bid_prices[level-1]
                depth_range_asks = ask_prices[level-1] - ask_prices[0]
                depth_features.update({
                    f'depth_range_bids_{level}': depth_range_bids,
                    f'depth_range_asks_{level}': depth_range_asks,
                    f'depth_volume_bids_{level}': cum_bid_volume[level-1],
                    f'depth_volume_asks_{level}': cum_ask_volume[level-1]
                })
        
        features = {
            'mid_price': mid_price,
            'spread': spread,
            'spread_pct': spread_pct,
            'volume_imbalance': volume_imbalance,
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            **depth_features
        }
        
        return features
        
    def collect_orderbook_data(self, duration_minutes: int = 60, interval_seconds: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Collecte les données du carnet d'ordres pour tous les symboles configurés.
        
        Args:
            duration_minutes (int): Durée de collecte en minutes
            interval_seconds (int): Intervalle entre chaque collecte
            
        Returns:
            Dict[str, pd.DataFrame]: Données collectées par symbole
        """
        logger.info(f"Début de la collecte des données pour {duration_minutes} minutes...")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        data_by_symbol = {f"{config['exchange']}_{config['symbol']}": [] for config in self.symbols_config}
        
        while datetime.now() < end_time:
            for config in self.symbols_config:
                exchange_id = config['exchange']
                symbol = config['symbol']
                limit = config.get('limit', 100)
                
                orderbook = self.fetch_orderbook(exchange_id, symbol, limit)
                if orderbook:
                    features = self.calculate_orderbook_features(orderbook)
                    features['timestamp'] = datetime.now()
                    features['symbol'] = symbol
                    features['exchange'] = exchange_id
                    data_by_symbol[f"{exchange_id}_{symbol}"].append(features)
            
            time.sleep(interval_seconds)
            
        # Conversion en DataFrames
        result = {}
        for key, data in data_by_symbol.items():
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Collecte terminée pour {key}. {len(df)} échantillons collectés.")
                result[key] = df
            
        return result
        
    def get_vwap_levels(self, orderbook: Dict, volume_threshold: float) -> Tuple[float, float]:
        """
        Calcule les niveaux VWAP (Volume-Weighted Average Price) pour un volume donné.
        
        Args:
            orderbook (Dict): Carnet d'ordres
            volume_threshold (float): Volume cible pour le calcul VWAP
            
        Returns:
            Tuple[float, float]: VWAP bid et ask
        """
        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])
        
        def calculate_vwap(orders, volume_target):
            cumulative_volume = 0
            weighted_price = 0
            
            for price, volume in orders:
                volume_to_use = min(volume, volume_target - cumulative_volume)
                weighted_price += price * volume_to_use
                cumulative_volume += volume_to_use
                
                if cumulative_volume >= volume_target:
                    break
                    
            return weighted_price / cumulative_volume if cumulative_volume > 0 else None
            
        vwap_bid = calculate_vwap(bids, volume_threshold)
        vwap_ask = calculate_vwap(asks, volume_threshold)
        
        return vwap_bid, vwap_ask 

    def calculate_slippage(self, orderbook: Dict, side: str, volume: float) -> float:
        """
        Calcule le slippage estimé pour un volume donné.
        
        Args:
            orderbook (Dict): Carnet d'ordres
            side (str): 'buy' ou 'sell'
            volume (float): Volume à exécuter
            
        Returns:
            float: Slippage estimé en pourcentage
        """
        if not orderbook:
            return 0.0
            
        orders = np.array(orderbook['asks'] if side == 'buy' else orderbook['bids'])
        best_price = orders[0][0]
        
        cumulative_volume = 0
        weighted_price = 0
        
        for price, size in orders:
            if cumulative_volume >= volume:
                break
            volume_to_use = min(size, volume - cumulative_volume)
            weighted_price += price * volume_to_use
            cumulative_volume += volume_to_use
            
        if cumulative_volume == 0:
            return 0.0
            
        average_price = weighted_price / cumulative_volume
        slippage = abs((average_price - best_price) / best_price) * 100
        
        return slippage
        
    def estimate_market_impact(self, orderbook: Dict, side: str, volume: float) -> Dict[str, float]:
        """
        Estime l'impact sur le marché d'un ordre donné.
        
        Args:
            orderbook (Dict): Carnet d'ordres
            side (str): 'buy' ou 'sell'
            volume (float): Volume à exécuter
            
        Returns:
            Dict[str, float]: Métriques d'impact estimées
        """
        if not orderbook:
            return {}
            
        orders = np.array(orderbook['asks'] if side == 'buy' else orderbook['bids'])
        total_volume = np.sum(orders[:, 1])
        
        # Calcul de l'impact immédiat
        immediate_impact = self.calculate_slippage(orderbook, side, volume)
        
        # Estimation de la résilience du marché
        volume_ratio = volume / total_volume
        
        # Estimation du temps de récupération (en secondes)
        recovery_time = 60 * (volume_ratio ** 2)  # Formule simplifiée
        
        return {
            'immediate_impact_pct': immediate_impact,
            'volume_ratio': volume_ratio,
            'estimated_recovery_time': recovery_time
        }
        
    def simulate_execution_delay(self, volume: float, current_load: float = 0.5) -> float:
        """
        Simule le délai d'exécution en fonction du volume et de la charge du marché.
        
        Args:
            volume (float): Volume à exécuter
            current_load (float): Charge actuelle du marché (0-1)
            
        Returns:
            float: Délai estimé en secondes
        """
        # Délai de base (en secondes)
        base_delay = 0.1
        
        # Facteurs d'ajustement
        volume_factor = np.log1p(volume) * 0.2 + (volume / 100)  # Ajout d'un terme linéaire pour les gros volumes
        load_factor = np.exp(current_load) - 1
        
        # Délai total estimé
        total_delay = base_delay * (1 + volume_factor) * (1 + load_factor)
        
        return total_delay
        
    def get_execution_metrics(self, orderbook: Dict, side: str, volume: float) -> Dict[str, float]:
        """
        Calcule toutes les métriques d'exécution pour un ordre donné.
        
        Args:
            orderbook (Dict): Carnet d'ordres
            side (str): 'buy' ou 'sell'
            volume (float): Volume à exécuter
            
        Returns:
            Dict[str, float]: Métriques d'exécution
        """
        # Calcul du slippage
        slippage = self.calculate_slippage(orderbook, side, volume)
        
        # Estimation de l'impact marché
        market_impact = self.estimate_market_impact(orderbook, side, volume)
        
        # Simulation du délai d'exécution
        # On utilise le ratio volume comme proxy de la charge du marché
        execution_delay = self.simulate_execution_delay(volume, market_impact['volume_ratio'])
        
        metrics = {
            'slippage_pct': slippage,
            'execution_delay_seconds': execution_delay,
            **market_impact
        }
        
        return metrics 

    def calibrate_models(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calibre les modèles d'impact et de délai avec des données historiques.
        
        Args:
            historical_data (pd.DataFrame): Données historiques avec colonnes:
                - timestamp
                - volume
                - price_before
                - price_after
                - execution_time
                
        Returns:
            Dict[str, float]: Paramètres calibrés
        """
        if len(historical_data) == 0:
            return {}
            
        # Calibration du modèle d'impact
        impact_params = self._calibrate_market_impact(historical_data)
        
        # Calibration du modèle de délai
        delay_params = self._calibrate_execution_delay(historical_data)
        
        # Calibration du modèle de slippage
        slippage_params = self._calibrate_slippage(historical_data)
        
        return {**impact_params, **delay_params, **slippage_params}
        
    def _calibrate_market_impact(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calibre le modèle d'impact marché."""
        # Calcul de l'impact réel
        data['real_impact'] = ((data['price_after'] - data['price_before']) / data['price_before']) * 100
        
        # Régression pour trouver la relation volume/impact
        volumes = data['volume'].values.reshape(-1, 1)
        impacts = data['real_impact'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(np.log1p(volumes), impacts)
        
        return {
            'impact_coefficient': float(model.coef_[0]),
            'impact_intercept': float(model.intercept_)
        }
        
    def _calibrate_execution_delay(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calibre le modèle de délai d'exécution."""
        # Analyse des délais réels
        delays = data['execution_time'].values
        volumes = data['volume'].values
        
        # Calcul des paramètres de base
        base_delay = np.percentile(delays, 10)  # 10ème percentile comme délai de base
        max_delay = np.percentile(delays, 90)   # 90ème percentile comme délai max
        
        # Régression pour la relation volume/délai
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(np.log1p(volumes).reshape(-1, 1), delays)
        
        return {
            'base_delay': float(base_delay),
            'max_delay': float(max_delay),
            'delay_coefficient': float(model.coef_[0])
        }
        
    def _calibrate_slippage(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calibre le modèle de slippage."""
        # Calcul du slippage réel
        data['real_slippage'] = abs((data['price_after'] - data['price_before']) / data['price_before']) * 100
        
        # Régression pour la relation volume/slippage
        volumes = data['volume'].values.reshape(-1, 1)
        slippages = data['real_slippage'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(np.log1p(volumes), slippages)
        
        return {
            'slippage_coefficient': float(model.coef_[0]),
            'slippage_intercept': float(model.intercept_)
        }
        
    def apply_calibration(self, params: Dict[str, float]):
        """
        Applique les paramètres calibrés aux modèles.
        
        Args:
            params (Dict[str, float]): Paramètres calibrés
        """
        self.model_params = params
        logger.info("Paramètres de modèle mis à jour avec les données calibrées") 