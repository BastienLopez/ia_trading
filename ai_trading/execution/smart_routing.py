"""
Routage intelligent des ordres pour l'exécution optimisée.

Ce module implémente la sélection optimale des venues d'exécution, l'arbitrage
des frais entre plateformes et l'agrégation de liquidité multi-exchange.
"""

import enum
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import heapq

logger = logging.getLogger(__name__)


class ExchangeVenue(enum.Enum):
    """Plateformes d'échange supportées."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BITFINEX = "bitfinex"
    FTX = "ftx"
    BITSTAMP = "bitstamp"
    HUOBI = "huobi"
    BYBIT = "bybit"
    DERIBIT = "deribit"
    OKX = "okx"


class ExchangePriority(enum.Enum):
    """Priorités pour la sélection des venues d'exécution."""
    FEES = "fees"           # Privilégie les frais les plus bas
    LIQUIDITY = "liquidity"  # Privilégie la meilleure liquidité
    SPEED = "speed"         # Privilégie la vitesse d'exécution
    SLIPPAGE = "slippage"   # Privilégie le moindre slippage
    SMART = "smart"         # Équilibre optimal basé sur tous les critères


class ExchangeInfo:
    """Informations sur une plateforme d'échange pour le routage d'ordres."""
    
    def __init__(self, 
                 venue: ExchangeVenue,
                 taker_fee: float,
                 maker_fee: float,
                 avg_latency_ms: float,
                 min_order_size: Dict[str, float] = None,
                 max_order_size: Dict[str, float] = None,
                 supports_market_orders: bool = True,
                 supports_limit_orders: bool = True,
                 supports_stop_orders: bool = True):
        """
        Initialise les informations sur la plateforme d'échange.
        
        Args:
            venue: Identifiant de la plateforme
            taker_fee: Frais pour les ordres taker (%)
            maker_fee: Frais pour les ordres maker (%)
            avg_latency_ms: Latence moyenne en millisecondes
            min_order_size: Taille minimale d'ordre par symbole
            max_order_size: Taille maximale d'ordre par symbole
            supports_market_orders: Si la plateforme supporte les ordres marché
            supports_limit_orders: Si la plateforme supporte les ordres limites
            supports_stop_orders: Si la plateforme supporte les ordres stop
        """
        self.venue = venue
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.avg_latency_ms = avg_latency_ms
        self.min_order_size = min_order_size or {}
        self.max_order_size = max_order_size or {}
        self.supports_market_orders = supports_market_orders
        self.supports_limit_orders = supports_limit_orders
        self.supports_stop_orders = supports_stop_orders
        
    def get_fee_for_order_type(self, is_taker: bool) -> float:
        """
        Retourne les frais applicables selon le type d'ordre.
        
        Args:
            is_taker: Vrai si l'ordre est taker, faux pour maker
            
        Returns:
            Les frais applicables (%)
        """
        return self.taker_fee if is_taker else self.maker_fee
    
    def can_execute_quantity(self, symbol: str, quantity: float) -> bool:
        """
        Vérifie si la quantité spécifiée peut être exécutée sur cette plateforme.
        
        Args:
            symbol: Symbole de l'actif
            quantity: Quantité à exécuter
            
        Returns:
            Vrai si la quantité est dans les limites acceptables
        """
        min_size = self.min_order_size.get(symbol, 0)
        max_size = self.max_order_size.get(symbol, float('inf'))
        
        return min_size <= quantity <= max_size


class OrderRoutingStrategy(ABC):
    """Classe abstraite définissant l'interface pour les stratégies de routage."""
    
    @abstractmethod
    def select_venue(self, symbol: str, order_type: str, quantity: float,
                    available_venues: List[ExchangeInfo]) -> ExchangeVenue:
        """
        Sélectionne la meilleure venue d'exécution selon la stratégie.
        
        Args:
            symbol: Symbole de l'actif
            order_type: Type d'ordre (market, limit, stop, etc.)
            quantity: Quantité à exécuter
            available_venues: Liste des venues disponibles avec leurs informations
            
        Returns:
            La venue d'exécution optimale
        """
        pass


class LowestFeeStrategy(OrderRoutingStrategy):
    """
    Stratégie de routage qui privilégie les frais les plus bas.
    Utile pour les traders sensibles aux coûts de transaction.
    """
    
    def select_venue(self, symbol: str, order_type: str, quantity: float,
                    available_venues: List[ExchangeInfo]) -> ExchangeVenue:
        is_taker = order_type.lower() == "market"
        
        # Filtrer les venues qui peuvent exécuter la quantité
        eligible_venues = [
            venue for venue in available_venues 
            if venue.can_execute_quantity(symbol, quantity)
        ]
        
        if not eligible_venues:
            raise ValueError(f"Aucune venue ne peut exécuter {quantity} de {symbol}")
        
        # Trier par frais (taker ou maker selon le type d'ordre)
        if is_taker:
            eligible_venues.sort(key=lambda v: v.taker_fee)
        else:
            eligible_venues.sort(key=lambda v: v.maker_fee)
        
        return eligible_venues[0].venue


class BestLiquidityStrategy(OrderRoutingStrategy):
    """
    Stratégie de routage qui privilégie la meilleure liquidité.
    Utilise la profondeur du carnet d'ordres pour minimiser le slippage.
    """
    
    def __init__(self, liquidity_data_provider):
        self.liquidity_provider = liquidity_data_provider
    
    def select_venue(self, symbol: str, order_type: str, quantity: float,
                    available_venues: List[ExchangeInfo]) -> ExchangeVenue:
        # Filtrer les venues qui peuvent exécuter la quantité
        eligible_venues = [
            venue for venue in available_venues 
            if venue.can_execute_quantity(symbol, quantity)
        ]
        
        if not eligible_venues:
            raise ValueError(f"Aucune venue ne peut exécuter {quantity} de {symbol}")
        
        # Calculer le slippage estimé pour chaque venue
        venue_slippage = []
        for venue in eligible_venues:
            # Obtenir les données de liquidité
            liquidity = self.liquidity_provider.get_liquidity_data(
                symbol=symbol, 
                venue=venue.venue
            )
            
            # Estimer le slippage
            estimated_slippage = self._estimate_slippage(
                liquidity=liquidity,
                quantity=quantity,
                is_buy=(order_type.lower() == "buy")
            )
            
            venue_slippage.append((venue, estimated_slippage))
        
        # Trier par slippage estimé (du plus faible au plus élevé)
        venue_slippage.sort(key=lambda x: x[1])
        return venue_slippage[0][0].venue
    
    def _estimate_slippage(self, liquidity: Dict, quantity: float, is_buy: bool) -> float:
        """
        Estime le slippage pour une quantité donnée sur base des données de liquidité.
        
        Args:
            liquidity: Données de liquidité (carnet d'ordres)
            quantity: Quantité à exécuter
            is_buy: Vrai pour un achat, faux pour une vente
            
        Returns:
            Slippage estimé en pourcentage
        """
        # À implémenter avec des données réelles de carnet d'ordres
        # Pour l'exemple, nous retournons une valeur simulée
        return 0.1  # 0.1% de slippage


class FastestExecutionStrategy(OrderRoutingStrategy):
    """
    Stratégie de routage qui privilégie la vitesse d'exécution.
    Utile pour les traders HFT ou les situations nécessitant une exécution rapide.
    """
    
    def select_venue(self, symbol: str, order_type: str, quantity: float,
                    available_venues: List[ExchangeInfo]) -> ExchangeVenue:
        # Filtrer les venues qui peuvent exécuter la quantité
        eligible_venues = [
            venue for venue in available_venues 
            if venue.can_execute_quantity(symbol, quantity)
        ]
        
        if not eligible_venues:
            raise ValueError(f"Aucune venue ne peut exécuter {quantity} de {symbol}")
        
        # Trier par latence (du plus rapide au plus lent)
        eligible_venues.sort(key=lambda v: v.avg_latency_ms)
        return eligible_venues[0].venue


class SmartRoutingStrategy(OrderRoutingStrategy):
    """
    Stratégie de routage intelligente qui équilibre tous les critères.
    Utilise une approche pondérée pour trouver la meilleure venue d'exécution.
    """
    
    def __init__(self, liquidity_data_provider, weights=None):
        self.liquidity_provider = liquidity_data_provider
        
        # Poids par défaut pour chaque critère
        self.weights = weights or {
            "fees": 0.3,          # Importance des frais
            "liquidity": 0.4,     # Importance de la liquidité
            "speed": 0.2,         # Importance de la vitesse
            "reliability": 0.1    # Importance de la fiabilité
        }
    
    def select_venue(self, symbol: str, order_type: str, quantity: float,
                    available_venues: List[ExchangeInfo]) -> ExchangeVenue:
        # Filtrer les venues qui peuvent exécuter la quantité
        eligible_venues = [
            venue for venue in available_venues 
            if venue.can_execute_quantity(symbol, quantity)
        ]
        
        if not eligible_venues:
            raise ValueError(f"Aucune venue ne peut exécuter {quantity} de {symbol}")
        
        # Calculer un score pour chaque venue
        venue_scores = []
        is_taker = order_type.lower() == "market"
        
        for venue in eligible_venues:
            # Score pour les frais (normalisé)
            fee = venue.get_fee_for_order_type(is_taker)
            max_fee = max(v.get_fee_for_order_type(is_taker) for v in eligible_venues)
            fee_score = 1 - (fee / max_fee) if max_fee > 0 else 1
            
            # Score pour la liquidité (via le slippage estimé)
            liquidity = self.liquidity_provider.get_liquidity_data(
                symbol=symbol, 
                venue=venue.venue
            )
            estimated_slippage = self._estimate_slippage(
                liquidity=liquidity,
                quantity=quantity,
                is_buy=(order_type.lower() == "buy")
            )
            max_slippage = 0.5  # 0.5% comme référence
            liquidity_score = 1 - (estimated_slippage / max_slippage)
            
            # Score pour la vitesse
            max_latency = max(v.avg_latency_ms for v in eligible_venues)
            speed_score = 1 - (venue.avg_latency_ms / max_latency) if max_latency > 0 else 1
            
            # Score pour la fiabilité (à implémenter avec des données réelles)
            reliability_score = 0.9  # Valeur par défaut
            
            # Score global pondéré
            total_score = (
                self.weights["fees"] * fee_score +
                self.weights["liquidity"] * liquidity_score +
                self.weights["speed"] * speed_score +
                self.weights["reliability"] * reliability_score
            )
            
            venue_scores.append((venue, total_score))
        
        # Trier par score (du plus élevé au plus faible)
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        return venue_scores[0][0].venue
    
    def _estimate_slippage(self, liquidity: Dict, quantity: float, is_buy: bool) -> float:
        """
        Estime le slippage pour une quantité donnée sur base des données de liquidité.
        
        Args:
            liquidity: Données de liquidité (carnet d'ordres)
            quantity: Quantité à exécuter
            is_buy: Vrai pour un achat, faux pour une vente
            
        Returns:
            Slippage estimé en pourcentage
        """
        # À implémenter avec des données réelles de carnet d'ordres
        # Pour l'exemple, nous retournons une valeur simulée
        return 0.1  # 0.1% de slippage


class SmartRouter:
    """
    Système de routage intelligent qui dirige les ordres vers les meilleures
    venues d'exécution en fonction de critères spécifiés.
    """
    
    def __init__(self, liquidity_data_provider=None):
        """
        Initialise le routeur intelligent.
        
        Args:
            liquidity_data_provider: Fournisseur de données de liquidité
        """
        self.liquidity_provider = liquidity_data_provider or DummyLiquidityProvider()
        
        # Initialiser les stratégies disponibles
        self.strategies = {
            ExchangePriority.FEES: LowestFeeStrategy(),
            ExchangePriority.LIQUIDITY: BestLiquidityStrategy(self.liquidity_provider),
            ExchangePriority.SPEED: FastestExecutionStrategy(),
            ExchangePriority.SMART: SmartRoutingStrategy(self.liquidity_provider)
        }
        
        # Informations de référence sur les exchanges
        self.exchange_info = self._initialize_exchange_info()
    
    def _initialize_exchange_info(self) -> Dict[ExchangeVenue, ExchangeInfo]:
        """
        Initialise les informations de référence sur les exchanges.
        Dans une implémentation réelle, ces données seraient obtenues
        à partir d'une API ou d'une base de données.
        
        Returns:
            Dictionnaire contenant les informations par venue
        """
        return {
            ExchangeVenue.BINANCE: ExchangeInfo(
                venue=ExchangeVenue.BINANCE,
                taker_fee=0.10,
                maker_fee=0.08,
                avg_latency_ms=50.0,
                min_order_size={"BTC/USD": 0.0001}
            ),
            ExchangeVenue.COINBASE: ExchangeInfo(
                venue=ExchangeVenue.COINBASE,
                taker_fee=0.30,
                maker_fee=0.15,
                avg_latency_ms=70.0,
                min_order_size={"BTC/USD": 0.0001}
            ),
            ExchangeVenue.KRAKEN: ExchangeInfo(
                venue=ExchangeVenue.KRAKEN,
                taker_fee=0.20,
                maker_fee=0.10,
                avg_latency_ms=100.0,
                min_order_size={"BTC/USD": 0.0002}
            ),
            ExchangeVenue.FTX: ExchangeInfo(
                venue=ExchangeVenue.FTX,
                taker_fee=0.07,
                maker_fee=0.02,
                avg_latency_ms=40.0,
                min_order_size={"BTC/USD": 0.0001}
            ),
            # Ajouter d'autres exchanges au besoin
        }
    
    def route_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = "market", priority: ExchangePriority = ExchangePriority.SMART,
                   exclude_venues: List[ExchangeVenue] = None) -> ExchangeVenue:
        """
        Route un ordre vers la meilleure venue d'exécution.
        
        Args:
            symbol: Symbole de l'actif
            side: Côté de l'ordre ('buy' ou 'sell')
            quantity: Quantité à exécuter
            order_type: Type d'ordre (market, limit, stop, etc.)
            priority: Priorité pour la sélection de la venue
            exclude_venues: Liste des venues à exclure
            
        Returns:
            La venue d'exécution optimale
        """
        logger.info(f"Routage d'ordre: {symbol} {side} {quantity} ({order_type}) avec priorité {priority.value}")
        
        # Filtrer les venues disponibles
        exclude_list = exclude_venues or []
        available_venues = [
            info for venue, info in self.exchange_info.items()
            if venue not in exclude_list
        ]
        
        # Sélectionner la stratégie appropriée
        strategy = self.strategies.get(priority)
        if not strategy:
            raise ValueError(f"Stratégie non supportée pour la priorité: {priority}")
        
        # Sélectionner la meilleure venue
        selected_venue = strategy.select_venue(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            available_venues=available_venues
        )
        
        logger.info(f"Venue sélectionnée: {selected_venue.value}")
        return selected_venue
    
    def smart_split_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", max_venues: int = 3) -> Dict[ExchangeVenue, float]:
        """
        Divise intelligemment un ordre entre plusieurs venues pour optimiser l'exécution.
        
        Args:
            symbol: Symbole de l'actif
            side: Côté de l'ordre ('buy' ou 'sell')
            quantity: Quantité totale à exécuter
            order_type: Type d'ordre
            max_venues: Nombre maximal de venues à utiliser
            
        Returns:
            Dictionnaire avec les venues et les quantités à exécuter sur chacune
        """
        logger.info(f"Smart split pour {quantity} {symbol} ({side})")
        
        # Obtenir toutes les venues disponibles
        available_venues = list(self.exchange_info.values())
        
        # Calculer un score pour chaque venue
        venue_scores = []
        is_taker = order_type.lower() == "market"
        
        for venue in available_venues:
            if not venue.can_execute_quantity(symbol, quantity / max_venues):
                continue
                
            # Score basé sur les frais
            fee = venue.get_fee_for_order_type(is_taker)
            fee_factor = 1 - (fee / 0.5)  # Normalisation avec 0.5% comme référence
            
            # Score basé sur la liquidité
            liquidity = self.liquidity_provider.get_liquidity_data(
                symbol=symbol, 
                venue=venue.venue
            )
            liquidity_factor = liquidity.get("depth_factor", 0.5)
            
            # Score composite
            score = (fee_factor * 0.4) + (liquidity_factor * 0.6)
            venue_scores.append((venue, score))
        
        # Trier les venues par score
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limiter au nombre maximum de venues
        top_venues = venue_scores[:max_venues]
        
        # Répartir les quantités selon les scores relatifs
        total_score = sum(score for _, score in top_venues)
        allocation = {}
        
        for venue, score in top_venues:
            venue_quantity = quantity * (score / total_score)
            allocation[venue.venue] = venue_quantity
            
        logger.info(f"Allocation: {[(v.value, q) for v, q in allocation.items()]}")
        return allocation
    
    def estimate_execution_cost(self, symbol: str, side: str, quantity: float,
                              venue: ExchangeVenue, order_type: str = "market") -> Dict:
        """
        Estime le coût total d'exécution pour un ordre sur une venue spécifique.
        
        Args:
            symbol: Symbole de l'actif
            side: Côté de l'ordre ('buy' ou 'sell')
            quantity: Quantité à exécuter
            venue: Venue d'exécution
            order_type: Type d'ordre
            
        Returns:
            Dictionnaire avec les coûts estimés (frais, slippage, impact)
        """
        # Obtenir les informations de la venue
        venue_info = self.exchange_info.get(venue)
        if not venue_info:
            raise ValueError(f"Venue non supportée: {venue}")
        
        # Calculer les frais
        is_taker = order_type.lower() == "market"
        fee_rate = venue_info.get_fee_for_order_type(is_taker)
        estimated_price = self._get_market_price(symbol, venue)
        fee_amount = estimated_price * quantity * (fee_rate / 100)
        
        # Estimer le slippage
        liquidity = self.liquidity_provider.get_liquidity_data(
            symbol=symbol, 
            venue=venue
        )
        slippage_pct = self._estimate_slippage(
            liquidity=liquidity,
            quantity=quantity,
            is_buy=(side.lower() == "buy")
        )
        slippage_amount = estimated_price * quantity * (slippage_pct / 100)
        
        # Estimer l'impact marché
        impact_pct = self._estimate_market_impact(
            symbol=symbol,
            quantity=quantity,
            venue=venue,
            side=side
        )
        impact_amount = estimated_price * quantity * (impact_pct / 100)
        
        # Coût total estimé
        total_cost = fee_amount + slippage_amount + impact_amount
        total_cost_pct = (total_cost / (estimated_price * quantity)) * 100
        
        return {
            "venue": venue.value,
            "estimated_price": estimated_price,
            "total_quantity": quantity,
            "fee_rate": fee_rate,
            "fee_amount": fee_amount,
            "slippage_pct": slippage_pct,
            "slippage_amount": slippage_amount,
            "impact_pct": impact_pct,
            "impact_amount": impact_amount,
            "total_cost": total_cost,
            "total_cost_pct": total_cost_pct
        }
        
    def _get_market_price(self, symbol: str, venue: ExchangeVenue) -> float:
        """
        Obtient le prix actuel du marché pour un symbole sur une venue spécifique.
        
        Args:
            symbol: Symbole de l'actif
            venue: Venue d'exécution
            
        Returns:
            Prix actuel du marché
        """
        # À implémenter avec des données réelles
        # Pour l'exemple, nous retournons une valeur simulée
        return 50000.0  # Prix fictif
    
    def _estimate_slippage(self, liquidity: Dict, quantity: float, is_buy: bool) -> float:
        """
        Estime le slippage pour une quantité donnée sur base des données de liquidité.
        
        Args:
            liquidity: Données de liquidité (carnet d'ordres)
            quantity: Quantité à exécuter
            is_buy: Vrai pour un achat, faux pour une vente
            
        Returns:
            Slippage estimé en pourcentage
        """
        # À implémenter avec des données réelles de carnet d'ordres
        # Pour l'exemple, nous retournons une valeur simulée
        return 0.1  # 0.1% de slippage
    
    def _estimate_market_impact(self, symbol: str, quantity: float, 
                               venue: ExchangeVenue, side: str) -> float:
        """
        Estime l'impact marché pour une quantité donnée.
        
        Args:
            symbol: Symbole de l'actif
            quantity: Quantité à exécuter
            venue: Venue d'exécution
            side: Côté de l'ordre ('buy' ou 'sell')
            
        Returns:
            Impact marché estimé en pourcentage
        """
        # À implémenter avec des données réelles
        # Pour l'exemple, nous retournons une valeur simulée
        return 0.05  # 0.05% d'impact marché


class DummyLiquidityProvider:
    """
    Fournisseur de données de liquidité factice pour les tests.
    Dans une implémentation réelle, ce service interrogerait les API des exchanges.
    """
    
    def get_liquidity_data(self, symbol: str, venue: ExchangeVenue) -> Dict:
        """
        Obtient les données de liquidité pour un symbole sur une venue spécifique.
        
        Args:
            symbol: Symbole de l'actif
            venue: Venue d'exécution
            
        Returns:
            Données de liquidité simulées
        """
        # Données simulées par venue
        liquidity_by_venue = {
            ExchangeVenue.BINANCE: {
                "bid_depth": 100.0,
                "ask_depth": 120.0,
                "spread": 0.02,
                "depth_factor": 0.8,  # Plus la valeur est élevée, meilleure est la liquidité
            },
            ExchangeVenue.COINBASE: {
                "bid_depth": 80.0,
                "ask_depth": 90.0,
                "spread": 0.03,
                "depth_factor": 0.7,
            },
            ExchangeVenue.KRAKEN: {
                "bid_depth": 60.0,
                "ask_depth": 70.0,
                "spread": 0.04,
                "depth_factor": 0.6,
            },
            ExchangeVenue.FTX: {
                "bid_depth": 110.0,
                "ask_depth": 130.0,
                "spread": 0.01,
                "depth_factor": 0.9,
            },
        }
        
        # Retourner les données pour la venue spécifiée, ou des données par défaut
        return liquidity_by_venue.get(venue, {
            "bid_depth": 50.0,
            "ask_depth": 50.0,
            "spread": 0.05,
            "depth_factor": 0.5,
        }) 