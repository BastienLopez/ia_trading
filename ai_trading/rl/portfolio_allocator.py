import numpy as np
from typing import Dict, List, Any
import logging

# Configuration du logger
logger = logging.getLogger("PortfolioAllocator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PortfolioAllocator:
    """
    Classe responsable de l'allocation du portefeuille entre différents actifs.
    
    Implémente plusieurs stratégies d'allocation:
    - Equal: Allocation équivalente entre tous les actifs actifs
    - Volatility: Allocation inversement proportionnelle à la volatilité
    - Momentum: Allocation proportionnelle au momentum récent (rendements)
    - Smart: Combinaison des stratégies précédentes avec une pondération intelligente
    """
    
    def __init__(self, method="equal", max_active_positions=None):
        """
        Initialise l'allocateur de portefeuille.
        
        Args:
            method (str): Méthode d'allocation à utiliser
                ("equal", "volatility", "momentum", "smart")
            max_active_positions (int, optional): Nombre maximum de positions actives
        """
        self.method = method
        self.max_active_positions = max_active_positions
        logger.info(f"PortfolioAllocator initialisé avec la méthode: {method}")
    
    def allocate(self, action_weights, symbols, prices=None, volatilities=None, returns=None):
        """
        Calcule l'allocation optimale du portefeuille.
        
        Args:
            action_weights (np.ndarray or Dict): Poids bruts suggérés par l'agent (entre 0 et 1)
            symbols (List[str]): Liste des symboles des actifs
            prices (Dict[str, float], optional): Prix actuels des actifs
            volatilities (Dict[str, float], optional): Volatilités des actifs (écart-type des rendements)
            returns (Dict[str, float], optional): Rendements récents des actifs
            
        Returns:
            Dict[str, float]: Allocation optimale pour chaque actif (somme = 1)
        """
        # Convertir action_weights en dictionnaire si ce n'est pas déjà le cas
        if isinstance(action_weights, np.ndarray):
            action_weights = {symbol: float(weight) for symbol, weight in zip(symbols, action_weights)}
        
        # Vérifier que nous avons des poids pour tous les symboles
        assert set(action_weights.keys()) == set(symbols), "Les poids d'action doivent correspondre aux symboles"
        
        # Appliquer la méthode d'allocation choisie
        if self.method == "equal":
            allocation = self._equal_allocation(action_weights)
        elif self.method == "volatility":
            assert volatilities is not None, "Les volatilités sont requises pour la méthode 'volatility'"
            allocation = self._volatility_allocation(action_weights, volatilities)
        elif self.method == "momentum":
            assert returns is not None, "Les rendements sont requis pour la méthode 'momentum'"
            allocation = self._momentum_allocation(action_weights, returns)
        elif self.method == "smart":
            assert volatilities is not None and returns is not None, "Volatilités et rendements requis pour la méthode 'smart'"
            allocation = self._smart_allocation(action_weights, volatilities, returns)
        else:
            logger.warning(f"Méthode d'allocation '{self.method}' non reconnue, utilisation de 'equal'")
            allocation = self._equal_allocation(action_weights)
        
        # Limiter le nombre de positions actives si spécifié
        if self.max_active_positions is not None and self.max_active_positions < len(symbols):
            allocation = self._limit_active_positions(allocation)
        
        # Normaliser l'allocation pour s'assurer que la somme est 1
        total_allocation = sum(allocation.values())
        if total_allocation > 0:
            normalized_allocation = {symbol: weight / total_allocation for symbol, weight in allocation.items()}
        else:
            # Si l'allocation totale est nulle, utiliser une allocation égale
            normalized_allocation = {symbol: 1.0 / len(symbols) for symbol in symbols}
        
        return normalized_allocation
    
    def _equal_allocation(self, action_weights):
        """
        Allocation équivalente entre tous les actifs actifs.
        
        Args:
            action_weights (Dict[str, float]): Poids bruts suggérés par l'agent (entre 0 et 1)
            
        Returns:
            Dict[str, float]: Allocation égale pour chaque actif actif
        """
        # Filtrer les actifs avec un poids positif
        active_assets = {symbol: weight for symbol, weight in action_weights.items() if weight > 0.1}
        
        if not active_assets:
            # Si aucun actif n'est sélectionné, utiliser une allocation égale
            return {symbol: 1.0 / len(action_weights) for symbol in action_weights}
        
        # Allocation égale entre les actifs actifs
        equal_weight = 1.0 / len(active_assets)
        return {symbol: equal_weight if symbol in active_assets else 0.0 for symbol in action_weights}
    
    def _volatility_allocation(self, action_weights, volatilities):
        """
        Allocation inversement proportionnelle à la volatilité.
        
        Args:
            action_weights (Dict[str, float]): Poids bruts suggérés par l'agent (entre 0 et 1)
            volatilities (Dict[str, float]): Volatilités des actifs
            
        Returns:
            Dict[str, float]: Allocation pondérée par la volatilité
        """
        # Filtrer les actifs avec un poids positif
        active_assets = {symbol: weight for symbol, weight in action_weights.items() if weight > 0.1}
        
        if not active_assets:
            return {symbol: 0.0 for symbol in action_weights}
        
        # Calculer l'inverse de la volatilité
        inv_volatility = {}
        for symbol in active_assets:
            vol = volatilities.get(symbol, 1.0)
            inv_volatility[symbol] = 1.0 / vol if vol > 0 else 0.0
        
        # Normaliser l'inverse de la volatilité
        total_inv_vol = sum(inv_volatility.values())
        if total_inv_vol > 0:
            for symbol in inv_volatility:
                inv_volatility[symbol] /= total_inv_vol
        
        # Combiner avec les poids de l'action
        allocation = {}
        for symbol in action_weights:
            if symbol in active_assets:
                allocation[symbol] = action_weights[symbol] * inv_volatility.get(symbol, 0.0)
            else:
                allocation[symbol] = 0.0
        
        return allocation
    
    def _momentum_allocation(self, action_weights, returns):
        """
        Allocation proportionnelle au momentum récent (rendements).
        
        Args:
            action_weights (Dict[str, float]): Poids bruts suggérés par l'agent (entre 0 et 1)
            returns (Dict[str, float]): Rendements récents des actifs
            
        Returns:
            Dict[str, float]: Allocation pondérée par le momentum
        """
        # Filtrer les actifs avec un poids positif
        active_assets = {symbol: weight for symbol, weight in action_weights.items() if weight > 0.1}
        
        if not active_assets:
            return {symbol: 0.0 for symbol in action_weights}
        
        # Exclure les rendements négatifs
        positive_returns = {}
        for symbol in active_assets:
            ret = returns.get(symbol, 0.0)
            positive_returns[symbol] = max(ret, 0.0)
        
        # Normaliser les rendements positifs
        total_pos_returns = sum(positive_returns.values())
        if total_pos_returns > 0:
            for symbol in positive_returns:
                positive_returns[symbol] /= total_pos_returns
        else:
            # Si tous les rendements sont négatifs ou nuls, utiliser une allocation égale
            for symbol in positive_returns:
                positive_returns[symbol] = 1.0 / len(active_assets)
        
        # Combiner avec les poids de l'action
        allocation = {}
        for symbol in action_weights:
            if symbol in active_assets:
                allocation[symbol] = action_weights[symbol] * positive_returns.get(symbol, 0.0)
            else:
                allocation[symbol] = 0.0
        
        return allocation
    
    def _smart_allocation(self, action_weights, volatilities, returns):
        """
        Allocation intelligente combinant plusieurs stratégies.
        
        Args:
            action_weights (Dict[str, float]): Poids bruts suggérés par l'agent (entre 0 et 1)
            volatilities (Dict[str, float]): Volatilités des actifs
            returns (Dict[str, float]): Rendements récents des actifs
            
        Returns:
            Dict[str, float]: Allocation intelligente
        """
        # Calculer les allocations avec différentes méthodes
        equal_alloc = self._equal_allocation(action_weights)
        vol_alloc = self._volatility_allocation(action_weights, volatilities)
        mom_alloc = self._momentum_allocation(action_weights, returns)
        
        # Calculer la moyenne des indicateurs de marché pour pondérer les stratégies
        avg_vol = np.mean([vol for vol in volatilities.values() if vol > 0])
        avg_ret = np.mean([ret for ret in returns.values()])
        
        # Ajuster les pondérations en fonction des conditions de marché
        # - Haute volatilité, faibles rendements -> favoriser l'allocation par volatilité
        # - Faible volatilité, hauts rendements -> favoriser l'allocation par momentum
        # - Conditions moyennes -> allocation équilibrée
        w_equal = 0.33
        w_vol = 0.33
        w_mom = 0.33
        
        # Ajuster si volatilité élevée
        if avg_vol > 0.25:  # 25% annualisé est considéré comme élevé
            w_vol += 0.2
            w_mom -= 0.1
            w_equal -= 0.1
        
        # Ajuster si rendements élevés
        if avg_ret > 0.10:  # 10% annualisé est considéré comme élevé
            w_mom += 0.2
            w_vol -= 0.1
            w_equal -= 0.1
        
        # Combinaison pondérée des allocations
        allocation = {}
        for symbol in action_weights:
            allocation[symbol] = (
                w_equal * equal_alloc.get(symbol, 0.0) +
                w_vol * vol_alloc.get(symbol, 0.0) +
                w_mom * mom_alloc.get(symbol, 0.0)
            )
        
        return allocation
    
    def _limit_active_positions(self, allocation):
        """
        Limite le nombre de positions actives au maximum spécifié.
        
        Args:
            allocation (Dict[str, float]): Allocation calculée
            
        Returns:
            Dict[str, float]: Allocation avec un nombre limité de positions actives
        """
        # Trier les actifs par poids décroissant
        sorted_positions = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
        
        # Garder uniquement les N positions les plus importantes
        limited_allocation = {}
        for symbol in allocation:
            limited_allocation[symbol] = 0.0
        
        # Ajouter les top positions
        for i, (symbol, weight) in enumerate(sorted_positions):
            if i < self.max_active_positions:
                limited_allocation[symbol] = weight
        
        return limited_allocation 