"""
Exemple d'optimisation de portefeuille pour les cryptomonnaies.

Ce script démontre comment optimiser l'allocation d'un portefeuille de cryptomonnaies
en utilisant différentes méthodes d'optimisation (Markowitz, risque-parité, etc.).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Import des modules d'optimisation
from ai_trading.optimization.portfolio.markowitz import MarkowitzOptimizer
from ai_trading.optimization.portfolio.risk_parity import RiskParityOptimizer
from ai_trading.optimization.portfolio.hierarchical import HierarchicalOptimizer
from ai_trading.optimization.portfolio.constraints import PortfolioConstraints
from ai_trading.optimization.portfolio.risk_metrics import RiskMetrics

def load_crypto_data():
    """Charge des données simulées de prix de plusieurs cryptomonnaies."""
    # Créer des données simulées
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # Créer des tendances pour différents actifs avec des corrélations variables
    assets = {
        'BTC': {
            'start_price': 30000,
            'volatility': 0.02,
            'trend': np.concatenate([
                np.linspace(1, 1.2, n//3),
                np.linspace(1.2, 1.1, n//3),
                np.linspace(1.1, 1.4, n - 2*(n//3))
            ])
        },
        'ETH': {
            'start_price': 2000,
            'volatility': 0.025,
            'trend': np.concatenate([
                np.linspace(1, 1.3, n//3),
                np.linspace(1.3, 1.2, n//3),
                np.linspace(1.2, 1.5, n - 2*(n//3))
            ])
        },
        'BNB': {
            'start_price': 300,
            'volatility': 0.018,
            'trend': np.concatenate([
                np.linspace(1, 1.1, n//3),
                np.linspace(1.1, 1.0, n//3),
                np.linspace(1.0, 1.3, n - 2*(n//3))
            ])
        },
        'SOL': {
            'start_price': 100,
            'volatility': 0.03,
            'trend': np.concatenate([
                np.linspace(1, 1.5, n//3),
                np.linspace(1.5, 1.3, n//3),
                np.linspace(1.3, 2.0, n - 2*(n//3))
            ])
        },
        'ADA': {
            'start_price': 0.5,
            'volatility': 0.022,
            'trend': np.concatenate([
                np.linspace(1, 0.9, n//3),
                np.linspace(0.9, 1.1, n//3),
                np.linspace(1.1, 1.2, n - 2*(n//3))
            ])
        },
        'DOT': {
            'start_price': 10,
            'volatility': 0.028,
            'trend': np.concatenate([
                np.linspace(1, 1.2, n//3),
                np.linspace(1.2, 0.9, n//3),
                np.linspace(0.9, 1.1, n - 2*(n//3))
            ])
        },
        'LINK': {
            'start_price': 15,
            'volatility': 0.026,
            'trend': np.concatenate([
                np.linspace(1, 1.1, n//3),
                np.linspace(1.1, 1.0, n//3),
                np.linspace(1.0, 1.4, n - 2*(n//3))
            ])
        },
        'XRP': {
            'start_price': 0.5,
            'volatility': 0.024,
            'trend': np.concatenate([
                np.linspace(1, 0.8, n//3),
                np.linspace(0.8, 1.0, n//3),
                np.linspace(1.0, 1.2, n - 2*(n//3))
            ])
        }
    }
    
    # Générer les prix
    prices = {}
    for asset, params in assets.items():
        # Générer les retours logarithmiques avec une corrélation de base
        base_returns = np.random.normal(0, params['volatility'], n)
        
        # Convertir en prix
        price = params['start_price'] * params['trend'] * np.exp(np.cumsum(base_returns))
        prices[asset] = price
    
    # Créer le DataFrame
    df = pd.DataFrame(prices, index=dates)
    
    return df

def compute_portfolio_metrics(weights, prices):
    """Calcule les métriques d'un portefeuille."""
    # Calculer les rendements
    returns = prices.pct_change().dropna()
    
    # Rendement attendu (annualisé)
    expected_return = np.sum(returns.mean() * weights) * 252
    
    # Volatilité (annualisée)
    cov_matrix = returns.cov() * 252
    portfolio_variance = weights.T @ cov_matrix @ weights
    volatility = np.sqrt(portfolio_variance)
    
    # Ratio de Sharpe (avec taux sans risque de 0% pour simplifier)
    sharpe_ratio = expected_return / volatility
    
    # VaR (Value at Risk) à 95%
    portfolio_returns = returns @ weights
    var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
    
    # Drawdown maximum
    portfolio_value = (1 + portfolio_returns).cumprod()
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value / running_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'max_drawdown': max_drawdown
    }

def portfolio_optimization_example():
    """Exemple principal d'optimisation de portefeuille."""
    # Charger les données
    print("Chargement des données de prix...")
    prices = load_crypto_data()
    
    # Créer le dossier de résultats
    output_dir = "examples/optimization/portfolio/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculer les rendements
    returns = prices.pct_change().dropna()
    
    # Afficher les statistiques de base
    print("\n=== Statistiques des actifs ===")
    stats = pd.DataFrame({
        'Rendement annuel (%)': returns.mean() * 252 * 100,
        'Volatilité annuelle (%)': returns.std() * np.sqrt(252) * 100,
        'Ratio de Sharpe': returns.mean() / returns.std() * np.sqrt(252)
    })
    print(stats)
    
    # 1. Optimisation selon Markowitz
    print("\n=== Optimisation selon Markowitz ===")
    
    # Configurer les contraintes
    constraints = PortfolioConstraints(
        min_weights=0.05,     # Au moins 5% par actif
        max_weights=0.30,     # Maximum 30% par actif
        sum_constraint=1.0    # Somme des poids = 100%
    )
    
    # Créer l'optimiseur
    markowitz = MarkowitzOptimizer(
        returns=returns,
        risk_free_rate=0.0,  # Taux sans risque (0% pour simplifier)
        target_return=None,  # Maximiser le ratio de Sharpe
        constraints=constraints
    )
    
    # Optimiser
    markowitz_weights = markowitz.optimize()
    
    # Afficher les poids
    print("\nPoids optimaux (Markowitz):")
    for asset, weight in markowitz_weights.items():
        print(f"  {asset}: {weight:.2%}")
    
    # Calculer les métriques
    markowitz_metrics = compute_portfolio_metrics(np.array(list(markowitz_weights.values())), prices)
    print("\nMétriques du portefeuille (Markowitz):")
    print(f"  Rendement attendu: {markowitz_metrics['expected_return']:.2%}")
    print(f"  Volatilité: {markowitz_metrics['volatility']:.2%}")
    print(f"  Ratio de Sharpe: {markowitz_metrics['sharpe_ratio']:.2f}")
    print(f"  VaR 95%: {-markowitz_metrics['var_95']:.2%}")
    print(f"  Drawdown maximum: {markowitz_metrics['max_drawdown']:.2%}")
    
    # 2. Optimisation par Risk Parity (parité de risque)
    print("\n=== Optimisation par Risk Parity ===")
    
    # Créer l'optimiseur
    risk_parity = RiskParityOptimizer(
        returns=returns,
        risk_measure='volatility',  # Utiliser la volatilité comme mesure de risque
        constraints=constraints
    )
    
    # Optimiser
    rp_weights = risk_parity.optimize()
    
    # Afficher les poids
    print("\nPoids optimaux (Risk Parity):")
    for asset, weight in rp_weights.items():
        print(f"  {asset}: {weight:.2%}")
    
    # Calculer les métriques
    rp_metrics = compute_portfolio_metrics(np.array(list(rp_weights.values())), prices)
    print("\nMétriques du portefeuille (Risk Parity):")
    print(f"  Rendement attendu: {rp_metrics['expected_return']:.2%}")
    print(f"  Volatilité: {rp_metrics['volatility']:.2%}")
    print(f"  Ratio de Sharpe: {rp_metrics['sharpe_ratio']:.2f}")
    print(f"  VaR 95%: {-rp_metrics['var_95']:.2%}")
    print(f"  Drawdown maximum: {rp_metrics['max_drawdown']:.2%}")
    
    # 3. Optimisation hiérarchique
    print("\n=== Optimisation hiérarchique ===")
    
    # Définir les groupes d'actifs
    groups = {
        'Large Cap': ['BTC', 'ETH'],
        'Mid Cap': ['BNB', 'SOL', 'ADA'],
        'Small Cap': ['DOT', 'LINK', 'XRP']
    }
    
    # Créer l'optimiseur
    hierarchical = HierarchicalOptimizer(
        returns=returns,
        groups=groups,
        group_weights={'Large Cap': 0.5, 'Mid Cap': 0.3, 'Small Cap': 0.2},
        risk_measure='contribution',  # Équilibrer les contributions au risque
        constraints=constraints
    )
    
    # Optimiser
    h_weights = hierarchical.optimize()
    
    # Afficher les poids
    print("\nPoids optimaux (Hiérarchique):")
    for asset, weight in h_weights.items():
        print(f"  {asset}: {weight:.2%}")
    
    # Calculer les métriques
    h_metrics = compute_portfolio_metrics(np.array(list(h_weights.values())), prices)
    print("\nMétriques du portefeuille (Hiérarchique):")
    print(f"  Rendement attendu: {h_metrics['expected_return']:.2%}")
    print(f"  Volatilité: {h_metrics['volatility']:.2%}")
    print(f"  Ratio de Sharpe: {h_metrics['sharpe_ratio']:.2f}")
    print(f"  VaR 95%: {-h_metrics['var_95']:.2%}")
    print(f"  Drawdown maximum: {h_metrics['max_drawdown']:.2%}")
    
    # Visualiser les allocations
    plt.figure(figsize=(15, 10))
    
    # Graphique des poids
    plt.subplot(2, 2, 1)
    
    # Préparer les données
    assets = list(markowitz_weights.keys())
    mark_weights = list(markowitz_weights.values())
    rp_weights_list = [rp_weights[asset] for asset in assets]
    h_weights_list = [h_weights[asset] for asset in assets]
    
    # Créer le graphique à barres
    x = np.arange(len(assets))
    width = 0.25
    
    plt.bar(x - width, mark_weights, width, label='Markowitz')
    plt.bar(x, rp_weights_list, width, label='Risk Parity')
    plt.bar(x + width, h_weights_list, width, label='Hiérarchique')
    
    plt.xlabel('Actifs')
    plt.ylabel('Poids')
    plt.title('Allocation des actifs par méthode')
    plt.xticks(x, assets, rotation=45)
    plt.legend()
    
    # Graphique comparatif des métriques
    plt.subplot(2, 2, 2)
    
    metrics = ['expected_return', 'volatility', 'sharpe_ratio']
    metric_labels = ['Rendement', 'Volatilité', 'Sharpe']
    
    data = [
        [markowitz_metrics[m] for m in metrics],
        [rp_metrics[m] for m in metrics],
        [h_metrics[m] for m in metrics]
    ]
    
    x = np.arange(len(metric_labels))
    width = 0.25
    
    plt.bar(x - width, data[0], width, label='Markowitz')
    plt.bar(x, data[1], width, label='Risk Parity')
    plt.bar(x + width, data[2], width, label='Hiérarchique')
    
    plt.xlabel('Métrique')
    plt.ylabel('Valeur')
    plt.title('Comparaison des métriques')
    plt.xticks(x, metric_labels)
    plt.legend()
    
    # Graphique de la frontière efficiente
    plt.subplot(2, 2, 3)
    
    # Générer la frontière efficiente
    returns_range = np.linspace(0.05, 0.30, 100)  # 5% à 30% de rendement annuel
    volatilities = []
    
    for target_return in returns_range:
        # Créer l'optimiseur avec un rendement cible
        optimizer = MarkowitzOptimizer(
            returns=returns,
            risk_free_rate=0.0,
            target_return=target_return / 252,  # Convertir en rendement quotidien
            constraints=PortfolioConstraints(min_weights=0, max_weights=1, sum_constraint=1.0)
        )
        
        try:
            # Optimiser
            weights = optimizer.optimize()
            
            # Calculer la volatilité
            cov_matrix = returns.cov() * 252
            weights_array = np.array(list(weights.values()))
            volatility = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
            
            volatilities.append(volatility)
        except:
            # En cas d'échec de l'optimisation (contraintes non satisfaites)
            volatilities.append(np.nan)
    
    # Filtrer les points valides
    valid_indices = ~np.isnan(volatilities)
    valid_returns = returns_range[valid_indices]
    valid_volatilities = np.array(volatilities)[valid_indices]
    
    # Tracer la frontière efficiente
    plt.plot(valid_volatilities, valid_returns, 'b-', label='Frontière efficiente')
    
    # Ajouter les points pour les trois portefeuilles
    plt.scatter(markowitz_metrics['volatility'], markowitz_metrics['expected_return'], 
                c='r', marker='o', s=100, label='Markowitz')
    plt.scatter(rp_metrics['volatility'], rp_metrics['expected_return'], 
                c='g', marker='s', s=100, label='Risk Parity')
    plt.scatter(h_metrics['volatility'], h_metrics['expected_return'], 
                c='purple', marker='^', s=100, label='Hiérarchique')
    
    # Ajouter les points pour les actifs individuels
    for i, asset in enumerate(assets):
        asset_return = returns[asset].mean() * 252
        asset_vol = returns[asset].std() * np.sqrt(252)
        plt.scatter(asset_vol, asset_return, c='gray', marker='x', s=50)
        plt.annotate(asset, (asset_vol, asset_return), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Volatilité annualisée')
    plt.ylabel('Rendement annualisé')
    plt.title('Frontière efficiente et portefeuilles optimisés')
    plt.grid(True)
    plt.legend()
    
    # Graphique de la composition par groupe (pour la méthode hiérarchique)
    plt.subplot(2, 2, 4)
    
    # Calculer les poids par groupe
    group_weights = {}
    for group, assets_list in groups.items():
        group_weights[group] = sum(h_weights[asset] for asset in assets_list)
    
    # Créer le graphique en camembert
    plt.pie(list(group_weights.values()), labels=list(group_weights.keys()), 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.axis('equal')
    plt.title('Allocation par groupe (Hiérarchique)')
    
    plt.tight_layout()
    
    # Enregistrer la figure
    plt.savefig(f"{output_dir}/portfolio_optimization_comparison.png")
    plt.close()
    
    print(f"\nGraphique de comparaison enregistré dans {output_dir}")
    
    # Retourner les résultats
    return {
        'markowitz': {
            'weights': markowitz_weights,
            'metrics': markowitz_metrics
        },
        'risk_parity': {
            'weights': rp_weights,
            'metrics': rp_metrics
        },
        'hierarchical': {
            'weights': h_weights,
            'metrics': h_metrics
        }
    }

if __name__ == "__main__":
    results = portfolio_optimization_example() 