import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from ai_trading.data_processor import DataProcessor
from ai_trading.rl_agent import RLAgent

def plot_training_results(metrics, save_path=None):
    """
    Trace les graphiques des résultats d'entraînement
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Tracer les récompenses
    rewards = metrics['rewards']
    ax1.plot(rewards, label='Récompense par étape')
    ax1.set_title('Récompenses pendant l\'entraînement')
    ax1.set_ylabel('Récompense')
    ax1.legend()
    ax1.grid(True)
    
    # Tracer la valeur du portefeuille
    if 'portfolio_values' in metrics and len(metrics['portfolio_values']) > 0:
        portfolio_values = metrics['portfolio_values']
        ax2.plot(portfolio_values, label='Valeur du portefeuille')
        ax2.set_title('Évolution du portefeuille pendant l\'entraînement')
        ax2.set_ylabel('Valeur ($)')
        ax2.set_xlabel('Étapes')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_backtest_results(results, save_path=None):
    """
    Trace les graphiques des résultats de backtest
    """
    df = results['results']
    
    # Créer la figure avec 3 sous-graphiques
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Tracer le prix et les actions
    ax1.plot(df.index, df['close'], label='Prix BTC', color='blue')
    ax1.set_title('Backtest de la stratégie RL')
    ax1.set_ylabel('Prix ($)')
    
    # Marquer les points d'achat et de vente
    buy_signals = df[df['action'] == 1].index
    sell_signals = df[df['action'] == 2].index
    
    ax1.scatter(buy_signals, df.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Achat')
    ax1.scatter(sell_signals, df.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Vente')
    
    ax1.legend()
    ax1.grid(True)
    
    # Tracer la valeur du portefeuille
    ax2.plot(df.index, df['portfolio_value'], label='Valeur du portefeuille', color='orange')
    ax2.set_ylabel('Valeur ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Tracer les positions
    ax3.step(df.index, df['position'], label='Position', color='purple')
    ax3.set_ylabel('Position (0=Aucune, 1=Long)')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Ajouter un texte avec les métriques
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    info_text = (
        f"Profit: {results['profit_pct']:.2f}%\n"
        f"Buy & Hold: {results['bh_profit_pct']:.2f}%\n"
        f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}\n"
        f"Nombre de trades: {len(results['trades'])}\n"
    )
    ax1.annotate(info_text, xy=(0.02, 0.85), xycoords='axes fraction', bbox=bbox_props)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Entraînement d\'un agent RL pour le trading de crypto')
    parser.add_argument('--download', action='store_true', help='Télécharger de nouvelles données')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange à utiliser')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Paire de trading')
    parser.add_argument('--timeframe', type=str, default='1h', help='Intervalle de temps')
    parser.add_argument('--days', type=int, default=60, help='Nombre de jours de données historiques')
    parser.add_argument('--timesteps', type=int, default=50000, help='Nombre d\'étapes d\'entraînement')
    parser.add_argument('--backtest', action='store_true', help='Effectuer un backtest après l\'entraînement')
    args = parser.parse_args()
    
    # Créer les dossiers nécessaires
    os.makedirs("ai_trading/data", exist_ok=True)
    os.makedirs("ai_trading/models", exist_ok=True)
    
    # Initialiser le processeur de données
    data_processor = DataProcessor(data_dir="ai_trading/data")
    
    # Préparation des données
    if args.download:
        # Télécharger de nouvelles données
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        df = data_processor.download_historical_data(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
    else:
        # Utiliser des données existantes
        data_path = f"ai_trading/data/{args.exchange}_{args.symbol.replace('/', '_')}_{args.timeframe}.csv"
        if not os.path.exists(data_path):
            print(f"Fichier de données {data_path} non trouvé. Utilisation de l'option --download pour télécharger des données.")
            return
        df = data_processor.load_data(data_path)
    
    # Ajouter les indicateurs techniques
    df = data_processor.add_indicators(df)
    
    # Préparer les données pour l'entraînement et le test
    train_data, test_data = data_processor.preprocess_for_training(df)
    
    # Initialiser l'agent RL
    agent = RLAgent(model_dir="ai_trading/models")
    
    # Entraîner l'agent
    model_name = f"{args.symbol.replace('/', '_')}_{args.timeframe}_rl_model"
    save_path = f"ai_trading/models/{model_name}.zip"
    
    print(f"Entraînement de l'agent sur {len(train_data)} points de données...")
    metrics = agent.train(train_data, total_timesteps=args.timesteps, save_path=save_path)
    
    # Tracer les résultats d'entraînement
    plot_training_results(metrics, save_path=f"ai_trading/models/{model_name}_training.png")
    
    # Effectuer un backtest si demandé
    if args.backtest:
        print("Exécution du backtest sur les données de test...")
        backtest_results = agent.backtest(test_data)
        
        # Tracer les résultats du backtest
        plot_backtest_results(backtest_results, save_path=f"ai_trading/models/{model_name}_backtest.png")
        
        # Afficher les métriques de performance
        print(f"Résultats du backtest:")
        print(f"Profit: {backtest_results['profit_pct']:.2f}%")
        print(f"Buy & Hold: {backtest_results['bh_profit_pct']:.2f}%")
        print(f"Ratio de Sharpe: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Nombre de trades: {len(backtest_results['trades'])}")
    
    print(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")

if __name__ == "__main__":
    main()