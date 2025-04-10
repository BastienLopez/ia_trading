import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.technical_indicators import TechnicalIndicators
from ai_trading.examples.visualize_indicators import generate_synthetic_data

class BasicIndicatorStrategy:
    """
    Classe pour implémenter et tester des stratégies de trading basées sur des indicateurs techniques.
    """
    
    def __init__(self, data, initial_balance=10000, transaction_fee=0.001):
        """
        Initialise la stratégie avec des données et des paramètres.
        
        Args:
            data (pandas.DataFrame): DataFrame contenant les données OHLCV.
            initial_balance (float): Solde initial.
            transaction_fee (float): Frais de transaction en pourcentage.
        """
        self.data = data
        self.indicators = TechnicalIndicators(data)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Initialiser les variables de suivi
        self.balance = initial_balance
        self.crypto_held = 0
        self.trades = []
        self.portfolio_values = []
    
    def reset(self):
        """Réinitialise la stratégie."""
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.trades = []
        self.portfolio_values = []
    
    def get_portfolio_value(self, step):
        """
        Calcule la valeur totale du portefeuille à un pas de temps donné.
        
        Args:
            step (int): Indice du pas de temps.
            
        Returns:
            float: Valeur du portefeuille.
        """
        crypto_value = self.crypto_held * self.data['close'].iloc[step]
        return self.balance + crypto_value
    
    def buy(self, step, amount_percent=1.0):
        """
        Exécute un achat.
        
        Args:
            step (int): Indice du pas de temps.
            amount_percent (float): Pourcentage du solde à utiliser pour l'achat.
            
        Returns:
            bool: True si l'achat a réussi, False sinon.
        """
        if step >= len(self.data):
            return False
        
        price = self.data['close'].iloc[step]
        amount_to_spend = self.balance * amount_percent
        
        if amount_to_spend <= 0:
            return False
        
        # Appliquer les frais de transaction
        fee = amount_to_spend * self.transaction_fee
        amount_to_spend -= fee
        
        # Calculer la quantité de crypto à acheter
        crypto_to_buy = amount_to_spend / price
        
        # Mettre à jour le solde et la quantité de crypto détenue
        self.balance -= (amount_to_spend + fee)
        self.crypto_held += crypto_to_buy
        
        # Enregistrer la transaction
        self.trades.append({
            'step': step,
            'date': self.data.index[step],
            'type': 'buy',
            'price': price,
            'amount': crypto_to_buy,
            'value': amount_to_spend,
            'fee': fee,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': self.get_portfolio_value(step)
        })
        
        return True
    
    def sell(self, step, amount_percent=1.0):
        """
        Exécute une vente.
        
        Args:
            step (int): Indice du pas de temps.
            amount_percent (float): Pourcentage de la crypto détenue à vendre.
            
        Returns:
            bool: True si la vente a réussi, False sinon.
        """
        if step >= len(self.data) or self.crypto_held <= 0:
            return False
        
        price = self.data['close'].iloc[step]
        crypto_to_sell = self.crypto_held * amount_percent
        
        if crypto_to_sell <= 0:
            return False
        
        # Calculer la valeur de la vente
        sale_value = crypto_to_sell * price
        
        # Appliquer les frais de transaction
        fee = sale_value * self.transaction_fee
        sale_value -= fee
        
        # Mettre à jour le solde et la quantité de crypto détenue
        self.balance += sale_value
        self.crypto_held -= crypto_to_sell
        
        # Enregistrer la transaction
        self.trades.append({
            'step': step,
            'date': self.data.index[step],
            'type': 'sell',
            'price': price,
            'amount': crypto_to_sell,
            'value': sale_value,
            'fee': fee,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': self.get_portfolio_value(step)
        })
        
        return True
    
    def run_strategy_macd_crossover(self):
        """
        Exécute une stratégie basée sur le croisement du MACD.
        
        Returns:
            pandas.DataFrame: DataFrame contenant les résultats de la stratégie.
        """
        self.reset()
        
        # Calculer le MACD
        macd_line, signal_line, _ = self.indicators.calculate_macd()
        
        # Parcourir les données
        for step in range(26, len(self.data)):
            # Calculer la valeur du portefeuille
            portfolio_value = self.get_portfolio_value(step)
            self.portfolio_values.append({
                'step': step,
                'date': self.data.index[step],
                'portfolio_value': portfolio_value
            })
            
            # Vérifier les croisements du MACD
            if macd_line.iloc[step] > signal_line.iloc[step] and macd_line.iloc[step-1] <= signal_line.iloc[step-1]:
                # Signal d'achat: MACD croise au-dessus de la ligne de signal
                self.buy(step, amount_percent=0.3)  # Utiliser 30% du solde
            
            elif macd_line.iloc[step] < signal_line.iloc[step] and macd_line.iloc[step-1] >= signal_line.iloc[step-1]:
                # Signal de vente: MACD croise en dessous de la ligne de signal
                self.sell(step, amount_percent=1.0)  # Vendre toute la crypto
        
        # Calculer les métriques de performance
        return self.calculate_performance()
    
    def run_strategy_rsi(self, oversold=30, overbought=70):
        """
        Exécute une stratégie basée sur le RSI.
        
        Args:
            oversold (int): Niveau de survente.
            overbought (int): Niveau de surachat.
            
        Returns:
            pandas.DataFrame: DataFrame contenant les résultats de la stratégie.
        """
        self.reset()
        
        # Calculer le RSI
        rsi = self.indicators.calculate_rsi()
        
        # Parcourir les données
        for step in range(14, len(self.data)):
            # Calculer la valeur du portefeuille
            portfolio_value = self.get_portfolio_value(step)
            self.portfolio_values.append({
                'step': step,
                'date': self.data.index[step],
                'portfolio_value': portfolio_value
            })
            
            # Vérifier les conditions du RSI
            if rsi.iloc[step] < oversold and rsi.iloc[step-1] >= oversold:
                # Signal d'achat: RSI passe sous le niveau de survente
                self.buy(step, amount_percent=0.3)  # Utiliser 30% du solde
            
            elif rsi.iloc[step] > overbought and rsi.iloc[step-1] <= overbought:
                # Signal de vente: RSI passe au-dessus du niveau de surachat
                self.sell(step, amount_percent=1.0)  # Vendre toute la crypto
        
        # Calculer les métriques de performance
        return self.calculate_performance()
    
    def run_strategy_bollinger_bands(self):
        """
        Exécute une stratégie basée sur les Bandes de Bollinger.
        
        Returns:
            pandas.DataFrame: DataFrame contenant les résultats de la stratégie.
        """
        self.reset()
        
        # Calculer les Bandes de Bollinger
        upper_bb, middle_bb, lower_bb = self.indicators.calculate_bollinger_bands()
        
        # Parcourir les données
        for step in range(20, len(self.data)):
            # Calculer la valeur du portefeuille
            portfolio_value = self.get_portfolio_value(step)
            self.portfolio_values.append({
                'step': step,
                'date': self.data.index[step],
                'portfolio_value': portfolio_value
            })
            
            price = self.data['close'].iloc[step]
            
            # Vérifier les conditions des Bandes de Bollinger
            if price < lower_bb.iloc[step]:
                # Signal d'achat: Prix sous la bande inférieure
                self.buy(step, amount_percent=0.3)  # Utiliser 30% du solde
            
            elif price > upper_bb.iloc[step]:
                # Signal de vente: Prix au-dessus de la bande supérieure
                self.sell(step, amount_percent=1.0)  # Vendre toute la crypto
        
        # Calculer les métriques de performance
        return self.calculate_performance()
    
    def run_strategy_ema_crossover(self):
        """
        Exécute une stratégie basée sur le croisement des EMA.
        
        Returns:
            pandas.DataFrame: DataFrame contenant les résultats de la stratégie.
        """
        self.reset()
        
        # Calculer les EMA
        ema9 = self.indicators.calculate_ema(period=9)
        ema21 = self.indicators.calculate_ema(period=21)
        
        # Parcourir les données
        for step in range(21, len(self.data)):
            # Calculer la valeur du portefeuille
            portfolio_value = self.get_portfolio_value(step)
            self.portfolio_values.append({
                'step': step,
                'date': self.data.index[step],
                'portfolio_value': portfolio_value
            })
            
            # Vérifier les croisements des EMA
            if ema9.iloc[step] > ema21.iloc[step] and ema9.iloc[step-1] <= ema21.iloc[step-1]:
                # Signal d'achat: EMA9 croise au-dessus de EMA21 (Golden Cross)
                self.buy(step, amount_percent=0.3)  # Utiliser 30% du solde
            
            elif ema9.iloc[step] < ema21.iloc[step] and ema9.iloc[step-1] >= ema21.iloc[step-1]:
                # Signal de vente: EMA9 croise en dessous de EMA21 (Death Cross)
                self.sell(step, amount_percent=1.0)  # Vendre toute la crypto
        
        # Calculer les métriques de performance
        return self.calculate_performance()
    
    def calculate_performance(self):
        """
        Calcule les métriques de performance de la stratégie.
        
        Returns:
            dict: Dictionnaire contenant les métriques de performance.
        """
        if not self.portfolio_values:
            return {
                'final_balance': self.balance,
                'final_crypto_held': self.crypto_held,
                'final_portfolio_value': self.balance,
                'total_return': 0,
                'total_return_percent': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_profit_per_trade': 0,
                'max_drawdown': 0
            }
        
        # Convertir les listes en DataFrames
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        portfolio_df = pd.DataFrame(self.portfolio_values)
        
        # Calculer la valeur finale du portefeuille
        final_step = portfolio_df['step'].iloc[-1]
        final_portfolio_value = self.get_portfolio_value(final_step)
        
        # Calculer le rendement total
        total_return = final_portfolio_value - self.initial_balance
        total_return_percent = (total_return / self.initial_balance) * 100
        
        # Calculer le nombre de transactions
        num_trades = len(trades_df) if not trades_df.empty else 0
        
        # Calculer le taux de réussite (win rate)
        if not trades_df.empty and num_trades > 0:
            # Calculer le profit/perte pour chaque transaction
            trades_df['profit'] = 0
            
            for i in range(1, len(trades_df)):
                if trades_df['type'].iloc[i-1] == 'buy' and trades_df['type'].iloc[i] == 'sell':
                    buy_value = trades_df['value'].iloc[i-1]
                    sell_value = trades_df['value'].iloc[i]
                    profit = sell_value - buy_value
                    trades_df.at[i, 'profit'] = profit
            
            # Calculer le taux de réussite
            winning_trades = (trades_df['profit'] > 0).sum()
            win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
            
            # Calculer le profit moyen par transaction
            avg_profit_per_trade = trades_df['profit'].mean() if num_trades > 0 else 0
        else:
            win_rate = 0
            avg_profit_per_trade = 0
        
        # Calculer le drawdown maximum
        if not portfolio_df.empty:
            portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax'] * 100
            max_drawdown = portfolio_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        return {
            'final_balance': self.balance,
            'final_crypto_held': self.crypto_held,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade,
            'max_drawdown': max_drawdown,
            'trades': trades_df,
            'portfolio_values': portfolio_df
        }
    
    def plot_performance(self, strategy_name):
        """
        Trace les performances de la stratégie.
        
        Args:
            strategy_name (str): Nom de la stratégie.
        """
        if not self.portfolio_values:
            print("Aucune donnée de performance à afficher.")
            return
        
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Créer une figure avec plusieurs sous-graphiques
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Graphique des prix et du portefeuille
        ax1.set_title(f'Performance de la stratégie: {strategy_name}')
        ax1.plot(self.data.index, self.data['close'], label='Prix', alpha=0.5)
        
        # Tracer la valeur du portefeuille
        portfolio_dates = [self.data.index[step] for step in portfolio_df['step']]
        ax1.plot(portfolio_dates, portfolio_df['portfolio_value'], label='Portefeuille', linewidth=2)
        
        # Marquer les transactions
        if not trades_df.empty:
            buy_dates = [self.data.index[step] for step, type in zip(trades_df['step'], trades_df['type']) if type == 'buy']
            buy_values = [value for type, value in zip(trades_df['type'], trades_df['portfolio_value']) if type == 'buy']
            
            sell_dates = [self.data.index[step] for step, type in zip(trades_df['step'], trades_df['type']) if type == 'sell']
            sell_values = [value for type, value in zip(trades_df['type'], trades_df['portfolio_value']) if type == 'sell']
            
            ax1.scatter(buy_dates, buy_values, color='green', marker='^', s=100, label='Achat')
            ax1.scatter(sell_dates, sell_values, color='red', marker='v', s=100, label='Vente')
        
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Graphique des rendements cumulés
        if not portfolio_df.empty:
            portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df['cumulative_return'] = (1 + portfolio_df['return']).cumprod() - 1
            
            ax2.set_title('Rendement cumulé')
            ax2.plot(portfolio_dates, portfolio_df['cumulative_return'] * 100, label='Rendement cumulé (%)', color='blue')
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            ax2.legend(loc='upper left')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{strategy_name}_performance.png')
        plt.show()

def main():
    """Fonction principale."""
    # Générer des données synthétiques
    data = generate_synthetic_data(n_samples=500, trend='bullish', volatility=0.02)
    
    # Initialiser la stratégie
    strategy = BasicIndicatorStrategy(data)
    
    # Exécuter différentes stratégies
    print("Exécution de la stratégie MACD Crossover...")
    macd_results = strategy.run_strategy_macd_crossover()
    strategy.plot_performance("MACD Crossover")
    
    print("\nExécution de la stratégie RSI...")
    rsi_results = strategy.run_strategy_rsi()
    strategy.plot_performance("RSI")
    
    print("\nExécution de la stratégie Bollinger Bands...")
    bb_results = strategy.run_strategy_bollinger_bands()
    strategy.plot_performance("Bollinger Bands")
    
    print("\nExécution de la stratégie EMA Crossover...")
    ema_results = strategy.run_strategy_ema_crossover()
    strategy.plot_performance("EMA Crossover")
    
    # Comparer les performances des stratégies
    strategies = ["MACD Crossover", "RSI", "Bollinger Bands", "EMA Crossover"]
    results = [macd_results, rsi_results, bb_results, ema_results]
    
    comparison = pd.DataFrame({
        'Strategy': strategies,
        'Final Portfolio Value': [r['final_portfolio_value'] for r in results],
        'Total Return (%)': [r['total_return_percent'] for r in results],
        'Number of Trades': [r['num_trades'] for r in results],
        'Win Rate (%)': [r['win_rate'] for r in results],
        'Avg Profit per Trade': [r['avg_profit_per_trade'] for r in results],
        'Max Drawdown (%)': [r['max_drawdown'] for r in results]
    })
    
    print("\nComparaison des stratégies:")
    print(comparison)
    
    # Tracer la comparaison des performances
    plt.figure(figsize=(12, 6))
    plt.bar(strategies, [r['total_return_percent'] for r in results])
    plt.title('Comparaison des rendements des stratégies')
    plt.ylabel('Rendement total (%)')
    plt.grid(axis='y')
    plt.savefig('strategy_comparison.png')
    plt.show()

if __name__ == "__main__":
    main() 