"""
Exemple de tableau de bord interactif pour le trading crypto.

Ce script démontre comment créer un tableau de bord interactif avec Dash
pour visualiser les données de marché, signaux de trading et performances.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import os

# Import des modules de visualisation
from ai_trading.visualization.dashboards.chart_components import CandlestickChart, IndicatorChart
from ai_trading.visualization.dashboards.performance_components import PerformanceMetrics
from ai_trading.visualization.dashboards.portfolio_components import PortfolioView

def load_market_data():
    """Charge des données de marché simulées pour la démonstration."""
    # Période de simulation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simuler une tendance de prix
    n = len(dates)
    base_trend = np.concatenate([
        np.linspace(40000, 45000, n//3),
        np.linspace(45000, 38000, n//3),
        np.linspace(38000, 50000, n - 2*(n//3))
    ])
    
    # Ajouter de la volatilité
    volatility = np.random.normal(0, 800, n)
    prices = base_trend + volatility
    
    # Créer les données OHLCV
    data = pd.DataFrame({
        'open': prices * (1 - np.random.uniform(0, 0.01, n)),
        'high': prices * (1 + np.random.uniform(0.005, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0.005, 0.02, n)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n) * (1 + 0.2 * np.sin(np.linspace(0, 10, n)))
    }, index=dates)
    
    return data

def generate_signals(data):
    """Génère des signaux de trading simulés."""
    # Calculer quelques indicateurs
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    # Générer des signaux basés sur les croisements de moyennes mobiles
    data['signal'] = 0
    data.loc[data['sma_20'] > data['sma_50'], 'signal'] = 1  # Achat
    data.loc[data['sma_20'] < data['sma_50'], 'signal'] = -1  # Vente
    
    # Ajouter des métriques de confiance (simulées)
    data['confidence'] = np.random.uniform(0.6, 0.95, len(data))
    
    # Ajouter un score de signal combiné
    data['signal_strength'] = data['signal'] * data['confidence']
    
    return data

def simulate_trades(data):
    """Simule des trades basés sur les signaux."""
    trades = []
    position = 0
    entry_price = 0
    
    for i in range(1, len(data)):
        date = data.index[i]
        signal = data['signal'].iloc[i]
        prev_signal = data['signal'].iloc[i-1]
        price = data['close'].iloc[i]
        
        # Entrée en position longue
        if signal == 1 and prev_signal != 1 and position <= 0:
            trades.append({
                'date': date,
                'type': 'buy',
                'price': price,
                'quantity': 1.0,
                'pnl': 0
            })
            position = 1
            entry_price = price
        
        # Sortie de position longue
        elif signal == -1 and prev_signal != -1 and position > 0:
            pnl = price - entry_price
            trades.append({
                'date': date,
                'type': 'sell',
                'price': price,
                'quantity': 1.0,
                'pnl': pnl
            })
            position = 0
            entry_price = 0
    
    # Convertir en DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculer le P&L cumulatif
    if not trades_df.empty:
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    return trades_df

def simulate_portfolio(initial_value, trades_df):
    """Simule l'évolution d'un portefeuille basé sur les trades."""
    if trades_df.empty:
        return pd.DataFrame()
    
    portfolio = []
    current_value = initial_value
    btc_position = 0
    
    # Trier les trades par date
    trades_df = trades_df.sort_values('date')
    
    for _, trade in trades_df.iterrows():
        if trade['type'] == 'buy':
            btc_position += trade['quantity']
            current_value -= trade['price'] * trade['quantity']  # Dépenser des USD
        elif trade['type'] == 'sell':
            btc_position -= trade['quantity']
            current_value += trade['price'] * trade['quantity']  # Recevoir des USD
        
        portfolio.append({
            'date': trade['date'],
            'cash': current_value,
            'btc_holdings': btc_position,
            'btc_value': btc_position * trade['price'],
            'total_value': current_value + (btc_position * trade['price'])
        })
    
    # Convertir en DataFrame
    portfolio_df = pd.DataFrame(portfolio)
    
    return portfolio_df

def get_performance_metrics(data, trades_df, portfolio_df):
    """Calcule les métriques de performance de la stratégie."""
    if trades_df.empty or portfolio_df.empty:
        return {}
    
    # Calculer les rendements
    initial_value = 100000
    final_value = portfolio_df['total_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value
    
    # Nombre de trades
    n_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] <= 0])
    
    # Taux de réussite
    win_rate = winning_trades / n_trades if n_trades > 0 else 0
    
    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown maximum
    portfolio_df['peak'] = portfolio_df['total_value'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown = abs(portfolio_df['drawdown'].min())
    
    # Ratio de Sharpe (simplifié, sans tenir compte du taux sans risque)
    if len(portfolio_df) >= 2:
        returns = portfolio_df['total_value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_return': total_return,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def create_dashboard(data, trades_df, portfolio_df, metrics):
    """Crée le tableau de bord interactif."""
    # Initialiser l'application Dash
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Définir la mise en page
    app.layout = html.Div([
        # En-tête
        html.Div([
            html.H1("Tableau de Bord de Trading Crypto", 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'margin-bottom': '10px'}),
            html.H3("BTC/USD - Analyse technique et Performance", 
                    style={'textAlign': 'center', 'color': '#7f8c8d', 'margin-top': '0px'})
        ]),
        
        # Contrôles
        html.Div([
            html.Label("Période d'affichage:"),
            dcc.Dropdown(
                id='timeframe-dropdown',
                options=[
                    {'label': 'Tout', 'value': 'all'},
                    {'label': 'Dernier mois', 'value': '30'},
                    {'label': 'Dernières 2 semaines', 'value': '14'},
                    {'label': 'Dernière semaine', 'value': '7'}
                ],
                value='all',
                clearable=False
            ),
            
            html.Label("Indicateurs techniques:"),
            dcc.Checklist(
                id='indicators-checklist',
                options=[
                    {'label': 'Moyennes mobiles', 'value': 'sma'},
                    {'label': 'Volume', 'value': 'volume'},
                    {'label': 'Signaux', 'value': 'signals'}
                ],
                value=['sma', 'volume', 'signals'],
                inline=True
            )
        ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '10px 0'}),
        
        # Graphique principal
        html.Div([
            dcc.Graph(id='market-chart')
        ]),
        
        # Métriques de performance
        html.Div([
            html.H3("Métriques de Performance", 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px 0 10px 0'}),
            
            html.Div([
                # Première ligne de métriques
                html.Div([
                    # Rendement total
                    html.Div([
                        html.H5("Rendement Total"),
                        html.H3(f"{metrics['total_return']:.2%}", 
                                style={'color': 'green' if metrics['total_return'] > 0 else 'red'})
                    ], className='metric-card'),
                    
                    # Nombre de trades
                    html.Div([
                        html.H5("Nombre de Trades"),
                        html.H3(f"{metrics['n_trades']}")
                    ], className='metric-card'),
                    
                    # Taux de réussite
                    html.Div([
                        html.H5("Taux de Réussite"),
                        html.H3(f"{metrics['win_rate']:.2%}")
                    ], className='metric-card'),
                ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                
                # Deuxième ligne de métriques
                html.Div([
                    # Profit factor
                    html.Div([
                        html.H5("Profit Factor"),
                        html.H3(f"{metrics['profit_factor']:.2f}")
                    ], className='metric-card'),
                    
                    # Drawdown maximum
                    html.Div([
                        html.H5("Drawdown Maximum"),
                        html.H3(f"{metrics['max_drawdown']:.2%}", 
                                style={'color': 'red'})
                    ], className='metric-card'),
                    
                    # Ratio de Sharpe
                    html.Div([
                        html.H5("Ratio de Sharpe"),
                        html.H3(f"{metrics['sharpe_ratio']:.2f}")
                    ], className='metric-card'),
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ])
        ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '20px 0'}),
        
        # Graphiques secondaires (PnL et Portefeuille)
        html.Div([
            html.Div([
                html.H3("Évolution du P&L", 
                        style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '10px 0'}),
                dcc.Graph(id='pnl-chart')
            ], style={'width': '48%'}),
            
            html.Div([
                html.H3("Composition du Portefeuille", 
                        style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '10px 0'}),
                dcc.Graph(id='portfolio-chart')
            ], style={'width': '48%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}),
        
        # Style CSS
        html.Style('''
            .metric-card {
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 30%;
                text-align: center;
            }
            h5 {
                color: #7f8c8d;
                margin-top: 0;
            }
            h3 {
                margin-bottom: 0;
            }
        ''')
    ])
    
    # Callback pour mettre à jour le graphique principal
    @app.callback(
        Output('market-chart', 'figure'),
        [Input('timeframe-dropdown', 'value'),
         Input('indicators-checklist', 'value')]
    )
    def update_market_chart(timeframe, indicators):
        # Filtrer les données selon la période sélectionnée
        if timeframe != 'all':
            filtered_data = data.iloc[-int(timeframe):]
        else:
            filtered_data = data
        
        # Créer le graphique
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        
        # Ajouter les chandeliers
        fig.add_trace(
            go.Candlestick(
                x=filtered_data.index,
                open=filtered_data['open'],
                high=filtered_data['high'],
                low=filtered_data['low'],
                close=filtered_data['close'],
                name="BTC/USD"
            ), row=1, col=1
        )
        
        # Ajouter les moyennes mobiles si sélectionnées
        if 'sma' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['sma_20'],
                    line=dict(color='blue', width=1),
                    name="SMA 20"
                ), row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['sma_50'],
                    line=dict(color='orange', width=1),
                    name="SMA 50"
                ), row=1, col=1
            )
        
        # Ajouter les signaux si sélectionnés
        if 'signals' in indicators:
            # Filtrer les signaux d'achat et de vente
            buy_signals = filtered_data[filtered_data['signal'] == 1]
            sell_signals = filtered_data[filtered_data['signal'] == -1]
            
            # Ajouter les signaux d'achat
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['low'] * 0.99,  # Placer sous les chandeliers
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name="Signal d'achat"
                ), row=1, col=1
            )
            
            # Ajouter les signaux de vente
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['high'] * 1.01,  # Placer au-dessus des chandeliers
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name="Signal de vente"
                ), row=1, col=1
            )
        
        # Ajouter le volume si sélectionné
        if 'volume' in indicators:
            fig.add_trace(
                go.Bar(
                    x=filtered_data.index,
                    y=filtered_data['volume'],
                    name="Volume",
                    marker=dict(color='rgba(0, 0, 255, 0.3)')
                ), row=2, col=1
            )
        
        # Ajouter les trades réalisés
        if not trades_df.empty:
            filtered_trades = trades_df[trades_df['date'].isin(filtered_data.index)]
            
            # Ajouter les achats
            buy_trades = filtered_trades[filtered_trades['type'] == 'buy']
            fig.add_trace(
                go.Scatter(
                    x=buy_trades['date'],
                    y=buy_trades['price'],
                    mode='markers',
                    marker=dict(symbol='circle', size=12, color='blue', line=dict(width=2, color='darkblue')),
                    name="Achat exécuté"
                ), row=1, col=1
            )
            
            # Ajouter les ventes
            sell_trades = filtered_trades[filtered_trades['type'] == 'sell']
            fig.add_trace(
                go.Scatter(
                    x=sell_trades['date'],
                    y=sell_trades['price'],
                    mode='markers',
                    marker=dict(symbol='circle', size=12, color='orange', line=dict(width=2, color='darkorange')),
                    name="Vente exécutée"
                ), row=1, col=1
            )
        
        # Mettre à jour la mise en page
        fig.update_layout(
            title='BTC/USD - Graphique des prix et indicateurs',
            xaxis_title='Date',
            yaxis_title='Prix (USD)',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    # Callback pour mettre à jour le graphique de P&L
    @app.callback(
        Output('pnl-chart', 'figure'),
        [Input('timeframe-dropdown', 'value')]
    )
    def update_pnl_chart(timeframe):
        # Créer la figure
        fig = go.Figure()
        
        if not trades_df.empty:
            # Filtrer selon la période si nécessaire
            if timeframe != 'all' and not trades_df.empty:
                cutoff_date = data.index[-int(timeframe)]
                filtered_trades = trades_df[trades_df['date'] >= cutoff_date]
            else:
                filtered_trades = trades_df
            
            # Ajouter le P&L cumulatif
            fig.add_trace(
                go.Scatter(
                    x=filtered_trades['date'],
                    y=filtered_trades['cumulative_pnl'],
                    mode='lines+markers',
                    line=dict(width=2, color='green'),
                    name="P&L Cumulatif"
                )
            )
            
            # Ajouter les trades individuels
            fig.add_trace(
                go.Bar(
                    x=filtered_trades['date'],
                    y=filtered_trades['pnl'],
                    marker=dict(color=filtered_trades['pnl'].apply(lambda x: 'green' if x > 0 else 'red')),
                    name="P&L par Trade"
                )
            )
        
        # Mise en page
        fig.update_layout(
            title='Évolution du P&L',
            xaxis_title='Date',
            yaxis_title='P&L (USD)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        
        return fig
    
    # Callback pour mettre à jour le graphique du portefeuille
    @app.callback(
        Output('portfolio-chart', 'figure'),
        [Input('timeframe-dropdown', 'value')]
    )
    def update_portfolio_chart(timeframe):
        # Créer la figure
        fig = go.Figure()
        
        if not portfolio_df.empty:
            # Filtrer selon la période si nécessaire
            if timeframe != 'all':
                cutoff_date = data.index[-int(timeframe)]
                filtered_portfolio = portfolio_df[portfolio_df['date'] >= cutoff_date]
            else:
                filtered_portfolio = portfolio_df
            
            # Ajouter le cash
            fig.add_trace(
                go.Bar(
                    x=filtered_portfolio['date'],
                    y=filtered_portfolio['cash'],
                    name="Cash (USD)",
                    marker=dict(color='lightblue')
                )
            )
            
            # Ajouter la valeur BTC
            fig.add_trace(
                go.Bar(
                    x=filtered_portfolio['date'],
                    y=filtered_portfolio['btc_value'],
                    name="BTC (USD)",
                    marker=dict(color='orange')
                )
            )
            
            # Ajouter la valeur totale
            fig.add_trace(
                go.Scatter(
                    x=filtered_portfolio['date'],
                    y=filtered_portfolio['total_value'],
                    mode='lines+markers',
                    line=dict(width=2, color='green'),
                    name="Valeur Totale"
                )
            )
        
        # Mise en page
        fig.update_layout(
            title='Composition du Portefeuille',
            xaxis_title='Date',
            yaxis_title='Valeur (USD)',
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        
        return fig
    
    return app

def trading_dashboard_example():
    """Exemple principal de tableau de bord de trading."""
    # Charger et préparer les données
    print("Chargement et préparation des données...")
    market_data = load_market_data()
    
    # Générer des signaux
    market_data_with_signals = generate_signals(market_data)
    
    # Simuler des trades
    trades_df = simulate_trades(market_data_with_signals)
    
    # Simuler l'évolution du portefeuille
    portfolio_df = simulate_portfolio(100000, trades_df)
    
    # Calculer les métriques de performance
    metrics = get_performance_metrics(market_data_with_signals, trades_df, portfolio_df)
    
    # Créer le tableau de bord
    print("Création du tableau de bord...")
    app = create_dashboard(market_data_with_signals, trades_df, portfolio_df, metrics)
    
    # Option pour exécuter l'application en mode développement
    print("\nTableau de bord prêt!")
    print("Pour exécuter le tableau de bord, décommentez la ligne app.run_server() à la fin du fichier.")
    print("Puis exécutez ce script et accédez à http://127.0.0.1:8050/ dans votre navigateur.")
    
    # Retourner l'application pour référence ultérieure
    return app, market_data_with_signals, trades_df, portfolio_df, metrics

if __name__ == "__main__":
    app, data, trades, portfolio, metrics = trading_dashboard_example()
    
    # Décommentez cette ligne pour exécuter l'application
    # app.run_server(debug=True) 