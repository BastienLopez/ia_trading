"""
Callbacks pour le dashboard interactif d'analyse post-trade.

Ce module définit les callbacks pour gérer les interactions de l'interface
utilisateur et mettre à jour dynamiquement les graphiques et tableaux.
"""

from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, html

from ai_trading.dashboard.data_loader import DataLoader
from ai_trading.dashboard.visualization_3d import Visualizer3D
from ai_trading.dashboard.trade_analysis import TradeAnalyzer


def register_callbacks(app):
    """
    Enregistre les callbacks pour l'application Dash.

    Args:
        app: Instance de l'application Dash
    """
    # Initialiser le chargeur de données
    data_loader = DataLoader()
    visualizer_3d = Visualizer3D(data_loader)
    trade_analyzer = TradeAnalyzer(data_loader)

    # Callbacks pour l'initialisation des données partagées
    @app.callback(Output("shared-data", "data"), Input("tabs", "active_tab"))
    def initialize_data(active_tab):
        """
        Initialise les données partagées au chargement du dashboard.

        Args:
            active_tab: Onglet actif (non utilisé mais nécessaire pour le callback)

        Returns:
            Données partagées
        """
        # Charger les données principales
        portfolio_history = data_loader.load_portfolio_history()
        transactions = data_loader.load_transactions()

        # Créer un dictionnaire de données à partager
        return {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "portfolio_id": "main",
            "date_range": {
                "start": portfolio_history["date"].min().strftime("%Y-%m-%d"),
                "end": portfolio_history["date"].max().strftime("%Y-%m-%d"),
            },
        }

    # Vue d'ensemble - Callbacks

    @app.callback(
        [
            Output("portfolio-date-range", "children"),
            Output("overview-value", "children"),
            Output("overview-value-change", "children"),
            Output("overview-performance", "children"),
            Output("overview-performance-period", "children"),
            Output("overview-sharpe", "children"),
            Output("overview-drawdown", "children"),
            Output("overview-drawdown-date", "children"),
        ],
        Input("shared-data", "data"),
    )
    def update_overview_metrics(shared_data):
        """
        Met à jour les métriques principales de la vue d'ensemble.

        Args:
            shared_data: Données partagées

        Returns:
            Valeurs mises à jour pour les métriques
        """
        # Charger les données
        portfolio_history = data_loader.load_portfolio_history(
            shared_data["portfolio_id"]
        )

        # Extraire la date la plus récente et la valeur
        latest_date = portfolio_history["date"].max()
        latest_value = portfolio_history.loc[
            portfolio_history["date"] == latest_date, "valeur"
        ].values[0]

        # Calculer la variation sur un jour
        prev_date = portfolio_history.iloc[-2]["date"]
        prev_value = portfolio_history.iloc[-2]["valeur"]
        daily_change_pct = (latest_value / prev_value - 1) * 100
        daily_change_text = (
            f"{daily_change_pct:+.2f}% depuis {prev_date.strftime('%d/%m/%Y')}"
        )

        # Calculer la performance totale
        initial_value = portfolio_history.iloc[0]["valeur"]
        total_performance_pct = (latest_value / initial_value - 1) * 100

        # Période de performance
        start_date = portfolio_history["date"].min()
        period_text = f"depuis {start_date.strftime('%d/%m/%Y')}"

        # Ratio de Sharpe
        risk_metrics = data_loader.load_risk_metrics()
        sharpe_ratio = risk_metrics.loc[
            risk_metrics["metric"] == "Ratio de Sharpe", "value"
        ].values[0]

        # Drawdown maximum
        max_drawdown = risk_metrics.loc[
            risk_metrics["metric"] == "Drawdown Maximum", "value"
        ].values[0]
        drawdown_duration = risk_metrics.loc[
            risk_metrics["metric"] == "Durée Drawdown (jours)", "value"
        ].values[0]
        drawdown_text = f"Durée: {drawdown_duration:.0f} jours"

        return [
            f"Période: {start_date.strftime('%d/%m/%Y')} - {latest_date.strftime('%d/%m/%Y')}",
            f"{latest_value:,.2f} €",
            daily_change_text,
            f"{total_performance_pct:+.2f}%",
            period_text,
            f"{sharpe_ratio:.2f}",
            f"{max_drawdown:.2f}%",
            drawdown_text,
        ]

    @app.callback(
        Output("overview-portfolio-chart", "figure"),
        [
            Input("shared-data", "data"),
            Input("btn-timeframe-1m", "n_clicks"),
            Input("btn-timeframe-3m", "n_clicks"),
            Input("btn-timeframe-6m", "n_clicks"),
            Input("btn-timeframe-1y", "n_clicks"),
            Input("btn-timeframe-max", "n_clicks"),
        ],
    )
    def update_portfolio_chart(shared_data, btn_1m, btn_3m, btn_6m, btn_1y, btn_max):
        """
        Met à jour le graphique de l'évolution du portefeuille.

        Args:
            shared_data: Données partagées
            btn_*: Clics sur les boutons de période (pour déterminer la période active)

        Returns:
            Figure mise à jour
        """
        # Déterminer quelle période est active en fonction du dernier bouton cliqué
        ctx = dash.callback_context
        if not ctx.triggered:
            timeframe = "6m"  # Par défaut
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "btn-timeframe-1m":
                timeframe = "1m"
            elif button_id == "btn-timeframe-3m":
                timeframe = "3m"
            elif button_id == "btn-timeframe-6m":
                timeframe = "6m"
            elif button_id == "btn-timeframe-1y":
                timeframe = "1y"
            elif button_id == "btn-timeframe-max":
                timeframe = "max"
            else:
                timeframe = "6m"  # Par défaut

        # Charger les données
        portfolio_history = data_loader.load_portfolio_history(
            shared_data["portfolio_id"]
        )

        # Filtrer en fonction de la période
        end_date = portfolio_history["date"].max()
        if timeframe == "1m":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "3m":
            start_date = end_date - timedelta(days=90)
        elif timeframe == "6m":
            start_date = end_date - timedelta(days=180)
        elif timeframe == "1y":
            start_date = end_date - timedelta(days=365)
        else:  # 'max'
            start_date = portfolio_history["date"].min()

        filtered_data = portfolio_history[portfolio_history["date"] >= start_date]

        # Créer le graphique
        fig = go.Figure()

        # Ajouter la trace principale de la valeur du portefeuille
        fig.add_trace(
            go.Scatter(
                x=filtered_data["date"],
                y=filtered_data["valeur"],
                mode="lines",
                name="Valeur",
                line=dict(color="#2C82FF", width=2),
            )
        )

        # Personnaliser le layout
        fig.update_layout(
            title=f"Évolution de la Valeur du Portefeuille ({timeframe.upper()})",
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    @app.callback(
        Output("overview-allocation-chart", "figure"), Input("shared-data", "data")
    )
    def update_allocation_chart(shared_data):
        """
        Met à jour le graphique d'allocation d'actifs.

        Args:
            shared_data: Données partagées

        Returns:
            Figure mise à jour
        """
        # Charger l'allocation la plus récente
        allocation = data_loader.load_portfolio_allocation(
            portfolio_id=shared_data["portfolio_id"]
        )

        # Transformer en format pour graphique
        allocation_data = allocation.drop("date", axis=1).iloc[0]

        # Filtrer les petites allocations pour la lisibilité
        threshold = 0.01  # 1%
        small_allocations = allocation_data[allocation_data < threshold]
        if not small_allocations.empty:
            allocation_data = allocation_data[allocation_data >= threshold]
            allocation_data["Autres"] = small_allocations.sum()

        # Créer le graphique
        colors = px.colors.qualitative.Plotly
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=allocation_data.index,
                    values=allocation_data.values,
                    hole=0.4,
                    marker=dict(colors=colors[: len(allocation_data)]),
                    textinfo="percent+label",
                    textposition="inside",
                    insidetextorientation="radial",
                )
            ]
        )

        # Personnaliser le layout
        fig.update_layout(
            title="Allocation Actuelle",
            height=300,
            margin=dict(l=20, r=20, t=30, b=0),
            showlegend=False,
            template="plotly_dark",
        )

        return fig

    @app.callback(
        Output("overview-recent-transactions", "children"), Input("shared-data", "data")
    )
    def update_recent_transactions(shared_data):
        """
        Met à jour la liste des transactions récentes.

        Args:
            shared_data: Données partagées

        Returns:
            Tableau des transactions récentes
        """
        # Charger les transactions
        transactions = data_loader.load_transactions(shared_data["portfolio_id"])

        # Extraire les 5 transactions les plus récentes
        recent_txs = transactions.sort_values("date", ascending=False).head(5)

        # Formater les données pour l'affichage
        table_data = []
        for _, tx in recent_txs.iterrows():
            tx_type_style = (
                {"color": "green"} if tx["type"] == "achat" else {"color": "red"}
            )
            row = [
                html.Td(tx["date"].strftime("%d/%m/%Y")),
                html.Td(tx["actif"], style={"fontWeight": "bold"}),
                html.Td(tx["type"].capitalize(), style=tx_type_style),
                html.Td(f"{tx['prix']:,.2f} €"),
                html.Td(f"{tx['quantite']:.4f}"),
                html.Td(f"{tx['montant']:,.2f} €"),
            ]
            table_data.append(html.Tr(row))

        # Créer le tableau
        return html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Date"),
                            html.Th("Actif"),
                            html.Th("Type"),
                            html.Th("Prix"),
                            html.Th("Quantité"),
                            html.Th("Montant"),
                        ]
                    )
                ),
                html.Tbody(table_data),
            ],
            className="table table-striped table-hover table-sm",
        )

    @app.callback(
        Output("overview-asset-performance", "figure"), Input("shared-data", "data")
    )
    def update_asset_performance(shared_data):
        """
        Met à jour le graphique de performance par actif.

        Args:
            shared_data: Données partagées

        Returns:
            Figure mise à jour
        """
        # Charger l'allocation
        allocation = data_loader.load_portfolio_allocation(
            portfolio_id=shared_data["portfolio_id"]
        )
        allocation_data = allocation.drop("date", axis=1).iloc[0]

        # Simuler des performances par actif (à remplacer par des données réelles)
        np.random.seed(42)
        assets = allocation_data.index
        performances = pd.Series(
            {asset: np.random.normal(0.1, 0.2) for asset in assets}
        )

        # Trier par performance
        performances = performances.sort_values(ascending=False)

        # Créer le graphique
        fig = go.Figure()

        # Déterminer les couleurs en fonction de la performance
        colors = ["green" if perf >= 0 else "red" for perf in performances]

        # Ajouter la trace principale
        fig.add_trace(
            go.Bar(
                x=performances.index,
                y=performances.values * 100,  # Convertir en pourcentage
                marker_color=colors,
                text=[f"{perf*100:+.1f}%" for perf in performances],
                textposition="auto",
            )
        )

        # Personnaliser le layout
        fig.update_layout(
            title="Performance par Actif",
            xaxis_title="Actif",
            yaxis_title="Performance (%)",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
        )

        return fig

    # Transaction - Callbacks

    @app.callback(Output("transactions-asset", "options"), Input("shared-data", "data"))
    def update_transaction_asset_options(shared_data):
        """
        Met à jour les options de filtre d'actifs pour les transactions.

        Args:
            shared_data: Données partagées

        Returns:
            Liste des options d'actifs
        """
        # Charger les transactions
        transactions = data_loader.load_transactions(shared_data["portfolio_id"])

        # Extraire les actifs uniques
        assets = sorted(transactions["actif"].unique())

        # Créer les options
        options = [{"label": "Tous", "value": "all"}]
        options.extend([{"label": asset, "value": asset} for asset in assets])

        return options

    @app.callback(
        Output("transactions-table", "children"),
        [
            Input("shared-data", "data"),
            Input("transactions-type", "value"),
            Input("transactions-asset", "value"),
            Input("transactions-date-range", "start_date"),
            Input("transactions-date-range", "end_date"),
        ],
    )
    def update_transactions_table(shared_data, tx_type, asset, start_date, end_date):
        """
        Met à jour le tableau détaillé des transactions.

        Args:
            shared_data: Données partagées
            tx_type: Type de transaction sélectionné
            asset: Actif sélectionné
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Tableau des transactions
        """
        # Charger les transactions
        transactions = data_loader.load_transactions(shared_data["portfolio_id"])

        # Appliquer les filtres
        if tx_type != "all":
            transactions = transactions[transactions["type"] == tx_type]

        if asset != "all":
            transactions = transactions[transactions["actif"] == asset]

        if start_date:
            transactions = transactions[
                transactions["date"] >= pd.to_datetime(start_date)
            ]

        if end_date:
            transactions = transactions[
                transactions["date"] <= pd.to_datetime(end_date)
            ]

        # Trier par date décroissante
        transactions = transactions.sort_values("date", ascending=False)

        # Limiter à 100 transactions maximum pour la performance
        if len(transactions) > 100:
            transactions = transactions.head(100)

        # Formater pour l'affichage
        table_data = []
        for _, tx in transactions.iterrows():
            tx_type_style = (
                {"color": "green"} if tx["type"] == "achat" else {"color": "red"}
            )
            
            # Gestion des performances relatives avec vérification correcte des NaN
            if "perf_relative" in tx and pd.notna(tx["perf_relative"]):
                if tx["perf_relative"] > 0:
                    perf_cell = html.Td(f"{tx['perf_relative']*100:+.2f}%", style={"color": "green"})
                else:
                    perf_cell = html.Td(f"{tx['perf_relative']*100:+.2f}%", style={"color": "red"})
            else:
                perf_cell = html.Td("N/A")

            row = [
                html.Td(tx["id"]),
                html.Td(tx["date"].strftime("%d/%m/%Y")),
                html.Td(tx["actif"], style={"fontWeight": "bold"}),
                html.Td(tx["type"].capitalize(), style=tx_type_style),
                html.Td(f"{tx['prix']:,.2f} €"),
                html.Td(f"{tx['quantite']:.4f}"),
                html.Td(f"{tx['montant']:,.2f} €"),
                html.Td(tx["signal"].replace("_", " ").title()),
                perf_cell,
            ]
            table_data.append(html.Tr(row))

        # Message si aucune transaction
        if not table_data:
            return html.Div(
                "Aucune transaction ne correspond aux critères sélectionnés.",
                className="alert alert-info",
            )

        # Créer le tableau
        return html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("ID"),
                            html.Th("Date"),
                            html.Th("Actif"),
                            html.Th("Type"),
                            html.Th("Prix"),
                            html.Th("Quantité"),
                            html.Th("Montant"),
                            html.Th("Signal"),
                            html.Th("Perf. Relative"),
                        ]
                    )
                ),
                html.Tbody(table_data),
            ],
            className="table table-striped table-hover table-sm",
        )

    # Risque - Callbacks

    @app.callback(
        Output("risk-metrics-summary", "children"), Input("shared-data", "data")
    )
    def update_risk_metrics_summary(shared_data):
        """
        Met à jour le résumé des métriques de risque.

        Args:
            shared_data: Données partagées

        Returns:
            Cartes des métriques de risque
        """
        # Charger les métriques de risque
        risk_metrics = data_loader.load_risk_metrics(
            portfolio_id=shared_data["portfolio_id"]
        )

        # Créer les cartes pour chaque métrique
        cards = []
        for _, metric in risk_metrics.iterrows():
            if metric["metric"] in [
                "VaR (95%)",
                "VaR (99%)",
                "Expected Shortfall (95%)",
                "Drawdown Maximum",
                "Ratio de Sharpe",
                "Ratio de Sortino",
            ]:
                color = "info"
                if (
                    "VaR" in metric["metric"]
                    or "Shortfall" in metric["metric"]
                    or "Drawdown" in metric["metric"]
                ):
                    color = "danger"
                elif "Sharpe" in metric["metric"] or "Sortino" in metric["metric"]:
                    color = "success"

                value_display = f"{metric['value']:,.2f}"
                if (
                    "VaR" in metric["metric"]
                    or "Shortfall" in metric["metric"]
                    or "Drawdown" in metric["metric"]
                ):
                    value_display = f"{metric['value']:,.2f}%"

                card = dbc.Card(
                    [
                        dbc.CardHeader(
                            metric["metric"], className=f"bg-{color} text-white"
                        ),
                        dbc.CardBody(
                            [
                                html.H4(value_display, className="card-title"),
                                html.P(
                                    metric["description"], className="card-text small"
                                ),
                            ]
                        ),
                    ],
                    className="mb-3",
                )

                cards.append(dbc.Col(card, width=4))

        # Organiser en rangées
        rows = []
        for i in range(0, len(cards), 3):
            row_cards = cards[i : i + 3]
            rows.append(dbc.Row(row_cards, className="mb-3"))

        return html.Div(rows)

    # ---------- VISUALISATIONS 3D CALLBACKS ----------

    @app.callback(
        [
            Output("viz3d-title", "children"),
            Output("viz3d-indicator-params", "style"),
            Output("viz3d-trajectory-params", "style"),
            Output("viz3d-clusters-params", "style"),
        ],
        [Input("viz3d-type", "value")],
    )
    def update_viz3d_params_visibility(viz_type):
        """
        Met à jour la visibilité des paramètres en fonction du type de visualisation.
        
        Args:
            viz_type: Type de visualisation 3D sélectionné
            
        Returns:
            Titre de la visualisation et visibilité des paramètres
        """
        indicator_style = {"display": "none"}
        trajectory_style = {"display": "none"}
        clusters_style = {"display": "none"}
        
        title = ""
        
        if viz_type == "indicator_surface":
            indicator_style = {"display": "block"}
            title = "Surface 3D d'indicateurs techniques"
        elif viz_type == "portfolio_trajectory":
            trajectory_style = {"display": "block"}
            title = "Trajectoire 3D du portefeuille"
        elif viz_type == "trade_clusters":
            clusters_style = {"display": "block"}
            title = "Clusters 3D des transactions"
        
        return title, indicator_style, trajectory_style, clusters_style


    @app.callback(
        Output("viz3d-config-collapse", "is_open"),
        [Input("viz3d-config-button", "n_clicks")],
        [State("viz3d-config-collapse", "is_open")],
    )
    def toggle_viz3d_config(n_clicks, is_open):
        """
        Ouvre/ferme le panneau de configuration des visualisations 3D.
        
        Args:
            n_clicks: Nombre de clics sur le bouton
            is_open: État actuel du panneau (ouvert/fermé)
            
        Returns:
            Nouvel état du panneau
        """
        if n_clicks:
            return not is_open
        return is_open


    @app.callback(
        [Output("viz3d-x-axis", "options"), 
         Output("viz3d-y-axis", "options"), 
         Output("viz3d-z-axis", "options")],
        [Input("shared-data", "data")],
    )
    def update_viz3d_axis_options(shared_data):
        """
        Met à jour les options des axes pour les visualisations 3D.
        
        Args:
            shared_data: Données partagées entre les composants
            
        Returns:
            Options pour les dropdowns des axes
        """
        if not shared_data:
            # Valeurs par défaut si pas de données
            default_options = [
                {"label": "RSI", "value": "rsi"},
                {"label": "MACD", "value": "macd"},
                {"label": "Volume", "value": "volume"},
                {"label": "Price", "value": "close"},
            ]
            return default_options, default_options, default_options
        
        # Charger les données
        market_data = data_loader.get_market_data()
        
        # Créer les options pour les axes
        if market_data is not None and not market_data.empty:
            # Sélectionner les colonnes numériques
            numeric_cols = market_data.select_dtypes(include=['float64', 'int64']).columns
            
            # Créer les options
            options = [{"label": col, "value": col} for col in numeric_cols]
            return options, options, options
        
        # Valeurs par défaut si pas de données
        default_options = [
            {"label": "RSI", "value": "rsi"},
            {"label": "MACD", "value": "macd"},
            {"label": "Volume", "value": "volume"},
            {"label": "Price", "value": "close"},
        ]
        return default_options, default_options, default_options


    @app.callback(
        [Output("viz3d-graph", "figure"), 
         Output("viz3d-explanation", "children")],
        [
            Input("viz3d-type", "value"),
            Input("viz3d-x-axis", "value"),
            Input("viz3d-y-axis", "value"),
            Input("viz3d-z-axis", "value"),
            Input("viz3d-window", "value"),
            Input("viz3d-clusters", "value"),
            Input("viz3d-date-range", "start_date"),
            Input("viz3d-date-range", "end_date"),
        ],
        [State("shared-data", "data")],
    )
    def update_viz3d_graph(
        viz_type, x_axis, y_axis, z_axis, window, n_clusters, 
        start_date, end_date, shared_data
    ):
        """
        Met à jour le graphique 3D en fonction des paramètres sélectionnés.
        
        Args:
            viz_type: Type de visualisation
            x_axis, y_axis, z_axis: Axes sélectionnés
            window: Taille de la fenêtre temporelle
            n_clusters: Nombre de clusters pour le clustering
            start_date, end_date: Période sélectionnée
            shared_data: Données partagées
            
        Returns:
            Figure Plotly et explication de la visualisation
        """
        # Vérifier que les valeurs nécessaires sont définies
        if not viz_type:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Sélectionnez un type de visualisation",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                template="plotly_dark"
            )
            empty_fig.add_annotation(
                text="Utilisez le menu ci-dessus pour sélectionner un type de visualisation",
                showarrow=False,
                font=dict(size=14)
            )
            return empty_fig, "Sélectionnez un type de visualisation pour commencer."
        
        # Récupérer les données
        try:
            # Utiliser get_market_data_for_visualization pour de meilleures performances
            market_data = data_loader.get_market_data_for_visualization()
            trades_data = data_loader.get_trades_data()
            
            print(f"Chargement des données: market_data shape={market_data.shape if market_data is not None else 'None'}, trades_data shape={trades_data.shape if trades_data is not None else 'None'}")
            
            # Vérifier que les données ne sont pas vides
            if market_data is None or market_data.empty:
                print("Erreur: Market data est vide ou None")
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Aucune donnée de marché disponible",
                    template="plotly_dark"
                )
                empty_fig.add_annotation(
                    text="Aucune donnée de marché disponible. Veuillez vérifier la connexion aux données.",
                    showarrow=False,
                    font=dict(size=14)
                )
                return empty_fig, "Aucune donnée de marché disponible pour la visualisation. Vérifiez le chargement des données."
            
            if (trades_data is None or trades_data.empty) and viz_type == "trade_clusters":
                print("Erreur: Trades data est vide ou None pour le clustering")
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Aucune donnée de transaction disponible",
                    template="plotly_dark"
                )
                empty_fig.add_annotation(
                    text="Aucune donnée de transaction disponible pour le clustering.",
                    showarrow=False,
                    font=dict(size=14)
                )
                return empty_fig, "Aucune donnée de transaction disponible pour la visualisation de clusters."
            
            # Filtrer par date si nécessaire
            if start_date and end_date:
                try:
                    if 'timestamp' in market_data.columns:
                        market_data = market_data[
                            (market_data['timestamp'] >= pd.to_datetime(start_date)) & 
                            (market_data['timestamp'] <= pd.to_datetime(end_date))
                        ]
                    elif 'date' in market_data.columns:
                        market_data = market_data[
                            (market_data['date'] >= pd.to_datetime(start_date)) & 
                            (market_data['date'] <= pd.to_datetime(end_date))
                        ]
                    
                    if trades_data is not None and 'entry_time' in trades_data.columns:
                        trades_data = trades_data[
                            (trades_data['entry_time'] >= pd.to_datetime(start_date)) & 
                            (trades_data['entry_time'] <= pd.to_datetime(end_date))
                        ]
                except Exception as date_filter_error:
                    print(f"Erreur lors du filtrage par date: {str(date_filter_error)}")
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
            error_fig = go.Figure()
            error_fig.update_layout(
                title=f"Erreur lors du chargement des données",
                template="plotly_dark"
            )
            error_fig.add_annotation(
                text=str(e),
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return error_fig, f"Erreur: {str(e)}"
        
        # Créer la visualisation en fonction du type
        explanation = ""
        
        try:
            if viz_type == "indicator_surface":
                # Vérifier les paramètres nécessaires
                if not all([x_axis, y_axis, z_axis]):
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        title="Paramètres incomplets",
                        template="plotly_dark"
                    )
                    empty_fig.add_annotation(
                        text="Veuillez sélectionner les axes X, Y et Z pour la surface 3D",
                        showarrow=False,
                        font=dict(size=14)
                    )
                    return empty_fig, "Veuillez sélectionner les axes X, Y et Z pour la surface 3D."
                
                # Vérifier que les colonnes existent
                missing_cols = []
                for col, axis_name in zip([x_axis, y_axis, z_axis], ["X", "Y", "Z"]):
                    if col not in market_data.columns:
                        missing_cols.append(f"{axis_name}:{col}")
                
                if missing_cols:
                    error_fig = go.Figure()
                    error_fig.update_layout(
                        title=f"Colonnes manquantes: {', '.join(missing_cols)}",
                        template="plotly_dark"
                    )
                    error_fig.add_annotation(
                        text=f"Les colonnes suivantes n'existent pas dans les données: {', '.join(missing_cols)}",
                        showarrow=False,
                        font=dict(size=14, color="red")
                    )
                    return error_fig, f"Erreur: Colonnes manquantes dans les données: {', '.join(missing_cols)}"
                
                # Vérifier les types de données
                for col in [x_axis, y_axis, z_axis]:
                    if not pd.api.types.is_numeric_dtype(market_data[col]):
                        market_data[col] = pd.to_numeric(market_data[col], errors='coerce')
                
                print(f"Création de surface d'indicateurs avec axes: X={x_axis}, Y={y_axis}, Z={z_axis}")
                fig = visualizer_3d.create_multi_indicator_surface(
                    market_data, x_axis, y_axis, z_axis
                )
                
                explanation = html.Div([
                    html.P([
                        "Cette visualisation montre la relation entre trois indicateurs techniques sous forme d'une surface 3D. ",
                        "Les axes X et Y représentent deux indicateurs, tandis que la hauteur (Z) montre un troisième indicateur. ",
                        "Les couleurs indiquent les valeurs du troisième indicateur, du plus bas (bleu) au plus élevé (jaune)."
                    ]),
                    html.P([
                        "Interprétation: ",
                        html.Ul([
                            html.Li("Les pics montrent où le troisième indicateur atteint ses valeurs maximales"),
                            html.Li("Les vallées indiquent les valeurs minimales"),
                            html.Li("Les plateaux suggèrent des zones de stabilité"),
                            html.Li("Les pentes abruptes révèlent des changements rapides"),
                        ])
                    ]),
                    html.P("Utilisez la souris pour faire pivoter la visualisation et explorer les différentes perspectives.")
                ])
                
            elif viz_type == "portfolio_trajectory":
                window = window or 30  # Valeur par défaut si None
                
                print(f"Création de trajectoire de portefeuille avec fenêtre: {window}")
                fig = visualizer_3d.create_portfolio_trajectory(
                    market_data, window=window, n_components=3
                )
                
                explanation = html.Div([
                    html.P([
                        "Cette visualisation montre la trajectoire 3D du portefeuille dans l'espace des caractéristiques au fil du temps. ",
                        "Une analyse en composantes principales (PCA) a été utilisée pour réduire la dimensionnalité des données ",
                        "et représenter visuellement l'évolution du marché et du portefeuille."
                    ]),
                    html.P([
                        "Interprétation: ",
                        html.Ul([
                            html.Li("Chaque point représente un moment dans le temps"),
                            html.Li("La ligne connecte les points chronologiquement"),
                            html.Li("Les couleurs évoluent du début (violet) à la fin (jaune) de la période"),
                            html.Li("Les regroupements de points indiquent des périodes de stabilité"),
                            html.Li("Les boucles peuvent suggérer des cycles de marché"),
                        ])
                    ]),
                    html.P([
                        f"PC1, PC2 et PC3 sont les trois composantes principales qui expliquent ",
                        f"le plus de variance dans les données multidimensionnelles."
                    ])
                ])
                
            elif viz_type == "trade_clusters":
                if trades_data is None or trades_data.empty:
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        title="Aucune donnée de transaction disponible",
                        template="plotly_dark"
                    )
                    empty_fig.add_annotation(
                        text="Données de transactions insuffisantes pour le clustering.",
                        showarrow=False,
                        font=dict(size=14)
                    )
                    return empty_fig, "Aucune donnée de transaction disponible pour l'analyse de clusters."
                
                n_clusters = n_clusters or 5  # Valeur par défaut si None
                
                print(f"Création de clusters de trades avec {n_clusters} clusters")
                fig = visualizer_3d.create_trade_clusters_3d(
                    trades_data, n_clusters=n_clusters
                )
                
                explanation = html.Div([
                    html.P([
                        "Cette visualisation regroupe les transactions similaires en clusters dans un espace 3D. ",
                        f"Les {n_clusters} clusters ont été créés avec l'algorithme K-means en utilisant ",
                        "le pourcentage de profit, la durée de détention et la volatilité à l'entrée comme dimensions."
                    ]),
                    html.P([
                        "Interprétation: ",
                        html.Ul([
                            html.Li("Chaque point représente une transaction"),
                            html.Li("Les points de même couleur appartiennent au même cluster"),
                            html.Li("Les clusters isolés indiquent des types de trades distincts"),
                            html.Li("Les clusters denses suggèrent des modèles récurrents"),
                            html.Li("Les valeurs aberrantes peuvent représenter des opportunités ou des risques exceptionnels"),
                        ])
                    ]),
                    html.P("Utilisez cette visualisation pour identifier les types de transactions qui ont tendance à être profitables.")
                ])
            
            else:
                error_fig = go.Figure()
                error_fig.update_layout(
                    title=f"Type de visualisation non reconnu: {viz_type}",
                    template="plotly_dark"
                )
                error_fig.add_annotation(
                    text=f"Le type de visualisation '{viz_type}' n'est pas supporté.",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                return error_fig, f"Type de visualisation non reconnu: {viz_type}"
                
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Erreur lors de la création de la visualisation: {str(e)}")
            print(f"Traceback: {traceback_str}")
            
            error_fig = go.Figure()
            error_fig.update_layout(
                title=f"Erreur lors de la création de la visualisation",
                template="plotly_dark"
            )
            error_fig.add_annotation(
                text=str(e),
                showarrow=False,
                font=dict(size=12, color="red")
            )
            return error_fig, f"Erreur lors de la création de la visualisation: {str(e)}"
        
        print(f"Visualisation {viz_type} créée avec succès")
        return fig, explanation


    # ---------- ANALYSE POST-MORTEM CALLBACKS ----------

    @app.callback(
        Output("postmortem-title", "children"),
        [Input("postmortem-analysis-type", "value")],
    )
    def update_postmortem_title(analysis_type):
        """
        Met à jour le titre de l'analyse post-mortem.
        
        Args:
            analysis_type: Type d'analyse sélectionné
            
        Returns:
            Titre de l'analyse
        """
        titles = {
            "performance_summary": "Résumé des performances des transactions",
            "win_loss_analysis": "Analyse des gains et pertes",
            "trade_attribution": "Attribution de la performance des transactions"
        }
        
        return titles.get(analysis_type, "Analyse des transactions")


    @app.callback(
        Output("postmortem-stats", "children"),
        [
            Input("postmortem-analysis-type", "value"),
            Input("postmortem-asset", "value"),
            Input("postmortem-direction", "value"),
            Input("postmortem-result", "value"),
            Input("postmortem-date-range", "start_date"),
            Input("postmortem-date-range", "end_date"),
        ],
    )
    def update_postmortem_stats(
        analysis_type, asset, direction, result, start_date, end_date
    ):
        """
        Met à jour les statistiques de l'analyse post-mortem.
        
        Args:
            analysis_type: Type d'analyse
            asset: Actif sélectionné
            direction: Direction des trades
            result: Résultat des trades
            start_date, end_date: Période sélectionnée
            
        Returns:
            Contenu HTML avec les statistiques
        """
        # Récupérer les données
        trades_data = data_loader.get_trades_data()
        
        # Appliquer les filtres
        trades_data = filter_trades(
            trades_data, asset, direction, result, start_date, end_date
        )
        
        if trades_data.empty:
            return html.Div("Aucune transaction ne correspond aux critères sélectionnés.")
        
        # Calculer les statistiques
        total_trades = len(trades_data)
        winning_trades = trades_data[trades_data['profit'] > 0]
        losing_trades = trades_data[trades_data['profit'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = trades_data['profit'].sum()
        avg_profit = trades_data['profit'].mean()
        
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if losing_trades['profit'].sum() != 0 else float('inf')
        
        # Créer les statistiques au format HTML
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4(f"{total_trades}"),
                        html.P("Total des transactions")
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H4(f"{win_rate:.2%}"),
                        html.P("Taux de réussite")
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H4(f"{total_profit:.2f}"),
                        html.P("Profit total")
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H4(f"{profit_factor:.2f}"),
                        html.P("Facteur de profit")
                    ], className="text-center")
                ], width=3),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4(f"{avg_profit:.2f}"),
                        html.P("Profit moyen")
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4(f"{avg_win:.2f}"),
                        html.P("Gain moyen")
                    ], className="text-center", style={"color": "green"})
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4(f"{avg_loss:.2f}"),
                        html.P("Perte moyenne")
                    ], className="text-center", style={"color": "red"})
                ], width=4),
            ]),
        ])


    def filter_trades(trades_df, asset, direction, result, start_date, end_date):
        """
        Filtre les transactions selon les critères spécifiés.
        
        Args:
            trades_df: DataFrame de transactions
            asset: Actif sélectionné
            direction: Direction des trades
            result: Résultat des trades
            start_date, end_date: Période sélectionnée
            
        Returns:
            DataFrame filtré
        """
        # Créer une copie pour éviter de modifier l'original
        filtered_df = trades_df.copy()
        
        # Filtrer par date
        if start_date and end_date and 'entry_time' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['entry_time'] >= start_date) & 
                (filtered_df['entry_time'] <= end_date)
            ]
        
        # Filtrer par actif
        if asset and asset != "all" and 'symbol' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['symbol'] == asset]
        
        # Filtrer par direction
        if direction and direction != "all" and 'direction' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['direction'] == direction]
        
        # Filtrer par résultat
        if result and result != "all" and 'profit' in filtered_df.columns:
            if result == "win":
                filtered_df = filtered_df[filtered_df['profit'] > 0]
            elif result == "loss":
                filtered_df = filtered_df[filtered_df['profit'] <= 0]
        
        return filtered_df


    @app.callback(
        Output("postmortem-graph", "figure"),
        [
            Input("postmortem-analysis-type", "value"),
            Input("postmortem-asset", "value"),
            Input("postmortem-direction", "value"),
            Input("postmortem-result", "value"),
            Input("postmortem-date-range", "start_date"),
            Input("postmortem-date-range", "end_date"),
        ],
    )
    def update_postmortem_graph(
        analysis_type, asset, direction, result, start_date, end_date
    ):
        """
        Met à jour le graphique de l'analyse post-mortem.
        
        Args:
            analysis_type: Type d'analyse
            asset: Actif sélectionné
            direction: Direction des trades
            result: Résultat des trades
            start_date, end_date: Période sélectionnée
            
        Returns:
            Figure Plotly pour l'analyse
        """
        # Vérifier le type d'analyse
        if not analysis_type:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Sélectionnez un type d'analyse",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                template="plotly_dark"
            )
            return empty_fig
        
        # Récupérer les données
        try:
            trades_data = data_loader.get_trades_data()
            
            # Appliquer les filtres
            filtered_trades = filter_trades(
                trades_data, asset, direction, result, start_date, end_date
            )
            
            if filtered_trades.empty:
                # Retourner un graphique vide avec un message
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Aucune transaction ne correspond aux critères sélectionnés",
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False),
                    template="plotly_dark"
                )
                empty_fig.add_annotation(
                    text="Modifiez les filtres pour voir des résultats",
                    showarrow=False,
                    font=dict(size=20)
                )
                return empty_fig
            
            # Récupérer les données de marché si nécessaire pour l'attribution
            market_data = None
            if analysis_type == "trade_attribution":
                market_data = data_loader.get_market_data_for_visualization()
        
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
            error_fig = go.Figure()
            error_fig.update_layout(
                title=f"Erreur lors du chargement des données: {str(e)}",
                template="plotly_dark"
            )
            return error_fig
        
        # Créer le graphique en fonction du type d'analyse
        try:
            if analysis_type == "performance_summary":
                return trade_analyzer.create_performance_summary(filtered_trades)
            
            elif analysis_type == "win_loss_analysis":
                return trade_analyzer.create_win_loss_analysis(filtered_trades)
            
            elif analysis_type == "trade_attribution":
                return trade_analyzer.create_trade_attribution(filtered_trades, market_data)
            
            else:
                # Type d'analyse non reconnu
                error_fig = go.Figure()
                error_fig.update_layout(
                    title="Type d'analyse non reconnu",
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False),
                    template="plotly_dark"
                )
                return error_fig
            
        except Exception as e:
            print(f"Erreur lors de la création du graphique: {str(e)}")
            error_fig = go.Figure()
            error_fig.update_layout(
                title=f"Erreur lors de la création du graphique",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                template="plotly_dark"
            )
            error_fig.add_annotation(
                text=str(e),
                showarrow=False,
                font=dict(size=12, color="red")
            )
            return error_fig


    @app.callback(
        Output("postmortem-insights", "children"),
        [
            Input("postmortem-analysis-type", "value"),
            Input("postmortem-graph", "figure"),
            Input("postmortem-stats", "children"),
        ],
    )
    def update_postmortem_insights(analysis_type, figure, stats):
        """
        Met à jour les insights et recommandations de l'analyse post-mortem.
        
        Args:
            analysis_type: Type d'analyse
            figure: Figure générée
            stats: Statistiques calculées
            
        Returns:
            Contenu HTML avec les insights
        """
        # Récupérer les données
        trades_data = data_loader.get_trades_data()
        
        if trades_data.empty:
            return html.Div("Aucune donnée disponible pour générer des insights.")
        
        # Générer des insights en fonction du type d'analyse
        insights = []
        
        if analysis_type == "performance_summary":
            # Analyser la distribution des profits
            winning_trades = trades_data[trades_data['profit'] > 0]
            win_rate = len(winning_trades) / len(trades_data) if len(trades_data) > 0 else 0
            
            if win_rate < 0.4:
                insights.append(html.P([
                    html.Strong("Taux de réussite faible: "),
                    f"Votre taux de réussite est de {win_rate:.2%}, ce qui est relativement bas. ",
                    "Considérez d'ajuster vos critères d'entrée et sortie, ou d'utiliser des stop-loss plus serrés."
                ]))
            
            # Analyser la performance par actif
            if 'symbol' in trades_data.columns:
                asset_perf = trades_data.groupby('symbol')['profit'].sum()
                best_asset = asset_perf.idxmax() if not asset_perf.empty else None
                worst_asset = asset_perf.idxmin() if not asset_perf.empty else None
                
                if best_asset and worst_asset:
                    insights.append(html.P([
                        html.Strong("Performance par actif: "),
                        f"{best_asset} a généré le plus de profits, tandis que {worst_asset} a causé le plus de pertes. ",
                        "Envisagez d'allouer plus de capital aux actifs les plus performants."
                    ]))
            
            # Analyser la performance par jour
            if 'entry_day' in trades_data.columns:
                day_perf = trades_data.groupby('entry_day')['profit'].mean()
                best_day = day_perf.idxmax() if not day_perf.empty else None
                worst_day = day_perf.idxmin() if not day_perf.empty else None
                
                if best_day and worst_day:
                    insights.append(html.P([
                        html.Strong("Performance par jour: "),
                        f"Les transactions ouvertes le {best_day} ont tendance à être plus profitables, ",
                        f"tandis que celles du {worst_day} sont moins performantes. ",
                        f"Considérez d'ajuster votre stratégie en fonction du jour de la semaine."
                    ]))
        
        elif analysis_type == "win_loss_analysis":
            # Analyser le ratio gain/perte
            winning_trades = trades_data[trades_data['profit'] > 0]
            losing_trades = trades_data[trades_data['profit'] <= 0]
            
            avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
            
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            if win_loss_ratio < 1.5:
                insights.append(html.P([
                    html.Strong("Ratio gain/perte insuffisant: "),
                    f"Votre ratio moyen gain/perte est de {win_loss_ratio:.2f}, ce qui est relativement bas. ",
                    "Pour être rentable avec votre taux de réussite actuel, visez un ratio d'au moins 2.0. ",
                    "Considérez de laisser courir vos profits plus longtemps et de couper vos pertes plus tôt."
                ]))
            
            # Analyser les séries de trades
            if len(trades_data) >= 10:
                insights.append(html.P([
                    html.Strong("Séquences de trades: "),
                    "Analysez les séquences de gains et pertes pour identifier d'éventuels biais psychologiques. ",
                    "Après une série de pertes, vous pourriez être enclin à prendre trop de risques ou à éviter les opportunités valables."
                ]))
        
        elif analysis_type == "trade_attribution":
            # Analyser les facteurs de performance
            if 'holding_period' in trades_data.columns:
                # Corrélation entre durée et profit
                corr = trades_data['holding_period'].corr(trades_data['profit'])
                
                if abs(corr) > 0.3:
                    direction = "positive" if corr > 0 else "négative"
                    recommendation = "plus longtemps" if corr > 0 else "moins longtemps"
                    
                    insights.append(html.P([
                        html.Strong("Impact de la durée: "),
                        f"Il existe une corrélation {direction} ({corr:.2f}) entre la durée de détention et le profit. ",
                        f"Considérez de conserver vos positions {recommendation} pour optimiser vos rendements."
                    ]))
            
            # Analyser l'impact de la taille de position
            if 'quantity' in trades_data.columns and 'profit_pct' in trades_data.columns:
                corr = trades_data['quantity'].corr(trades_data['profit_pct'])
                
                if abs(corr) > 0.3:
                    direction = "positive" if corr > 0 else "négative"
                    recommendation = "plus grandes" if corr > 0 else "plus petites"
                    
                    insights.append(html.P([
                        html.Strong("Impact de la taille: "),
                        f"Il existe une corrélation {direction} ({corr:.2f}) entre la taille des positions et le profit en pourcentage. ",
                        f"Envisagez d'utiliser des positions {recommendation} pour les prochaines transactions."
                    ]))
        
        # Si aucun insight n'a été généré
        if not insights:
            insights.append(html.P("Pas assez de données pour générer des insights spécifiques."))
        
        # Ajouter des recommandations générales
        insights.append(html.Hr())
        insights.append(html.H6("Recommandations générales:"))
        insights.append(html.Ul([
            html.Li("Tenez un journal de trading détaillé et analysez-le régulièrement"),
            html.Li("Testez différentes heures d'entrée et de sortie pour optimiser vos résultats"),
            html.Li("Ajustez votre taille de position en fonction de la volatilité de l'actif"),
            html.Li("Développez un plan de trading clair et respectez-le à chaque transaction"),
        ]))
        
        return html.Div(insights)


import dash  # Nécessaire pour le contexte de callback
