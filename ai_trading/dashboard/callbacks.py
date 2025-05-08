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
from dash import Input, Output, html

from ai_trading.dashboard.data_loader import DataLoader


def register_callbacks(app):
    """
    Enregistre les callbacks pour l'application Dash.

    Args:
        app: Instance de l'application Dash
    """
    # Initialiser le chargeur de données
    data_loader = DataLoader()

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
            perf_cell = (
                html.Td(f"{tx['perf_relative']*100:+.2f}%", style={"color": "green"})
                if tx["perf_relative"] is not None and tx["perf_relative"] > 0
                else (
                    html.Td(f"{tx['perf_relative']*100:+.2f}%", style={"color": "red"})
                    if tx["perf_relative"] is not None
                    else html.Td("N/A")
                )
            )

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

    # Autres callbacks à ajouter pour les autres onglets

    # ...


import dash  # Nécessaire pour le contexte de callback
