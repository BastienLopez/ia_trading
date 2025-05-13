"""
Module d'analyse post-mortem des trades.

Ce module fournit des outils pour analyser en profondeur les transactions passées:
- Analyse détaillée de la performance des trades
- Décomposition des trades par caractéristiques (durée, actifs, moment, etc.)
- Identification des schémas de succès/échec
- Attribution de performance
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


class TradeAnalyzer:
    """Classe pour l'analyse post-mortem des trades."""

    def __init__(self, data_loader=None):
        """
        Initialise l'analyseur de trades.

        Args:
            data_loader: Instance du chargeur de données
        """
        self.data_loader = data_loader

    def prepare_trade_data(self, trades_df):
        """
        Prépare et enrichit les données de trades pour l'analyse.

        Args:
            trades_df: DataFrame contenant les transactions

        Returns:
            DataFrame enrichi
        """
        # Copier pour éviter de modifier l'original
        df = trades_df.copy()

        # Assurer que les colonnes nécessaires existent
        required_cols = [
            "entry_time",
            "exit_time",
            "symbol",
            "direction",
            "entry_price",
            "exit_price",
            "quantity",
            "profit",
            "status",
        ]

        # Créer des colonnes dérivées
        if "entry_time" in df.columns and "exit_time" in df.columns:
            # Calculer la durée de détention
            df["holding_period"] = (
                pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])
            ).dt.total_seconds() / 3600  # en heures

            # Extraire les informations temporelles
            df["entry_hour"] = pd.to_datetime(df["entry_time"]).dt.hour
            df["entry_day"] = pd.to_datetime(df["entry_time"]).dt.day_name()
            df["entry_month"] = pd.to_datetime(df["entry_time"]).dt.month_name()

        # Calculer le pourcentage de profit si nécessaire
        if "profit" in df.columns and "profit_pct" not in df.columns:
            if "entry_value" in df.columns:
                df["profit_pct"] = df["profit"] / df["entry_value"] * 100
            elif "entry_price" in df.columns and "quantity" in df.columns:
                df["entry_value"] = df["entry_price"] * df["quantity"]
                df["profit_pct"] = df["profit"] / df["entry_value"] * 100

        # Classer les trades
        if "profit" in df.columns:
            df["trade_result"] = np.where(df["profit"] > 0, "Gain", "Perte")

            # Catégoriser les trades par magnitude de gain/perte
            df["result_category"] = pd.cut(
                df["profit_pct"] if "profit_pct" in df.columns else df["profit"],
                bins=[-float("inf"), -5, -1, 0, 1, 5, float("inf")],
                labels=[
                    "Perte majeure",
                    "Perte modérée",
                    "Perte mineure",
                    "Gain mineur",
                    "Gain modéré",
                    "Gain majeur",
                ],
            )

        return df

    def create_performance_summary(self, trades_df):
        """
        Crée un résumé de performance des trades.

        Args:
            trades_df: DataFrame contenant les transactions

        Returns:
            Figure Plotly avec plusieurs sous-graphiques
        """
        # Préparer les données
        df = self.prepare_trade_data(trades_df)

        # Créer une figure avec des sous-graphiques
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Distribution des profits",
                "Performance par actif",
                "Performance par jour de la semaine",
                "Performance par durée de détention",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # 1. Distribution des profits
        fig.add_trace(
            go.Histogram(
                x=df["profit_pct"] if "profit_pct" in df.columns else df["profit"],
                name="Distribution des profits",
                marker_color="skyblue",
                opacity=0.7,
                nbinsx=30,
            ),
            row=1,
            col=1,
        )

        # 2. Performance par actif
        if "symbol" in df.columns:
            asset_perf = df.groupby("symbol", observed=False)["profit"].agg(
                ["sum", "mean", "count"]
            )
            asset_perf = asset_perf.sort_values("sum", ascending=False)

            fig.add_trace(
                go.Bar(
                    x=asset_perf.index[:10],  # Top 10
                    y=asset_perf["sum"][:10],
                    name="Profit total par actif",
                    marker_color="lightgreen",
                ),
                row=1,
                col=2,
            )

        # 3. Performance par jour de la semaine
        if "entry_day" in df.columns:
            day_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            day_order_fr = [
                "Lundi",
                "Mardi",
                "Mercredi",
                "Jeudi",
                "Vendredi",
                "Samedi",
                "Dimanche",
            ]

            day_map = dict(zip(day_order, day_order_fr))
            if "entry_day" in df.columns:
                df["entry_day_fr"] = df["entry_day"].map(day_map)

                day_perf = (
                    df.groupby("entry_day_fr", observed=False)["profit"]
                    .sum()
                    .reindex(day_order_fr)
                )

                fig.add_trace(
                    go.Bar(
                        x=day_perf.index,
                        y=day_perf.values,
                        name="Profit par jour",
                        marker_color="coral",
                    ),
                    row=2,
                    col=1,
                )

        # 4. Performance par durée de détention
        if "holding_period" in df.columns:
            # Créer des bins pour la durée de détention
            df["holding_period_bin"] = pd.cut(
                df["holding_period"],
                bins=[0, 1, 6, 24, 72, 168, float("inf")],
                labels=["<1h", "1-6h", "6-24h", "1-3j", "3-7j", ">7j"],
            )

            duration_perf = df.groupby("holding_period_bin")["profit"].mean()

            fig.add_trace(
                go.Scatter(
                    x=duration_perf.index,
                    y=duration_perf.values,
                    mode="lines+markers",
                    name="Profit moyen par durée",
                    marker_color="mediumpurple",
                    line=dict(width=3),
                ),
                row=2,
                col=2,
            )

        # Mise à jour du layout
        fig.update_layout(
            title="Analyse de performance des transactions",
            height=800,
            width=1000,
            showlegend=False,
        )

        return fig

    def create_win_loss_analysis(self, trades_df):
        """
        Crée une analyse détaillée des gains et pertes.

        Args:
            trades_df: DataFrame contenant les transactions

        Returns:
            Figure Plotly
        """
        # Préparer les données
        df = self.prepare_trade_data(trades_df)

        # Calculer les métriques
        total_trades = len(df)
        winning_trades = df[df["profit"] > 0]
        losing_trades = df[df["profit"] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = winning_trades["profit"].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades["profit"].mean() if len(losing_trades) > 0 else 0

        max_win = winning_trades["profit"].max() if len(winning_trades) > 0 else 0
        max_loss = losing_trades["profit"].min() if len(losing_trades) > 0 else 0

        profit_factor = (
            abs(winning_trades["profit"].sum() / losing_trades["profit"].sum())
            if losing_trades["profit"].sum() != 0
            else float("inf")
        )

        # Créer la figure avec des sous-graphiques
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Répartition Gains/Pertes",
                "Profit moyen par résultat",
                "Série de transactions",
                "Courbe d'équité",
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}],
            ],
        )

        # 1. Répartition Gains/Pertes
        fig.add_trace(
            go.Pie(
                labels=["Gains", "Pertes"],
                values=[len(winning_trades), len(losing_trades)],
                marker=dict(colors=["green", "red"]),
                textinfo="percent+label",
                hole=0.3,
            ),
            row=1,
            col=1,
        )

        # 2. Profit moyen par résultat
        fig.add_trace(
            go.Bar(
                x=["Gain moyen", "Perte moyenne", "Gain max", "Perte max"],
                y=[avg_win, avg_loss, max_win, max_loss],
                marker_color=["green", "red", "darkgreen", "darkred"],
            ),
            row=1,
            col=2,
        )

        # 3. Série de transactions
        # Créer une matrice de résultats (1 pour gain, 0 pour perte) sur une grille 2D
        series_length = min(50, len(df))  # Limiter à 50 derniers trades
        results = (df["profit"] > 0).iloc[-series_length:].astype(int).values

        # Réorganiser en grille 5x10
        rows_count = 5
        cols_count = 10
        matrix = np.zeros((rows_count, cols_count))

        for i in range(min(rows_count * cols_count, len(results))):
            row = i // cols_count
            col = i % cols_count
            matrix[row, col] = results[i]

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=[[0, "red"], [1, "green"]],
                showscale=False,
                text=[
                    [f"Trade {i*cols_count+j+1}" for j in range(cols_count)]
                    for i in range(rows_count)
                ],
                hoverinfo="text+z",
            ),
            row=2,
            col=1,
        )

        # 4. Courbe d'équité
        df_sorted = df.sort_values(
            "entry_time" if "entry_time" in df.columns else df.index
        )
        equity_curve = df_sorted["profit"].cumsum()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(equity_curve))),
                y=equity_curve.values,
                mode="lines",
                line=dict(color="blue", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 0, 255, 0.1)",
            ),
            row=2,
            col=2,
        )

        # Ajouter des annotations avec les métriques clés
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            text=f"Win Rate: {win_rate:.2%} | Profit Factor: {profit_factor:.2f} | Total Trades: {total_trades}",
            showarrow=False,
            font=dict(size=14),
        )

        # Mise à jour du layout
        fig.update_layout(title="Analyse des gains et pertes", height=800, width=1000)

        return fig

    def create_trade_attribution(self, trades_df, market_data=None):
        """
        Crée une analyse d'attribution des performances des trades.

        Args:
            trades_df: DataFrame contenant les transactions
            market_data: DataFrame avec les données de marché (optionnel)

        Returns:
            Figure Plotly
        """
        # Préparer les données
        df = self.prepare_trade_data(trades_df)

        # Créer la figure avec des sous-graphiques
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Attribution par type de trade",
                "Performance vs. Volatilité du marché",
                "Distribution des durées",
                "Corrélation taille vs. performance",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}],
            ],
        )

        # 1. Attribution par type de trade
        if "direction" in df.columns and "result_category" in df.columns:
            # Compter les trades par direction et résultat
            direction_result = (
                df.groupby(["direction", "result_category"])
                .size()
                .unstack(fill_value=0)
            )

            # Pour chaque direction
            for direction in direction_result.index:
                fig.add_trace(
                    go.Bar(
                        x=direction_result.columns,
                        y=direction_result.loc[direction],
                        name=direction,
                    ),
                    row=1,
                    col=1,
                )

        # 2. Performance vs. Volatilité du marché
        if "entry_volatility" in df.columns or (
            market_data is not None and "volatility" in market_data.columns
        ):
            # Utiliser les données directement du DataFrame de trades ou joindre avec market_data
            if "entry_volatility" in df.columns:
                volatility_data = df[["entry_volatility", "profit_pct"]].copy()
                volatility_col = "entry_volatility"
            elif "entry_time" in df.columns and market_data is not None:
                # Faire correspondre chaque trade avec la volatilité du marché au moment de l'entrée
                volatility_data = pd.merge_asof(
                    df.sort_values("entry_time"),
                    market_data[["timestamp", "volatility"]].sort_values("timestamp"),
                    left_on="entry_time",
                    right_on="timestamp",
                    direction="nearest",
                )
                volatility_col = "volatility"

            # Tracer un scatter plot
            if "volatility_col" in locals():
                fig.add_trace(
                    go.Scatter(
                        x=volatility_data[volatility_col],
                        y=volatility_data["profit_pct"],
                        mode="markers",
                        marker=dict(
                            color=volatility_data["profit_pct"],
                            colorscale="RdYlGn",
                            size=8,
                            showscale=True,
                            colorbar=dict(title="Profit %", x=0.46),
                        ),
                        name="Trades",
                    ),
                    row=1,
                    col=2,
                )

                # Ajouter une ligne de régression
                if len(volatility_data) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        volatility_data[volatility_col], volatility_data["profit_pct"]
                    )

                    x_range = np.linspace(
                        volatility_data[volatility_col].min(),
                        volatility_data[volatility_col].max(),
                        100,
                    )
                    y_range = slope * x_range + intercept

                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                            name=f"R²={r_value**2:.2f}",
                        ),
                        row=1,
                        col=2,
                    )

        # 3. Distribution des durées
        if "holding_period" in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df["holding_period"],
                    nbinsx=20,
                    marker_color="lightblue",
                    opacity=0.7,
                    name="Durée des trades",
                ),
                row=2,
                col=1,
            )

        # 4. Corrélation taille vs. performance
        if "quantity" in df.columns and "profit_pct" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["quantity"],
                    y=df["profit_pct"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=df["profit_pct"],
                        colorscale="RdYlGn",
                        opacity=0.7,
                    ),
                    name="Taille vs profit",
                ),
                row=2,
                col=2,
            )

            # Ajouter une ligne de tendance
            if len(df) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df["quantity"], df["profit_pct"]
                )

                x_range = np.linspace(df["quantity"].min(), df["quantity"].max(), 100)
                y_range = slope * x_range + intercept

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name=f"Tendance (R²={r_value**2:.2f})",
                    ),
                    row=2,
                    col=2,
                )

        # Mise à jour du layout
        fig.update_layout(
            title="Analyse d'attribution des performances",
            height=800,
            width=1000,
            showlegend=True,
        )

        return fig
