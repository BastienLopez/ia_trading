"""
Module de visualisations 3D pour le dashboard.

Ce module permet de créer des visualisations en 3D interactives pour analyser:
- Les relations entre multiples indicateurs techniques
- L'évolution du portefeuille dans l'espace des facteurs
- Les clusters de transactions similaires
"""

import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Visualizer3D:
    """Classe pour générer des visualisations 3D interactives."""

    def __init__(self, data_loader=None):
        """
        Initialise le visualiseur 3D.

        Args:
            data_loader: Instance du chargeur de données
        """
        self.data_loader = data_loader

    def create_multi_indicator_surface(
        self, df, x_col, y_col, z_col, colorscale="Viridis"
    ):
        """
        Crée une surface 3D montrant la relation entre trois indicateurs.

        Args:
            df: DataFrame contenant les données
            x_col: Colonne pour l'axe X
            y_col: Colonne pour l'axe Y
            z_col: Colonne pour l'axe Z (hauteur)
            colorscale: Échelle de couleur à utiliser

        Returns:
            Figure Plotly
        """
        # Gérer les valeurs manquantes
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=[x_col, y_col, z_col])

        # S'assurer qu'il y a suffisamment de données
        if len(df_clean) < 10:
            raise ValueError(
                f"Pas assez de données valides après élimination des valeurs manquantes (restant: {len(df_clean)})"
            )

        # Créer une grille 2D pour la surface
        x_unique = np.sort(df_clean[x_col].unique())
        y_unique = np.sort(df_clean[y_col].unique())

        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.zeros_like(X)

        # Remplir la matrice Z
        for i, x_val in enumerate(x_unique):
            for j, y_val in enumerate(y_unique):
                mask = (df_clean[x_col] == x_val) & (df_clean[y_col] == y_val)
                if mask.any():
                    Z[j, i] = df_clean.loc[mask, z_col].mean()

        # Remplacer les zéros par des NaN pour une meilleure visualisation
        Z[Z == 0] = np.nan

        # Interpoler les valeurs manquantes pour une surface plus lisse
        # Créer des coordonnées pour les valeurs non-NaN
        valid_mask = ~np.isnan(Z)
        points = np.vstack([X[valid_mask].ravel(), Y[valid_mask].ravel()]).T
        values = Z[valid_mask].ravel()

        # Utiliser une méthode d'interpolation si nous avons suffisamment de points
        if len(points) > 3:
            try:
                # Cubic si possible, sinon linéaire
                method = "cubic" if len(points) > 10 else "linear"
                grid_z = interpolate.griddata(
                    points, values, (X, Y), method=method, fill_value=np.nan
                )

                # Combler les trous restants avec une méthode plus simple
                if np.isnan(grid_z).any():
                    mask = np.isnan(grid_z)
                    grid_z_nearest = interpolate.griddata(
                        points, values, (X, Y), method="nearest"
                    )
                    grid_z[mask] = grid_z_nearest[mask]

                Z = grid_z
            except Exception:
                # En cas d'erreur avec l'interpolation, utiliser les données originales
                pass

        # Créer la figure
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale=colorscale)])

        fig.update_layout(
            title=f"Relation 3D entre {x_col}, {y_col} et {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=800,
            height=700,
            margin=dict(l=0, r=0, b=10, t=50),
        )

        return fig

    def create_portfolio_trajectory(self, df, window=30, n_components=3):
        """
        Crée une visualisation 3D de la trajectoire du portefeuille
        en utilisant PCA pour réduire les dimensions.

        Args:
            df: DataFrame avec les données du portefeuille et les indicateurs
            window: Taille de la fenêtre temporelle pour le calcul
            n_components: Nombre de composantes pour la PCA

        Returns:
            Figure Plotly
        """
        # Sélectionner les colonnes numériques pour la PCA
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

        # Exclure certaines colonnes non pertinentes
        exclude_cols = ["timestamp", "date", "time", "close", "open", "high", "low"]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # S'assurer qu'il y a suffisamment de caractéristiques
        if len(feature_cols) < n_components:
            raise ValueError(
                f"Pas assez de colonnes numériques ({len(feature_cols)}) pour {n_components} composantes"
            )

        # Gérer les valeurs manquantes
        data_for_pca = df[feature_cols].copy()
        data_for_pca = data_for_pca.fillna(data_for_pca.mean())

        # Standardiser les données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_pca)

        # Appliquer PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data_scaled)

        # Créer la figure
        fig = go.Figure()

        # Ajouter la ligne de trajectoire
        fig.add_trace(
            go.Scatter3d(
                x=components[:, 0],
                y=components[:, 1],
                z=components[:, 2],
                mode="lines",
                line=dict(color=df.index, colorscale="Viridis", width=4),
                name="Trajectoire",
            )
        )

        # Ajouter les points pour certaines périodes significatives
        markers_indices = np.linspace(0, len(df) - 1, 20, dtype=int)
        fig.add_trace(
            go.Scatter3d(
                x=components[markers_indices, 0],
                y=components[markers_indices, 1],
                z=components[markers_indices, 2],
                mode="markers",
                marker=dict(
                    size=5, color=markers_indices, colorscale="Viridis", opacity=0.8
                ),
                text=[
                    (
                        df.index[i].strftime("%Y-%m-%d")
                        if hasattr(df.index[i], "strftime")
                        else str(df.index[i])
                    )
                    for i in markers_indices
                ],
                name="Points clés",
            )
        )

        # Mettre à jour le layout
        fig.update_layout(
            title="Trajectoire 3D du portefeuille dans l'espace des caractéristiques",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.2%})",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=800,
            height=700,
            margin=dict(l=0, r=0, b=10, t=50),
        )

        return fig

    def create_trade_clusters_3d(self, trades_df, features=None, n_clusters=5):
        """
        Visualise les clusters de trades dans un espace 3D.

        Args:
            trades_df: DataFrame contenant les transactions
            features: Liste des caractéristiques à utiliser pour le clustering
            n_clusters: Nombre de clusters à créer

        Returns:
            Figure Plotly
        """
        from sklearn.cluster import KMeans

        # Définir les caractéristiques par défaut si non spécifiées
        if features is None:
            features = ["profit_pct", "holding_period", "entry_volatility"]

            # Assurer que toutes les colonnes existent, sinon utiliser des alternatives
            available_cols = trades_df.columns
            if "profit_pct" not in available_cols and "profit" in available_cols:
                features[0] = "profit"
            if "holding_period" not in available_cols and "duration" in available_cols:
                features[1] = "duration"
            if "entry_volatility" not in available_cols and "volume" in available_cols:
                features[2] = "volume"

        # Extraire les données pour le clustering
        X = trades_df[features].copy()

        # Gérer les valeurs manquantes
        X = X.dropna()

        # S'assurer qu'il y a suffisamment de données
        if len(X) < n_clusters:
            # Si pas assez de données, réduire le nombre de clusters
            n_clusters = max(2, len(X) // 2)
            if len(X) < 3:
                raise ValueError(
                    f"Pas assez de données valides après élimination des valeurs manquantes (restant: {len(X)})"
                )

        # Normaliser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Appliquer KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Créer la figure
        fig = go.Figure()

        # Ajouter un trace pour chaque cluster
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id

            fig.add_trace(
                go.Scatter3d(
                    x=X[mask][features[0]],
                    y=X[mask][features[1]],
                    z=X[mask][features[2]],
                    mode="markers",
                    marker=dict(size=5, opacity=0.7),
                    name=f"Cluster {cluster_id+1}",
                )
            )

        # Mettre à jour le layout
        fig.update_layout(
            title="Clusters 3D des transactions",
            scene=dict(
                xaxis_title=features[0],
                yaxis_title=features[1],
                zaxis_title=features[2],
            ),
            width=800,
            height=700,
            margin=dict(l=0, r=0, b=10, t=50),
        )

        return fig
