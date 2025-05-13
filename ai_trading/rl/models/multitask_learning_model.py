import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_trading.rl.models.multi_horizon_transformer import (
    MultiHorizonTemporalTransformer,
)

logger = logging.getLogger(__name__)


class SharedEncoder(nn.Module):
    """
    Encodeur partagé basé sur Transformer pour extraire des caractéristiques
    temporelles communes à toutes les tâches.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 500,
    ):
        super().__init__()

        self.transformer = MultiHorizonTemporalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=max_seq_len,
            forecast_horizons=[1],  # Un seul horizon pour l'encodeur partagé
            output_dim=d_model,
        )

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de l'encodeur.

        Args:
            x: Tenseur d'entrée [batch_size, seq_len, input_dim]

        Returns:
            Caractéristiques encodées [batch_size, d_model]
        """
        # Le transformer retourne un dict avec les horizons comme clés
        transformer_outputs = self.transformer(x)

        # Extraire les caractéristiques de l'horizon 1
        return transformer_outputs[1]


class PricePredictionHead(nn.Module):
    """
    Tête de modèle pour la prédiction de prix et volumes.
    Permet de prédire différents horizons futurs.
    """

    def __init__(
        self,
        d_model: int,
        horizons: List[int] = [1, 5, 10, 20],
        price_targets: int = 4,  # OHLC
        volume_targets: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizons = horizons
        self.price_targets = price_targets
        self.volume_targets = volume_targets

        # Couches partagées pour tous les horizons
        self.shared_layer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Têtes de prédiction spécifiques à chaque horizon
        self.price_heads = nn.ModuleDict(
            {f"h{h}": nn.Linear(hidden_dim, price_targets) for h in horizons}
        )

        self.volume_heads = nn.ModuleDict(
            {f"h{h}": nn.Linear(hidden_dim, volume_targets) for h in horizons}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass pour la prédiction de prix et volumes.

        Args:
            x: Caractéristiques extraites [batch_size, d_model]

        Returns:
            Dictionnaire des prédictions pour chaque horizon:
            {
                'h1': {'price': tensor, 'volume': tensor},
                'h5': {'price': tensor, 'volume': tensor},
                ...
            }
        """
        shared_features = self.shared_layer(x)

        predictions = {}
        for h in self.horizons:
            horizon_key = f"h{h}"

            price_pred = self.price_heads[horizon_key](shared_features)
            volume_pred = self.volume_heads[horizon_key](shared_features)

            predictions[horizon_key] = {"price": price_pred, "volume": volume_pred}

        return predictions


class TrendClassificationHead(nn.Module):
    """
    Tête de modèle pour la classification des tendances de marché.
    Peut classifier différentes échelles temporelles.
    """

    def __init__(
        self,
        d_model: int,
        horizons: List[int] = [1, 5, 10, 20],
        num_classes: int = 3,  # Haussier, Baissier, Neutre
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizons = horizons
        self.num_classes = num_classes

        # Couche partagée
        self.shared_layer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Têtes de classification par horizon
        self.classification_heads = nn.ModuleDict(
            {f"h{h}": nn.Linear(hidden_dim, num_classes) for h in horizons}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass pour la classification de tendances.

        Args:
            x: Caractéristiques extraites [batch_size, d_model]

        Returns:
            Dictionnaire des logits de classification par horizon:
            {
                'h1': tensor[batch_size, num_classes],
                'h5': tensor[batch_size, num_classes],
                ...
            }
        """
        shared_features = self.shared_layer(x)

        classifications = {}
        for h in self.horizons:
            horizon_key = f"h{h}"

            logits = self.classification_heads[horizon_key](shared_features)
            classifications[horizon_key] = logits

        return classifications


class PortfolioOptimizationHead(nn.Module):
    """
    Tête de modèle pour l'optimisation de portefeuille.
    Prend en compte plusieurs actifs et prédit l'allocation optimale.
    """

    def __init__(
        self,
        d_model: int,
        num_assets: int = 10,
        hidden_dim: int = 128,
        use_softmax: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.use_softmax = use_softmax

        self.allocation_network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour l'optimisation de portefeuille.

        Args:
            x: Caractéristiques extraites [batch_size, d_model]

        Returns:
            Allocations de portefeuille [batch_size, num_assets]
        """
        allocations = self.allocation_network(x)

        if self.use_softmax:
            # Utiliser softmax pour garantir que la somme des allocations = 1
            allocations = F.softmax(allocations, dim=-1)

        return allocations


class RiskManagementHead(nn.Module):
    """
    Tête de modèle pour la gestion des risques.
    Prédit des paramètres de gestion de risque comme le stop-loss,
    le take-profit, et la taille des positions.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.risk_network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Différents paramètres de risque
        self.stop_loss_head = nn.Linear(hidden_dim // 2, 1)
        self.take_profit_head = nn.Linear(hidden_dim // 2, 1)
        self.position_size_head = nn.Linear(hidden_dim // 2, 1)
        self.risk_score_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass pour la gestion des risques.

        Args:
            x: Caractéristiques extraites [batch_size, d_model]

        Returns:
            Dictionnaire des paramètres de risque:
            {
                'stop_loss': tensor[batch_size, 1],
                'take_profit': tensor[batch_size, 1],
                'position_size': tensor[batch_size, 1],
                'risk_score': tensor[batch_size, 1]
            }
        """
        risk_features = self.risk_network(x)

        # Calculer différents paramètres de risque
        stop_loss = (
            torch.sigmoid(self.stop_loss_head(risk_features)) * 0.2
        )  # 0-20% de stop loss
        take_profit = (
            torch.sigmoid(self.take_profit_head(risk_features)) * 0.5
        )  # 0-50% de take profit
        position_size = torch.sigmoid(
            self.position_size_head(risk_features)
        )  # 0-100% de la taille de position
        risk_score = torch.sigmoid(
            self.risk_score_head(risk_features)
        )  # 0-1 score de risque

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "risk_score": risk_score,
        }


class MultitaskLearningModel(nn.Module):
    """
    Modèle d'apprentissage multi-tâches qui combine:
    1. Prédiction de prix et volumes
    2. Classification de tendances
    3. Optimisation de portefeuille
    4. Gestion des risques

    Utilise un encodeur partagé basé sur Transformer et des têtes spécifiques à chaque tâche.
    """

    def __init__(
        self,
        input_dim: int,
        # Paramètres de l'encodeur
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 500,
        # Paramètres des têtes
        prediction_horizons: List[int] = [1, 5, 10, 20],
        num_trend_classes: int = 3,
        num_assets: int = 10,
        # Paramètres d'apprentissage
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        # Encodeur partagé
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=max_seq_len,
        )

        # Têtes spécifiques aux tâches
        self.price_prediction = PricePredictionHead(
            d_model=d_model,
            horizons=prediction_horizons,
            hidden_dim=d_model // 2,
            dropout=dropout,
        )

        self.trend_classification = TrendClassificationHead(
            d_model=d_model,
            horizons=prediction_horizons,
            num_classes=num_trend_classes,
            hidden_dim=d_model // 2,
            dropout=dropout,
        )

        self.portfolio_optimization = PortfolioOptimizationHead(
            d_model=d_model,
            num_assets=num_assets,
            hidden_dim=d_model // 2,
            dropout=dropout,
        )

        self.risk_management = RiskManagementHead(
            d_model=d_model,
            hidden_dim=d_model // 2,
            dropout=dropout,
        )

        # Poids des tâches pour la perte combinée
        default_weights = {
            "price_prediction": 1.0,
            "trend_classification": 1.0,
            "portfolio_optimization": 1.0,
            "risk_management": 1.0,
        }
        self.task_weights = (
            task_weights if task_weights is not None else default_weights
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Union[Dict, torch.Tensor]]:
        """
        Forward pass du modèle multi-tâches.

        Args:
            x: Séquence d'entrée [batch_size, seq_len, input_dim]

        Returns:
            Dictionnaire des sorties de chaque tâche
        """
        # Extraire les caractéristiques avec l'encodeur partagé
        encoded_features = self.encoder(x)

        # Appliquer chaque tête de tâche
        price_preds = self.price_prediction(encoded_features)
        trend_preds = self.trend_classification(encoded_features)
        portfolio_alloc = self.portfolio_optimization(encoded_features)
        risk_params = self.risk_management(encoded_features)

        return {
            "price_prediction": price_preds,
            "trend_classification": trend_preds,
            "portfolio_optimization": portfolio_alloc,
            "risk_management": risk_params,
        }

    def compute_price_volume_loss(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule la perte pour la prédiction de prix et volumes.

        Args:
            predictions: Prédictions du modèle
            targets: Valeurs cibles

        Returns:
            Dictionnaire des pertes par horizon
        """
        loss_per_horizon = {}

        for horizon_key, horizon_preds in predictions.items():
            if horizon_key not in targets:
                continue

            # Calcul des pertes MSE pour prix et volumes
            price_loss = F.mse_loss(
                horizon_preds["price"], targets[horizon_key]["price"]
            )

            volume_loss = F.mse_loss(
                horizon_preds["volume"], targets[horizon_key]["volume"]
            )

            # Combiner les pertes
            combined_loss = price_loss + 0.5 * volume_loss
            loss_per_horizon[horizon_key] = combined_loss

        # Calculer la perte moyenne sur tous les horizons
        if loss_per_horizon:
            avg_loss = torch.mean(torch.stack(list(loss_per_horizon.values())))
        else:
            avg_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        return {"per_horizon": loss_per_horizon, "avg": avg_loss}

    def compute_trend_classification_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule la perte pour la classification de tendances.

        Args:
            predictions: Prédictions du modèle
            targets: Valeurs cibles

        Returns:
            Dictionnaire des pertes par horizon
        """
        loss_per_horizon = {}

        for horizon_key, horizon_preds in predictions.items():
            if horizon_key not in targets:
                continue

            # Calcul de la perte d'entropie croisée pour la classification
            class_loss = F.cross_entropy(horizon_preds, targets[horizon_key])

            loss_per_horizon[horizon_key] = class_loss

        # Calculer la perte moyenne sur tous les horizons
        if loss_per_horizon:
            avg_loss = torch.mean(torch.stack(list(loss_per_horizon.values())))
        else:
            avg_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        return {"per_horizon": loss_per_horizon, "avg": avg_loss}

    def compute_portfolio_optimization_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcule la perte pour l'optimisation de portefeuille.

        Args:
            predictions: Allocations prédites [batch_size, num_assets]
            targets: Allocations cibles [batch_size, num_assets]
            returns: Rendements des actifs [batch_size, num_assets] (optionnel)

        Returns:
            Perte d'optimisation de portefeuille
        """
        # Vérifier et adapter les dimensions si nécessaire
        if predictions.size(1) != targets.size(1):
            logger.warning(
                f"Dimensions différentes: predictions {predictions.shape}, targets {targets.shape}"
            )

            # Cas 1: Si les cibles ont plus de dimensions que les prédictions
            if predictions.size(1) < targets.size(1):
                # On prend seulement les premières dimensions des cibles
                targets = targets[:, : predictions.size(1)]
            # Cas 2: Si les prédictions ont plus de dimensions que les cibles
            else:
                # On prend seulement les premières dimensions des prédictions
                predictions = predictions[:, : targets.size(1)]

        # MSE entre les allocations prédites et cibles
        allocation_loss = F.mse_loss(predictions, targets)

        # Si les rendements sont fournis, on peut calculer une perte basée sur les rendements
        if returns is not None:
            # Adapter les dimensions des rendements si nécessaire
            if returns.size(1) != predictions.size(1):
                returns = returns[:, : predictions.size(1)]

            # Rendement du portefeuille prédit
            pred_portfolio_return = torch.sum(predictions * returns, dim=1)

            # Rendement du portefeuille cible
            target_portfolio_return = torch.sum(targets * returns, dim=1)

            # Perte de rendement (on veut maximiser le rendement, donc minimiser le négatif)
            return_loss = torch.mean(target_portfolio_return - pred_portfolio_return)

            # Combiner les pertes
            return allocation_loss + return_loss

        return allocation_loss

    def compute_risk_management_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule la perte pour la gestion des risques.

        Args:
            predictions: Paramètres de risque prédits
            targets: Paramètres de risque cibles

        Returns:
            Dictionnaire des pertes par paramètre
        """
        loss_per_param = {}

        for param_key, param_pred in predictions.items():
            if param_key not in targets:
                continue

            # Calcul de la perte MSE pour chaque paramètre de risque
            param_loss = F.mse_loss(param_pred, targets[param_key])
            loss_per_param[param_key] = param_loss

        # Calculer la perte moyenne sur tous les paramètres
        if loss_per_param:
            avg_loss = torch.mean(torch.stack(list(loss_per_param.values())))
        else:
            avg_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        return {"per_param": loss_per_param, "avg": avg_loss}

    def compute_combined_loss(
        self,
        outputs: Dict[str, Union[Dict, torch.Tensor]],
        targets: Dict[str, Union[Dict, torch.Tensor]],
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcule la perte combinée pour toutes les tâches.

        Args:
            outputs: Sorties du modèle
            targets: Valeurs cibles
            returns: Rendements des actifs (optionnel)

        Returns:
            Tuple (perte totale, dictionnaire des pertes par tâche)
        """
        task_losses = {}

        # Calcul des pertes par tâche
        if "price_prediction" in targets:
            price_loss = self.compute_price_volume_loss(
                outputs["price_prediction"], targets["price_prediction"]
            )
            task_losses["price_prediction"] = price_loss["avg"]

        if "trend_classification" in targets:
            trend_loss = self.compute_trend_classification_loss(
                outputs["trend_classification"], targets["trend_classification"]
            )
            task_losses["trend_classification"] = trend_loss["avg"]

        if "portfolio_optimization" in targets:
            portfolio_loss = self.compute_portfolio_optimization_loss(
                outputs["portfolio_optimization"],
                targets["portfolio_optimization"],
                returns,
            )
            task_losses["portfolio_optimization"] = portfolio_loss

        if "risk_management" in targets:
            risk_loss = self.compute_risk_management_loss(
                outputs["risk_management"], targets["risk_management"]
            )
            task_losses["risk_management"] = risk_loss["avg"]

        # Perte totale pondérée
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for task_name, loss in task_losses.items():
            if task_name in self.task_weights:
                total_loss += self.task_weights[task_name] * loss

        return total_loss, task_losses
