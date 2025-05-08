"""
Exemple d'utilisation du meta-learning et transfer learning pour l'adaptation inter-marchés.

Cet exemple démontre:
1. L'utilisation de MAML pour l'adaptation rapide entre différents marchés
2. Le transfer learning pour adapter un modèle d'un marché à un autre
3. L'adaptation de domaine pour réduire les écarts entre marchés
4. La comparaison des performances entre différentes approches
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_trading.rl.models.market_task_generator import MarketTaskGenerator

# Importer les modules de meta-learning et transfer learning
from ai_trading.rl.models.meta_learning import MAML
from ai_trading.rl.models.temporal_transformer import FinancialTemporalTransformer
from ai_trading.rl.models.transfer_learning import (
    DomainAdaptation,
    MarketTransferLearning,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_market_data(
    markets: List[str], timeframe: str = "daily"
) -> Dict[str, pd.DataFrame]:
    """
    Charge les données de plusieurs marchés.

    Args:
        markets: Liste des marchés à charger
        timeframe: Timeframe des données ('daily', 'hourly', etc.)

    Returns:
        Dictionnaire de DataFrames avec les données par marché
    """
    data_dir = Path(__file__).parent.parent / "data"
    market_data = {}

    for market in markets:
        try:
            # Charger les données de ce marché
            file_path = data_dir / f"{market.lower().replace('-', '_')}_{timeframe}.csv"

            if file_path.exists():
                df = pd.read_csv(file_path, parse_dates=["timestamp"])
                df.set_index("timestamp", inplace=True)
                market_data[market] = df
                logger.info(f"Données chargées pour {market}: {len(df)} points")
            else:
                logger.warning(f"Données non disponibles pour {market}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données pour {market}: {e}")

    # Si aucune donnée n'a été chargée, générer des données synthétiques
    if not market_data:
        logger.warning(
            "Aucune donnée réelle chargée, génération de données synthétiques"
        )
        market_data = generate_synthetic_data(markets, 500)

    return market_data


def generate_synthetic_data(
    markets: List[str], n_points: int = 500
) -> Dict[str, pd.DataFrame]:
    """
    Génère des données synthétiques pour les tests.

    Args:
        markets: Liste des marchés
        n_points: Nombre de points de données par marché

    Returns:
        Dictionnaire de DataFrames avec les données par marché
    """
    np.random.seed(42)

    market_data = {}

    for i, market in enumerate(markets):
        # Paramètres spécifiques au marché pour la simulation
        volatility = 0.01 + (i * 0.005)  # Volatilité différente par marché
        trend = 0.0002 * (i - len(markets) / 2)  # Tendance différente par marché
        start_price = 100 * (1 + 0.2 * i)  # Prix de départ différent

        # Générer une série temporelle
        prices = [start_price]
        for _ in range(n_points - 1):
            # Mouvement de prix avec tendance et volatilité spécifiques
            price_change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)

        # Créer un DataFrame avec OHLCV
        df = pd.DataFrame()
        df["close"] = prices
        df["open"] = df["close"].shift(1) * (1 + np.random.normal(0, 0.002, n_points))
        df["high"] = df[["open", "close"]].max(axis=1) * (
            1 + np.abs(np.random.normal(0, 0.003, n_points))
        )
        df["low"] = df[["open", "close"]].min(axis=1) * (
            1 - np.abs(np.random.normal(0, 0.003, n_points))
        )
        df["volume"] = np.random.gamma(shape=2.0, scale=1000, size=n_points) * (
            1 + 0.1 * i
        )

        # Remplacer les NaN
        df.fillna(method="bfill", inplace=True)

        # Ajouter quelques indicateurs techniques simples
        df["rsi"] = 50 + 15 * np.sin(np.linspace(0, 10 + i, n_points))
        df["volatility"] = np.abs(
            df["close"].pct_change().rolling(window=20).std().fillna(0)
        )
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        market_data[market] = df

    return market_data


def create_simple_model(input_features: int, sequence_length: int) -> nn.Module:
    """
    Crée un modèle simple pour la prédiction de séries temporelles.

    Args:
        input_features: Nombre de features d'entrée
        sequence_length: Longueur des séquences

    Returns:
        Modèle PyTorch
    """

    class SimpleModel(nn.Module):
        def __init__(self, input_dim, seq_len, hidden_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            # x shape: [batch_size, seq_len, input_dim]
            lstm_out, _ = self.lstm(x)
            # Prendre la dernière sortie
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out)

    return SimpleModel(input_features, sequence_length)


def create_transformer_model(input_features: int, sequence_length: int) -> nn.Module:
    """
    Crée un modèle Transformer pour la prédiction de séries temporelles.

    Args:
        input_features: Nombre de features d'entrée
        sequence_length: Longueur des séquences

    Returns:
        Modèle Transformer
    """
    # Utiliser la classe FinancialTemporalTransformer
    model = FinancialTemporalTransformer(
        input_dim=input_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_dim=1,
    )

    return model


def evaluate_model(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> Tuple[float, float]:
    """
    Évalue un modèle sur un ensemble de données.

    Args:
        model: Modèle à évaluer
        dataloader: DataLoader pour l'évaluation
        device: Appareil à utiliser ('cuda' ou 'cpu')

    Returns:
        Tuple (MSE, MAE)
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Calculer les erreurs
            mse = F.mse_loss(y_pred, y)
            mae = F.l1_loss(y_pred, y)

            total_mse += mse.item()
            total_mae += mae.item()

    # Moyennes
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)

    return avg_mse, avg_mae


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience: int = 5,
) -> Dict[str, List[float]]:
    """
    Entraîne un modèle de base sans meta-learning.

    Args:
        model: Modèle à entraîner
        train_loader: DataLoader pour l'entraînement
        val_loader: DataLoader pour la validation
        epochs: Nombre d'epochs d'entraînement
        learning_rate: Taux d'apprentissage
        device: Appareil à utiliser ('cuda' ou 'cpu')
        early_stopping_patience: Patience pour l'early stopping

    Returns:
        Historique d'entraînement
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        if val_loader:
            val_mse, val_mae = evaluate_model(model, val_loader, device)
            history["val_loss"].append(val_mse)
            history["val_mae"].append(val_mae)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f}, Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}"
            )

            # Early stopping
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping à l'epoch {epoch+1}")
                break
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f}")

    # Restaurer le meilleur modèle
    if best_model_state:
        model.load_state_dict(best_model_state)

    return history


def run_meta_learning_example():
    """Exécute l'exemple de meta-learning entre différents marchés."""
    logger.info("Démarrage de l'exemple de meta-learning inter-marchés")

    # Définir les marchés à utiliser
    markets = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD", "AVAX-USD"]

    # Charger les données
    market_data = load_market_data(markets)
    logger.info(f"Données chargées pour {len(market_data)} marchés")

    # Définir les colonnes de features et la cible
    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi",
        "volatility",
        "ema_20",
        "ema_50",
    ]
    target_column = "close"

    # Paramètres pour les modèles et l'entraînement
    sequence_length = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Utilisation de {device} pour l'entraînement")

    # Préparer le générateur de tâches
    task_generator = MarketTaskGenerator(
        market_data=market_data,
        target_column=target_column,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        prediction_horizon=1,
        normalization="task",
        random_seed=42,
    )

    # Dimension d'entrée pour les modèles
    input_dim = (
        len(feature_columns)
        if feature_columns
        else len(market_data[list(market_data.keys())[0]].columns) - 1
    )

    # Créer un modèle de base pour les tests
    base_model = create_transformer_model(input_dim, sequence_length)
    logger.info(f"Modèle créé: {type(base_model).__name__}")

    # 1. Entraînement de base sur un seul marché
    logger.info("\n1. Entraînement de base sur un seul marché")
    source_market = "BTC-USD"
    target_market = "ETH-USD"

    # Créer des dataloaders pour les marchés source et cible
    source_train_loader, source_val_loader = task_generator.create_market_dataloader(
        source_market, batch_size=32, test_size=0.2
    )

    target_train_loader, target_val_loader = task_generator.create_market_dataloader(
        target_market, batch_size=32, test_size=0.2
    )

    # Entraînement de base sur le marché source
    model_source = create_transformer_model(input_dim, sequence_length)
    history_source = train_model(
        model_source,
        source_train_loader,
        source_val_loader,
        epochs=50,
        learning_rate=0.001,
        device=device,
    )

    # Évaluer le modèle source sur le marché cible (sans adaptation)
    source_on_target_mse, source_on_target_mae = evaluate_model(
        model_source, target_val_loader, device
    )

    logger.info(
        f"Modèle source sur marché cible - MSE: {source_on_target_mse:.6f}, MAE: {source_on_target_mae:.6f}"
    )

    # 2. Transfer learning simple (fine-tuning)
    logger.info("\n2. Transfer learning simple (fine-tuning)")

    transfer_model = MarketTransferLearning(
        base_model=model_source,
        fine_tune_layers=None,  # Fine-tuner toutes les couches
        learning_rate=0.0005,
        feature_mapping=True,
        device=device,
    )

    # Fine-tuning sur le marché cible
    transfer_history = transfer_model.fine_tune(
        train_loader=target_train_loader,
        val_loader=target_val_loader,
        epochs=20,
        early_stopping_patience=5,
    )

    # Évaluer le modèle fine-tuné
    transfer_mse, transfer_mae = evaluate_model(
        transfer_model.base_model, target_val_loader, device
    )

    logger.info(f"Modèle fine-tuné - MSE: {transfer_mse:.6f}, MAE: {transfer_mae:.6f}")

    # 3. Meta-learning avec MAML
    logger.info("\n3. Meta-learning avec MAML")

    # Créer le modèle MAML
    maml_model = create_transformer_model(input_dim, sequence_length)
    maml = MAML(
        model=maml_model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=3,
        first_order=True,
        device=device,
    )

    # Définir la fonction de génération de tâches pour MAML
    def task_sampler(support_size, query_size):
        # Sélectionner un marché aléatoire (sauf le marché cible)
        available_markets = [m for m in market_data.keys() if m != target_market]
        market = np.random.choice(available_markets)
        return task_generator.generate_task(market, support_size, query_size)

    # Entraînement meta-learning
    maml_history = maml.meta_train(
        task_generator=task_sampler,
        num_tasks=100,
        num_epochs=30,
        support_size=20,
        query_size=20,
        batch_size=4,
    )

    # Adaptation rapide au marché cible
    # Extraire quelques exemples du marché cible pour l'adaptation
    support_samples = []
    for i, (x, y) in enumerate(target_train_loader):
        if i == 0:  # Prendre le premier batch
            support_x = x[:20].to(device)  # Limiter à 20 exemples
            support_y = y[:20].to(device)
            break

    # Adapter le modèle MAML au marché cible
    adapted_model = maml.adapt(support_x, support_y, steps=5)

    # Évaluer le modèle adapté
    adapted_maml_mse, adapted_maml_mae = evaluate_model(
        adapted_model, target_val_loader, device
    )

    logger.info(
        f"Modèle MAML adapté - MSE: {adapted_maml_mse:.6f}, MAE: {adapted_maml_mae:.6f}"
    )

    # 4. Domain Adaptation
    logger.info("\n4. Domain adaptation entre marchés")

    domain_adaptation = DomainAdaptation(
        source_model=model_source,
        adaptation_type="dann",
        lambda_param=0.2,
        device=device,
    )

    # Créer des itérateurs pour les données d'entraînement
    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)

    # Entraînement avec adaptation de domaine (version simplifiée)
    for epoch in range(10):
        epoch_metrics = {"task_loss": 0.0, "domain_loss": 0.0, "total_loss": 0.0}
        num_batches = 0

        # Réinitialiser les itérateurs si nécessaire
        if num_batches >= len(source_train_loader):
            source_iter = iter(source_train_loader)
        if num_batches >= len(target_train_loader):
            target_iter = iter(target_train_loader)

        # Limiter à quelques batchs pour l'exemple
        for _ in range(min(10, len(source_train_loader))):
            try:
                source_x, source_y = next(source_iter)
                target_x, _ = next(target_iter)

                # Adaptation de domaine
                batch_metrics = domain_adaptation.train_step(
                    source_data=source_x, source_labels=source_y, target_data=target_x
                )

                # Mettre à jour les métriques
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

                num_batches += 1

            except StopIteration:
                # Réinitialiser les itérateurs
                source_iter = iter(source_train_loader)
                target_iter = iter(target_train_loader)
                source_x, source_y = next(source_iter)
                target_x, _ = next(target_iter)

        # Calculer les moyennes
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        logger.info(
            f"Epoch {epoch+1}/10 - "
            f"Task Loss: {epoch_metrics['task_loss']:.6f}, "
            f"Domain Loss: {epoch_metrics['domain_loss']:.6f}, "
            f"Total Loss: {epoch_metrics['total_loss']:.6f}"
        )

    # Évaluer le modèle adapté
    domain_adapted_mse, domain_adapted_mae = evaluate_model(
        domain_adaptation.source_model, target_val_loader, device
    )

    logger.info(
        f"Modèle avec domain adaptation - MSE: {domain_adapted_mse:.6f}, MAE: {domain_adapted_mae:.6f}"
    )

    # 5. Comparaison des méthodes
    logger.info("\n5. Comparaison des différentes approches")

    # Entraînement direct sur le marché cible (baseline)
    model_target = create_transformer_model(input_dim, sequence_length)
    history_target = train_model(
        model_target,
        target_train_loader,
        target_val_loader,
        epochs=50,
        learning_rate=0.001,
        device=device,
    )

    target_mse, target_mae = evaluate_model(model_target, target_val_loader, device)

    logger.info(
        f"Modèle entraîné directement sur cible - MSE: {target_mse:.6f}, MAE: {target_mae:.6f}"
    )

    # Afficher le tableau comparatif
    results = {
        "Sans adaptation (source → cible)": (
            source_on_target_mse,
            source_on_target_mae,
        ),
        "Transfer Learning (fine-tuning)": (transfer_mse, transfer_mae),
        "Meta-Learning (MAML)": (adapted_maml_mse, adapted_maml_mae),
        "Domain Adaptation": (domain_adapted_mse, domain_adapted_mae),
        "Entraînement direct (cible)": (target_mse, target_mae),
    }

    logger.info("\nRésultats comparatifs:")
    logger.info(f"{'Méthode':<30} | {'MSE':<10} | {'MAE':<10}")
    logger.info("-" * 54)

    for method, (mse, mae) in results.items():
        logger.info(f"{method:<30} | {mse:<10.6f} | {mae:<10.6f}")

    # Visualisation des résultats
    visualize_comparison(results, f"{source_market} → {target_market}")

    return results


def visualize_comparison(results: Dict[str, Tuple[float, float]], title: str) -> None:
    """
    Visualise la comparaison des différentes méthodes.

    Args:
        results: Dictionnaire des résultats par méthode
        title: Titre du graphique
    """
    methods = list(results.keys())
    mse_values = [res[0] for res in results.values()]
    mae_values = [res[1] for res in results.values()]

    # Créer les figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique MSE
    bars1 = ax1.bar(methods, mse_values, color="skyblue")
    ax1.set_title(f"MSE par méthode - {title}")
    ax1.set_ylabel("MSE (Erreur quadratique moyenne)")
    ax1.tick_params(axis="x", rotation=45)

    # Ajouter les valeurs
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0001,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Graphique MAE
    bars2 = ax2.bar(methods, mae_values, color="lightgreen")
    ax2.set_title(f"MAE par méthode - {title}")
    ax2.set_ylabel("MAE (Erreur absolue moyenne)")
    ax2.tick_params(axis="x", rotation=45)

    # Ajouter les valeurs
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0001,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    # Sauvegarder la figure
    plt.savefig("meta_learning_comparison.png")
    plt.close()

    logger.info(
        "Graphique des résultats sauvegardé dans 'meta_learning_comparison.png'"
    )


if __name__ == "__main__":
    results = run_meta_learning_example()
