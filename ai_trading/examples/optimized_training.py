"""
Exemple d'utilisation du DataLoader optimisé et des optimisations de stockage
pour l'entraînement des modèles financiers.
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.trading_system import RLTradingSystem

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Définir les chemins
BASE_DIR = Path(__file__).parent.parent.parent
INFO_RETOUR_DIR = BASE_DIR / "ai_trading" / "info_retour"
EXAMPLES_OUTPUT_DIR = INFO_RETOUR_DIR / "examples" / "optimized_training"
DATA_DIR = EXAMPLES_OUTPUT_DIR / "data"
MODELS_DIR = EXAMPLES_OUTPUT_DIR / "models"
VISUALIZATION_DIR = EXAMPLES_OUTPUT_DIR / "visualizations"

# Créer les répertoires nécessaires
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
VISUALIZATION_DIR.mkdir(exist_ok=True, parents=True)


def main():
    """Fonction principale de l'exemple."""
    # Créer les répertoires si nécessaires
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Générer des données synthétiques pour l'exemple
    logger.info("Génération des données synthétiques...")
    data = generate_synthetic_market_data(
        n_points=10000,
        trend=0.0001,
        volatility=0.01,
        cyclic_pattern=True,
        depreciation_start=0.7,  # Introduire une dépréciation à 70% des données
    )

    # Préparer les données pour le modèle transformer
    target_col = "close"
    feature_cols = ["open", "high", "low", "close", "volume"]

    # Calculer des caractéristiques techniques supplémentaires
    data["returns"] = data["close"].pct_change().fillna(0)
    data["volatility"] = data["returns"].rolling(20).std().fillna(0)
    data["ma_20"] = data["close"].rolling(20).mean().fillna(method="bfill")
    data["ma_50"] = data["close"].rolling(50).mean().fillna(method="bfill")

    # Ajouter ces caractéristiques aux colonnes de features
    feature_cols.extend(["returns", "volatility", "ma_20", "ma_50"])

    # Paramètres d'entraînement
    sequence_length = 50
    batch_size = 64
    num_workers = 4  # Utilisez le nombre de cœurs de votre CPU

    # Préparation des cibles (par exemple, prédiction du prix de clôture à t+1)
    # Utilisation du DataLoader optimisé
    logger.info("Préparation du dataset et DataLoader optimisés...")
    dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        target_column=target_col,
        feature_columns=feature_cols,
        is_train=True,
        predict_n_ahead=1,  # Prédire un pas de temps dans le futur
        use_shared_memory=True,
    )

    # Initialisation du système de trading RL
    trading_system = RLTradingSystem(
        asset_symbol="BTC/USDT",
        initial_balance=10000.0,
        trading_fee=0.001,
        models_dir=str(MODELS_DIR),
    )

    # Créer le modèle transformer
    input_dim = len(feature_cols)
    trading_system.create_transformer(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=sequence_length,
        output_dim=1,  # Prédiction de la prochaine valeur
    )

    # Comparer l'efficacité des différentes configurations de DataLoader
    configurations = [
        {"name": "Single Worker", "workers": 0, "pin_memory": False, "prefetch": None},
        {
            "name": "Multi Worker",
            "workers": num_workers,
            "pin_memory": False,
            "prefetch": 2,
        },
        {
            "name": "Multi Worker + Pin Memory",
            "workers": num_workers,
            "pin_memory": True,
            "prefetch": 2,
        },
    ]

    # Résultats de performance
    results = []

    for config in configurations:
        logger.info(f"Test avec configuration: {config['name']}")

        # Créer le DataLoader avec la configuration actuelle
        dataloader = get_financial_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["workers"],
            prefetch_factor=config["prefetch"],
            pin_memory=config["pin_memory"],
        )

        # Mesurer le temps de chargement pour quelques époques
        start_time = time.time()

        # Simuler quelques époques
        num_epochs = 3
        total_batches = 0

        for epoch in range(num_epochs):
            batch_times = []

            for i, (features, targets) in enumerate(dataloader):
                batch_start = time.time()

                # Simuler un traitement de modèle (transfert GPU si disponible)
                if torch.cuda.is_available():
                    features = features.cuda()
                    targets = targets.cuda()

                # Simuler un calcul forward
                with torch.no_grad():
                    _ = trading_system._transformer(features)

                # Enregistrer le temps de traitement du batch
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)
                total_batches += 1

                # Limiter le nombre de batch pour la démonstration
                if i >= 20:
                    break

        end_time = time.time()
        total_time = end_time - start_time
        avg_batch_time = np.mean(batch_times)

        # Enregistrer les résultats
        results.append(
            {
                "config": config["name"],
                "total_time": total_time,
                "avg_batch_time": avg_batch_time,
                "total_batches": total_batches,
            }
        )

        logger.info(
            f"Configuration {config['name']}: "
            f"Temps total: {total_time:.2f}s, "
            f"Temps moyen par batch: {avg_batch_time*1000:.2f}ms"
        )

    # Afficher les résultats
    logger.info("\nRésultats de performance:")
    for result in results:
        logger.info(
            f"{result['config']}: "
            f"Temps total: {result['total_time']:.2f}s, "
            f"Temps moyen par batch: {result['avg_batch_time']*1000:.2f}ms"
        )

    # Graphique comparatif
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = [r["config"] for r in results]
    batch_times = [r["avg_batch_time"] * 1000 for r in results]

    ax.bar(configs, batch_times, color="skyblue")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Temps moyen par batch (ms)")
    ax.set_title("Comparaison des performances de DataLoader")
    ax.grid(axis="y", alpha=0.3)

    # Enregistrer le graphique
    plt.tight_layout()
    performance_graph_path = VISUALIZATION_DIR / "dataloader_performance.png"
    plt.savefig(performance_graph_path)
    logger.info(f"Graphique de performance enregistré dans '{performance_graph_path}'")

    # Exemple d'entrainement complet avec DataLoader optimisé
    logger.info("\nDémarrage de l'entraînement du modèle Transformer...")

    # Utilisation du DataLoader optimisé pour l'entraînement
    optimized_dataloader = get_financial_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
    )

    # Entraînement avec le nouveau DataLoader optimisé
    start_time = time.time()

    # Nombre d'époques limité pour l'exemple
    epochs = 5
    batch_size = 64
    learning_rate = 1e-4

    # Entraîner directement avec des batches du DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(
        trading_system._transformer.parameters(), lr=learning_rate
    )
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        trading_system._transformer.train()
        epoch_loss = 0.0
        num_batches = 0

        for features, targets in optimized_dataloader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(
                1
            )  # Ajouter une dimension pour le modèle

            # Forward pass
            optimizer.zero_grad()
            outputs, _ = trading_system._transformer(features)

            # Calculer la perte
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Limiter le nombre de batches pour la démonstration
            if num_batches >= 20:
                break

        avg_loss = epoch_loss / num_batches
        logger.info(f"Époque {epoch+1}/{epochs}, Perte moyenne: {avg_loss:.6f}")

    end_time = time.time()
    logger.info(f"Entraînement terminé en {end_time - start_time:.2f} secondes")

    # Sauvegarder le modèle
    model_path = MODELS_DIR / "transformer_optimized.pt"
    torch.save(trading_system._transformer.state_dict(), model_path)
    logger.info(f"Modèle enregistré dans {model_path}")


if __name__ == "__main__":
    main()
