"""
Exemple d'utilisation du gestionnaire de checkpoints.

Ce script démontre comment utiliser le gestionnaire de checkpoints pour sauvegarder
et charger des modèles, des sessions et d'autres données.
"""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim

from ai_trading.utils.advanced_logging import get_logger
from ai_trading.utils.checkpoint_manager import get_checkpoint_manager

# Configurer le logger
logger = get_logger("ai_trading.examples.checkpoint")


# Définir un modèle simple pour la démonstration
class SimpleModel(nn.Module):
    """Modèle PyTorch simple pour la démonstration."""

    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Fonction qui simule l'entraînement d'un modèle
def train_model(model, epochs=5):
    """
    Simule l'entraînement d'un modèle.

    Args:
        model: Modèle PyTorch
        epochs: Nombre d'époques

    Returns:
        Dictionnaire contenant les métriques d'entraînement
    """
    logger.info(f"Entraînement du modèle pour {epochs} époques")

    # Créer des données d'entraînement aléatoires
    input_size = model.fc1.in_features
    X = torch.randn(100, input_size)
    y = torch.randint(0, 2, (100,))

    # Configurer l'optimiseur et la fonction de perte
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Simuler l'entraînement
    losses = []
    accuracies = []

    for epoch in range(epochs):
        # Mettre le modèle en mode entraînement
        model.train()

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculer la précision
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)

        # Enregistrer les métriques
        losses.append(loss.item())
        accuracies.append(accuracy)

        logger.info(
            f"Époque {epoch+1}/{epochs}, Perte: {loss.item():.4f}, Précision: {accuracy:.4f}"
        )

        # Simuler une pause entre les époques
        time.sleep(0.5)

    # Retourner les métriques
    return {
        "loss": losses[-1],
        "accuracy": accuracies[-1],
        "epochs": epochs,
        "all_losses": losses,
        "all_accuracies": accuracies,
    }


# Fonction qui simule une session de trading
def create_trading_session():
    """
    Crée une session de trading simulée.

    Returns:
        Dictionnaire contenant les données de la session
    """
    logger.info("Création d'une session de trading simulée")

    # Simuler des données de marché
    num_days = 30
    prices = np.random.normal(loc=100, scale=5, size=num_days)
    prices = np.cumsum(np.random.normal(loc=0, scale=1, size=num_days)) + 100

    # Simuler un portefeuille
    portfolio = {
        "initial_balance": 10000.0,
        "current_balance": 10000.0 + random.uniform(-500, 1500),
        "positions": [
            {"symbol": "BTC", "amount": 0.5, "entry_price": 45000.0},
            {"symbol": "ETH", "amount": 5.0, "entry_price": 3000.0},
        ],
        "trades": [
            {
                "symbol": "BTC",
                "type": "BUY",
                "amount": 0.5,
                "price": 45000.0,
                "timestamp": time.time() - 86400,
            },
            {
                "symbol": "ETH",
                "type": "BUY",
                "amount": 5.0,
                "price": 3000.0,
                "timestamp": time.time() - 43200,
            },
        ],
    }

    # Simuler des paramètres de stratégie
    strategy_params = {
        "risk_factor": 0.2,
        "take_profit": 0.05,
        "stop_loss": 0.03,
        "timeframe": "1h",
        "indicators": ["RSI", "MACD", "Bollinger"],
    }

    # Créer la session complète
    session = {
        "id": f"session_{int(time.time())}",
        "start_time": time.time() - 86400,
        "last_update": time.time(),
        "market_data": {
            "prices": prices.tolist(),
            "volumes": (prices * np.random.uniform(0.8, 1.2, size=num_days)).tolist(),
            "timestamps": [
                time.time() - (num_days - i) * 86400 / num_days for i in range(num_days)
            ],
        },
        "portfolio": portfolio,
        "strategy_params": strategy_params,
        "performance": {
            "roi": random.uniform(-5, 15),
            "max_drawdown": random.uniform(0, 10),
            "sharpe_ratio": random.uniform(0, 2),
        },
    }

    return session


# Fonction principale pour la démonstration
def main():
    """Fonction principale pour démontrer l'utilisation du gestionnaire de checkpoints."""
    logger.info("Démarrage de la démonstration du gestionnaire de checkpoints")

    # 1. Obtenir le gestionnaire de checkpoints
    checkpoint_manager = get_checkpoint_manager()
    logger.info(
        f"Gestionnaire de checkpoints initialisé dans {checkpoint_manager.root_dir}"
    )

    # 2. Créer et entraîner un modèle
    logger.info("Création d'un modèle simple")
    model = SimpleModel()
    metrics = train_model(model, epochs=3)

    # 3. Sauvegarder le modèle
    logger.info("Sauvegarde du modèle")
    model_id = checkpoint_manager.save_model(
        model=model,
        name="simple_model",
        description="Modèle simple pour la démonstration",
        metrics=metrics,
    )

    if model_id:
        logger.info(f"Modèle sauvegardé avec l'ID: {model_id}")
    else:
        logger.error("Échec de la sauvegarde du modèle")
        return

    # 4. Créer et sauvegarder une session de trading
    logger.info("Création d'une session de trading")
    session = create_trading_session()

    session_id = checkpoint_manager.create_session_snapshot(
        session_data=session,
        name="trading_session",
        description="Session de trading simulée",
        metrics=session["performance"],
    )

    if session_id:
        logger.info(f"Session sauvegardée avec l'ID: {session_id}")
    else:
        logger.error("Échec de la sauvegarde de la session")

    # 5. Lister tous les checkpoints
    logger.info("Liste de tous les checkpoints")
    all_checkpoints = checkpoint_manager.list_checkpoints()

    logger.info(f"Nombre total de checkpoints: {len(all_checkpoints)}")
    for i, checkpoint in enumerate(all_checkpoints):
        logger.info(
            f"Checkpoint {i+1}: {checkpoint['checkpoint_id']} "
            f"(Type: {checkpoint['type']}, Date: {checkpoint['timestamp']})"
        )

    # 6. Charger le modèle sauvegardé
    logger.info("Chargement du modèle sauvegardé")
    new_model = SimpleModel()  # Créer une nouvelle instance

    if checkpoint_manager.load_model(model_id, new_model):
        logger.info("Modèle chargé avec succès")

        # Vérifier que les paramètres sont identiques
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            if not torch.equal(p1, p2):
                logger.error("Les paramètres du modèle ne correspondent pas!")
                break
        else:
            logger.info("Les paramètres du modèle correspondent parfaitement")
    else:
        logger.error("Échec du chargement du modèle")

    # 7. Charger la session sauvegardée
    logger.info("Chargement de la session sauvegardée")
    loaded_data = checkpoint_manager.load_checkpoint(session_id)

    if "portfolio" in loaded_data:
        logger.info("Session chargée avec succès")
        logger.info(f"Balance initiale: {loaded_data['portfolio']['initial_balance']}")
        logger.info(f"Balance actuelle: {loaded_data['portfolio']['current_balance']}")
        logger.info(
            f"Nombre de positions: {len(loaded_data['portfolio']['positions'])}"
        )
    else:
        logger.error("Échec du chargement de la session")

    # 8. Exporter un checkpoint
    export_path = Path(__file__).parent / "exported_checkpoint.zip"
    logger.info(f"Exportation du checkpoint {model_id} vers {export_path}")

    if checkpoint_manager.export_checkpoint(model_id, export_path):
        logger.info("Checkpoint exporté avec succès")
    else:
        logger.error("Échec de l'exportation du checkpoint")

    # 9. Importer un checkpoint
    if export_path.exists():
        logger.info(f"Importation du checkpoint depuis {export_path}")
        imported_id = checkpoint_manager.import_checkpoint(export_path)

        if imported_id:
            logger.info(f"Checkpoint importé avec l'ID: {imported_id}")
        else:
            logger.error("Échec de l'importation du checkpoint")

    # 10. Supprimer un checkpoint
    if len(all_checkpoints) > 0:
        checkpoint_to_delete = all_checkpoints[-1]["checkpoint_id"]
        logger.info(f"Suppression du checkpoint {checkpoint_to_delete}")

        if checkpoint_manager.delete_checkpoint(checkpoint_to_delete):
            logger.info("Checkpoint supprimé avec succès")
        else:
            logger.error("Échec de la suppression du checkpoint")

    logger.info("Démonstration du gestionnaire de checkpoints terminée")


if __name__ == "__main__":
    main()
