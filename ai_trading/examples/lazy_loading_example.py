#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du module de chargement paresseux (lazy loading).

Ce script démontre:
1. L'utilisation du LazyFileReader pour charger des fichiers volumineux par morceaux
2. La création et l'utilisation d'un LazyDataset pour l'entraînement efficace
3. L'utilisation du cache de transformations pour éviter les calculs répétitifs
4. L'optimisation des inférences par lots (batch inference)
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ajouter le chemin racine au PYTHONPATH pour les imports relatifs
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Importer les modules de lazy loading
from ai_trading.data.lazy_loading import (
    BatchInferenceOptimizer,
    LazyDataset,
    LazyFileReader,
    get_cache_transform_fn,
    get_lazy_dataloader,
)

# Importer les utilitaires pour générer des données synthétiques
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data

# Répertoires de données et résultats
DATA_DIR = Path("ai_trading/info_retour/data/demo")
RESULTS_DIR = Path("ai_trading/info_retour/visualisations/misc")


def create_demo_data(n_points=100000, save_dir=None):
    """
    Crée des données synthétiques volumineuses pour la démonstration.

    Args:
        n_points: Nombre de points de données à générer.
        save_dir: Répertoire où sauvegarder les données.

    Returns:
        Chemin vers le fichier CSV généré.
    """
    logger.info(f"Génération de {n_points} points de données synthétiques...")

    # Générer des données synthétiques
    data = generate_synthetic_market_data(
        n_points=n_points,
        trend=0.0001,
        volatility=0.01,
        start_price=100.0,
        include_volume=True,
        cyclic_pattern=True,
        seasonal_periods=20,
        with_anomalies=True,
    )

    # Ajouter quelques indicateurs techniques simples
    logger.info("Calcul des indicateurs techniques...")

    # Moyennes mobiles
    data["ma_5"] = data["close"].rolling(5).mean().fillna(method="bfill")
    data["ma_20"] = data["close"].rolling(20).mean().fillna(method="bfill")

    # RSI simplifié
    delta = data["close"].diff().fillna(0)
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss

    avg_gain = gain.rolling(14).mean().fillna(0)
    avg_loss = loss.rolling(14).mean().fillna(0)

    rs = avg_gain / avg_loss.replace(0, 1e-8)  # Éviter division par zéro
    data["rsi"] = 100 - (100 / (1 + rs))

    logger.info(f"Dataset créé avec {len(data)} lignes et {len(data.columns)} colonnes")

    # Sauvegarder en CSV si un répertoire est spécifié
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        csv_path = save_dir / "demo_data.csv"
        data.to_csv(csv_path)
        logger.info(f"Données sauvegardées dans: {csv_path}")

        return str(csv_path)

    return None


@get_cache_transform_fn(cache_size=100)
def calculate_technical_indicators(tensor):
    """
    Calcule des indicateurs techniques additionnels.
    Cette fonction est décorée pour mettre en cache les résultats.

    Args:
        tensor: Tenseur d'entrée contenant des séquences temporelles.

    Returns:
        Tenseur avec les indicateurs additionnels calculés.
    """
    # Simuler un calcul coûteux
    time.sleep(0.01)

    # Extraire les colonnes (supposées être close, open, high, low, volume)
    close = tensor[:, 0].unsqueeze(1)

    # Calculer la volatilité locale (écart-type mobile)
    volatility = torch.zeros_like(close)
    window_size = 5
    for i in range(window_size, len(close)):
        window = close[i - window_size : i]
        volatility[i] = torch.std(window)

    # Calculer la tendance (pente de la droite de régression)
    trend = torch.zeros_like(close)
    window_size = 10
    for i in range(window_size, len(close)):
        window = close[i - window_size : i]
        x = torch.arange(window_size, dtype=torch.float32).unsqueeze(1)
        # Formule simplifiée de la pente
        x_mean = torch.mean(x)
        y_mean = torch.mean(window)
        numerator = torch.sum((x - x_mean) * (window - y_mean))
        denominator = torch.sum((x - x_mean) ** 2)
        if denominator != 0:
            trend[i] = numerator / denominator

    # Concaténer avec les données originales
    return torch.cat([tensor, volatility, trend], dim=1)


class SimpleTimeSeriesModel(nn.Module):
    """
    Modèle simple pour la prédiction de séries temporelles.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        """
        Initialise le modèle.

        Args:
            input_dim: Dimension d'entrée (nombre de features).
            hidden_dim: Dimension des couches cachées.
            num_layers: Nombre de couches LSTM.
            output_dim: Dimension de sortie (nombre de valeurs à prédire).
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Passe avant du modèle.

        Args:
            x: Tenseur d'entrée de forme [batch_size, seq_len, input_dim].

        Returns:
            Tenseur de sortie de forme [batch_size, output_dim].
        """
        # LSTM retourne (output, (h_n, c_n))
        lstm_out, _ = self.lstm(x)

        # Utiliser seulement la dernière sortie temporelle
        output = self.fc(lstm_out[:, -1, :])

        return output


def train_model_with_lazy_loading(
    data_path, sequence_length=50, batch_size=32, epochs=5
):
    """
    Entraîne un modèle en utilisant le chargement paresseux des données.

    Args:
        data_path: Chemin vers le fichier de données.
        sequence_length: Longueur des séquences.
        batch_size: Taille des batchs.
        epochs: Nombre d'époques d'entraînement.

    Returns:
        Modèle entraîné.
    """
    logger.info(f"Initialisation du dataset avec lazy loading: {data_path}")

    # Créer un dataset avec lazy loading
    lazy_dataset = LazyDataset(
        file_path=data_path,
        sequence_length=sequence_length,
        target_column="close",  # Prédire le prix de clôture
        transform=calculate_technical_indicators,  # Utiliser la transformation avec cache
        chunk_size=5000,  # Charger 5000 lignes à la fois
        cache_size=5,  # Garder 5 chunks en cache
    )

    # Créer un DataLoader optimisé
    dataloader = get_lazy_dataloader(
        file_path=data_path,
        sequence_length=sequence_length,
        target_column="close",
        transform=calculate_technical_indicators,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )

    # Obtenir la dimension d'entrée à partir du premier lot
    sample_batch = next(iter(dataloader))
    input_dim = sample_batch[0].shape[2]  # [batch_size, seq_len, input_dim]

    # Créer le modèle
    model = SimpleTimeSeriesModel(input_dim=input_dim)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraîner le modèle
    logger.info("Début de l'entraînement...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        start_time = time.time()
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumuler la perte
            epoch_loss += loss.item()
            batch_count += 1

            # Afficher la progression tous les 10 batchs
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}"
                )

        # Afficher la perte moyenne de l'époque
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / batch_count
        logger.info(
            f"Epoch {epoch+1}/{epochs} terminée en {epoch_time:.2f}s, Loss moyenne: {avg_loss:.6f}"
        )

    logger.info("Entraînement terminé!")
    return model


def test_batch_inference(model, data_path, sequence_length=50, batch_size=64):
    """
    Teste l'inférence par lots sur un modèle entraîné.

    Args:
        model: Modèle entraîné.
        data_path: Chemin vers le fichier de données.
        sequence_length: Longueur des séquences.
        batch_size: Taille des batchs pour l'inférence.

    Returns:
        DataFrame avec les prédictions.
    """
    logger.info("Test de l'inférence par lots...")

    # Créer un dataset pour le test (sans transformation pour comparer les performances)
    test_dataset = LazyDataset(
        file_path=data_path,
        sequence_length=sequence_length,
        target_column="close",
        transform=None,  # Pas de transformation pour comparer
        chunk_size=5000,
        cache_size=5,
    )

    # Créer un DataLoader standard
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Créer l'optimiseur d'inférence par lots
    batch_optimizer = BatchInferenceOptimizer(
        model=model,
        batch_size=batch_size,
        use_half_precision=False,  # Désactivé pour la simplicité
        optimize_for_inference=True,
        num_workers=2,
    )

    # Mesurer le temps d'inférence standard
    logger.info("Exécution de l'inférence standard...")
    start_time = time.time()

    standard_predictions = []
    with torch.no_grad():
        for sequences, _ in test_dataloader:
            outputs = model(sequences)
            standard_predictions.append(outputs.cpu())

    standard_time = time.time() - start_time
    standard_predictions = torch.cat(standard_predictions, dim=0).numpy()

    # Mesurer le temps d'inférence optimisée
    logger.info("Exécution de l'inférence optimisée par lots...")
    start_time = time.time()

    # Extraire toutes les séquences de test dans un seul tenseur
    # (dans un cas réel, on pourrait directement utiliser un générateur ou un fichier)
    all_sequences = torch.cat([seq for seq, _ in test_dataloader], dim=0)

    # Utiliser l'optimiseur d'inférence par lots
    optimized_predictions = batch_optimizer.predict(all_sequences, return_numpy=True)

    optimized_time = time.time() - start_time

    # Afficher les résultats
    logger.info(f"Temps d'inférence standard: {standard_time:.4f}s")
    logger.info(f"Temps d'inférence optimisée: {optimized_time:.4f}s")
    logger.info(f"Accélération: {standard_time / optimized_time:.2f}x")

    # Vérifier que les prédictions sont similaires
    prediction_diff = np.abs(standard_predictions - optimized_predictions).mean()
    logger.info(f"Différence moyenne entre les prédictions: {prediction_diff:.6f}")

    # Créer un DataFrame avec les prédictions
    result_df = pd.DataFrame(
        {
            "standard_predictions": standard_predictions.flatten(),
            "optimized_predictions": optimized_predictions.flatten(),
        }
    )

    return result_df


def test_lazy_file_reader(data_path):
    """
    Teste le LazyFileReader pour l'accès efficace aux fichiers volumineux.

    Args:
        data_path: Chemin vers le fichier de données.
    """
    logger.info(f"Test du LazyFileReader sur: {data_path}")

    # Créer un lecteur de fichier paresseux
    reader = LazyFileReader(
        file_path=data_path,
        chunk_size=1000,  # Charger 1000 lignes à la fois
        cache_size=3,  # Garder 3 chunks en cache
    )

    # Afficher les informations sur le fichier
    logger.info(f"Type de fichier détecté: {reader._file_type}")
    logger.info(f"Nombre de lignes dans le fichier: {reader.get_length()}")
    logger.info(f"Colonnes: {reader.get_column_names()}")

    # Test d'accès aléatoire
    logger.info("\nTest d'accès aléatoire:")

    # Lire des lignes individuelles à différentes positions
    positions = [10, 100, 1000, 5000]
    for pos in positions:
        if pos < reader.get_length():
            start_time = time.time()
            row = reader.get_row(pos)
            elapsed = time.time() - start_time
            logger.info(f"Ligne {pos}: chargée en {elapsed:.6f}s")
            logger.info(f"  Première colonne: {row.iloc[0]}")

    # Test de lecture de plages
    logger.info("\nTest de lecture de plages:")

    # Lire des plages de lignes
    ranges = [(0, 10), (500, 520), (2000, 2100)]
    for start, end in ranges:
        if end <= reader.get_length():
            start_time = time.time()
            rows = reader.get_rows(start, end)
            elapsed = time.time() - start_time
            logger.info(
                f"Lignes {start}-{end}: {len(rows)} lignes chargées en {elapsed:.6f}s"
            )

    # Test de l'efficacité du cache
    logger.info("\nTest de l'efficacité du cache:")

    # Lire le même chunk plusieurs fois
    chunk_idx = 2
    logger.info(f"Lecture répétée du chunk {chunk_idx}:")

    # Première lecture (cold)
    start_time = time.time()
    chunk = reader.get_chunk(chunk_idx)
    cold_time = time.time() - start_time
    logger.info(f"  Première lecture (cold): {cold_time:.6f}s")

    # Deuxième lecture (warmed cache)
    start_time = time.time()
    chunk = reader.get_chunk(chunk_idx)
    warm_time = time.time() - start_time
    logger.info(f"  Deuxième lecture (warm): {warm_time:.6f}s")
    logger.info(f"  Accélération du cache: {cold_time / warm_time:.2f}x")


def test_cached_transform():
    """
    Teste l'efficacité du cache pour les transformations.
    """
    logger.info("Test de la mise en cache des transformations...")

    # Créer des données d'exemple
    data = torch.randn(1000, 5)

    # Définir une transformation avec décorateur de cache
    @get_cache_transform_fn(cache_size=10)
    def complex_transform(x):
        # Simuler un calcul coûteux
        time.sleep(0.1)
        return x**2 + torch.sin(x)

    # Tester la transformation sans cache
    logger.info("Transformation sans cache:")

    # Fonction équivalente sans cache
    def uncached_transform(x):
        time.sleep(0.1)
        return x**2 + torch.sin(x)

    # Mesurer le temps pour 5 appels sans cache
    start_time = time.time()
    for _ in range(5):
        _ = uncached_transform(data)
    uncached_time = time.time() - start_time
    logger.info(f"  Temps total sans cache: {uncached_time:.4f}s")

    # Tester la transformation avec cache
    logger.info("Transformation avec cache:")

    # Premier appel (cold cache)
    start_time = time.time()
    _ = complex_transform(data)
    cold_cache_time = time.time() - start_time
    logger.info(f"  Premier appel (cold cache): {cold_cache_time:.4f}s")

    # Mesurer le temps pour 5 appels avec cache
    start_time = time.time()
    for _ in range(5):
        _ = complex_transform(data)
    cached_time = time.time() - start_time
    logger.info(f"  Temps total avec cache (5 appels): {cached_time:.4f}s")
    logger.info(f"  Accélération du cache: {uncached_time / cached_time:.2f}x")

    # Afficher les statistiques du cache
    cache_info = complex_transform.cache_info()
    logger.info(f"  Informations du cache: {cache_info}")


def main():
    """Point d'entrée principal du script."""
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(
        description="Exemple de lazy loading et optimisations associées"
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=100000,
        help="Nombre de points de données à générer",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Longueur des séquences pour le modèle",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille des batchs pour l'entraînement",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Nombre d'époques d'entraînement"
    )

    args = parser.parse_args()

    # Créer les répertoires
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Générer les données de démonstration
    data_path = create_demo_data(n_points=args.data_size, save_dir=DATA_DIR)

    if data_path:
        # Tester le LazyFileReader
        test_lazy_file_reader(data_path)

        # Tester le cache des transformations
        test_cached_transform()

        # Entraîner un modèle avec lazy loading
        model = train_model_with_lazy_loading(
            data_path,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

        # Tester l'inférence par lots
        result_df = test_batch_inference(
            model,
            data_path,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size
            * 2,  # Utiliser des batchs plus grands pour l'inférence
        )

        # Sauvegarder les résultats
        result_path = RESULTS_DIR / "batch_inference_results.csv"
        result_df.to_csv(result_path)
        logger.info(f"Résultats sauvegardés dans: {result_path}")
    else:
        logger.error("Échec de la génération des données de démonstration.")


if __name__ == "__main__":
    main()
