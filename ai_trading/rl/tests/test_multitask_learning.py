import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ajuster les chemins d'importation
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import des modules
from ai_trading.rl.models.multitask_learning_model import MultitaskLearningModel
from ai_trading.rl.trainer.multitask_trainer import FinancialMultitaskDataset, MultitaskTrainer

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "ai_trading" / "info_retour" / "visualisations" / "multitask"
DATA_DIR = BASE_DIR / "ai_trading" / "info_retour" / "data" / "processed"

# Assurer que le répertoire d'output existe
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=1000, num_assets=5):
    """
    Génère des données financières synthétiques pour les tests.
    
    Args:
        n_samples: Nombre d'échantillons à générer
        num_assets: Nombre d'actifs à simuler
        
    Returns:
        DataFrame avec les données synthétiques
    """
    # Créer un index de dates
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1H")
    
    # Dictionnaire pour stocker les données
    data = {}
    
    # Simuler les prix et volumes pour chaque actif
    for i in range(num_assets):
        # Simuler un mouvement de prix avec tendance, bruit et cycles
        trend = np.linspace(0, 30, n_samples) * (0.8 + 0.4 * np.random.rand())
        noise = np.random.normal(0, 5, n_samples)
        sine = 10 * np.sin(np.linspace(0, 5, n_samples) + i)
        
        # Pour simuler des changements de régime
        regime_changes = np.zeros(n_samples)
        change_points = np.random.choice(range(n_samples), size=5, replace=False)
        for point in change_points:
            regime_changes[point:] += np.random.uniform(-10, 10)
        
        # Combinaison des composantes
        base_price = 100 * (1 + 0.2 * i)
        prices = base_price + trend + noise + sine + regime_changes
        
        # Générer les données OHLCV
        asset_prefix = f"asset_{i+1}_"
        data[asset_prefix + "open"] = prices * 0.99
        data[asset_prefix + "high"] = prices * 1.02
        data[asset_prefix + "low"] = prices * 0.98
        data[asset_prefix + "close"] = prices
        data[asset_prefix + "volume"] = np.random.randint(1000, 10000, n_samples)
    
    # Colonnes supplémentaires pour l'actif principal
    data["open"] = data["asset_1_open"]
    data["high"] = data["asset_1_high"]
    data["low"] = data["asset_1_low"]
    data["close"] = data["asset_1_close"]
    data["volume"] = data["asset_1_volume"]
    
    # Création du DataFrame
    df = pd.DataFrame(data, index=dates)
    
    return df


def prepare_dataloaders(df, batch_size=32, val_split=0.2, window_size=50, num_assets=5):
    """
    Prépare les DataLoaders pour l'entraînement et la validation.
    
    Args:
        df: DataFrame avec les données
        batch_size: Taille des batchs
        val_split: Proportion des données pour la validation
        window_size: Taille de la fenêtre d'observation
        num_assets: Nombre d'actifs
        
    Returns:
        Tuple (train_loader, val_loader)
    """
    # Identifier les colonnes pour chaque actif
    asset_columns = []
    for i in range(1, num_assets + 1):
        asset_prefix = f"asset_{i}_"
        asset_columns.extend([
            asset_prefix + "open",
            asset_prefix + "high",
            asset_prefix + "low",
            asset_prefix + "close",
            asset_prefix + "volume"
        ])
    
    # Déterminer le point de séparation pour l'ensemble de validation
    split_idx = int(len(df) * (1 - val_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Créer les datasets
    train_dataset = FinancialMultitaskDataset(
        data=train_df,
        window_size=window_size,
        prediction_horizons=[1, 5, 10, 20],
        asset_columns=asset_columns,
        num_trend_classes=3,
        trend_threshold=0.005,
    )
    
    val_dataset = FinancialMultitaskDataset(
        data=val_df,
        window_size=window_size,
        prediction_horizons=[1, 5, 10, 20],
        asset_columns=asset_columns,
        num_trend_classes=3,
        trend_threshold=0.005,
    )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    return train_loader, val_loader


def visualize_training_history(history, title="Historique d'entraînement"):
    """
    Visualise l'historique d'entraînement.
    
    Args:
        history: Dictionnaire contenant les historiques de pertes
        title: Titre du graphique
    """
    plt.figure(figsize=(15, 12))
    
    # Perte totale
    plt.subplot(3, 2, 1)
    plt.plot(history['train_losses'], label="Entraînement")
    plt.plot(history['val_losses'], label="Validation")
    plt.title("Perte totale")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid(True)
    
    # Pertes par tâche
    tasks = ['price_prediction', 'trend_classification', 'portfolio_optimization', 'risk_management']
    task_titles = {
        'price_prediction': "Prédiction de prix et volumes",
        'trend_classification': "Classification de tendances",
        'portfolio_optimization': "Optimisation de portefeuille",
        'risk_management': "Gestion des risques",
    }
    
    for i, task in enumerate(tasks):
        plt.subplot(3, 2, i+2)
        plt.plot(history['task_train_losses'][task], label="Entraînement")
        plt.plot(history['task_val_losses'][task], label="Validation")
        plt.title(task_titles[task])
        plt.xlabel("Époque")
        plt.ylabel("Perte")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multitask_learning_history.png")
    plt.close()


def test_model_predictions(model, test_loader, device):
    """
    Teste les prédictions du modèle sur un ensemble de données.
    
    Args:
        model: Modèle entraîné
        test_loader: DataLoader pour les données de test
        device: Périphérique pour l'inférence
        
    Returns:
        Dictionnaire des prédictions et métriques
    """
    model.eval()
    all_preds = {
        'price_prediction': {},
        'trend_classification': {},
        'portfolio_optimization': [],
        'risk_management': {},
    }
    
    all_targets = {
        'price_prediction': {},
        'trend_classification': {},
        'portfolio_optimization': [],
        'risk_management': {},
    }
    
    metrics = {
        'price_prediction_mse': {},
        'trend_classification_accuracy': {},
        'portfolio_optimization_mse': 0.0,
        'risk_management_mse': {},
    }
    
    batch_count = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Collecter les prédictions et les cibles
            # Prix et volumes
            for h_key, h_preds in outputs['price_prediction'].items():
                if h_key not in all_preds['price_prediction']:
                    all_preds['price_prediction'][h_key] = {'price': [], 'volume': []}
                    all_targets['price_prediction'][h_key] = {'price': [], 'volume': []}
                    metrics['price_prediction_mse'][h_key] = {'price': 0.0, 'volume': 0.0}
                
                if h_key in targets['price_prediction']:
                    # Prix
                    pred_price = h_preds['price'].cpu().numpy()
                    target_price = targets['price_prediction'][h_key]['price'].numpy()
                    all_preds['price_prediction'][h_key]['price'].append(pred_price)
                    all_targets['price_prediction'][h_key]['price'].append(target_price)
                    
                    # MSE pour les prix
                    mse_price = ((pred_price - target_price) ** 2).mean()
                    metrics['price_prediction_mse'][h_key]['price'] += mse_price
                    
                    # Volume
                    pred_volume = h_preds['volume'].cpu().numpy()
                    target_volume = targets['price_prediction'][h_key]['volume'].numpy()
                    all_preds['price_prediction'][h_key]['volume'].append(pred_volume)
                    all_targets['price_prediction'][h_key]['volume'].append(target_volume)
                    
                    # MSE pour les volumes
                    mse_volume = ((pred_volume - target_volume) ** 2).mean()
                    metrics['price_prediction_mse'][h_key]['volume'] += mse_volume
            
            # Classification de tendances
            for h_key, h_preds in outputs['trend_classification'].items():
                if h_key not in all_preds['trend_classification']:
                    all_preds['trend_classification'][h_key] = []
                    all_targets['trend_classification'][h_key] = []
                    metrics['trend_classification_accuracy'][h_key] = 0.0
                
                if h_key in targets['trend_classification']:
                    # Prédictions de classe
                    _, pred_class = torch.max(h_preds, dim=1)
                    pred_class = pred_class.cpu().numpy()
                    target_class = targets['trend_classification'][h_key].squeeze().numpy()
                    
                    all_preds['trend_classification'][h_key].append(pred_class)
                    all_targets['trend_classification'][h_key].append(target_class)
                    
                    # Précision
                    accuracy = (pred_class == target_class).mean()
                    metrics['trend_classification_accuracy'][h_key] += accuracy
            
            # Optimisation de portefeuille
            pred_portfolio = outputs['portfolio_optimization'].cpu().numpy()
            target_portfolio = targets['portfolio_optimization'].numpy()
            
            all_preds['portfolio_optimization'].append(pred_portfolio)
            all_targets['portfolio_optimization'].append(target_portfolio)
            
            # MSE pour le portefeuille
            mse_portfolio = ((pred_portfolio - target_portfolio) ** 2).mean()
            metrics['portfolio_optimization_mse'] += mse_portfolio
            
            # Gestion des risques
            for param_key, param_preds in outputs['risk_management'].items():
                if param_key not in all_preds['risk_management']:
                    all_preds['risk_management'][param_key] = []
                    all_targets['risk_management'][param_key] = []
                    metrics['risk_management_mse'][param_key] = 0.0
                
                if param_key in targets['risk_management']:
                    pred_param = param_preds.cpu().numpy()
                    target_param = targets['risk_management'][param_key].numpy()
                    
                    all_preds['risk_management'][param_key].append(pred_param)
                    all_targets['risk_management'][param_key].append(target_param)
                    
                    # MSE pour les paramètres de risque
                    mse_param = ((pred_param - target_param) ** 2).mean()
                    metrics['risk_management_mse'][param_key] += mse_param
            
            batch_count += 1
    
    # Calculer les moyennes des métriques
    for h_key in metrics['price_prediction_mse']:
        metrics['price_prediction_mse'][h_key]['price'] /= batch_count
        metrics['price_prediction_mse'][h_key]['volume'] /= batch_count
        
    for h_key in metrics['trend_classification_accuracy']:
        metrics['trend_classification_accuracy'][h_key] /= batch_count
        
    metrics['portfolio_optimization_mse'] /= batch_count
    
    for param_key in metrics['risk_management_mse']:
        metrics['risk_management_mse'][param_key] /= batch_count
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'metrics': metrics,
    }


def visualize_predictions(results, title="Prédictions multi-tâches"):
    """
    Visualise les prédictions du modèle.
    
    Args:
        results: Résultats des prédictions
        title: Titre du graphique
    """
    plt.figure(figsize=(15, 20))
    
    # 1. Prédiction de prix
    plt.subplot(4, 1, 1)
    horizon_key = 'h1'  # Horizon de 1 pas
    
    if horizon_key in results['predictions']['price_prediction']:
        # Concaténer toutes les prédictions
        pred_prices = np.concatenate(results['predictions']['price_prediction'][horizon_key]['price'])
        target_prices = np.concatenate(results['targets']['price_prediction'][horizon_key]['price'])
        
        # Tracer les prix pour quelques exemples
        n_examples = min(100, len(pred_prices))
        plt.plot(pred_prices[:n_examples, 3], label="Prédiction (close)")  # Indice 3 pour le prix de clôture
        plt.plot(target_prices[:n_examples, 3], label="Cible (close)")
        plt.title(f"Prédiction de prix (horizon {horizon_key})")
        plt.xlabel("Échantillon")
        plt.ylabel("Prix normalisé")
        plt.legend()
        plt.grid(True)
    
    # 2. Classification de tendances
    plt.subplot(4, 1, 2)
    if horizon_key in results['predictions']['trend_classification']:
        # Concaténer toutes les prédictions
        pred_trends = np.concatenate(results['predictions']['trend_classification'][horizon_key])
        target_trends = np.concatenate(results['targets']['trend_classification'][horizon_key])
        
        # Tracer la matrice de confusion sous forme d'histogramme
        trend_labels = ["Baissier", "Neutre", "Haussier"]
        confusion = np.zeros((3, 3))
        
        for i in range(len(pred_trends)):
            confusion[target_trends[i], pred_trends[i]] += 1
            
        # Normaliser par ligne
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion_norm = confusion / row_sums
        
        # Afficher
        x = np.arange(len(trend_labels))
        width = 0.2
        
        for i in range(3):
            plt.bar(x + i*width, confusion_norm[:,i], width, label=f'Prédit {trend_labels[i]}')
            
        plt.xlabel('Tendance réelle')
        plt.ylabel('Proportion')
        plt.title(f'Matrice de confusion pour la classification de tendances (horizon {horizon_key})')
        plt.xticks(x + width, trend_labels)
        plt.legend()
        plt.grid(True, axis='y')
    
    # 3. Optimisation de portefeuille
    plt.subplot(4, 1, 3)
    if results['predictions']['portfolio_optimization']:
        # Concaténer toutes les prédictions
        pred_allocations = np.concatenate(results['predictions']['portfolio_optimization'])
        target_allocations = np.concatenate(results['targets']['portfolio_optimization'])
        
        # Calculer l'allocation moyenne par actif
        avg_pred_allocation = pred_allocations.mean(axis=0)
        avg_target_allocation = target_allocations.mean(axis=0)
        
        # Tracer
        x = np.arange(len(avg_pred_allocation))
        width = 0.35
        
        plt.bar(x - width/2, avg_target_allocation, width, label='Cible')
        plt.bar(x + width/2, avg_pred_allocation, width, label='Prédiction')
        
        plt.xlabel('Actif')
        plt.ylabel('Allocation moyenne')
        plt.title('Optimisation de portefeuille: Allocations moyennes')
        plt.xticks(x, [f'Actif {i+1}' for i in range(len(avg_pred_allocation))])
        plt.legend()
        plt.grid(True, axis='y')
    
    # 4. Gestion des risques
    plt.subplot(4, 1, 4)
    risk_params = ['stop_loss', 'take_profit', 'position_size', 'risk_score']
    
    if all(param in results['predictions']['risk_management'] for param in risk_params):
        # Préparer les données pour le tracé
        param_values = {}
        for param in risk_params:
            pred_values = np.concatenate(results['predictions']['risk_management'][param])
            target_values = np.concatenate(results['targets']['risk_management'][param])
            
            param_values[param] = {
                'pred': pred_values.mean(),
                'target': target_values.mean(),
            }
        
        # Tracer
        x = np.arange(len(risk_params))
        width = 0.35
        
        plt.bar(x - width/2, [param_values[p]['target'] for p in risk_params], width, label='Cible')
        plt.bar(x + width/2, [param_values[p]['pred'] for p in risk_params], width, label='Prédiction')
        
        plt.xlabel('Paramètre de risque')
        plt.ylabel('Valeur moyenne')
        plt.title('Gestion des risques: Paramètres moyens')
        plt.xticks(x, [p.replace('_', ' ').title() for p in risk_params])
        plt.legend()
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multitask_learning_predictions.png")
    plt.close()


def main():
    """Fonction principale pour tester le modèle d'apprentissage multi-tâches."""
    # Définir le périphérique
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Utilisation du périphérique: {device}")
    
    # Générer des données synthétiques
    logger.info("Génération de données synthétiques...")
    data = generate_synthetic_data(n_samples=2000, num_assets=5)
    logger.info(f"Données générées: {data.shape}")
    
    # Préparer les DataLoaders
    logger.info("Préparation des DataLoaders...")
    train_loader, val_loader = prepare_dataloaders(
        df=data,
        batch_size=32,
        val_split=0.2,
        window_size=50,
        num_assets=5,
    )
    logger.info(f"DataLoaders préparés: {len(train_loader)} batches d'entraînement, {len(val_loader)} batches de validation")
    
    # Déterminer la dimension d'entrée
    input_dim = next(iter(train_loader))[0].shape[2]
    logger.info(f"Dimension d'entrée: {input_dim}")
    
    # Créer le modèle
    logger.info("Création du modèle d'apprentissage multi-tâches...")
    model = MultitaskLearningModel(
        input_dim=input_dim,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=50,
        prediction_horizons=[1, 5, 10, 20],
        num_trend_classes=3,
        num_assets=5,
    )
    logger.info(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Créer l'entraîneur
    logger.info("Initialisation de l'entraîneur...")
    trainer = MultitaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001, 'weight_decay': 1e-5},
        lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={'factor': 0.5, 'patience': 3, 'verbose': True},
        device=device,
        save_dir=str(RESULTS_DIR),
    )
    
    # Entraîner le modèle
    logger.info("Début de l'entraînement...")
    history = trainer.train(
        num_epochs=20,
        early_stopping=True,
        patience=5,
        model_name='multitask_model',
    )
    
    # Visualiser l'historique d'entraînement
    logger.info("Visualisation de l'historique d'entraînement...")
    visualize_training_history(history)
    
    # Tester le modèle
    logger.info("Test du modèle...")
    test_results = test_model_predictions(model, val_loader, device)
    
    # Afficher les métriques
    logger.info("Métriques de test:")
    logger.info(f"MSE Prix (h1): {test_results['metrics']['price_prediction_mse'].get('h1', {}).get('price', 'N/A')}")
    logger.info(f"Précision Classification (h1): {test_results['metrics']['trend_classification_accuracy'].get('h1', 'N/A')}")
    logger.info(f"MSE Portefeuille: {test_results['metrics']['portfolio_optimization_mse']}")
    for param, mse in test_results['metrics']['risk_management_mse'].items():
        logger.info(f"MSE {param}: {mse}")
    
    # Visualiser les prédictions
    logger.info("Visualisation des prédictions...")
    visualize_predictions(test_results)
    
    logger.info("Test d'apprentissage multi-tâches terminé avec succès!")
    
    return model, trainer, history, test_results


if __name__ == "__main__":
    main() 