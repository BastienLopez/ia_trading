import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ajuster les chemins d'importation
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import des modules
from ai_trading.rl.models.multitask_learning_model import MultitaskLearningModel
from ai_trading.rl.agents.multitask_agent import MultitaskTradingAgent

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_multitask_model():
    """Teste le modèle d'apprentissage multi-tâches."""
    # Paramètres
    batch_size = 2
    seq_len = 10
    input_dim = 20
    prediction_horizons = [1, 5]
    num_trend_classes = 3
    num_assets = 5
    
    # Créer des données d'entrée simulées
    inputs = torch.randn(batch_size, seq_len, input_dim)
    
    # Créer le modèle
    model = MultitaskLearningModel(
        input_dim=input_dim,
        d_model=64,
        n_heads=2,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.1,
        activation="gelu",
        max_seq_len=seq_len,
        prediction_horizons=prediction_horizons,
        num_trend_classes=num_trend_classes,
        num_assets=num_assets,
    )
    
    # Forward pass
    outputs = model(inputs)
    
    # Vérifier les formes des sorties
    assert 'price_prediction' in outputs, "La sortie price_prediction est manquante"
    assert 'trend_classification' in outputs, "La sortie trend_classification est manquante"
    assert 'portfolio_optimization' in outputs, "La sortie portfolio_optimization est manquante"
    assert 'risk_management' in outputs, "La sortie risk_management est manquante"
    
    # Vérifier les sorties de prédiction de prix
    for h in prediction_horizons:
        h_key = f"h{h}"
        assert h_key in outputs['price_prediction'], f"Horizon {h_key} manquant dans price_prediction"
        assert 'price' in outputs['price_prediction'][h_key], f"price manquant dans {h_key}"
        assert 'volume' in outputs['price_prediction'][h_key], f"volume manquant dans {h_key}"
        
        assert outputs['price_prediction'][h_key]['price'].shape == (batch_size, 4), \
            f"Forme incorrecte pour {h_key} price: {outputs['price_prediction'][h_key]['price'].shape}"
        assert outputs['price_prediction'][h_key]['volume'].shape == (batch_size, 1), \
            f"Forme incorrecte pour {h_key} volume: {outputs['price_prediction'][h_key]['volume'].shape}"
    
    # Vérifier les sorties de classification de tendances
    for h in prediction_horizons:
        h_key = f"h{h}"
        assert h_key in outputs['trend_classification'], f"Horizon {h_key} manquant dans trend_classification"
        assert outputs['trend_classification'][h_key].shape == (batch_size, num_trend_classes), \
            f"Forme incorrecte pour {h_key} trend: {outputs['trend_classification'][h_key].shape}"
    
    # Vérifier les sorties d'optimisation de portefeuille
    assert outputs['portfolio_optimization'].shape == (batch_size, num_assets), \
        f"Forme incorrecte pour portfolio_optimization: {outputs['portfolio_optimization'].shape}"
    
    # Vérifier les sorties de gestion des risques
    risk_params = ['stop_loss', 'take_profit', 'position_size', 'risk_score']
    for param in risk_params:
        assert param in outputs['risk_management'], f"Paramètre {param} manquant dans risk_management"
        assert outputs['risk_management'][param].shape == (batch_size, 1), \
            f"Forme incorrecte pour {param}: {outputs['risk_management'][param].shape}"
    
    logger.info("✅ Test du modèle d'apprentissage multi-tâches réussi!")


def test_multitask_agent():
    """Teste l'agent de trading multi-tâches."""
    # Paramètres
    state_dim = 20
    action_dim = 3
    num_assets = 5
    max_seq_len = 10
    
    # Créer l'agent
    agent = MultitaskTradingAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_assets=num_assets,
        d_model=64,
        n_heads=2,
        num_layers=1,
        max_seq_len=max_seq_len,
        device='cpu',
        risk_aversion=0.5,
        exploration_rate=0.1,
        lr=0.001,
    )
    
    # Tester reset_state_buffer
    agent.reset_state_buffer()
    assert len(agent.state_buffer) == 0, "Le buffer d'état devrait être vide"
    
    # Tester act
    state = np.random.random(state_dim)
    action = agent.act(state, explore=False)
    assert action.shape == (action_dim,), f"Forme incorrecte pour action: {action.shape}"
    
    # Tester l'ajout d'un état au buffer
    assert len(agent.state_buffer) == 1, "Le buffer devrait contenir un état"
    
    # Tester update
    states = [np.random.random(state_dim) for _ in range(5)]
    actions = [np.random.random(action_dim) for _ in range(5)]
    rewards = [np.random.random() for _ in range(5)]
    next_states = [np.random.random(state_dim) for _ in range(5)]
    dones = [False] * 4 + [True]
    
    # Tester que l'update ne plante pas
    try:
        losses = agent.update(states, actions, rewards, next_states, dones)
        assert isinstance(losses, dict), "Les pertes devraient être un dictionnaire"
        logger.info("✅ Update réussi!")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'update: {e}")
        raise
    
    logger.info("✅ Test de l'agent multi-tâches réussi!")


def main():
    """Fonction principale pour tester l'intégration du modèle multi-tâches."""
    logger.info("Début des tests d'intégration multi-tâches...")
    
    # Tester le modèle multi-tâches
    try:
        test_multitask_model()
    except Exception as e:
        logger.error(f"❌ Test du modèle échoué: {e}")
        return
    
    # Tester l'agent multi-tâches
    try:
        test_multitask_agent()
    except Exception as e:
        logger.error(f"❌ Test de l'agent échoué: {e}")
        return
    
    logger.info("🎉 Tous les tests d'intégration multi-tâches sont réussis!")


if __name__ == "__main__":
    main() 