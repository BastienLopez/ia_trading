"""
Exemples d'utilisation du module sentiment_utils.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ai_trading.llm.sentiment_analysis.sentiment_utils import (
    SentimentCache,
    SentimentVisualizer,
    calculate_sentiment_metrics,
    get_llm_client,
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_cache_usage():
    """Exemple d'utilisation du cache de sentiment."""
    logger.info("Démonstration de l'utilisation du cache...")
    
    # Initialisation du cache
    cache = SentimentCache()
    
    # Données de test
    test_data = {
        "text": "Bitcoin atteint un nouveau record historique !",
        "sentiment": "positive",
        "score": 0.95,
        "timestamp": datetime.now().isoformat()
    }
    
    # Sauvegarde dans le cache
    cache.save("test_key", test_data)
    logger.info("Données sauvegardées dans le cache")
    
    # Chargement depuis le cache
    loaded_data = cache.load("test_key")
    logger.info(f"Données chargées depuis le cache: {loaded_data}")

def example_visualization():
    """Exemple d'utilisation des visualisations."""
    logger.info("Démonstration des visualisations...")
    
    # Création de données de test
    dates = pd.date_range(start=datetime.now(), periods=10, freq='D')
    sentiments = np.random.choice(['positive', 'neutral', 'negative'], size=10)
    scores = np.random.uniform(0, 1, size=10)
    
    df = pd.DataFrame({
        'date': dates,
        'sentiment': sentiments,
        'score': scores
    })
    
    # Initialisation du visualiseur
    visualizer = SentimentVisualizer()
    
    # Génération des graphiques
    visualizer.plot_sentiment_trends(df)
    logger.info("Graphique des tendances généré")
    
    visualizer.plot_sentiment_distribution(df)
    logger.info("Graphique de distribution généré")

def example_sentiment_analysis():
    """Exemple d'analyse de sentiment."""
    logger.info("Démonstration de l'analyse de sentiment...")
    
    # Obtention du client LLM
    client = get_llm_client()
    if client is None:
        logger.error("Impossible d'initialiser le client LLM")
        return
    
    # Texte de test
    text = "Le marché des cryptomonnaies est en pleine expansion !"
    
    try:
        # Analyse du sentiment
        result = client(text)
        # Le résultat est une liste de dictionnaires avec les scores pour chaque label
        sentiment_scores = {item['label']: item['score'] for item in result[0]}
        
        # Calcul des métriques
        metrics = calculate_sentiment_metrics([sentiment_scores])
        logger.info(f"Métriques calculées : {metrics}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse : {e}")

def main():
    """Point d'entrée principal."""
    logger.info("Démarrage des exemples d'utilisation de sentiment_utils...")
    
    try:
        example_cache_usage()
        example_visualization()
        example_sentiment_analysis()
        logger.info("Tous les exemples ont été exécutés avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des exemples: {e}")

if __name__ == "__main__":
    main() 