"""
Tests unitaires pour le module de collecte de données (Phase 1.1).
"""

import unittest
import os
import pandas as pd
from datetime import datetime, timedelta
import sys

# Ajout du chemin absolu vers le répertoire ai_trading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Puis importer normalement
from ai_trading.utils.minimal_data_collector import MinimalDataCollector

class TestMinimalDataCollector(unittest.TestCase):
    """Tests pour la classe MinimalDataCollector."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.collector = MinimalDataCollector()
    
    def test_initialization(self):
        """Teste l'initialisation du collecteur."""
        self.assertIsNotNone(self.collector.coingecko)
    
    def test_get_crypto_prices(self):
        """Teste la récupération des prix de cryptomonnaies."""
        # Test avec un petit nombre de jours pour accélérer le test
        df = self.collector.get_crypto_prices(coin_id='bitcoin', days=3)
        
        # Vérification que le DataFrame n'est pas vide
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Vérification des colonnes
        self.assertIn('price', df.columns)
        self.assertIn('volume', df.columns)
        
        # Vérification des types de données
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['price']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['volume']))
    
    def test_get_trending_coins(self):
        """Teste la récupération des cryptomonnaies tendance."""
        trending = self.collector.get_trending_coins()
        
        # Vérification que la liste n'est pas vide
        self.assertIsNotNone(trending)
        
        # Si l'API renvoie des données, vérifier la structure
        if trending:
            first_coin = trending[0]
            self.assertIn('item', first_coin)
            self.assertIn('name', first_coin['item'])
            self.assertIn('symbol', first_coin['item'])
    
    def test_save_data(self):
        """Teste la sauvegarde des données."""
        # Création d'un petit DataFrame de test
        test_data = pd.DataFrame({
            'price': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        # Sauvegarde des données
        test_file = 'data/test_save.csv'
        self.collector.save_data(test_data, 'test_save.csv')
        
        # Vérification que le fichier existe
        self.assertTrue(os.path.exists(test_file))
        
        # Vérification du contenu
        loaded_data = pd.read_csv(test_file, index_col=0, parse_dates=True)
        self.assertEqual(len(loaded_data), len(test_data))
        self.assertListEqual(list(loaded_data.columns), list(test_data.columns))
        
        # Nettoyage
        os.remove(test_file)


if __name__ == '__main__':
    unittest.main() 