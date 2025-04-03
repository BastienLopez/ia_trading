"""
Tests unitaires pour le module data_processor
"""
import os
import unittest
import logging
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Assurez-vous que le module est accessible dans le chemin
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print("\n[DEBUG TEST_DATA_PROCESSOR] Début du test")
print(f"[DEBUG TEST_DATA_PROCESSOR] Chemin Python: {sys.path}")
print(f"[DEBUG TEST_DATA_PROCESSOR] Répertoire courant: {os.getcwd()}")

try:
    print("[DEBUG TEST_DATA_PROCESSOR] Tentative d'import de ccxt")
    import ccxt
    print(f"[DEBUG TEST_DATA_PROCESSOR] Module ccxt importé avec succès, version: {ccxt.__version__}")
except ImportError as e:
    print(f"[DEBUG TEST_DATA_PROCESSOR] ERREUR: Impossible d'importer ccxt: {e}")
    print("[DEBUG TEST_DATA_PROCESSOR] Solution: installer avec 'pip install ccxt'")

try:
    print("[DEBUG TEST_DATA_PROCESSOR] Tentative d'import de ai_trading.data_processor")
    from ai_trading.data_processor import DataProcessor
    print("[DEBUG TEST_DATA_PROCESSOR] Module DataProcessor importé avec succès")
except ImportError as e:
    print(f"[DEBUG TEST_DATA_PROCESSOR] ERREUR: Impossible d'importer DataProcessor: {e}")
    print(f"[DEBUG TEST_DATA_PROCESSOR] Détails de l'erreur: {e.__class__.__name__}: {str(e)}")
    print(f"[DEBUG TEST_DATA_PROCESSOR] Modules chargés: {sys.modules.keys()}")

# Configuration d'un logger pour ce module de test
logger = logging.getLogger(__name__)

class TestDataProcessor(unittest.TestCase):
    """Tests pour le DataProcessor"""

    def setUp(self):
        """Configuration avant chaque test"""
        print(f"\n[SETUP] Test: {self._testMethodName}")
        logger.info("Configuration du test DataProcessor")
        try:
            self.data_processor = DataProcessor(data_dir="test_data")
            print("[SETUP] DataProcessor initialisé avec succès")
        except Exception as e:
            print(f"[SETUP] ERREUR lors de l'initialisation du DataProcessor: {e}")
            raise
        
        # Créer un DataFrame test avec les colonnes nécessaires
        logger.debug("Création du DataFrame de test")
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
            'open': np.random.rand(100) * 100 + 42000,
            'high': np.random.rand(100) * 100 + 42100,
            'low': np.random.rand(100) * 100 + 41900,
            'close': np.random.rand(100) * 100 + 42000,
            'volume': np.random.rand(100) * 1000
        })
        self.test_df.set_index('timestamp', inplace=True)
        logger.debug(f"DataFrame de test créé avec shape: {self.test_df.shape}")
        print(f"[SETUP] DataFrame créé avec {len(self.test_df)} lignes et {len(self.test_df.columns)} colonnes")

    @patch('os.makedirs')
    def test_init(self, mock_makedirs):
        """Tester l'initialisation du DataProcessor"""
        print(f"\n[TEST] Exécution de test_init")
        logger.info("Test d'initialisation du DataProcessor")
        try:
            data_processor = DataProcessor(data_dir="custom_dir")
            logger.debug(f"Répertoire de données configuré: {data_processor.data_dir}")
            self.assertEqual(data_processor.data_dir, "custom_dir")
            mock_makedirs.assert_called_once_with("custom_dir", exist_ok=True)
            logger.info("Test d'initialisation du DataProcessor réussi")
            print(f"[TEST] test_init réussi")
        except Exception as e:
            print(f"[TEST ERROR] test_init a échoué: {e}")
            raise

    @patch('ai_trading.data_processor.ccxt')
    def test_download_historical_data(self, mock_ccxt):
        """Tester la fonction de téléchargement des données historiques"""
        print(f"\n[TEST] Exécution de test_download_historical_data")
        logger.info("Test de download_historical_data")
        
        # Créer un mock pour l'exchange
        logger.debug("Configuration du mock pour l'exchange")
        mock_exchange = MagicMock()
        # Définir un comportement qui ne va pas boucler indéfiniment
        mock_exchange.fetch_ohlcv.side_effect = [
            # Premier appel
            [
                [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],  # 2024-01-01 00:00:00
                [1704070800000, 42050.0, 42150.0, 41950.0, 42100.0, 110.0]   # 2024-01-01 01:00:00
            ],
            # Deuxième appel - retourner une liste vide pour sortir de la boucle
            []
        ]
        
        # Configurer le mock ccxt pour retourner notre mock_exchange
        mock_ccxt.binance.return_value = mock_exchange
        
        # Paramètres de la requête
        symbol = 'BTC/USDT'
        timeframe = '1h'
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        logger.debug(f"Appel de download_historical_data avec paramètres: symbol={symbol}, timeframe={timeframe}")
        logger.debug(f"Dates: start={start_date}, end={end_date}")
        print(f"[TEST] Paramètres: {symbol}, {timeframe}, start={start_date}, end={end_date}")
        
        # Appeler la fonction avec des paramètres spécifiques
        try:
            print(f"[TEST] Tentative d'appel de download_historical_data")
            result = self.data_processor.download_historical_data(
                exchange_id='binance',
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                save=False
            )
            
            # Vérifications
            logger.debug(f"Résultat: type={type(result)}, shape={result.shape}")
            logger.debug(f"Colonnes: {result.columns.tolist()}")
            print(f"[TEST] Résultat shape: {result.shape}")
            print(f"[TEST] Colonnes: {result.columns.tolist()}")
            
            # Vérifier que le DataFrame résultant a la bonne structure
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)  # 2 entrées de données
            self.assertListEqual(list(result.columns), ['open', 'high', 'low', 'close', 'volume'])
            
            # Vérifier que l'appel à fetch_ohlcv a été fait correctement
            mock_exchange.fetch_ohlcv.assert_called()
            # On pourrait vérifier les arguments exacts de l'appel si nécessaire
            logger.debug("Paramètres d'appel à fetch_ohlcv: " + 
                       str(mock_exchange.fetch_ohlcv.call_args))
            
            logger.info("Test de download_historical_data réussi")
            print(f"[TEST] test_download_historical_data réussi")
        except Exception as e:
            print(f"[TEST ERROR] test_download_historical_data a échoué: {e}")
            import traceback
            print(traceback.format_exc())
            raise

    @patch('builtins.open', new_callable=mock_open, read_data='timestamp,open,high,low,close,volume\n2024-01-01 00:00:00,42000,42100,41900,42050,100\n2024-01-01 01:00:00,42050,42150,41950,42100,110')
    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv, mock_file):
        """Tester le chargement de données depuis un fichier CSV"""
        print(f"\n[TEST] Exécution de test_load_data")
        # Configurer le mock pour pd.read_csv
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=2, freq='1h'),
            'open': [42000.0, 42050.0],
            'high': [42100.0, 42150.0],
            'low': [41900.0, 41950.0],
            'close': [42050.0, 42100.0],
            'volume': [100.0, 110.0]
        })
        mock_df.set_index('timestamp', inplace=True)
        mock_read_csv.return_value = mock_df
        
        # Appeler la fonction load_data
        print(f"[TEST] Appel de load_data avec fichier: test_file.csv")
        try:
            result = self.data_processor.load_data('test_file.csv')
            
            # Vérifier que read_csv a été appelé avec les bons paramètres
            mock_read_csv.assert_called_once_with('test_file.csv', index_col='timestamp', parse_dates=True)
            print(f"[TEST] read_csv appelé avec les bons paramètres")
            
            # Vérifier que le résultat est identique à notre mock_df
            pd.testing.assert_frame_equal(result, mock_df)
            print(f"[TEST] DataFrames identiques - test_load_data réussi")
        except Exception as e:
            print(f"[TEST ERROR] test_load_data a échoué: {e}")
            raise

    def test_add_indicators(self):
        """Tester l'ajout d'indicateurs techniques"""
        # Informations de débogage avant d'appeler la fonction
        print(f"\n[TEST] Exécution de test_add_indicators")
        print(f"[DEBUG] Test d'ajout d'indicateurs")
        print(f"[DEBUG] DataFrame initial shape: {self.test_df.shape}")
        print(f"[DEBUG] Colonnes initiales: {self.test_df.columns.tolist()}")
        print(f"[DEBUG] Premières valeurs de 'close': {self.test_df['close'].head(3).tolist()}")
        
        logger.info("Test de l'ajout d'indicateurs techniques")
        logger.debug(f"DataFrame avant traitement: shape={self.test_df.shape}")
        
        # Appeler la fonction add_indicators
        try:
            logger.debug("Appel de la fonction add_indicators")
            print(f"[TEST] Appel de add_indicators")
            result = self.data_processor.add_indicators(self.test_df.copy())
            
            # Vérifier que tous les indicateurs attendus sont présents
            expected_indicators = [
                'rsi', 'macd', 'signal_line', 'hist_line', 'ema9', 'ema21',
                'tenkan', 'kijun', 'senkou_span_a', 'senkou_span_b',
                'bb_upper', 'bb_middle', 'bb_lower', 'volume_ratio'
            ]
            
            logger.debug(f"Vérification des indicateurs: {expected_indicators}")
            logger.debug(f"Colonnes résultantes: {result.columns.tolist()}")
            print(f"[TEST] Colonnes résultantes: {result.columns.tolist()}")
            
            for indicator in expected_indicators:
                self.assertIn(indicator, result.columns)
                logger.debug(f"Indicateur {indicator} présent: OK")
                # Afficher quelques valeurs pour vérification
                print(f"[DEBUG] Premières valeurs de '{indicator}': {result[indicator].head(3).tolist()}")
            
            # Vérifier qu'il n'y a pas de valeurs NaN après le traitement (dropna() a été appliqué)
            has_nulls = result.isnull().any().any()
            logger.debug(f"Présence de valeurs nulles: {has_nulls}")
            self.assertFalse(has_nulls)
            
            logger.info("Test de l'ajout d'indicateurs techniques réussi")
            print(f"[TEST] test_add_indicators réussi")
        except Exception as e:
            logger.error(f"Erreur lors du test d'ajout d'indicateurs: {str(e)}", exc_info=True)
            print(f"[TEST] ERREUR dans test_add_indicators: {str(e)}")
            raise

    def test_preprocess_for_training(self):
        """Tester le prétraitement des données pour l'entraînement"""
        print(f"\n[TEST] Exécution de test_preprocess_for_training")
        # Ajouter d'abord les indicateurs au DataFrame de test
        try:
            df_with_indicators = self.data_processor.add_indicators(self.test_df.copy())
            print(f"[TEST] DataFrame avec indicateurs créé: {df_with_indicators.shape}")
            
            # Appeler la fonction preprocess_for_training
            print(f"[TEST] Appel de preprocess_for_training avec train_ratio=0.8")
            train_data, test_data = self.data_processor.preprocess_for_training(df_with_indicators, train_ratio=0.8)
            
            # Vérifier la division train/test
            expected_train_size = int(len(df_with_indicators) * 0.8)
            print(f"[TEST] Taille attendue de train: {expected_train_size}, obtenue: {len(train_data)}")
            print(f"[TEST] Taille attendue de test: {len(df_with_indicators) - expected_train_size}, obtenue: {len(test_data)}")
            
            self.assertEqual(len(train_data), expected_train_size)
            self.assertEqual(len(test_data), len(df_with_indicators) - len(train_data))
            
            # Vérifier que les colonnes normalisées sont présentes
            normalized_columns = ['open_norm', 'high_norm', 'low_norm', 'close_norm']
            print(f"[TEST] Vérification des colonnes normalisées: {normalized_columns}")
            for col in normalized_columns:
                self.assertIn(col, train_data.columns)
                self.assertIn(col, test_data.columns)
                print(f"[TEST] Colonne {col} présente dans les deux ensembles")
            
            print(f"[TEST] test_preprocess_for_training réussi")
        except Exception as e:
            print(f"[TEST ERROR] test_preprocess_for_training a échoué: {e}")
            raise

    def test_prepare_backtesting_data(self):
        """Tester la préparation des données pour le backtesting"""
        print(f"\n[TEST] Exécution de test_prepare_backtesting_data")
        try:
            # Tester lorsque les indicateurs sont déjà présents
            df_with_indicators = self.data_processor.add_indicators(self.test_df.copy())
            print(f"[TEST] DataFrame avec indicateurs créé: {df_with_indicators.shape}")
            
            print(f"[TEST] Appel de prepare_backtesting_data avec données déjà préparées")
            result1 = self.data_processor.prepare_backtesting_data(df_with_indicators)
            
            # Vérifier que le DataFrame n'a pas été modifié (les indicateurs étaient déjà présents)
            self.assertEqual(len(result1.columns), len(df_with_indicators.columns))
            print(f"[TEST] Le DataFrame n'a pas été modifié comme attendu")
            
            # Tester lorsque les indicateurs ne sont pas présents
            with patch.object(self.data_processor, 'add_indicators', return_value=df_with_indicators) as mock_add:
                print(f"[TEST] Appel de prepare_backtesting_data avec données non préparées")
                result2 = self.data_processor.prepare_backtesting_data(self.test_df.copy())
                mock_add.assert_called_once()
                print(f"[TEST] add_indicators a bien été appelé")
                
                # Vérifier que tous les indicateurs sont présents
                for indicator in ['rsi', 'macd', 'ema9']:
                    self.assertIn(indicator, result2.columns)
                    print(f"[TEST] Indicateur {indicator} présent dans le résultat")
            
            print(f"[TEST] test_prepare_backtesting_data réussi")
        except Exception as e:
            print(f"[TEST ERROR] test_prepare_backtesting_data a échoué: {e}")
            raise


if __name__ == '__main__':
    print("[DEBUG TEST_DATA_PROCESSOR] Exécution directe du fichier de test")
    unittest.main() 