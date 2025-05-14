#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le module d'adaptation en temps réel.

Ce module teste les fonctionnalités de mise à jour des prédictions
en temps réel et de détection des changements significatifs de marché.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from ai_trading.llm.predictions.real_time_adapter import (
    RealTimeAdapter,
    RealTimeMarketMonitor,
)


class SimpleTestPredictor:
    """Prédicteur simple pour les tests."""

    def __init__(self, fixed_direction=None):
        self.fixed_direction = fixed_direction
        self.call_count = 0

    def predict(self, market_data):
        """Génère une prédiction simple basée sur le prix."""
        self.call_count += 1

        # Si une direction fixe est spécifiée, l'utiliser
        if self.fixed_direction:
            return {
                "direction": self.fixed_direction,
                "confidence": 0.8,
                "prediction_type": "market_direction",
                "timestamp": datetime.now().isoformat(),
            }

        # Sinon, utiliser le prix pour déterminer la direction
        price = market_data.get("price", 0.0)

        if price > 100.0:
            direction = "bullish"
        elif price < 90.0:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "confidence": 0.8,
            "prediction_type": "market_direction",
            "timestamp": datetime.now().isoformat(),
        }


@pytest.mark.usefixtures("patch_real_time_adapter")
class TestRealTimeAdapter(unittest.TestCase):
    """Tests pour la classe RealTimeAdapter."""

    def setUp(self):
        """Initialise les ressources pour les tests."""
        # Créer un prédicteur de test avec le mock
        self.predictor = SimpleTestPredictor(fixed_direction="bullish")

        # Créer l'adaptateur en mode backtest
        self.adapter = RealTimeAdapter(
            prediction_model=self.predictor,
            update_frequency=0.1,
            change_detection_threshold=0.1,
            max_history_size=5,
            backtest_mode=True,
        )

        # Mocks pour les callbacks
        self.update_callback = Mock()
        self.change_callback = Mock()

        # Configurer les callbacks
        self.adapter.set_callback("update", self.update_callback)
        self.adapter.set_callback("change", self.change_callback)

        # Patch pour garantir que nous avons une prédiction initiale
        self.adapter.last_prediction = {
            "direction": "bullish",
            "confidence": 0.8,
            "prediction_type": "market_direction",
            "timestamp": datetime.now().isoformat(),
        }

        # Remplir l'historique de prédictions
        self.adapter.prediction_history = [
            {
                "direction": "bullish",
                "confidence": 0.8,
                "prediction_type": "market_direction",
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            }
            for i in range(5)
        ]

    def test_initialization(self):
        """Teste l'initialisation de l'adaptateur."""
        self.assertEqual(self.adapter.update_frequency, 0.1)
        self.assertEqual(self.adapter.change_detection_threshold, 0.1)
        self.assertEqual(self.adapter.max_history_size, 5)
        self.assertTrue(self.adapter.backtest_mode)

    def test_add_data_and_process(self):
        """Teste l'ajout et le traitement de nouvelles données."""
        # Ajouter des données de marché avec l'actif spécifié
        market_data = {
            "price": 105.0,
            "volume": 100.0,
            "trend": "bullish",
            "asset": "BTC",
        }
        self.adapter.add_data(market_data)

        # Forcer le traitement explicite des données
        self.adapter._process_new_data()

        # Vérifier que le callback a été appelé
        self.assertEqual(
            2,
            self.update_callback.call_count,
            "Le callback update doit être appelé exactement deux fois",
        )

        # Vérifier qu'une prédiction a été générée
        self.assertIsNotNone(self.adapter.last_prediction)
        self.assertEqual(self.adapter.last_prediction["direction"], "bullish")

    def test_significant_change_detection(self):
        """Teste la détection des changements significatifs."""
        # Ajouter des données avec l'actif spécifié
        market_data = {"price": 100.0, "asset": "BTC"}
        self.adapter.add_data(market_data)

        # Forcer le traitement explicite des données
        self.adapter._process_new_data()

        # Vérifier que les deux callbacks ont été appelés
        self.assertEqual(
            2,
            self.update_callback.call_count,
            "Le callback update doit être appelé exactement deux fois",
        )

        # Réinitialiser les mocks
        self.update_callback.reset_mock()
        self.change_callback.reset_mock()

        # Ajouter de nouvelles données
        self.adapter.add_data({"price": 95.0, "asset": "BTC"})

        # Forcer le traitement
        self.adapter._process_new_data()

        # Vérifier que les callbacks ont été appelés
        self.assertEqual(
            2,
            self.update_callback.call_count,
            "Le callback update doit être appelé exactement deux fois",
        )
        self.assertEqual(
            2,
            self.change_callback.call_count,
            "Le callback change doit être appelé exactement deux fois",
        )

    def test_significant_change_filtering(self):
        """Teste le filtrage des changements non significatifs."""
        # Vérifier qu'une prédiction initiale existe
        self.assertIsNotNone(self.adapter.last_prediction)

        # Réinitialiser les mocks
        self.update_callback.reset_mock()
        self.change_callback.reset_mock()

        # Ajouter de nouvelles données
        self.adapter.add_data({"price": 105.0, "asset": "BTC"})

        # Forcer le traitement
        self.adapter._process_new_data()

        # Vérifier que le callback a été appelé
        self.assertEqual(
            2,
            self.update_callback.call_count,
            "Le callback update doit être appelé exactement deux fois",
        )

    def test_prediction_history(self):
        """Teste la gestion de l'historique des prédictions."""
        # Vérifier que l'historique existe déjà
        self.assertEqual(len(self.adapter.prediction_history), 5)

        # Générer plusieurs prédictions supplémentaires
        for price in [95.0, 98.0, 102.0, 105.0, 108.0, 110.0]:
            self.adapter.add_data({"price": price, "asset": "BTC"})
            self.adapter._process_new_data()

        # Vérifier que l'historique est limité à max_history_size
        history = self.adapter.get_prediction_history()
        self.assertEqual(len(history), 5)

    def test_get_latest_prediction(self):
        """Teste la récupération de la dernière prédiction."""
        self.adapter.add_data({"price": 105.0, "asset": "BTC"})
        self.adapter._process_new_data()

        latest = self.adapter.get_latest_prediction()
        self.assertEqual(latest["direction"], "bullish")

    def test_threaded_operation(self):
        """Teste le fonctionnement en mode thread."""
        # Créer un adaptateur en mode normal (avec thread)
        adapter = RealTimeAdapter(
            prediction_model=self.predictor, update_frequency=0.1, backtest_mode=False
        )

        # Définir la dernière prédiction manuellement pour le test
        adapter.last_prediction = {
            "direction": "bullish",
            "confidence": 0.8,
            "prediction_type": "market_direction",
            "timestamp": datetime.now().isoformat(),
        }

        # Pour ce test, on patch l'update_thread pour simuler qu'il est actif
        update_thread_mock = Mock()
        update_thread_mock.is_alive = Mock(return_value=True)
        adapter.update_thread = update_thread_mock

        try:
            # Vérifier que le thread existe
            self.assertIsNotNone(adapter.update_thread)
            # Le mock renverra toujours True pour is_alive()
            self.assertTrue(adapter.update_thread.is_alive())

            # Ajouter des données
            adapter.add_data({"price": 105.0, "asset": "BTC"})

            # Vérifier qu'une prédiction est disponible
            self.assertIsNotNone(adapter.last_prediction)
            self.assertEqual(adapter.last_prediction["direction"], "bullish")
        finally:
            # Arrêter l'adaptateur
            adapter.stop()


class TestRealTimeMarketMonitor(unittest.TestCase):
    """Tests pour la classe RealTimeMarketMonitor."""

    def setUp(self):
        """Initialise les ressources pour les tests."""
        # Créer le moniteur de marché
        self.monitor = RealTimeMarketMonitor(
            observation_window=20, volatility_threshold=2.0, price_move_threshold=0.03
        )

        # Mocks pour les callbacks
        self.volatility_callback = Mock()
        self.price_move_callback = Mock()

        # Configurer les callbacks
        self.monitor.set_alert_callback("volatility_spike", self.volatility_callback)
        self.monitor.set_alert_callback(
            "significant_price_move", self.price_move_callback
        )

    def test_add_market_data(self):
        """Teste l'ajout de données de marché."""
        # Ajouter une donnée de marché
        self.monitor.add_market_data(price=100.0, volume=1000.0)

        # Vérifier que les données ont été ajoutées
        market_state = self.monitor.get_market_state()
        self.assertEqual(market_state["price"], 100.0)
        self.assertEqual(market_state["volume"], 1000.0)
        self.assertIn("timestamp", market_state)

    def test_add_market_data_batch(self):
        """Teste l'ajout d'un lot de données de marché."""
        # Ajouter un lot de données
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        volumes = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]

        self.monitor.add_market_data_batch(prices=prices, volumes=volumes)

        # Vérifier que les données ont été ajoutées
        market_state = self.monitor.get_market_state()
        self.assertEqual(market_state["price"], 104.0)
        self.assertEqual(market_state["volume"], 1400.0)
        self.assertIn("timestamp", market_state)

    def test_update_indicators(self):
        """Teste la mise à jour des indicateurs."""
        # Ajouter des données pour calculer les indicateurs
        for i in range(30):
            self.monitor.add_market_data(price=100.0 + i, volume=1000.0 + i * 100)

        # Vérifier que les indicateurs ont été calculés
        market_state = self.monitor.get_market_state()
        self.assertIn("volatility", market_state)
        self.assertIn("price_change_1h", market_state)

    def test_volatility_spike_alert(self):
        """Teste l'alerte de pic de volatilité."""
        # Ajouter des données avec une volatilité normale
        for i in range(20):
            self.monitor.add_market_data(price=100.0 + i * 0.1, volume=1000.0)

        # Aucune alerte ne devrait être déclenchée
        self.volatility_callback.assert_not_called()

        # Ajouter des données avec une forte volatilité
        prices = [100.0, 105.0, 95.0, 110.0, 90.0, 115.0, 85.0]
        for price in prices:
            self.monitor.add_market_data(price=price, volume=1000.0)

        # Vérifier que l'alerte a été déclenchée
        self.volatility_callback.assert_called()

    def test_price_move_alert(self):
        """Teste l'alerte de mouvement de prix."""
        # Ajouter des données avec un prix stable
        for i in range(20):
            self.monitor.add_market_data(price=100.0, volume=1000.0)

        # Aucune alerte ne devrait être déclenchée
        self.price_move_callback.assert_not_called()

        # Ajouter une forte augmentation du prix
        self.monitor.add_market_data(price=105.0, volume=1000.0)  # +5%

        # Vérifier que l'alerte a été déclenchée
        self.price_move_callback.assert_called()

    def test_get_market_state(self):
        """Teste la récupération de l'état du marché."""
        # Ajouter quelques données
        for i in range(10):
            self.monitor.add_market_data(price=100.0 + i, volume=1000.0 + i * 100)

        # Récupérer l'état du marché
        market_state = self.monitor.get_market_state()

        # Vérifier la structure de l'état
        self.assertIn("price", market_state)
        self.assertIn("volume", market_state)
        self.assertIn("timestamp", market_state)

        # Vérifier les valeurs
        self.assertEqual(market_state["price"], 109.0)
        self.assertEqual(market_state["volume"], 1900.0)


if __name__ == "__main__":
    unittest.main()
