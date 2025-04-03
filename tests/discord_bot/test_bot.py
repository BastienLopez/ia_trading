"""
Tests unitaires pour le bot Discord
"""
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import discord
from discord.ext import commands

# Assurez-vous que le module est accessible dans le chemin
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import du module à tester
from discord_bot.bot import api_request, check_signals


class TestDiscordBot(unittest.TestCase):
    """Tests pour le bot Discord"""

    def setUp(self):
        """Configuration avant chaque test"""
        # Créer des mocks pour les modules discord
        self.mock_bot = MagicMock()
        self.mock_ctx = MagicMock()
        self.mock_ctx.send = AsyncMock()
        self.mock_ctx.channel = MagicMock()
        self.mock_ctx.channel.id = 12345

    @patch('discord_bot.bot.requests.get')
    @patch('discord_bot.bot.requests.post')
    async def test_api_request_get(self, mock_post, mock_get):
        """Tester la fonction api_request avec méthode GET"""
        # Configurer le mock pour requests.get
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "test_data"}
        mock_get.return_value = mock_response
        
        # Appeler la fonction api_request avec méthode GET
        result = await api_request("test-endpoint", method="GET")
        
        # Vérifier que requests.get a été appelé avec la bonne URL
        mock_get.assert_called_once()
        self.assertIn("test-endpoint", mock_get.call_args[0][0])
        
        # Vérifier que le résultat est celui attendu
        self.assertEqual(result, {"success": True, "data": "test_data"})

    @patch('discord_bot.bot.requests.get')
    @patch('discord_bot.bot.requests.post')
    async def test_api_request_post(self, mock_post, mock_get):
        """Tester la fonction api_request avec méthode POST"""
        # Configurer le mock pour requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "test_data"}
        mock_post.return_value = mock_response
        
        # Appeler la fonction api_request avec méthode POST
        result = await api_request("test-endpoint", method="POST", data={"param": "value"})
        
        # Vérifier que requests.post a été appelé avec les bons paramètres
        mock_post.assert_called_once()
        self.assertIn("test-endpoint", mock_post.call_args[0][0])
        self.assertEqual(mock_post.call_args[1]["json"], {"param": "value"})
        
        # Vérifier que le résultat est celui attendu
        self.assertEqual(result, {"success": True, "data": "test_data"})

    @patch('discord_bot.bot.requests.get')
    async def test_api_request_error_status(self, mock_get):
        """Tester la fonction api_request avec un code d'erreur"""
        # Configurer le mock pour requests.get avec un code d'erreur
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_get.return_value = mock_response
        
        # Appeler la fonction api_request
        result = await api_request("test-endpoint")
        
        # Vérifier que le résultat contient l'erreur
        self.assertTrue(result["error"])
        self.assertIn("404", result["message"])
        self.assertEqual(result["details"], "Not found")

    @patch('discord_bot.bot.requests.get')
    async def test_api_request_exception(self, mock_get):
        """Tester la fonction api_request avec une exception"""
        # Configurer le mock pour requests.get pour lever une exception
        mock_get.side_effect = Exception("Connection error")
        
        # Appeler la fonction api_request
        result = await api_request("test-endpoint")
        
        # Vérifier que le résultat contient l'erreur
        self.assertTrue(result["error"])
        self.assertIn("Connection error", result["message"])

    @patch('discord_bot.bot.api_request')
    async def test_check_signals_with_signal(self, mock_api_request):
        """Tester la fonction check_signals avec un signal de trading"""
        # Créer un contexte pour le test de check_signals
        signal_channels = {"BTC/USDT_1h": 12345}
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()
        
        # Configurer les mocks
        self.mock_bot.get_channel.return_value = mock_channel
        
        # Configurer le mock pour api_request
        mock_api_request.return_value = {
            "recommendation": "ACHETER",
            "timestamp": "2023-01-01T12:00:00",
            "market_data": {
                "price": 20000.0,
                "rsi": 30.0,
                "macd": 0.5
            }
        }
        
        # Appeler check_signals
        with patch('discord_bot.bot.bot', self.mock_bot):
            with patch.dict('discord_bot.bot.signal_channels', signal_channels, clear=True):
                await check_signals.__wrapped__()
        
        # Vérifier que send a été appelé pour envoyer le message d'alerte
        mock_channel.send.assert_called_once()
        
        # Vérifier que l'appel contenait un embed
        self.assertIsInstance(mock_channel.send.call_args[1]['embed'], discord.Embed)

    @patch('discord_bot.bot.api_request')
    async def test_check_signals_no_signal(self, mock_api_request):
        """Tester la fonction check_signals sans signal de trading"""
        # Créer un contexte pour le test de check_signals
        signal_channels = {"BTC/USDT_1h": 12345}
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()
        
        # Configurer les mocks
        self.mock_bot.get_channel.return_value = mock_channel
        
        # Configurer le mock pour api_request - pas de signal d'achat/vente
        mock_api_request.return_value = {
            "recommendation": "ATTENDRE",
            "timestamp": "2023-01-01T12:00:00",
            "market_data": {
                "price": 20000.0,
                "rsi": 50.0,
                "macd": 0.1
            }
        }
        
        # Appeler check_signals
        with patch('discord_bot.bot.bot', self.mock_bot):
            with patch.dict('discord_bot.bot.signal_channels', signal_channels, clear=True):
                await check_signals.__wrapped__()
        
        # Vérifier que send n'a pas été appelé (pas d'alerte pour "ATTENDRE")
        mock_channel.send.assert_not_called()

    @patch('discord_bot.bot.api_request')
    async def test_price_command(self, mock_api_request):
        """Tester la commande !prix"""
        # Importer la commande price pour le test
        from discord_bot.bot import price
        
        # Configurer le mock pour api_request
        mock_api_request.return_value = {
            "market_data": {
                "price": 20000.0
            }
        }
        
        # Appeler la commande price
        await price(self.mock_ctx, "BTC")
        
        # Vérifier que ctx.send a été appelé deux fois (message d'attente + résultat)
        self.assertEqual(self.mock_ctx.send.call_count, 2)
        
        # Vérifier que l'appel pour le résultat contenait un embed
        last_call = self.mock_ctx.send.call_args_list[-1]
        self.assertIn('embed', last_call[1])
        self.assertIsInstance(last_call[1]['embed'], discord.Embed)

    @patch('discord_bot.bot.api_request')
    async def test_predict_command(self, mock_api_request):
        """Tester la commande !predire"""
        # Importer la commande predict pour le test
        from discord_bot.bot import predict
        
        # Créer un mock pour le message d'attente
        mock_message = MagicMock()
        mock_message.edit = AsyncMock()
        self.mock_ctx.send.return_value = mock_message
        
        # Configurer le mock pour api_request
        mock_api_request.return_value = {
            "recommendation": "ACHETER",
            "timestamp": "2023-01-01T12:00:00",
            "confidence": 0.8,
            "market_data": {
                "price": 20000.0,
                "rsi": 30.0,
                "macd": 0.5
            }
        }
        
        # Appeler la commande predict
        await predict(self.mock_ctx, "BTC/USDT", "1h")
        
        # Vérifier que ctx.send a été appelé pour le message d'attente
        self.mock_ctx.send.assert_called_once()
        
        # Vérifier que message.edit a été appelé pour mettre à jour avec le résultat
        mock_message.edit.assert_called_once()
        
        # Vérifier que l'appel pour le résultat contenait un embed
        self.assertIn('embed', mock_message.edit.call_args[1])
        self.assertIsInstance(mock_message.edit.call_args[1]['embed'], discord.Embed)

    @patch('discord_bot.bot.api_request')
    async def test_backtest_command(self, mock_api_request):
        """Tester la commande !backtest"""
        # Importer la commande backtest pour le test
        from discord_bot.bot import backtest
        
        # Créer un mock pour le message d'attente
        mock_message = MagicMock()
        mock_message.edit = AsyncMock()
        self.mock_ctx.send.return_value = mock_message
        
        # Configurer le mock pour api_request
        mock_api_request.return_value = {
            "symbol": "BTC/USDT",
            "period": "2023-01-01 to 2023-01-30",
            "profit_pct": 5.2,
            "buy_hold_pct": 3.1,
            "sharpe_ratio": 1.2,
            "num_trades": 10,
            "win_rate": 0.6,
            "trades_summary": [
                {"profit_pct": 2.5, "win": True},
                {"profit_pct": -1.2, "win": False}
            ]
        }
        
        # Appeler la commande backtest
        await backtest(self.mock_ctx, "BTC/USDT", "1h", "30")
        
        # Vérifier que ctx.send a été appelé pour le message d'attente
        self.mock_ctx.send.assert_called_once()
        
        # Vérifier que message.edit a été appelé pour mettre à jour avec le résultat
        mock_message.edit.assert_called_once()
        
        # Vérifier que l'appel pour le résultat contenait un embed
        self.assertIn('embed', mock_message.edit.call_args[1])
        self.assertIsInstance(mock_message.edit.call_args[1]['embed'], discord.Embed)


if __name__ == '__main__':
    unittest.main() 