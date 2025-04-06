import pytest
from web_app.services.price_service import PriceService
from unittest.mock import patch, MagicMock

@pytest.fixture
def price_service():
    return PriceService()

def test_get_real_crypto_prices(price_service):
    """Test de récupération des prix réels des cryptomonnaies"""
    with patch('requests.get') as mock_get:
        # Simuler une réponse réussie de l'API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bitcoin': {'usd': 50000.0},
            'ethereum': {'usd': 3000.0}
        }
        mock_get.return_value = mock_response

        prices = price_service.get_real_crypto_prices(['BTC', 'ETH'])
        assert prices is not None
        assert 'BTC' in prices
        assert 'ETH' in prices
        assert prices['BTC'] == 50000.0
        assert prices['ETH'] == 3000.0

def test_get_real_crypto_prices_api_error(price_service):
    """Test de gestion d'erreur de l'API"""
    with patch('requests.get') as mock_get:
        # Simuler une erreur de l'API
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        prices = price_service.get_real_crypto_prices(['BTC', 'ETH'])
        assert prices is None

def test_get_price_for_symbol(price_service):
    """Test de récupération du prix pour un symbole spécifique"""
    with patch.object(price_service, 'get_real_crypto_prices') as mock_get_prices:
        mock_get_prices.return_value = {
            'BTC': 50000.0,
            'ETH': 3000.0
        }

        price = price_service.get_price_for_symbol('BTC')
        assert price == 50000.0

def test_get_price_for_symbol_not_found(price_service):
    """Test de gestion d'un symbole non trouvé"""
    with patch.object(price_service, 'get_real_crypto_prices') as mock_get_prices:
        mock_get_prices.return_value = {
            'BTC': 50000.0,
            'ETH': 3000.0
        }

        price = price_service.get_price_for_symbol('INVALID')
        assert price is None 