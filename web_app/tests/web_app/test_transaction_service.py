import pytest
from web_app.services.transaction_service import TransactionService
from web_app.services.price_service import PriceService
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

def test_add_transaction(transaction_service):
    """Test d'ajout d'une transaction"""
    with patch.object(transaction_service.price_service, 'get_price_for_symbol') as mock_get_price:
        mock_get_price.return_value = 50000.0
        
        transaction = {
            'symbol': 'BTC',
            'action': 'ACHETER',
            'amount': 1.0,
            'price': None  # Prix automatique
        }
        
        result = transaction_service.add_transaction(transaction)
        assert result['success'] is True
        assert result['transaction']['symbol'] == 'BTC'
        assert result['transaction']['price'] == 50000.0

def test_add_transaction_manual_price(transaction_service):
    """Test d'ajout d'une transaction avec prix manuel"""
    transaction = {
        'symbol': 'BTC',
        'action': 'ACHETER',
        'amount': 1.0,
        'price': 45000.0  # Prix manuel
    }
    
    result = transaction_service.add_transaction(transaction)
    assert result['success'] is True
    assert result['transaction']['symbol'] == 'BTC'
    assert result['transaction']['price'] == 45000.0

def test_get_transactions(transaction_service):
    """Test de récupération des transactions"""
    # Ajouter quelques transactions de test
    transactions = [
        {
            'symbol': 'BTC',
            'action': 'ACHETER',
            'amount': 1.0,
            'price': 50000.0
        },
        {
            'symbol': 'ETH',
            'action': 'ACHETER',
            'amount': 2.0,
            'price': 3000.0
        }
    ]
    
    for tx in transactions:
        transaction_service.add_transaction(tx)
    
    result = transaction_service.get_transactions()
    assert len(result) >= 2
    assert any(tx['symbol'] == 'BTC' for tx in result)
    assert any(tx['symbol'] == 'ETH' for tx in result)

def test_delete_transaction(transaction_service):
    """Test de suppression d'une transaction"""
    # Ajouter une transaction
    transaction = {
        'symbol': 'BTC',
        'action': 'ACHETER',
        'amount': 1.0,
        'price': 50000.0
    }
    add_result = transaction_service.add_transaction(transaction)
    timestamp = add_result['transaction']['timestamp']
    
    # Supprimer la transaction
    delete_result = transaction_service.delete_transaction(timestamp)
    assert delete_result['success'] is True
    
    # Vérifier que la transaction a été supprimée
    transactions = transaction_service.get_transactions()
    assert not any(tx['timestamp'] == timestamp for tx in transactions)

def test_delete_nonexistent_transaction(transaction_service):
    """Test de suppression d'une transaction inexistante"""
    result = transaction_service.delete_transaction('nonexistent_timestamp')
    assert result['success'] is False
    assert 'error' in result 