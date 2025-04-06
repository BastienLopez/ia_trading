import pytest
from web_app.app import app
import json
from datetime import datetime

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test de la route principale"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Tableau de bord' in response.data

def test_dashboard_route(client):
    """Test de la route du tableau de bord"""
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'Portefeuille' in response.data
    assert b'Performance globale' in response.data

def test_add_transaction(client):
    """Test d'ajout d'une transaction"""
    transaction = {
        'symbol': 'BTC',
        'action': 'ACHETER',
        'amount': 1.0,
        'price': 50000.0
    }
    response = client.post('/api/transactions', 
                         data=json.dumps(transaction),
                         content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True

def test_get_transactions(client):
    """Test de récupération des transactions"""
    response = client.get('/api/transactions')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_delete_transaction(client):
    """Test de suppression d'une transaction"""
    # D'abord ajouter une transaction
    transaction = {
        'symbol': 'ETH',
        'action': 'ACHETER',
        'amount': 1.0,
        'price': 3000.0
    }
    add_response = client.post('/api/transactions', 
                             data=json.dumps(transaction),
                             content_type='application/json')
    assert add_response.status_code == 200
    
    # Récupérer les transactions pour avoir le timestamp
    get_response = client.get('/api/transactions')
    transactions = json.loads(get_response.data)
    assert len(transactions) > 0
    
    # Supprimer la transaction
    timestamp = transactions[0]['timestamp']
    delete_response = client.delete('/api/transactions',
                                  data=json.dumps({'timestamp': timestamp}),
                                  content_type='application/json')
    assert delete_response.status_code == 200
    data = json.loads(delete_response.data)
    assert data['success'] is True 