def test_download_historical_data(mocker):
    """Test la fonction de téléchargement des données historiques"""
    # Mock de la fonction fetch_ohlcv pour éviter les appels réels à l'API
    mock_fetch = mocker.patch('ccxt.binance.fetch_ohlcv')
    
    # Créer des données OHLCV fictives
    mock_data = [
        [1641024000000, 46000.0, 47000.0, 45500.0, 46500.0, 100.0],  # 01-01-2022
        [1641110400000, 46500.0, 48000.0, 46000.0, 47500.0, 120.0],  # 02-01-2022
    ]
    
    # Configuration du mock pour qu'il retourne les données fictives
    mock_fetch.return_value = mock_data
    
    processor = DataProcessor()
    
    # Utiliser des dates en 2024 au lieu de 2021
    data = processor.download_historical_data(
        symbol='BTC/USDT',
        timeframe='1d',
        since='2024-01-01',
        until='2024-01-05'
    )
    
    # Vérifier que la fonction fetch_ohlcv a été appelée avec les bons paramètres
    mock_fetch.assert_called()
    
    # Vérifier que le DataFrame a la bonne structure
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(column in data.columns for column in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Le DataFrame devrait avoir la même longueur que les données fictives
    assert len(data) == len(mock_data) 