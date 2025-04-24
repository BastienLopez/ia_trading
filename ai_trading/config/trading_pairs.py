from typing import Dict, List

# Configuration des exchanges et des paires disponibles
AVAILABLE_EXCHANGES = {
    'binance': {
        'crypto': ['BTC/USDT', 'ETH/USDT'],
        'api_key': '',  # À remplir avec votre clé API
        'api_secret': ''  # À remplir avec votre secret API
    },
    'kraken': {
        'crypto': ['BTC/USD', 'ETH/USD'],
        'api_key': '',
        'api_secret': ''
    },
    'oanda': {  # Pour l'or
        'forex': ['XAU/USD'],  # XAU/USD = Or en once
        'api_key': '',
        'account_id': ''
    }
}

# Configuration par défaut des paires de trading
DEFAULT_TRADING_PAIRS = [
    {'exchange': 'binance', 'symbol': 'BTC/USDT', 'limit': 100},
    {'exchange': 'binance', 'symbol': 'ETH/USDT', 'limit': 100},
    {'exchange': 'kraken', 'symbol': 'BTC/USD', 'limit': 100},
    {'exchange': 'kraken', 'symbol': 'ETH/USD', 'limit': 100},
    {'exchange': 'oanda', 'symbol': 'XAU/USD', 'limit': 50}
]

def get_trading_pairs(selected_pairs: List[str] = None) -> List[Dict]:
    """
    Retourne la configuration des paires de trading sélectionnées.
    
    Args:
        selected_pairs (List[str]): Liste des paires souhaitées (ex: ['BTC/USDT', 'ETH/USD'])
        
    Returns:
        List[Dict]: Configuration des paires sélectionnées
    """
    if not selected_pairs:
        return DEFAULT_TRADING_PAIRS
        
    selected_config = []
    for pair in selected_pairs:
        for exchange, config in AVAILABLE_EXCHANGES.items():
            if pair in config.get('crypto', []) or pair in config.get('forex', []):
                selected_config.append({
                    'exchange': exchange,
                    'symbol': pair,
                    'limit': 100 if exchange != 'oanda' else 50
                })
                break
                
    return selected_config or DEFAULT_TRADING_PAIRS 