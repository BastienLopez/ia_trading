import json
import logging
import os
from datetime import datetime
from web_app.config import Config

logger = logging.getLogger(__name__)

class TransactionService:
    def __init__(self, price_service, data_dir=None):
        self.price_service = price_service
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'info_retour', 'data')
        self.transactions_file = os.path.join(self.data_dir, 'transactions.json')
        
        # Créer le dossier data s'il n'existe pas
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "info_retour/logs"), exist_ok=True)
            
        # Créer le fichier s'il n'existe pas
        if not os.path.exists(self.transactions_file):
            with open(self.transactions_file, 'w') as f:
                json.dump([], f)
                
        logger.info(f"Fichier de transactions configuré: {self.transactions_file}")

        self.load_transactions()

    def load_transactions(self):
        """Charge les transactions depuis le fichier JSON"""
        try:
            with open(self.transactions_file, 'r') as f:
                self.transactions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.transactions = []
            self.save_transactions()

    def save_transactions(self):
        """Sauvegarde les transactions dans le fichier JSON"""
        with open(self.transactions_file, 'w') as f:
            json.dump(self.transactions, f, indent=2)

    @staticmethod
    def validate_transaction(transaction):
        """Valide les données d'une transaction"""
        required_fields = ['symbol', 'action', 'amount']
        missing_fields = [field for field in required_fields if field not in transaction]
        if missing_fields:
            raise ValueError(f"Champs manquants: {', '.join(missing_fields)}")

        try:
            amount = float(transaction['amount'])
            if amount <= 0:
                raise ValueError("Le montant doit être positif")
        except ValueError as e:
            raise ValueError(f"Montant invalide: {str(e)}")

        if transaction['symbol'] not in Config.SUPPORTED_CRYPTOS:
            raise ValueError(f"Symbole non supporté: {transaction['symbol']}")

    def add_transaction(self, transaction_data):
        """Ajoute une nouvelle transaction"""
        try:
            symbol = transaction_data['symbol']
            action = transaction_data['action']
            amount = float(transaction_data['amount'])
            
            # Utiliser le prix fourni ou récupérer le prix actuel
            if transaction_data.get('price'):
                price = float(transaction_data['price'])
            else:
                price = self.price_service.get_price_for_symbol(symbol)
                if price is None:
                    return {'success': False, 'error': f"Impossible de récupérer le prix pour {symbol}"}

            timestamp = datetime.now().isoformat()
            
            transaction = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'amount': amount,
                'price': price
            }
            
            self.transactions.append(transaction)
            self.save_transactions()
            
            return {
                'success': True,
                'transaction': transaction
            }
            
        except KeyError as e:
            return {
                'success': False,
                'error': f"Champ manquant: {str(e)}"
            }
        except ValueError as e:
            return {
                'success': False,
                'error': f"Erreur de format: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Erreur inattendue: {str(e)}"
            }

    def get_transactions(self):
        """Récupère toutes les transactions"""
        return self.transactions

    def delete_transaction(self, timestamp):
        """Supprime une transaction par son timestamp"""
        initial_length = len(self.transactions)
        self.transactions = [t for t in self.transactions if t['timestamp'] != timestamp]
        
        if len(self.transactions) < initial_length:
            self.save_transactions()
            return {'success': True}
        else:
            return {
                'success': False,
                'error': "Transaction non trouvée"
            } 