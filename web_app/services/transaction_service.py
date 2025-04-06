import json
import logging
import os
from datetime import datetime
from web_app.config import Config

logger = logging.getLogger(__name__)

class TransactionService:
    @staticmethod
    def load_data():
        """Charge les données depuis le fichier JSON"""
        try:
            if os.path.exists(Config.DATA_FILE):
                with open(Config.DATA_FILE, 'r') as f:
                    data = json.load(f)
                    if "transactions" not in data:
                        data["transactions"] = []
                    if "portfolio" not in data:
                        data["portfolio"] = {"current_value": 0, "starting_value": 0, "assets": []}
                    return data
            return {"transactions": [], "portfolio": {"current_value": 0, "starting_value": 0, "assets": []}}
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            return {"transactions": [], "portfolio": {"current_value": 0, "starting_value": 0, "assets": []}}

    @staticmethod
    def save_data(data):
        """Sauvegarde les données dans le fichier JSON"""
        try:
            with open(Config.DATA_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")
            raise

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

    @staticmethod
    def add_transaction(transaction, current_price):
        """Ajoute une nouvelle transaction"""
        data = TransactionService.load_data()
        
        # Utiliser le prix fourni ou le prix actuel du marché
        price = transaction.get('price', current_price)
        
        transaction["timestamp"] = datetime.now().timestamp()
        transaction["price"] = price
        transaction["value"] = float(transaction['amount']) * price
        transaction["date"] = datetime.now().isoformat()
        
        data["transactions"].append(transaction)
        TransactionService.save_data(data)
        
        return transaction

    @staticmethod
    def delete_transaction(timestamp):
        """Supprime une transaction"""
        data = TransactionService.load_data()
        
        initial_length = len(data['transactions'])
        data['transactions'] = [t for t in data['transactions'] if t['timestamp'] != timestamp]
        
        if len(data['transactions']) == initial_length:
            raise ValueError("Transaction non trouvée")
        
        TransactionService.save_data(data)
        return True 