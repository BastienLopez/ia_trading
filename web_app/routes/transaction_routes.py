from flask import Blueprint, request, jsonify
import logging
from web_app.services.transaction_service import TransactionService
from web_app.services.price_service import PriceService

logger = logging.getLogger(__name__)
transaction_bp = Blueprint('transactions', __name__)

@transaction_bp.route('/api/transactions', methods=['GET', 'POST', 'PUT', 'DELETE'])
def handle_transactions():
    try:
        if request.method == 'GET':
            data = TransactionService.load_data()
            logger.info(f"Récupération des transactions: {json.dumps(data['transactions'], indent=2)}")
            return jsonify(data["transactions"])
            
        elif request.method == 'POST':
            try:
                transaction = request.json
                logger.info(f"Nouvelle transaction reçue: {json.dumps(transaction, indent=2)}")
                
                # Valider la transaction
                TransactionService.validate_transaction(transaction)
                
                # Récupérer le prix actuel
                current_price = PriceService.get_price_for_symbol(
                    transaction['symbol'],
                    TransactionService.load_data()['transactions']
                )
                
                # Ajouter la transaction
                TransactionService.add_transaction(transaction, current_price)
                
                logger.info(f"Transaction ajoutée avec succès")
                return jsonify({"success": True})
                
            except ValueError as e:
                logger.error(f"Erreur de validation: {str(e)}")
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la transaction: {str(e)}")
                return jsonify({"error": str(e)}), 500
            
        elif request.method == 'DELETE':
            try:
                delete_data = request.json
                if not delete_data or 'timestamp' not in delete_data:
                    logger.error("Données de suppression invalides")
                    return jsonify({"error": "Timestamp manquant"}), 400
                
                timestamp = float(delete_data['timestamp'])
                logger.info(f"Tentative de suppression de la transaction avec timestamp: {timestamp}")
                
                # Supprimer la transaction
                TransactionService.delete_transaction(timestamp)
                
                logger.info("Transaction supprimée avec succès")
                return jsonify({"success": True})
                
            except ValueError as e:
                logger.error(f"Erreur de format: {str(e)}")
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error(f"Erreur lors de la suppression: {str(e)}")
                return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Erreur lors de la gestion des transactions: {str(e)}")
        return jsonify({"error": "Erreur lors de la gestion des transactions"}), 500 