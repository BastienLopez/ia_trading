import pytest
import os
import shutil
from web_app.services.price_service import PriceService
from web_app.services.transaction_service import TransactionService

@pytest.fixture
def price_service():
    return PriceService()

@pytest.fixture
def transaction_service(price_service):
    service = TransactionService(price_service)
    yield service
    
    # Nettoyage après chaque test
    if os.path.exists('data'):
        shutil.rmtree('data') 