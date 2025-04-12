class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, lookback_window=30):
        self.data = data
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        # ... reste du code ... 