from typing import List, Union
from pathlib import Path
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
from ai_trading.utils.enhanced_preprocessor import EnhancedTextDataPreprocessor
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.agents.sac_agent import SACAgent

class MultiPeriodTrainer:
    def __init__(
        self,
        symbol: str,
        days: int,
        periods: List[int],
        agent_type: str = "sac",
        use_gru: bool = True,
        initial_balance: float = 10000,
        save_dir: Union[str, Path] = None,
        use_curriculum: bool = True,
        epochs_per_period: int = 10,
        episodes_per_epoch: int = 100,
        validation_ratio: float = 0.2,
        include_sentiment: bool = False,
    ):
        """
        Initialise le MultiPeriodTrainer.

        Args:
            symbol: Symbole de la cryptomonnaie
            days: Nombre de jours d'historique
            periods: Liste des périodes en minutes
            agent_type: Type d'agent à utiliser
            use_gru: Utiliser GRU ou non
            initial_balance: Solde initial
            save_dir: Dossier de sauvegarde
            use_curriculum: Utiliser l'apprentissage par curriculum
            epochs_per_period: Nombre d'époques par période
            episodes_per_epoch: Nombre d'épisodes par époque
            validation_ratio: Ratio de validation
            include_sentiment: Inclure les données de sentiment
        """
        self.symbol = symbol
        self.days = days
        self.periods = periods
        self.agent_type = agent_type
        self.use_gru = use_gru
        self.initial_balance = initial_balance
        self.save_dir = Path(save_dir) if save_dir else Path("models")
        self.use_curriculum = use_curriculum
        self.epochs_per_period = epochs_per_period
        self.episodes_per_epoch = episodes_per_epoch
        self.validation_ratio = validation_ratio
        self.include_sentiment = include_sentiment

        # Créer le répertoire de sauvegarde s'il n'existe pas
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def collect_data(self):
        """Collecte les données de marché et de sentiment."""
        collector = EnhancedDataCollector()
        market_data = collector.get_merged_price_data(self.symbol, days=self.days)

        sentiment_data = None
        if self.include_sentiment:
            text_preprocessor = EnhancedTextDataPreprocessor()
            sentiment_data = text_preprocessor.preprocess_social_data(collector.get_social_data())

        return market_data, sentiment_data

    def create_env(self, market_data, sentiment_data=None):
        """Crée l'environnement de trading."""
        env = TradingEnvironment(
            market_data=market_data,
            sentiment_data=sentiment_data,
            initial_balance=self.initial_balance,
            include_sentiment=self.include_sentiment
        )
        return env

    def prepare_datasets(self, market_data):
        """Prépare les ensembles d'entraînement et de validation."""
        split_idx = int(len(market_data) * (1 - self.validation_ratio))
        train_data = market_data[:split_idx]
        val_data = market_data[split_idx:]
        return train_data, val_data

    def save_agent(self, agent, name):
        """Sauvegarde l'agent."""
        save_path = self.save_dir / f"{name}.h5"
        agent.save(save_path)

    def load_agent(self, name):
        """Charge un agent sauvegardé."""
        load_path = self.save_dir / f"{name}.h5"
        agent = SACAgent()
        agent.load(load_path)
        return agent 