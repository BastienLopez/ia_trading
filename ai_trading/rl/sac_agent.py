class SACAgent:
    """Version simplifiée du SACAgent pour les tests."""

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.001,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def train(self, env):
        """Simule l'entraînement pour les tests."""
        return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}

    def evaluate(self, env):
        """Simule l'évaluation pour les tests."""
        return {"sharpe_ratio": 1.0, "max_drawdown": -0.1, "profit_pct": 0.05}
