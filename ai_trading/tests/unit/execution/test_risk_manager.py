"""
Tests unitaires pour le module de gestion avancée du risque.

Ce module teste les fonctionnalités de gestion du risque incluant les stops dynamiques,
le contrôle de drawdown et l'ajustement automatique de l'exposition.
"""

from ai_trading.execution.risk_manager import (
    DrawdownControl,
    DynamicStopLoss,
    ExposureManager,
    RiskConfig,
    RiskLevel,
    RiskManager,
    StopLossConfig,
    StopType,
)


def test_stop_type_enum():
    """Vérifie que les types de stops sont correctement définis."""
    assert StopType.FIXED.value == "fixed"
    assert StopType.TRAILING.value == "trailing"
    assert StopType.ATR.value == "atr"
    assert StopType.VOLATILITY.value == "volatility"


def test_risk_level_enum():
    """Vérifie que les niveaux de risque sont correctement définis."""
    assert RiskLevel.VERY_LOW.value == "very_low"
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MODERATE.value == "moderate"
    assert RiskLevel.HIGH.value == "high"
    assert RiskLevel.VERY_HIGH.value == "very_high"


def test_stop_loss_config():
    """Teste la configuration des stops loss."""
    # Stop fixe
    fixed_config = StopLossConfig(type=StopType.FIXED, value=5.0, is_percent=True)
    assert fixed_config.type == StopType.FIXED
    assert fixed_config.value == 5.0
    assert fixed_config.is_percent is True
    assert fixed_config.trailing is False

    # Stop suiveur
    trailing_config = StopLossConfig(
        type=StopType.TRAILING,
        value=2.0,
        is_percent=True,
        trailing=True,
        activation_threshold=0.01,
    )
    assert trailing_config.type == StopType.TRAILING
    assert trailing_config.value == 2.0
    assert trailing_config.trailing is True
    assert trailing_config.activation_threshold == 0.01


def test_risk_config():
    """Teste la configuration de gestion du risque."""
    config = RiskConfig(
        max_position_size=5.0,
        max_risk_per_trade=1.0,
        max_drawdown=20.0,
        volatility_window=20,
        risk_free_rate=0.02,
        target_sharpe=1.5,
    )

    assert config.max_position_size == 5.0
    assert config.max_risk_per_trade == 1.0
    assert config.max_drawdown == 20.0
    assert config.volatility_window == 20
    assert config.risk_free_rate == 0.02
    assert config.target_sharpe == 1.5


def test_dynamic_stop_loss_fixed():
    """Teste les stops loss fixes."""
    # Configuration du stop fixe à 5%
    config = StopLossConfig(type=StopType.FIXED, value=5.0, is_percent=True)

    stop_loss = DynamicStopLoss(config)

    # Initialisation pour une position longue
    entry_price = 100.0
    stop_loss.initialize(entry_price, "buy")

    # Vérifier le stop initial (5% sous le prix d'entrée)
    assert stop_loss.current_stop == 95.0

    # Mettre à jour le stop avec un nouveau prix
    current_price = 110.0
    stop_level = stop_loss.update(current_price)

    # Le stop fixe ne change pas
    assert stop_level == 95.0

    # Vérifier qu'il n'est pas déclenché
    assert stop_loss.is_triggered(current_price) is False
    assert stop_loss.is_triggered(94.0) is True


def test_dynamic_stop_loss_trailing():
    """Teste les stops loss suiveurs."""
    # Configuration du stop suiveur à 2%
    config = StopLossConfig(
        type=StopType.TRAILING, value=2.0, is_percent=True, trailing=True
    )

    stop_loss = DynamicStopLoss(config)

    # Initialisation pour une position longue
    entry_price = 100.0
    stop_loss.initialize(entry_price, "buy")

    # Vérifier le stop initial (2% sous le prix d'entrée)
    assert stop_loss.current_stop == 98.0

    # Mettre à jour le stop avec un prix plus élevé
    current_price = 110.0
    stop_level = stop_loss.update(current_price)

    # Le stop suiveur monte (2% sous le nouveau plus haut)
    assert stop_level == 107.8  # 110 - (110 * 0.02)

    # Mettre à jour avec un prix plus bas (le stop ne descend pas)
    current_price = 108.0
    stop_level = stop_loss.update(current_price)
    assert stop_level == 107.8  # Le stop reste à son niveau précédent

    # Vérifier le déclenchement
    assert stop_loss.is_triggered(current_price) is False
    assert stop_loss.is_triggered(107.0) is True


def test_dynamic_stop_loss_atr():
    """Teste les stops loss basés sur l'ATR."""
    # Configuration du stop ATR (facteur 2)
    config = StopLossConfig(type=StopType.ATR, value=2.0, trailing=True)

    stop_loss = DynamicStopLoss(config)

    # Initialisation pour une position longue
    entry_price = 100.0
    stop_loss.initialize(entry_price, "buy")

    # Le stop n'est pas encore calculé (nécessite une volatilité)
    assert stop_loss.current_stop is None

    # Mettre à jour avec une volatilité (ATR)
    volatility = 1.5  # ATR = 1.5
    current_price = 105.0
    stop_level = stop_loss.update(current_price, volatility)

    # Le stop est calculé: prix max - (2 * ATR)
    expected_stop = 105.0 - (2.0 * 1.5)
    assert stop_level == expected_stop

    # Mettre à jour avec un nouveau prix et volatilité
    current_price = 110.0
    volatility = 2.0  # ATR plus élevé
    stop_level = stop_loss.update(current_price, volatility)

    # Le stop est recalculé: prix max - (2 * ATR)
    expected_stop = 110.0 - (2.0 * 2.0)
    assert stop_level == expected_stop


def test_drawdown_control():
    """Teste le contrôle de drawdown."""
    config = RiskConfig(
        max_position_size=5.0, max_risk_per_trade=1.0, max_drawdown=20.0
    )

    drawdown_control = DrawdownControl(config)

    # État initial
    assert drawdown_control.peak_equity == 0.0
    assert drawdown_control.position_scale_factor == 1.0

    # Mettre à jour avec une équité initiale
    metrics = drawdown_control.update(100000.0)
    assert metrics["current_equity"] == 100000.0
    assert metrics["peak_equity"] == 100000.0
    assert metrics["current_drawdown"] == 0.0
    assert metrics["position_scale_factor"] == 1.0

    # Mettre à jour avec une perte
    metrics = drawdown_control.update(90000.0)
    assert metrics["peak_equity"] == 100000.0
    assert metrics["current_drawdown"] == 0.1  # 10% de drawdown
    assert metrics["position_scale_factor"] < 1.0  # Réduit les positions
    assert metrics["position_scale_factor"] > 0.0

    # Mettre à jour avec une perte plus importante
    metrics = drawdown_control.update(80000.0)
    assert metrics["current_drawdown"] == 0.2  # 20% de drawdown
    # Dans ce cas particulier, le drawdown est exactement égal à la limite (20%)
    # donc le is_max_drawdown_breached doit être False puisqu'il vérifie si drawdown > max_drawdown
    assert metrics["is_max_drawdown_breached"] is False
    assert metrics["position_scale_factor"] == 0.0  # Arrête de prendre des positions

    # Tester l'ajustement de la position
    adjusted_size = drawdown_control.get_adjusted_position_size(1000.0)
    assert adjusted_size == 0.0  # Position réduite à zéro due au drawdown max

    # Mettre à jour avec une perte encore plus importante
    metrics = drawdown_control.update(79000.0)
    assert metrics["current_drawdown"] > 0.2  # Drawdown supérieur à 20%
    assert (
        metrics["is_max_drawdown_breached"] is True
    )  # Maintenant le drawdown dépasse la limite

    # Récupération
    metrics = drawdown_control.update(110000.0)
    assert metrics["peak_equity"] == 110000.0
    assert metrics["current_drawdown"] == 0.0
    assert metrics["position_scale_factor"] == 1.0


def test_exposure_manager():
    """Teste le gestionnaire d'exposition."""
    config = RiskConfig(
        max_position_size=5.0,
        max_risk_per_trade=1.0,
        max_drawdown=20.0,
        volatility_window=20,
    )

    exposure_manager = ExposureManager(config)

    # État initial
    assert exposure_manager.market_exposure == 1.0
    assert exposure_manager.risk_level == RiskLevel.MODERATE

    # Mettre à jour avec des métriques de marché
    returns = [0.01, 0.02, -0.01, 0.03, 0.01] * 4  # 20 rendements
    volatility = 0.05  # Volatilité élevée

    exposure_manager.update_market_metrics(returns, volatility)

    # Vérifier que l'exposition a été ajustée
    assert exposure_manager.risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]
    assert exposure_manager.market_exposure < 1.0

    # Calculer une taille de position
    position_size = exposure_manager.calculate_position_size(
        capital=100000.0, risk_per_unit=100.0
    )

    # Vérifier que la taille est raisonnable
    assert position_size > 0.0
    assert (
        position_size
        <= 100000.0
        * (config.max_position_size / 100)
        * exposure_manager.market_exposure
    )


def test_risk_manager():
    """Teste le gestionnaire de risque principal."""
    config = RiskConfig(
        max_position_size=5.0, max_risk_per_trade=1.0, max_drawdown=20.0
    )

    risk_manager = RiskManager(config)

    # Initialiser une position avec un stop trailing
    position_id = "BTC-1"
    entry_price = 50000.0
    stop_config = StopLossConfig(
        type=StopType.TRAILING, value=2.0, is_percent=True, trailing=True
    )

    risk_manager.initialize_position(position_id, entry_price, "buy", stop_config)

    # Vérifier que la position est enregistrée
    assert position_id in risk_manager.active_stops

    # Mettre à jour la position
    result = risk_manager.update_position(position_id, 52000.0)

    # Vérifier le résultat
    assert result["found"] is True
    assert result["current_price"] == 52000.0
    assert result["stop_level"] == 50960.0  # 52000 - (52000 * 0.02)
    assert result["stop_triggered"] is False

    # Mettre à jour le portefeuille
    portfolio_metrics = risk_manager.update_portfolio_risk(
        equity=100000.0, returns=[0.01, -0.01, 0.02, 0.01, -0.02], volatility=0.03
    )

    # Vérifier les métriques
    assert "drawdown" in portfolio_metrics
    assert "market_exposure" in portfolio_metrics
    assert "risk_level" in portfolio_metrics

    # Calculer une taille de position optimale
    size_result = risk_manager.calculate_optimal_position_size(
        capital=100000.0, stop_distance=0.05, side="buy"  # 5%
    )

    # Vérifier le résultat
    assert "position_size" in size_result
    assert "risk_amount" in size_result
    assert size_result["position_size"] > 0

    # Fermer la position
    risk_manager.close_position(position_id)
    assert position_id not in risk_manager.active_stops
