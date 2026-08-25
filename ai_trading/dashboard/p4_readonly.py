"""Vue dashboard P4 strictement informative : aucune action de trading."""

from __future__ import annotations

from typing import Any, Dict

from ai_trading.llm.predictions.p4_registry import registry


def p4_snapshot(asset: str, timeframe: str) -> Dict[str, Any]:
    prediction = registry.prediction(asset, timeframe)
    if prediction is None:
        return {"available": False, "asset": asset.upper(), "timeframe": timeframe, "trading_enabled": False}
    return {"available": True, "asset": prediction["asset"], "timeframe": prediction["timeframe"],
            "direction": prediction.get("direction"), "confidence": prediction.get("confidence"),
            "abstain": prediction.get("abstain"), "as_of": prediction.get("as_of"),
            "freshness_seconds": prediction.get("sentiment_freshness_seconds"),
            "uncertainty": prediction.get("uncertainty"), "metrics": prediction.get("performance_metrics"),
            "market_safety": prediction.get("market_safety"), "trading_enabled": False}


def build_p4_readonly_layout(asset: str, timeframe: str):
    """Construit un composant Dash seulement si le dashboard est lancé."""
    from dash import html
    data = p4_snapshot(asset, timeframe)
    return html.Div([
        html.H3(f"P4 — {data['asset']} {data['timeframe']}"),
        html.P(f"Direction: {data.get('direction', 'indisponible')}"),
        html.P(f"Confiance: {data.get('confidence', 'indisponible')}"),
        html.P(f"Abstention: {data.get('abstain', True)}"),
        html.P("Lecture seule — trading réel désactivé."),
    ], id="p4-readonly-panel")
