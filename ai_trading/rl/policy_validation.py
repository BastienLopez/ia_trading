"""Garde-fous mesurables contre les politiques de trading dégénérées."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


class PolicyDiversityError(ValueError):
    """Levée quand une politique ne produit pas assez d'actions distinctes."""


def action_diversity_metrics(
    actions: Iterable[int | float],
    min_unique_actions: int = 2,
    max_dominant_share: float = 0.85,
    action_type: str = "discrete",
    n_discrete_actions: int = 5,
) -> dict:
    """Mesure la diversité exécutée sans confondre action et trade clôturé."""
    if action_type not in {"discrete", "continuous"}:
        raise ValueError("Type d'action inconnu")
    if action_type == "discrete":
        values = [int(action) for action in actions]
        directions = [
            "hold" if action == 0
            else "buy" if 1 <= action <= n_discrete_actions
            else "sell"
            for action in values
        ]
    else:
        # Les actions continues sont ramenées en vente / hold / achat pour
        # mesurer une diversité de décisions, pas une diversité numérique.
        values = [
            -1 if float(np.asarray(action).reshape(-1)[0]) < -0.05
            else 1 if float(np.asarray(action).reshape(-1)[0]) > 0.05
            else 0
            for action in actions
        ]
        directions = ["sell" if action < 0 else "buy" if action > 0 else "hold" for action in values]
    if min_unique_actions < 1 or n_discrete_actions < 1 or not 0 < max_dominant_share <= 1:
        raise ValueError("Paramètres de diversité invalides")
    counts = Counter(values)
    direction_counts = Counter(directions)
    total = len(values)
    dominant_share = max(counts.values()) / total if total else 1.0
    direction_dominant_share = max(direction_counts.values()) / total if total else 1.0
    numeric_passed = bool(total and len(counts) >= min_unique_actions and dominant_share <= max_dominant_share)
    directional_passed = bool(
        total and len(direction_counts) >= 2 and direction_dominant_share <= max_dominant_share
    )
    return {
        "action_count": total,
        "unique_action_count": len(counts),
        "dominant_action": int(max(counts, key=counts.get)) if counts else None,
        "dominant_action_share": float(dominant_share),
        "min_unique_actions": int(min_unique_actions),
        "max_dominant_share": float(max_dominant_share),
        # Conservé pour la sélection validation historique : plusieurs tailles
        # d'achat restent des actions distinctes, mais le rapport expose aussi
        # la diversité sémantique buy/hold/sell pour l'audit P4.
        "passed": numeric_passed,
        "directional_passed": directional_passed,
        "direction_counts": {direction: int(count) for direction, count in sorted(direction_counts.items())},
        "direction_dominant_share": float(direction_dominant_share),
        "counts": {str(action): int(count) for action, count in sorted(counts.items())},
    }


def require_action_diversity(
    actions: Iterable[int | float],
    min_unique_actions: int = 2,
    max_dominant_share: float = 0.85,
    action_type: str = "discrete",
) -> dict:
    """Retourne les métriques ou refuse explicitement une politique mono-action."""
    metrics = action_diversity_metrics(actions, min_unique_actions, max_dominant_share, action_type)
    if not metrics["passed"] or not metrics["directional_passed"]:
        raise PolicyDiversityError(
            "Politique dégénérée: "
            f"{metrics['unique_action_count']} action(s), {len(metrics['direction_counts'])} direction(s), "
            f"part dominante numérique {metrics['dominant_action_share']:.1%}, "
            f"directionnelle {metrics['direction_dominant_share']:.1%}"
        )
    return metrics
