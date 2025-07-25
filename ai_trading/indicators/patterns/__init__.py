"""
Sous-module dédié à la détection de patterns chartistes avancés.

Ce package fournit:
- Reconnaissance de motifs chartistes par deep learning (CNN)
- Classification des formations techniques
- Quantification de la fiabilité des patterns
"""

from ai_trading.indicators.patterns.classical_patterns import (
    ClassicalPatternDetector,
    reliability_score,
)
from ai_trading.indicators.patterns.cnn_detector import (
    ChartPatternType,
    CNNPatternDetector,
    PatternInstance,
)

__all__ = [
    "CNNPatternDetector",
    "ChartPatternType",
    "PatternInstance",
    "ClassicalPatternDetector",
    "reliability_score",
]
