"""Controles de qualite reutilisables pour les DataFrame OHLCV.

Le module ne depend pas de pandas au chargement afin de pouvoir etre verifie
dans les outils legers. Il accepte tout objet expose comme un DataFrame.
"""

from __future__ import annotations


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataQualityError(ValueError):
    """Les donnees ne sont pas exploitables pour features ou validation."""


def validate_ohlcv_frame(frame: object) -> None:
    """Valide les invariants minimaux avant calcul d'indicateurs.

    La fonction leve une erreur explicite plutot que de laisser un calcul de
    feature ou un split temporel produire silencieusement un resultat invalide.
    """

    if len(frame) == 0:  # type: ignore[arg-type]
        raise MarketDataQualityError("Le jeu OHLCV est vide.")

    columns = set(frame.columns)  # type: ignore[attr-defined]
    missing = set(REQUIRED_OHLCV_COLUMNS) - columns
    if missing:
        raise MarketDataQualityError(f"Colonnes OHLCV manquantes: {sorted(missing)}")

    index = frame.index  # type: ignore[attr-defined]
    if not index.is_monotonic_increasing:
        raise MarketDataQualityError("L'index temporel OHLCV doit etre croissant.")
    if index.has_duplicates:
        raise MarketDataQualityError("L'index temporel OHLCV contient des doublons.")

    for column in REQUIRED_OHLCV_COLUMNS:
        values = frame[column]  # type: ignore[index]
        if values.isna().any():
            raise MarketDataQualityError(f"La colonne OHLCV {column!r} contient des valeurs manquantes.")
        if not values.map(lambda value: isinstance(value, (int, float))).all():
            raise MarketDataQualityError(f"La colonne OHLCV {column!r} contient une valeur non numerique.")

    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():  # type: ignore[index]
        raise MarketDataQualityError("high doit etre superieur ou egal a open, close et low.")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():  # type: ignore[index]
        raise MarketDataQualityError("low doit etre inferieur ou egal a open, close et high.")
    if (frame["volume"] < 0).any():  # type: ignore[index]
        raise MarketDataQualityError("volume ne peut pas etre negatif.")
