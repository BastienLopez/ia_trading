# Prompt pour le nettoyage du code et la gestion de la dépréciation

## Objectif
- Scanner tous les modules du projet (`ai_trading/`, `tests/`, `ai_trading/examples/`, etc.).
- Identifier et gérer correctement les fonctions, classes et méthodes obsolètes.
- Nettoyer le code tout en maintenant la rétrocompatibilité.

## Actions à réaliser

### 1. Identification des éléments obsolètes
- Rechercher les commentaires contenant "TODO", "DEPRECATED", "OBSOLETE", etc.
- Identifier les fonctions et classes dupliquées ou avec des noms similaires.
- Repérer les imports non utilisés et le code mort.

### 2. Stratégie de dépréciation
- Pour chaque élément obsolète, appliquer une stratégie de dépréciation:
  1. **Documenter** la dépréciation avec des docstrings appropriés.
  2. **Ajouter des avertissements** avec `warnings.warn()`.
  3. **Rediriger** vers les nouvelles implémentations.
  4. **Archiver** les anciens fichiers dans `clean_repo/ai_trading/examples/`.

### 3. Exemple de dépréciation de fonction
```python
import warnings
import functools

def deprecated(reason):
    """
    Décorateur pour marquer des fonctions comme dépréciées.
    
    Args:
        reason (str): Raison de la dépréciation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"La fonction {func.__name__} est dépréciée: {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        wrapper.__deprecated__ = True
        wrapper.__deprecated_reason__ = reason
        return wrapper
    return decorator

# Exemple d'utilisation
@deprecated("Utiliser new_function() à la place")
def old_function():
    """Fonction obsolète."""
    pass
```

### 4. Exemple de dépréciation de classe
```python
class OldClass:
    """
    Classe obsolète.
    
    .. deprecated:: 2.0.0
       Utiliser :class:`NewClass` à la place.
    """
    
    def __init__(self):
        warnings.warn(
            "La classe OldClass est dépréciée. Utiliser NewClass à la place.",
            category=DeprecationWarning,
            stacklevel=2
        )
```

### 5. Stratégie de nettoyage
- Créer un fichier `deprecated.py` dans chaque module pour regrouper les éléments obsolètes.
- Documenter clairement la date prévue de suppression (généralement après 2 versions majeures).
- Mettre à jour la documentation pour refléter ces changements.

## Résultat attendu
- Code plus propre avec une gestion claire des éléments obsolètes.
- Rétrocompatibilité maintenue pour les utilisateurs existants.
- Documentation mise à jour reflétant ces changements.
- Plan clair pour la suppression future des éléments obsolètes.
