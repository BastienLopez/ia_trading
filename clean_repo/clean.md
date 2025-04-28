# Guide de Nettoyage du Repository

Ce guide explique comment utiliser les scripts de nettoyage pour maintenir le repository propre et organisé.

## Scripts Disponibles

1. `clean_repo.ps1` - Nettoyage global du repository
2. `clean_ai_trading.ps1` - Nettoyage spécifique du dossier ai_trading
3. `clean_ai_trading_structure.ps1` - Nettoyage de la structure interne de ai_trading

## Ordre d'Exécution Recommandé

Pour un nettoyage complet, exécutez les scripts dans l'ordre suivant :

```powershell
# 1. Nettoyer la structure interne de ai_trading
.\clean_repo\clean_ai_trading_structure.ps1

# 2. Nettoyer le dossier ai_trading
.\clean_repo\clean_ai_trading.ps1

# 3. Nettoyer l'ensemble du repository
.\clean_repo\clean_repo.ps1
```

## Ce que font les Scripts

### clean_ai_trading_structure.ps1
- Supprime les dossiers vides ou inutiles dans ai_trading
- Nettoie les fichiers de cache Python
- Supprime les fichiers obsolètes
- Réorganise la structure si nécessaire

### clean_ai_trading.ps1
- Supprime les sous-dossiers redondants
- Nettoie les fichiers de cache
- Fusionne les fichiers .env si nécessaire
- Supprime les checkpoints Jupyter

### clean_repo.ps1
- Nettoie les dossiers temporaires à la racine
- Supprime les fichiers de cache et de logs
- Conserve les dossiers et fichiers essentiels
- Nettoie les fichiers de cache Python

## Dossiers et Fichiers Conservés

Les scripts conservent toujours :
- `ai_trading/` (code principal)
- `web_app/` (frontend)
- `tradingview/` (scripts Pine)
- `data/` (données)
- `tests/` (tests)
- `clean_repo/` (scripts de nettoyage)
- `.gitignore`
- `requirements.txt`
- `README.md`
- `setup_env.bat`

## Notes Importantes

1. Assurez-vous d'avoir les permissions nécessaires pour exécuter les scripts
2. Les scripts sont conçus pour Windows PowerShell
3. Les scripts incluent une gestion des erreurs et des logs détaillés
4. Les fichiers de configuration importants ne sont jamais supprimés

## En Cas d'Erreur

Si un script rencontre une erreur :
1. Vérifiez les logs affichés dans la console
2. Assurez-vous que vous êtes dans le bon répertoire
3. Vérifiez que le repository Git est correctement initialisé
4. Contactez l'administrateur si le problème persiste 