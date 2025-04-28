# Guide d'utilisation de CUDA dans le projet

Ce document explique comment configurer, tester et nettoyer l'environnement CUDA pour le projet.

## Configuration de l'environnement

Pour configurer l'environnement CUDA correctement, utilisez le script `setup_env.bat` :

```powershell
.\setup_env.bat
```

Ce script va :
- Configurer les variables d'environnement nécessaires
- Activer CUDA pour PyTorch et TensorFlow
- Vérifier que les GPUs sont correctement détectés

## Exécution des tests

### Tests standard

Pour exécuter les tests standards :

```powershell
make test
```

Cette commande lance tous les tests dans le répertoire `ai_trading/tests/`.

### Tests avec nettoyage CUDA automatique

Pour exécuter les tests avec nettoyage automatique des ressources CUDA :

```powershell
make test-clean
```

Cette commande exécute les tests puis nettoie automatiquement la mémoire CUDA.

## Nettoyage des ressources CUDA

### Nettoyage simple

Pour nettoyer uniquement les ressources CUDA :

```powershell
make clean-cuda
```

Cette commande libère la mémoire CUDA de PyTorch et TensorFlow, et force le garbage collector.

### Nettoyage manuel avec le script Python

Vous pouvez également utiliser directement le script de nettoyage :

```powershell
python scripts/clean_cuda.py
```

Ce script effectue :
1. Le nettoyage de la mémoire CUDA pour PyTorch
2. La libération des ressources GPU pour TensorFlow
3. L'exécution forcée du garbage collector

### Nettoyage complet

Pour effectuer un nettoyage complet (fichiers temporaires + ressources CUDA) :

```powershell
make deep-clean
```

## Dépannage

### Vérification de l'état des GPUs

Pour vérifier l'état des GPUs NVIDIA :

```powershell
nvidia-smi
```

### Problèmes courants

1. **TensorFlow ne détecte pas la GPU** : Redémarrez l'environnement avec `setup_env.bat`
2. **Erreurs de mémoire insuffisante** : Exécutez `make clean-cuda` pour libérer la mémoire
3. **Tests qui échouent avec des erreurs CUDA** : Essayez `make deep-clean` puis réexécutez les tests

## Bonnes pratiques

- Toujours exécuter `clean-cuda` après de longs sessions de tests
- Utiliser `test-clean` au lieu de `test` pour des exécutions automatisées
- Si vous rencontrez des problèmes de stabilité, exécutez `deep-clean` avant de redémarrer vos tests 