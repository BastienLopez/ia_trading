### 🎯 Objectif :
Faire un audit et un ménage du dépôt AI Trading pour :
1. **Identifier et supprimer les éléments obsolètes, redondants ou inutiles** dans les tests, les exemples et la production.
2. **S'assurer que tous les fichiers de sortie** (rapports, PNG, graphiques, logs, etc.) générés par les tests, les exemples et la production **soient enregistrés dans `ai_trading/info_retour/...`**, dans des sous-dossiers correspondant au contexte (test, prod, exemple, optimisation, etc.).

---

### 🧠 Prompt pour AI ou tâche automatisée :
```plaintext
Analyse ce dépôt AI Trading :

1. Passe en revue tous les dossiers de tests (`tests/`), d'exemples (`ai_trading/examples/`), de production (`ai_trading/` ou équivalents) et d'optimisations.
2. Vérifie s'il existe des fichiers, fonctions, classes, ou tests obsolètes, redondants ou inutilisés. Supprime les
3. Identifie tous les scripts ou modules qui génèrent des fichiers de sortie (rapports, images, logs, graphes, etc.).
4. S'assure que tous ces fichiers de sortie sont systématiquement enregistrés dans `ai_trading/info_retour/`, dans des sous-dossiers selon leur origine :
   - `info_retour/tests/` pour les fichiers générés par les tests
   - `info_retour/examples/` pour les exemples et démos
   - `info_retour/prod/` pour les scripts en production
   - `info_retour/optimisation/` pour les logs d'optimisations, profiling, etc.
   - `info_retour/visualisations/` pour les graphiques et visualisations
   - `info_retour/models/` pour les modèles entraînés
   - `info_retour/checkpoints/` pour les checkpoints des modèles
   - `info_retour/data/` pour les données générées/transformées avec sous-dossiers appropriés
   - `info_retour/logs/` pour tous les fichiers de logs
5. Vérifie et corrige tous les emplacements où des chemins sont définis, en particulier :
   - Les variables de chemins en dur ou chemins absolus
   - Les appels à `os.makedirs`, `os.mkdir`, `Path().mkdir()`
   - Les sauvegardes de fichiers avec `to_csv`, `to_json`, `save`, `savefig`, etc.
   - Les chemins définis pour Ray, DeepSpeed ou autres frameworks
   - Les emplacements de logs, checkpoints, et résultats d'optimisation
6. Utilise des chemins relatifs basés sur la position du fichier, par exemple :
   ```python
   # Bonne pratique de définition de chemins
   import os
   from pathlib import Path
   
   # À mettre en début de fichier
   # Option 1: Définir le chemin de base à partir du fichier courant
   BASE_DIR = Path(__file__).parent.parent  # Remonte au répertoire racine ai_trading
   INFO_RETOUR_DIR = BASE_DIR / "info_retour"
   
   # Option 2: Pour un script dans examples/
   EXAMPLE_OUTPUT_DIR = Path(__file__).parent.parent / "info_retour" / "examples" / "nom_exemple"
   
   # Toujours créer les répertoires avant utilisation
   os.makedirs(EXAMPLE_OUTPUT_DIR, exist_ok=True)
   ```
7. Assure-toi que RIEN n'est jamais écrit à la racine du projet ou en dehors de `ai_trading/info_retour/`
8. Vérifie particulièrement les modules suivants qui génèrent souvent des fichiers :
   - Les scripts d'exemples dans `ai_trading/examples/`
   - Les scripts d'optimisation comme Ray Tune et Ray RLlib
   - Les modules de visualisation et d'évaluation
   - Les modules d'entraînement de modèles

Résultat attendu : 
- Un dépôt propre et bien structuré
- Une séparation claire entre code et fichiers de sortie
- Tous les fichiers générés sont dans `ai_trading/info_retour/` avec une organisation cohérente
- Aucun fichier généré à la racine du projet ou dans d'autres répertoires
```

### 📊 Structure attendue des fichiers de sortie :

```
ai_trading/
├── info_retour/
│   ├── tests/                  # Sorties des tests
│   ├── examples/               # Sorties des exemples 
│   │   ├── ray_tune/           # Sorties spécifiques à chaque exemple
│   │   ├── sentiment/
│   │   └── ...
│   ├── prod/                   # Sorties de production
│   ├── optimisation/           # Sorties d'optimisation
│   │   ├── profiling/          # Résultats de profiling
│   │   ├── ray_checkpoints/    # Checkpoints Ray
│   │   └── ...
│   ├── visualisations/         # Graphiques et visualisations
│   │   ├── sentiment/
│   │   ├── performance/
│   │   └── ...
│   ├── models/                 # Modèles entraînés
│   ├── checkpoints/            # Points de sauvegarde
│   ├── data/                   # Données transformées
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── sentiment/
│   │   └── ...
│   └── logs/                   # Tous les logs
```
