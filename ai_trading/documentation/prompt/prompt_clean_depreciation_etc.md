Objectif : effectuer un nettoyage complet et structuré du dépôt AI Trading, incluant :

1. 🔧 Correction des dépréciations :
   - Scanner tous les modules du projet (`ai_trading/`, `tests/`, `examples/`, etc.).
   - Identifier et corriger les usages dépréciés de bibliothèques (e.g. NumPy, Pandas, PyTorch, TensorFlow, etc.).

2. ✅ Nettoyage des tests :
   - Supprimer les tests redondants, obsolètes ou inutiles dans `tests/`.
   - Garder uniquement les tests valides, pertinents et exécutables.
   - Mettre de côté les fichiers supprimés ou fusionnés dans `clean_repo/tests/`.

3. 📘 Nettoyage des exemples :
   - Supprimer ou corriger les notebooks/scripts non exécutables ou obsolètes.
   - Centraliser les exemples utiles et fonctionnels dans un sous-dossier clair.
   - Archiver les anciens fichiers dans `clean_repo/examples/`.

4. 🧼 Nettoyage automatisé :
   - Mettre a jour les fichier de clean (.\clean_repo\...)avec les avancés du projet 
   - Exécuter les scripts PowerShell fournis dans l’ordre recommandé :
     ```powershell
     .\clean_repo\clean_ai_trading_structure.ps1
     .\clean_repo\clean_ai_trading.ps1
     .\clean_repo\clean_repo.ps1
     ```

5. 🎨 Formatage automatique du code :
   - Supprimer les imports inutilisés :
     ```bash
     autoflake --in-place --remove-all-unused-imports --recursive ai_trading/
     ```
   - Réorganiser les imports :
     ```bash
     isort ai_trading/
     ```
   - Reformater tout le code :
     ```bash
     black ai_trading/
     ```

6. 📦 Résultats, PNG, rapports :
   - Tous les fichiers générés (rapports, PNG, graphiques, JSON, CSV, etc.) doivent être enregistrés exclusivement dans :
     ```
     ai_trading/info_retour/{tests|prod|examples}/
     ```
   - Modifier tous les scripts si nécessaire pour rediriger correctement les sorties.

7. ♻️ Nettoyage CUDA et modèles :
   - Utiliser `clean_cuda.py` pour nettoyer les ressources GPU, modèles obsolètes (>30j), et fichiers temporaires.

8. Correction des Warning : 
   - Effectuer les test et corriger les warning

Résultat attendu : un dépôt épuré, organisé, formaté, avec tous les fichiers anciens archivés dans `clean_repo/` et toutes les sorties bien redirigées dans `ai_trading/info_retour/`.
