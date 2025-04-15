# Script pour nettoyer la structure du dossier ai_trading

# Dossiers à supprimer car vides ou inutiles
$emptyDirsToRemove = @(
    "ai_trading/environments",   # Dossier vide
    "ai_trading/agents",         # Dossier vide, remplacé par ai_trading/rl/agents
    "ai_trading/config",         # Dossier vide, remplacé par ai_trading/config.py
    "ai_trading/__pycache__",    # Fichiers de cache Python
    "ai_trading/utils/__pycache__",
    "ai_trading/llm/sentiment_analysis/__pycache__",
    "ai_trading/rl/__pycache__",
    "ai_trading/rl/agents/__pycache__"
)

# Fichiers à supprimer car obsolètes ou remplacés
$filesToRemove = @(
    "ai_trading/download_nltk_data.py",   # Remplacé par des appels directs dans le code ou dans le workflow GitHub Actions
    "ai_trading/utils/data_collector.py", # Remplacé par enhanced_data_collector.py
    "ai_trading/utils/rl_data_integrator.py"  # Semble être un fichier vide ou obsolète
)

Write-Host "Nettoyage de la structure du dossier ai_trading..."

# Supprimer les dossiers vides ou inutiles
foreach ($dir in $emptyDirsToRemove) {
    if (Test-Path $dir) {
        Write-Host "Suppression du dossier: $dir"
        Remove-Item -Path $dir -Recurse -Force
    } else {
        Write-Host "Le dossier $dir n'existe pas."
    }
}

# Supprimer les fichiers obsolètes
foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Write-Host "Suppression du fichier: $file"
        Remove-Item -Path $file -Force
    } else {
        Write-Host "Le fichier $file n'existe pas."
    }
}

# Création des mappings de réorganisation des fichiers
$filesToMove = @{
    # Aucun fichier à déplacer pour l'instant, la structure actuelle semble cohérente avec les phases du projet
}

# Déplacer les fichiers si nécessaire
foreach ($source in $filesToMove.Keys) {
    $destination = $filesToMove[$source]
    if (Test-Path $source) {
        $destDir = Split-Path -Parent $destination
        if (-not (Test-Path $destDir)) {
            Write-Host "Création du dossier: $destDir"
            New-Item -Path $destDir -ItemType Directory -Force | Out-Null
        }
        Write-Host "Déplacement de $source vers $destination"
        Move-Item -Path $source -Destination $destination -Force
    } else {
        Write-Host "Le fichier source $source n'existe pas."
    }
}

Write-Host "Nettoyage terminé. Structure de ai_trading optimisée selon les phases du projet." 