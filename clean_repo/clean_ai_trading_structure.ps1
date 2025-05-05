# Script pour nettoyer la structure du dossier ai_trading

# Dossiers à supprimer car vides ou inutiles
$emptyDirsToRemove = @(
    "ai_trading/ai_trading",     # Structure redondante
    "ai_trading/environments",   # Dossier vide
    "ai_trading/agents",         # Dossier vide, remplacé par ai_trading/rl/agents
    "ai_trading/config",         # Dossier vide, remplacé par ai_trading/config.py
    "ai_trading/__pycache__",    # Fichiers de cache Python
    "ai_trading/utils/__pycache__",
    "ai_trading/llm/sentiment_analysis/__pycache__",
    "ai_trading/rl/__pycache__",
    "ai_trading/rl/agents/__pycache__",
    "ai_trading/models/__pycache__",
    "ai_trading/data/__pycache__",
    "ai_trading/tests/__pycache__",
    "ai_trading/tmp_test_models", # Dossier temporaire de test
    "ai_trading/.pytest_cache"    # Cache pytest
)

# Fichiers à supprimer car obsolètes ou remplacés
$filesToRemove = @(
    "ai_trading/download_nltk_data.py",   # Remplacé par des appels directs dans le code
    "ai_trading/utils/data_collector.py", # Remplacé par enhanced_data_collector.py
    "ai_trading/utils/rl_data_integrator.py",  # Obsolète
    "ai_trading/utils/gpu_cleanup.py"     # Remplacé par clean_cuda.py
)

Write-Host "Nettoyage de la structure du dossier ai_trading..."

# Vérifier que nous sommes dans le bon répertoire
if (-not (Test-Path "ai_trading")) {
    Write-Host "Erreur: Le dossier ai_trading n'existe pas dans le répertoire courant."
    exit 1
}

# Supprimer les dossiers vides ou inutiles
foreach ($dir in $emptyDirsToRemove) {
    if (Test-Path $dir) {
        try {
            Write-Host "Suppression du dossier: $dir"
            Remove-Item -Path $dir -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Host "Erreur lors de la suppression de $dir : $_"
        }
    } else {
        Write-Host "Le dossier $dir n'existe pas."
    }
}

# Supprimer les fichiers obsolètes
foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        try {
            Write-Host "Suppression du fichier: $file"
            Remove-Item -Path $file -Force -ErrorAction Stop
        } catch {
            Write-Host "Erreur lors de la suppression de $file : $_"
        }
    } else {
        Write-Host "Le fichier $file n'existe pas."
    }
}

# Création des mappings de réorganisation des fichiers
$filesToMove = @{
    # Aucun fichier à déplacer pour l'instant, la structure actuelle semble cohérente
}

# Déplacer les fichiers si nécessaire
foreach ($source in $filesToMove.Keys) {
    $destination = $filesToMove[$source]
    if (Test-Path $source) {
        try {
            $destDir = Split-Path -Parent $destination
            if (-not (Test-Path $destDir)) {
                Write-Host "Création du dossier: $destDir"
                New-Item -Path $destDir -ItemType Directory -Force | Out-Null
            }
            Write-Host "Déplacement de $source vers $destination"
            Move-Item -Path $source -Destination $destination -Force -ErrorAction Stop
        } catch {
            Write-Host "Erreur lors du déplacement de $source : $_"
        }
    } else {
        Write-Host "Le fichier source $source n'existe pas."
    }
}

Write-Host "Nettoyage terminé. Structure de ai_trading optimisée selon les phases du projet." 