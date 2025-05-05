# Script pour nettoyer le dossier ai_trading

# Liste des dossiers à supprimer
$dirsToRemove = @(
    # "ai_trading/ai_trading",         # Sous-dossier redondant (déjà nettoyé)
    "ai_trading/__pycache__",        # Fichiers de cache Python
    "ai_trading/.pytest_cache",      # Cache pytest
    "ai_trading/tmp_test_models",    # Dossier temporaire de test
    "ai_trading/.ipynb_checkpoints"  # Checkpoints Jupyter
)

# Liste des fichiers à conserver et leurs chemins de destination
$filesToMove = @{
    "ai_trading/.env" = ".env"  # Fusionner avec le fichier .env à la racine
}

Write-Host "Nettoyage du dossier ai_trading..."

# Vérifier que nous sommes dans le bon répertoire
if (-not (Test-Path "ai_trading")) {
    Write-Host "Erreur: Le dossier ai_trading n'existe pas dans le répertoire courant."
    exit 1
}

# Supprimer les dossiers inutiles
foreach ($dir in $dirsToRemove) {
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

# Fusionner les fichiers .env
if ((Test-Path "ai_trading/.env") -and (Test-Path ".env")) {
    Write-Host "Fusion des fichiers .env..."
    try {
        $rootEnv = Get-Content ".env" -ErrorAction Stop
        $aiTradingEnv = Get-Content "ai_trading/.env" -ErrorAction Stop
        
        # Vérifier si le fichier .env à la racine contient déjà les informations du fichier dans ai_trading
        $needsUpdate = $false
        $newVars = @()
        foreach ($line in $aiTradingEnv) {
            if ($line.Trim() -ne "" -and -not ($rootEnv -contains $line)) {
                $needsUpdate = $true
                $newVars += $line
            }
        }
        
        if ($needsUpdate) {
            Write-Host "Ajout des variables d'environnement manquantes au fichier .env à la racine..."
            Add-Content -Path ".env" -Value "`n# Merged from ai_trading/.env on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"
            foreach ($line in $newVars) {
                Add-Content -Path ".env" -Value $line
            }
            Write-Host "Ajout de $($newVars.Count) nouvelles variables d'environnement"
        } else {
            Write-Host "Aucune nouvelle variable d'environnement à ajouter"
        }
        
        # Supprimer le fichier .env dans ai_trading
        Remove-Item -Path "ai_trading/.env" -Force -ErrorAction Stop
    } catch {
        Write-Host "Erreur lors de la fusion des fichiers .env : $_"
    }
}

# Nettoyer les fichiers de cache Python
Get-ChildItem -Path "ai_trading" -Recurse -Include "*.pyc", "*.pyo", "*.pyd" | ForEach-Object {
    try {
        Write-Host "Suppression du fichier de cache: $($_.FullName)"
        Remove-Item $_.FullName -Force -ErrorAction Stop
    } catch {
        Write-Host "Erreur lors de la suppression de $($_.FullName) : $_"
    }
}

Write-Host "Nettoyage terminé. Structure de ai_trading optimisée." 