# Script pour nettoyer le dossier ai_trading

# Liste des dossiers à supprimer
$dirsToRemove = @(
    "ai_trading/ai_trading",         # Sous-dossier redondant
    "ai_trading/__pycache__"         # Fichiers de cache Python
)

# Liste des fichiers à conserver et leurs chemins de destination
$filesToMove = @{
    "ai_trading/.env" = ".env"  # Fusionner avec le fichier .env à la racine
}

Write-Host "Nettoyage du dossier ai_trading..."

# Supprimer les dossiers inutiles
foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        Write-Host "Suppression du dossier: $dir"
        Remove-Item -Path $dir -Recurse -Force
    } else {
        Write-Host "Le dossier $dir n'existe pas."
    }
}

# Fusionner les fichiers .env
if ((Test-Path "ai_trading/.env") -and (Test-Path ".env")) {
    Write-Host "Fusion des fichiers .env..."
    $rootEnv = Get-Content ".env"
    $aiTradingEnv = Get-Content "ai_trading/.env"
    
    # Vérifier si le fichier .env à la racine contient déjà les informations du fichier dans ai_trading
    $needsUpdate = $false
    foreach ($line in $aiTradingEnv) {
        if ($line.Trim() -ne "" -and -not ($rootEnv -contains $line)) {
            $needsUpdate = $true
            break
        }
    }
    
    if ($needsUpdate) {
        Write-Host "Ajout des variables d'environnement manquantes au fichier .env à la racine..."
        Add-Content -Path ".env" -Value "`n# Merged from ai_trading/.env`n"
        foreach ($line in $aiTradingEnv) {
            if ($line.Trim() -ne "" -and -not ($rootEnv -contains $line)) {
                Add-Content -Path ".env" -Value $line
            }
        }
    }
    
    # Supprimer le fichier .env dans ai_trading
    Remove-Item -Path "ai_trading/.env" -Force
}

Write-Host "Nettoyage terminé. Structure de ai_trading optimisée." 