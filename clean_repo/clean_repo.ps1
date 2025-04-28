# Script pour nettoyer le repository

# Liste des dossiers à supprimer
$dirsToRemove = @(
    ".cursor",
    ".pytest_cache",
    "htmlcov",
    "sentiment_cache",
    "logs",
    "bin",
    "cache",
    "test_data",
    "reports",
    "results",
    ".github",
    "ai_trading.egg-info",
    "__pycache__",
    ".ipynb_checkpoints",
    "tmp_test_models",
    "model_checkpoints",
    "saved_models",
    "tensorboard",
    ".tf_profile",
    ".keras"
)

# Liste des fichiers à supprimer
$filesToRemove = @(
    ".coverage",
    "coverage.xml",
    "web_app.log",
    "pytest.ini",
    "settings.json",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*.prof",
    "*.coverage",
    "*.egg-info",
    "*.bak",
    "*.tmp",
    "*.temp",
    "*.swp",
    "*~"
)

Write-Host "Début du nettoyage du repository..."

# Vérifier que nous sommes dans le bon répertoire
if (-not (Test-Path ".git")) {
    Write-Host "Erreur: Le répertoire courant n'est pas un dépôt Git."
    exit 1
}

# Supprimer les dossiers
foreach ($dir in $dirsToRemove) {
    Get-ChildItem -Path "." -Recurse -Directory -Include $dir | ForEach-Object {
        try {
            Write-Host "Suppression du dossier: $($_.FullName)"
            Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Host "Erreur lors de la suppression de $($_.FullName) : $_"
        }
    }
}

# Supprimer les fichiers
foreach ($file in $filesToRemove) {
    Get-ChildItem -Path "." -Recurse -File -Include $file | ForEach-Object {
        try {
            # Ne pas supprimer le fichier test.yml de la CI/CD
            if (-not ($_.FullName -like "*\.github\workflows\test.yml")) {
                Write-Host "Suppression du fichier: $($_.FullName)"
                Remove-Item -Path $_.FullName -Force -ErrorAction Stop
            }
        } catch {
            Write-Host "Erreur lors de la suppression de $($_.FullName) : $_"
        }
    }
}

# Nettoyer les fichiers de cache Python
Get-ChildItem -Path "." -Recurse -Include "*.pyc", "*.pyo", "*.pyd" | ForEach-Object {
    try {
        Write-Host "Suppression du fichier de cache: $($_.FullName)"
        Remove-Item $_.FullName -Force -ErrorAction Stop
    } catch {
        Write-Host "Erreur lors de la suppression de $($_.FullName) : $_"
    }
}

Write-Host "`nNettoyage terminé. Dossiers essentiels conservés:"
Write-Host "- ai_trading (code principal)"
Write-Host "- web_app (frontend)"
Write-Host "- tradingview (scripts Pine)"
Write-Host "- data (données)"
Write-Host "- tests (tests)"
Write-Host "- clean_repo (scripts de nettoyage)"
Write-Host "- .github/workflows/test.yml (CI/CD)"
Write-Host "`nEt les fichiers de configuration:"
Write-Host "- .gitignore"
Write-Host "- requirements.txt"
Write-Host "- README.md"
Write-Host "- setup_env.bat" 