from pathlib import Path

# Définition du répertoire racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Définition du répertoire info_retour
INFO_RETOUR_DIR = PROJECT_ROOT / "info_retour"

# Création du répertoire s'il n'existe pas
INFO_RETOUR_DIR.mkdir(exist_ok=True) 