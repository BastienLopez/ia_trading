import os
import re
import sys


def fix_fstrings_in_file(filepath):
    """Corrige les f-strings sans placeholders dans un fichier."""
    print(f"Traitement du fichier: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Motif pour trouver les f-strings sans placeholders (guillemets doubles)
    pattern1 = r'f"([^{}"]*)"'
    replacement1 = r'"\1"'

    # Motif pour les f-strings avec apostrophes
    pattern2 = r"f'([^{}']*)'"
    replacement2 = r"'\1'"

    # Recherche et remplacement
    new_content = re.sub(pattern1, replacement1, content)
    new_content = re.sub(pattern2, replacement2, new_content)

    # Si des modifications ont été apportées
    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  Corrigé: {filepath}")
        return True

    print(f"  Aucune modification nécessaire pour: {filepath}")
    return False


def fix_fstrings_in_directory(directory):
    """Corrige les f-strings sans placeholders dans tous les fichiers Python du répertoire."""
    fixed_files = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    if fix_fstrings_in_file(filepath):
                        fixed_files += 1
                except Exception as e:
                    print(f"  Erreur lors du traitement de {filepath}: {str(e)}")

    return fixed_files


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "ai_trading"

    print(f"Recherche de f-strings sans placeholders dans: {directory}")
    fixed_count = fix_fstrings_in_directory(directory)
    print(f"Terminé! {fixed_count} fichiers corrigés.")
