import os
import sys


def fix_fstrings_in_file(filepath):
    """Corrige les f-strings sans placeholders dans un fichier."""
    print(f"Traitement du fichier: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for line in lines:
        # Recherche de f-strings sans placeholders
        if 'f"' in line and "{" not in line:
            new_line = line.replace('f"', '"')
            modified = True
            new_lines.append(new_line)
        elif "f'" in line and "{" not in line:
            new_line = line.replace("f'", "'")
            modified = True
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
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
