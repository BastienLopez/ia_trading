import ssl

import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Téléchargement des ressources NLTK nécessaires
print("Téléchargement de punkt...")
nltk.download("punkt")
print("Téléchargement de stopwords...")
nltk.download("stopwords")
print("Téléchargement de wordnet...")
nltk.download("wordnet")
print("Ressources NLTK téléchargées avec succès!")
