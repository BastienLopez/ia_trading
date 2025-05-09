"""
Dashboard interactif avec analyse post-trade pour le système de trading IA.

Ce module implémente une application web interactif permettant de:
- Visualiser les performances des portefeuilles
- Analyser les transactions effectuées
- Examiner les métriques de risque
- Explorer les expositions aux facteurs
- Suivre les allocations d'actifs
"""

import os
import dash
import dash_bootstrap_components as dbc

from ai_trading.dashboard.callbacks import register_callbacks
from ai_trading.dashboard.layouts import create_layout

# Initialisation de l'application Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    suppress_callback_exceptions=True,
)

app.title = "Crypto Trading - Dashboard d'Analyse Post-Trade"

# Création du layout principal de l'application
app.layout = create_layout(app)

# Enregistrement des callbacks
register_callbacks(app)

# Point d'entrée pour l'exécution du serveur
if __name__ == "__main__":
    # Récupérer les variables d'environnement ou utiliser des valeurs par défaut
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "False").lower() == "true"
    
    print(f"Démarrage du serveur sur {host}:{port} (debug={debug})")
    
    # Création d'une route de santé pour les healthchecks
    @app.server.route("/health")
    def health():
        return "OK"

    app.run_server(
        host=host,
        port=port,
        debug=debug
    )
