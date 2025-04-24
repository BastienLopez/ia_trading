import os

os.makedirs(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "info_retour/data"),
    exist_ok=True,
)
