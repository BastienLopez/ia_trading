import logging

# Configuration du logger
logger = logging.getLogger("DataProcessor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def prepare_data_for_rl(
    market_data, sentiment_data=None, window_size=5, test_split=0.2
):
    """
    Prépare les données pour l'apprentissage par renforcement.

    Args:
        market_data (DataFrame): Données de marché prétraitées
        sentiment_data (DataFrame, optional): Données de sentiment
        window_size (int): Taille de la fenêtre d'observation
        test_split (float): Proportion des données à utiliser pour le test

    Returns:
        tuple: (train_data, test_data) DataFrames prêts pour l'RL
    """
    # Vérifier que les données contiennent au moins la colonne 'close'
    if "close" not in market_data.columns:
        raise ValueError("Les données de marché doivent contenir une colonne 'close'")

    # Copier les données pour éviter de modifier l'original
    df = market_data.copy()

    # Fusionner avec les données de sentiment si disponibles
    if sentiment_data is not None:
        logger.info("Fusion des données de marché et de sentiment")
        # Assurez-vous que les index correspondent
        sentiment_data = sentiment_data.reindex(df.index, method="ffill")
        # Fusionner les DataFrames
        for col in sentiment_data.columns:
            df[f"sentiment_{col}"] = sentiment_data[col]

    # Ajouter des indicateurs techniques de base si non présents
    if "returns" not in df.columns:
        logger.info("Calcul des rendements")
        df["returns"] = df["close"].pct_change()

    # Remplir les valeurs NaN
    df = df.ffill().bfill()

    # Normaliser les données
    logger.info("Normalisation des données")
    for col in df.columns:
        if col not in ["date", "timestamp"]:
            # Éviter la division par zéro
            if df[col].max() - df[col].min() > 0:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Diviser en ensembles d'entraînement et de test
    train_size = int(len(df) * (1 - test_split))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    logger.info(
        f"Données préparées: {len(train_data)} points d'entraînement, {len(test_data)} points de test"
    )

    return train_data, test_data
