class TemporalCrossValidator:
    def __init__(self, n_splits=5, train_ratio=0.6, gap_ratio=0.2):
        """
        Initialise le validateur de validation croisée temporelle.

        Args:
            n_splits (int): Nombre de splits
            train_ratio (float): Ratio de données d'entraînement
            gap_ratio (float): Ratio de données dans le gap entre train et test
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap_ratio = gap_ratio

    def split(self, df):
        """
        Génère les indices pour la validation croisée temporelle.

        Args:
            df (pd.DataFrame): DataFrame contenant les données

        Yields:
            tuple: (indices_train, indices_test)
        """
        total_size = len(df)
        train_size = int(total_size * self.train_ratio)
        gap_size = int(total_size * self.gap_ratio)
        test_size = total_size - train_size - gap_size

        # Calculer le pas pour chaque split
        step = (
            (total_size - train_size - gap_size - test_size) // (self.n_splits - 1)
            if self.n_splits > 1
            else 0
        )

        for i in range(self.n_splits):
            start_train = i * step
            end_train = start_train + train_size
            start_test = end_train + gap_size
            end_test = start_test + test_size

            if end_test > total_size:
                break

            train_indices = list(range(start_train, end_train))
            test_indices = list(range(start_test, end_test))

            yield train_indices, test_indices
