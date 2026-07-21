class TemporalCrossValidator:
    def __init__(self, n_splits=5, train_ratio=0.6, gap_ratio=0.2):
        """
        Initialise le validateur de validation croisée temporelle.

        Args:
            n_splits (int): Nombre de splits
            train_ratio (float): Ratio de données d'entraînement
            gap_ratio (float): Ratio de données dans le gap entre train et test
        """
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits doit etre un entier positif.")
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio doit etre strictement compris entre 0 et 1.")
        if not 0 <= gap_ratio < 1:
            raise ValueError("gap_ratio doit etre compris entre 0 et 1.")
        if train_ratio + gap_ratio >= 1:
            raise ValueError("train_ratio + gap_ratio doit laisser une zone de test.")
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
        remaining_size = total_size - train_size - gap_size
        if train_size < 1 or remaining_size < 1:
            raise ValueError("Jeu de donnees trop court pour les ratios demandes.")
        if self.n_splits > remaining_size:
            raise ValueError("n_splits est trop eleve pour la zone de test disponible.")

        test_size = remaining_size if self.n_splits == 1 else max(1, remaining_size // self.n_splits)
        last_start = total_size - train_size - gap_size - test_size
        step = 0 if self.n_splits == 1 else max(1, last_start // (self.n_splits - 1))

        for i in range(self.n_splits):
            start_train = i * step
            end_train = start_train + train_size
            start_test = end_train + gap_size
            end_test = start_test + test_size

            if end_test > total_size:
                raise ValueError("Les splits temporels depassent le jeu de donnees.")

            train_indices = list(range(start_train, end_train))
            test_indices = list(range(start_test, end_test))

            yield train_indices, test_indices
