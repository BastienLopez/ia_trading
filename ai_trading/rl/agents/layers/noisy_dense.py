import tensorflow as tf
from tensorflow.keras.layers import Layer


class NoisyDense(Layer):
    """
    Implémentation d'une couche dense bruitée avec bruit factoriel.

    Cette couche est utilisée dans les Noisy Networks pour améliorer l'exploration
    en ajoutant du bruit paramétrable aux poids et biais de la couche.

    Référence:
        Fortunato et al., "Noisy Networks for Exploration", 2018
        https://arxiv.org/abs/1706.10295
    """

    def __init__(
        self, units, activation=None, sigma_init=0.5, bias_regularizer=None, **kwargs
    ):
        """
        Initialise une couche dense bruitée.

        Args:
            units (int): Nombre d'unités (neurones) dans la couche
            activation (callable, optional): Fonction d'activation
            sigma_init (float): Valeur initiale pour le paramètre sigma (écart-type du bruit)
            bias_regularizer: Régulariseur pour le biais
            **kwargs: Arguments supplémentaires pour la classe de base
        """
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma_init = sigma_init
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        super(NoisyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Construit la couche.

        Args:
            input_shape: Forme de l'entrée
        """
        # Taille de l'entrée
        self.input_dim = input_shape[-1]

        # Initialiser les paramètres mu (moyenne) pour les poids et les biais
        mu_range = 1 / tf.math.sqrt(tf.cast(self.input_dim, tf.float32))

        # Paramètres de poids déterministes (mu)
        self.weight_mu = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
            name="weight_mu",
            trainable=True,
        )

        # Paramètres de poids bruités (sigma)
        self.weight_sigma = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=tf.keras.initializers.Constant(
                self.sigma_init / tf.math.sqrt(tf.cast(self.input_dim, tf.float32))
            ),
            name="weight_sigma",
            trainable=True,
        )

        # Paramètres de biais déterministes (mu)
        self.bias_mu = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
            name="bias_mu",
            trainable=True,
            regularizer=self.bias_regularizer,
        )

        # Paramètres de biais bruités (sigma)
        self.bias_sigma = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(
                self.sigma_init / tf.math.sqrt(tf.cast(self.input_dim, tf.float32))
            ),
            name="bias_sigma",
            trainable=True,
        )

        # Variables pour le bruit factoriel
        self.epsilon_in = None
        self.epsilon_out = None

        super(NoisyDense, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Appelle la couche avec les entrées spécifiées.

        Args:
            inputs: Entrées de la couche
            training: Indicateur si nous sommes en mode entraînement

        Returns:
            Sortie de la couche
        """

        def noisy_call():
            # Générer le bruit factoriel si en mode entraînement
            # Fonctions auxiliaires pour le bruit factoriel
            def f(x):
                # f(x) = sign(x) * sqrt(abs(x))
                return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x))

            # Générer des échantillons de bruit indépendants
            eps_in = tf.random.normal([self.input_dim])
            eps_out = tf.random.normal([self.units])

            # Calculer le bruit factoriel à partir des échantillons indépendants
            self.epsilon_in = f(eps_in)
            self.epsilon_out = f(eps_out)

            # Bruit factoriel pour les poids
            epsilon_w = tf.tensordot(self.epsilon_in, self.epsilon_out, axes=0)

            # Calculer les poids bruités = mu + sigma * epsilon
            weights = self.weight_mu + self.weight_sigma * epsilon_w
            bias = self.bias_mu + self.bias_sigma * self.epsilon_out

            return weights, bias

        def non_noisy_call():
            # Utiliser simplement les paramètres mu (moyenne) sans bruit
            return self.weight_mu, self.bias_mu

        # Sélectionner la fonction à utiliser en fonction du mode d'entraînement
        weights, bias = tf.cond(
            (
                tf.constant(training, dtype=tf.bool)
                if training is not None
                else tf.constant(True, dtype=tf.bool)
            ),
            noisy_call,
            non_noisy_call,
        )

        # Calcul de la sortie
        output = tf.matmul(inputs, weights) + bias

        # Appliquer la fonction d'activation si spécifiée
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        """
        Retourne la configuration de la couche.

        Returns:
            dict: Configuration de la couche
        """
        config = super(NoisyDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "sigma_init": self.sigma_init,
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config
