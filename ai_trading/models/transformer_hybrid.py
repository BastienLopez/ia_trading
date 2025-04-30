import gc
import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model

# Configuration du logger
logger = logging.getLogger("TransformerHybrid")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def optimize_memory():
    """
    Libère la mémoire non utilisée et vide le cache TensorFlow si disponible.
    """
    # Collecter les objets garbage
    gc.collect()

    # Nettoyer la mémoire TensorFlow
    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        logger.warning(f"Erreur lors du nettoyage de la session TensorFlow: {e}")


class TransformerBlock(tf.keras.layers.Layer):
    """
    Bloc Transformer standard avec multi-head attention et feed forward network
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        # Application de l'auto-attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Application du feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Encodage positionnel pour les séquences dans un Transformer
    """

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )

        # Appliquer sin aux indices pairs
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Appliquer cos aux indices impairs
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        # Correction: Convertir l'encodage positionnel au même type que les inputs pour éviter les erreurs
        input_dtype = inputs.dtype
        pos_encoding_cast = tf.cast(
            self.pos_encoding[:, : tf.shape(inputs)[1], :], input_dtype
        )
        return inputs + pos_encoding_cast


class TransformerGRUModel(Model):
    """
    Modèle hybride combinant Transformer et GRU pour l'analyse de séries temporelles financières
    """

    def __init__(
        self,
        input_shape,
        output_dim,
        embed_dim=64,
        num_heads=4,
        ff_dim=64,
        num_transformer_blocks=2,
        gru_units=64,
        dropout_rate=0.1,
        recurrent_dropout=0.0,
        sequence_length=None,
        use_mixed_precision=True,  # Activer la précision mixte par défaut
        **kwargs,
    ):
        """
        Initialise un modèle hybride Transformer-GRU.

        Args:
            input_shape: Forme des données d'entrée (seq_len, features)
            output_dim: Dimension de sortie (nombre d'actions pour les agents RL)
            embed_dim: Dimension d'embedding pour le Transformer
            num_heads: Nombre de têtes d'attention
            ff_dim: Dimension du réseau feed-forward dans le Transformer
            num_transformer_blocks: Nombre de blocs Transformer à empiler
            gru_units: Nombre d'unités dans la couche GRU
            dropout_rate: Taux de dropout
            recurrent_dropout: Taux de dropout récurrent pour GRU
            sequence_length: Longueur de la séquence (pour l'encodage positionnel)
            use_mixed_precision: Utiliser la précision mixte (float16) pour l'entraînement
        """
        # Activer la politique de précision mixte si demandé
        if use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Précision mixte (float16) activée pour le modèle")
            except Exception as e:
                logger.warning(f"Impossible d'activer la précision mixte: {e}")

        super(TransformerGRUModel, self).__init__(**kwargs)

        # Si la séquence n'est pas spécifiée, utiliser la première dimension de input_shape
        if sequence_length is None:
            sequence_length = input_shape[0]

        # Calculer la dimension d'entrée
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            input_dim = input_shape[1]
        else:
            input_dim = input_shape

        # Couche d'entrée pour projeter les features dans l'espace d'embedding
        self.embedding = Dense(embed_dim, activation="relu")

        # Encodage positionnel
        self.pos_encoding = PositionalEncoding(sequence_length, embed_dim)

        # Blocs Transformer
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]

        # Couche GRU bidirectionnelle
        self.bi_gru = Bidirectional(
            GRU(
                gru_units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
            )
        )

        # Couche GRU finale pour l'agrégation de séquence
        self.gru = GRU(gru_units)

        # Couches de sortie
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(output_dim)

        # Enregistrer les hyperpermètres
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.output_dim = output_dim
        self.use_mixed_precision = use_mixed_precision

        logger.info(
            f"Modèle hybride Transformer-GRU initialisé: "
            f"{num_transformer_blocks} blocs transformer, "
            f"{gru_units} unités GRU, "
            f"{embed_dim} dim d'embedding, "
            f"{num_heads} têtes d'attention, "
            f"{'mixed_float16' if use_mixed_precision else 'float32'}"
        )

    def call(self, inputs, training=False):
        # Projection de l'entrée dans l'espace d'embedding
        x = self.embedding(inputs)

        # Ajouter l'encodage positionnel
        x = self.pos_encoding(x)

        # Passer à travers les blocs Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)

        # Passer à travers la couche GRU bidirectionnelle
        x = self.bi_gru(x, training=training)

        # Passer à travers la couche GRU finale
        x = self.gru(x)

        # Dropout final
        x = self.dropout(x, training=training)

        # Couche de sortie
        return self.output_layer(x)

    def compile_model(self, optimizer="adam", loss="mse", metrics=None):
        """
        Compile le modèle avec les paramètres spécifiés et active les optimisations de performance.
        """
        if metrics is None:
            metrics = ["mae"]

        # Optimisations TensorFlow pour les performances
        try:
            # Activer XLA pour l'accélération de la compilation
            tf.config.optimizer.set_jit(True)

            # Paralléliser les opérations quand possible
            tf.config.threading.set_intra_op_parallelism_threads(8)
            tf.config.threading.set_inter_op_parallelism_threads(8)

            logger.info("Optimisations TensorFlow activées pour le modèle")
        except Exception as e:
            logger.warning(
                f"Erreur lors de l'activation des optimisations TensorFlow: {e}"
            )

        # Compiler le modèle
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return self


class TransformerLSTMModel(Model):
    """
    Modèle hybride combinant Transformer et LSTM pour l'analyse de séries temporelles financières
    """

    def __init__(
        self,
        input_shape,
        output_dim,
        embed_dim=64,
        num_heads=4,
        ff_dim=64,
        num_transformer_blocks=2,
        lstm_units=64,
        dropout_rate=0.1,
        recurrent_dropout=0.0,
        sequence_length=None,
        use_mixed_precision=True,  # Activer la précision mixte par défaut
        **kwargs,
    ):
        """
        Initialise un modèle hybride Transformer-LSTM.

        Args:
            input_shape: Forme des données d'entrée (seq_len, features)
            output_dim: Dimension de sortie
            embed_dim: Dimension d'embedding pour le Transformer
            num_heads: Nombre de têtes d'attention
            ff_dim: Dimension du réseau feed-forward dans le Transformer
            num_transformer_blocks: Nombre de blocs Transformer à empiler
            lstm_units: Nombre d'unités dans la couche LSTM
            dropout_rate: Taux de dropout
            recurrent_dropout: Taux de dropout récurrent pour LSTM
            sequence_length: Longueur de la séquence (pour l'encodage positionnel)
            use_mixed_precision: Utiliser la précision mixte (float16) pour l'entraînement
        """
        # Activer la politique de précision mixte si demandé
        if use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Précision mixte (float16) activée pour le modèle")
            except Exception as e:
                logger.warning(f"Impossible d'activer la précision mixte: {e}")

        super(TransformerLSTMModel, self).__init__(**kwargs)

        # Si la séquence n'est pas spécifiée, utiliser la première dimension de input_shape
        if sequence_length is None:
            sequence_length = input_shape[0]

        # Calculer la dimension d'entrée
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            input_dim = input_shape[1]
        else:
            input_dim = input_shape

        # Couche d'entrée pour projeter les features dans l'espace d'embedding
        self.embedding = Dense(embed_dim, activation="relu")

        # Encodage positionnel
        self.pos_encoding = PositionalEncoding(sequence_length, embed_dim)

        # Blocs Transformer
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]

        # Couche LSTM bidirectionnelle
        self.bi_lstm = Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
            )
        )

        # Couche LSTM finale pour l'agrégation de séquence
        self.lstm = LSTM(lstm_units)

        # Couches de sortie
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(output_dim)

        # Enregistrer les hyperpermètres
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.output_dim = output_dim
        self.use_mixed_precision = use_mixed_precision

        logger.info(
            f"Modèle hybride Transformer-LSTM initialisé: "
            f"{num_transformer_blocks} blocs transformer, "
            f"{lstm_units} unités LSTM, "
            f"{embed_dim} dim d'embedding, "
            f"{num_heads} têtes d'attention, "
            f"{'mixed_float16' if use_mixed_precision else 'float32'}"
        )

    def call(self, inputs, training=False):
        # Projection de l'entrée dans l'espace d'embedding
        x = self.embedding(inputs)

        # Ajouter l'encodage positionnel
        x = self.pos_encoding(x)

        # Passer à travers les blocs Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)

        # Passer à travers la couche LSTM bidirectionnelle
        x = self.bi_lstm(x, training=training)

        # Passer à travers la couche LSTM finale
        x = self.lstm(x)

        # Dropout et couche de sortie
        x = self.dropout(x, training=training)
        return self.output_layer(x)


def create_transformer_hybrid_model(
    model_type,
    input_shape,
    output_dim,
    embed_dim=64,
    num_heads=4,
    ff_dim=64,
    num_transformer_blocks=2,
    rnn_units=64,
    dropout_rate=0.1,
    recurrent_dropout=0.0,
    sequence_length=None,
):
    """
    Crée un modèle hybride basé sur Transformer.

    Args:
        model_type: Type de modèle ('gru' ou 'lstm')
        input_shape: Forme des données d'entrée
        output_dim: Dimension de sortie
        embed_dim: Dimension d'embedding pour le Transformer
        num_heads: Nombre de têtes d'attention
        ff_dim: Dimension du réseau feed-forward dans le Transformer
        num_transformer_blocks: Nombre de blocs Transformer à empiler
        rnn_units: Nombre d'unités dans la couche RNN
        dropout_rate: Taux de dropout
        recurrent_dropout: Taux de dropout récurrent
        sequence_length: Longueur de la séquence

    Returns:
        Model: Instance du modèle hybride
    """
    if model_type.lower() == "gru":
        return TransformerGRUModel(
            input_shape=input_shape,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            gru_units=rnn_units,
            dropout_rate=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            sequence_length=sequence_length,
        )
    elif model_type.lower() == "lstm":
        return TransformerLSTMModel(
            input_shape=input_shape,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            lstm_units=rnn_units,
            dropout_rate=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            sequence_length=sequence_length,
        )
    else:
        raise ValueError(
            f"Type de modèle non reconnu: {model_type}. Utilisez 'gru' ou 'lstm'."
        )
