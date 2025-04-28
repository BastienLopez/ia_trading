"""
Module contenant les modèles Transformer hybrides pour l'apprentissage par renforcement.
Ces modèles combinent des architectures Transformer avec des couches récurrentes (GRU ou LSTM)
pour traiter efficacement les séries temporelles dans un contexte d'RL.
"""

import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model

# Configuration du logger
logger = logging.getLogger("TransformerModels")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    """
    Bloc Transformer standard avec attention multi-têtes et réseau feed-forward.

    Args:
        inputs: Tensor d'entrée
        embed_dim: Dimension d'embedding
        num_heads: Nombre de têtes d'attention
        ff_dim: Dimension du réseau feed-forward
        dropout_rate: Taux de dropout

    Returns:
        Tensor: Sortie du bloc Transformer
    """
    # Normalisation de couche
    attention_input = LayerNormalization(epsilon=1e-6)(inputs)

    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads
    )(attention_input, attention_input)

    # Dropout sur la sortie d'attention
    attention_output = Dropout(dropout_rate)(attention_output)

    # Connexion résiduelle
    attention_output = tf.keras.layers.add([inputs, attention_output])

    # Feed-forward network
    ffn_input = LayerNormalization(epsilon=1e-6)(attention_output)
    ffn_output = Dense(ff_dim, activation="relu")(ffn_input)
    ffn_output = Dense(embed_dim)(ffn_output)

    # Dropout sur la sortie du feed-forward
    ffn_output = Dropout(dropout_rate)(ffn_output)

    # Connexion résiduelle finale
    return tf.keras.layers.add([attention_output, ffn_output])


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
    sequence_length=20,
):
    """
    Crée un modèle hybride combinant des blocs Transformer et une couche récurrente (GRU ou LSTM).

    Args:
        model_type: Type de modèle récurrent ('gru' ou 'lstm')
        input_shape: Forme des entrées (séquence, features)
        output_dim: Dimension de sortie
        embed_dim: Dimension d'embedding pour le Transformer
        num_heads: Nombre de têtes d'attention
        ff_dim: Dimension du réseau feed-forward dans les blocs Transformer
        num_transformer_blocks: Nombre de blocs Transformer
        rnn_units: Nombre d'unités dans la couche récurrente
        dropout_rate: Taux de dropout
        recurrent_dropout: Taux de dropout récurrent
        sequence_length: Longueur de la séquence

    Returns:
        Model: Modèle Keras
    """
    # Définir l'entrée
    inputs = Input(shape=input_shape)

    # Embedded projection
    x = Dense(embed_dim)(inputs)

    # Ajouter des blocs Transformer
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, embed_dim, num_heads, ff_dim, dropout_rate)

    # Ajouter une couche récurrente (GRU ou LSTM)
    if model_type.lower() == "gru":
        rnn_layer = GRU(
            units=rnn_units,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
        )
    elif model_type.lower() == "lstm":
        rnn_layer = LSTM(
            units=rnn_units,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
        )
    else:
        raise ValueError(
            f"Type de modèle non supporté: {model_type}. Utilisez 'gru' ou 'lstm'."
        )

    x = rnn_layer(x)

    # Couche de sortie
    outputs = Dense(output_dim)(x)

    # Créer le modèle
    model = Model(inputs=inputs, outputs=outputs)

    logger.info(
        f"Modèle hybride Transformer-{model_type.upper()} créé avec "
        f"{num_transformer_blocks} blocs Transformer, {num_heads} têtes d'attention, "
        f"dimension d'embedding {embed_dim}, {rnn_units} unités récurrentes"
    )

    return model
