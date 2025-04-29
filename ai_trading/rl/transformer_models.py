"""
Module contenant les modèles Transformer hybrides pour l'apprentissage par renforcement.
Ces modèles combinent des architectures Transformer avec des couches récurrentes (GRU ou LSTM)
pour traiter efficacement les séries temporelles dans un contexte d'RL.
"""

import logging

import torch
import torch.nn as nn

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


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class TransformerHybridModel(nn.Module):
    def __init__(
        self,
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
        super().__init__()

        # Projection initiale
        self.projection = nn.Linear(input_shape[-1], embed_dim)

        # Blocs Transformer
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
                for _ in range(num_transformer_blocks)
            ]
        )

        # Couche récurrente
        if model_type.lower() == "gru":
            self.rnn = nn.GRU(
                embed_dim,
                rnn_units,
                batch_first=True,
                dropout=dropout_rate if num_transformer_blocks > 1 else 0,
            )
        elif model_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                embed_dim,
                rnn_units,
                batch_first=True,
                dropout=dropout_rate if num_transformer_blocks > 1 else 0,
            )
        else:
            raise ValueError(
                f"Type de modèle non supporté: {model_type}. Utilisez 'gru' ou 'lstm'."
            )

        # Couche de sortie
        self.output_layer = nn.Linear(rnn_units, output_dim)

        # Optimisations
        self.sequence_length = sequence_length

        logger.info(
            f"Modèle hybride Transformer-{model_type.upper()} créé avec "
            f"{num_transformer_blocks} blocs Transformer, {num_heads} têtes d'attention, "
            f"dimension d'embedding {embed_dim}, {rnn_units} unités récurrentes"
        )

    def forward(self, x):
        # Projection initiale
        x = self.projection(x)

        # Blocs Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Couche récurrente
        if isinstance(self.rnn, nn.GRU):
            _, h_n = self.rnn(x)
        else:  # LSTM
            _, (h_n, _) = self.rnn(x)

        # Dernière sortie de la séquence
        x = h_n[-1]

        # Couche de sortie
        return self.output_layer(x)

    def compile(self, optimizer, loss_fn):
        """Configure l'optimiseur et la fonction de perte"""
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, batch):
        """Effectue une étape d'entraînement"""
        self.train()
        self.optimizer.zero_grad()

        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, x):
        """Fait des prédictions en mode évaluation"""
        self.eval()
        with torch.no_grad():
            return self(x)
