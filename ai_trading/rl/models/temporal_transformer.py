import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel pour les séquences temporelles.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class TemporalTransformerBlock(nn.Module):
    """
    Bloc Transformer adapté pour l'analyse temporelle des données financières.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass avec retour des poids d'attention.
        """
        # Appliquer l'attention multi-têtes
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)

        # Les poids d'attention ont la forme (batch_size * nhead, seq_len, seq_len)
        # Nous gardons les poids tels quels, sans essayer de les redimensionner
        # attn_weights a déjà la bonne dimension pour torch.nn.MultiheadAttention

        # Dropout sur la sortie de l'attention
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(src + attn_output)

        # Feed-forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(out1))))
        ff_output = self.dropout2(ff_output)
        out2 = self.norm2(out1 + ff_output)

        return out2, attn_weights


class FinancialTemporalTransformer(nn.Module):
    """
    Transformer temporel spécialisé pour l'analyse financière.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        output_dim: int = 1,
    ):
        super().__init__()

        # Couche d'entrée avec normalisation
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )

        # Encodage positionnel
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Blocs Transformer
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Couche de sortie améliorée
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim),
        )

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """
        Initialisation améliorée des poids pour une meilleure convergence.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialisation spéciale pour la dernière couche
        if isinstance(self.output_layer[-1], nn.Linear):
            nn.init.zeros_(self.output_layer[-1].bias)
            nn.init.xavier_uniform_(self.output_layer[-1].weight, gain=0.01)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass du modèle.

        Args:
            src: Tenseur d'entrée de forme (batch_size, seq_len, input_dim)
            src_mask: Masque d'attention optionnel
            src_key_padding_mask: Masque de padding optionnel

        Returns:
            Tuple contenant:
            - Tenseur de sortie de forme (batch_size, output_dim)
            - Liste des matrices d'attention de chaque couche
        """
        # Normalisation des données d'entrée
        src_mean = src.mean(dim=1, keepdim=True)
        src_std = src.std(dim=1, keepdim=True) + 1e-8
        src = (src - src_mean) / src_std

        # Projection et encodage positionnel
        x = self.input_projection(src)
        x = self.pos_encoder(x)

        # Passer à travers les blocs Transformer
        attention_weights = []
        for block in self.transformer_blocks:
            x, weights = block(x, src_mask, src_key_padding_mask)
            # Pas besoin de normalisation supplémentaire ici
            attention_weights.append(weights)

        # Prendre la dernière position de la séquence
        x = x[:, -1, :]

        # Couche de sortie
        output = self.output_layer(x)

        return output, attention_weights
