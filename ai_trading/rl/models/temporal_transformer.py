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
        # Créer l'encodage positionnel de base
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # Format flexible pour l'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ajoute l'encodage positionnel aux entrées.

        Args:
            x: Tenseur d'entrée de forme [batch_size, seq_len, d_model]
               ou [seq_len, batch_size, d_model]

        Returns:
            Tenseur avec encodage positionnel ajouté
        """
        if len(x.size()) == 4:  # Pour les entrées [batch, seq_len, autre_dim, d_model]
            # Obtenir les dimensions
            batch_size = x.size(0)
            seq_len = x.size(1)
            other_dim = x.size(2)

            # Vérifier que les dimensions sont compatibles
            if seq_len > self.max_len:
                raise ValueError(f"Séquence trop longue: {seq_len} > {self.max_len}")
            if x.size(-1) != self.d_model:
                raise ValueError(
                    f"Dimension du modèle incorrecte: {x.size(-1)} != {self.d_model}"
                )

            # Ajuster l'encodage positionnel
            pos_encoding = self.pe[:seq_len].unsqueeze(0).unsqueeze(2)
            pos_encoding = pos_encoding.expand(batch_size, seq_len, other_dim, -1)
            return x + pos_encoding

        elif len(x.size()) == 3:  # Pour les entrées [batch_size, seq_len, d_model]
            batch_size = x.size(0)
            seq_len = x.size(1)

            # Vérifier la taille de séquence
            if seq_len > self.max_len:
                # Si la séquence est trop longue, on la sous-échantillonne
                print(
                    f"Sous-échantillonnage de la séquence de longueur {seq_len} à {self.max_len}"
                )
                stride = max(1, seq_len // self.max_len)
                indices = torch.arange(0, seq_len, stride)[: self.max_len]
                x = x[:, indices, :]
                seq_len = x.size(1)

            # Vérifier à nouveau la dimension du modèle
            if x.size(2) != self.d_model:
                raise ValueError(
                    f"Dimension du modèle incorrecte: {x.size(2)} != {self.d_model}"
                )

            # Encoder avec broadcast
            return x + self.pe[:seq_len].unsqueeze(0)

        elif len(x.size()) == 2:  # Pour les entrées [seq_len, d_model]
            seq_len = x.size(0)
            if seq_len > self.max_len:
                # Si la séquence est trop longue, on la sous-échantillonne
                print(
                    f"Sous-échantillonnage de la séquence de longueur {seq_len} à {self.max_len}"
                )
                stride = max(1, seq_len // self.max_len)
                indices = torch.arange(0, seq_len, stride)[: self.max_len]
                x = x[indices, :]
                seq_len = x.size(0)
            return x + self.pe[:seq_len]

        else:
            raise ValueError(f"Format d'entrée non supporté: {x.size()}")


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
        max_seq_len: int = 5000,
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
        # Vérifier et adapter le format d'entrée
        if src.dim() > 3:
            # Si l'entrée est 4D [batch, seq_len, autre_dim, features]
            batch_size, seq_len, other_dim, input_dim = src.shape

            # Au lieu d'aplatir, sous-échantillonner pour réduire la longueur de séquence
            if seq_len * other_dim > 5000:
                # Prendre un échantillon plus petit en utilisant la première et dernière partie
                samples_per_dim = min(
                    50, seq_len
                )  # Limiter à 50 échantillons par dimension
                stride = max(1, seq_len // samples_per_dim)

                # Sélectionner un sous-ensemble des séquences
                indices = torch.arange(0, seq_len, stride)[:samples_per_dim]
                src = src[:, indices, :, :]

                # Mettre à jour les dimensions
                _, seq_len, _, _ = src.shape
                print(
                    f"Séquence sous-échantillonnée: de {batch_size}x{src.size(1)}x{other_dim}x{input_dim}"
                )

            # Aplatir les dimensions
            src = src.reshape(batch_size, seq_len * other_dim, input_dim)
            print(f"Format d'entrée final: {src.shape}")

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
            attention_weights.append(weights)

        # Prendre la dernière position de la séquence
        x = x[:, -1, :]

        # Couche de sortie
        output = self.output_layer(x)

        return output, attention_weights
