from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Module d'attention temporelle pour les séries temporelles.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        """
        Initialise le module d'attention temporelle.

        Args:
            input_dim: Dimension des entrées
            hidden_dim: Dimension cachée
            num_heads: Nombre de têtes d'attention
            dropout: Taux de dropout
            max_len: Longueur maximale de la séquence
        """
        super().__init__()
        assert (
            hidden_dim % num_heads == 0
        ), "La dimension cachée doit être divisible par le nombre de têtes"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Projections linéaires
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # Position encoding
        self.position_encoding = nn.Parameter(torch.randn(max_len, hidden_dim))

        # Dropout et normalisation
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Projection finale
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant du module d'attention.

        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_len, input_dim)
            mask: Masque d'attention optionnel

        Returns:
            Tuple contenant:
            - Tenseur de sortie de forme (batch_size, seq_len, hidden_dim)
            - Matrice d'attention de forme (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Projections linéaires
        q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key(x)  # (batch_size, seq_len, hidden_dim)
        v = self.value(x)  # (batch_size, seq_len, hidden_dim)

        # Ajout du position encoding
        pos_enc = self.position_encoding[:seq_len].unsqueeze(
            0
        )  # (1, seq_len, hidden_dim)
        q = q + pos_enc
        k = k + pos_enc

        # Redimensionner pour l'attention multi-têtes
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calcul des scores d'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Application du masque si fourni
        if mask is not None:
            # Redimensionner le masque pour correspondre aux dimensions de l'attention
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax et dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Application de l'attention
        output = torch.matmul(
            attention_weights, v
        )  # (batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(
            batch_size, seq_len, self.hidden_dim
        )  # (batch_size, seq_len, hidden_dim)

        # Projection et normalisation
        output = self.output(output)
        output = self.layer_norm(output)

        return output, attention_weights


class TemporalAttentionModel(nn.Module):
    """
    Modèle complet utilisant l'attention temporelle.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        """
        Initialise le modèle d'attention temporelle.

        Args:
            input_dim: Dimension des entrées
            hidden_dim: Dimension cachée
            output_dim: Dimension de sortie
            num_layers: Nombre de couches d'attention
            num_heads: Nombre de têtes d'attention
            dropout: Taux de dropout
            max_len: Longueur maximale de la séquence
        """
        super().__init__()

        # Couche d'entrée
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Couches d'attention
        self.attention_layers = nn.ModuleList(
            [
                TemporalAttention(hidden_dim, hidden_dim, num_heads, dropout, max_len)
                for _ in range(num_layers)
            ]
        )

        # Couches de feed-forward
        self.ff_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # Couche de sortie
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Passe avant du modèle.

        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_len, input_dim)
            mask: Masque d'attention optionnel

        Returns:
            Tuple contenant:
            - Tenseur de sortie de forme (batch_size, output_dim)
            - Liste des matrices d'attention de chaque couche
        """
        # Projection initiale
        x = self.input_projection(x)

        attention_weights = []

        # Application des couches d'attention
        for attention_layer, ff_layer in zip(self.attention_layers, self.ff_layers):
            # Attention
            x, weights = attention_layer(x, mask)
            attention_weights.append(weights)

            # Feed-forward
            x = x + ff_layer(x)

        # Dernière position de la séquence
        x = x[:, -1, :]

        # Couche de sortie
        output = self.output_layer(x)

        return output, attention_weights
