from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriceGraphCNN(nn.Module):
    """
    CNN pour l'analyse des graphiques de prix.
    Utilise des convolutions 1D pour extraire des motifs temporels.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        kernel_sizes: list = [3, 5, 7],
        n_filters: list = [64, 128, 256],
        dropout: float = 0.2,
    ):
        """
        Initialise le CNN pour l'analyse des graphiques.

        Args:
            input_channels: Nombre de canaux d'entrée (OHLCV = 5)
            output_dim: Dimension de sortie
            kernel_sizes: Tailles des noyaux de convolution
            n_filters: Nombre de filtres par couche
            dropout: Taux de dropout
        """
        super().__init__()
        assert len(kernel_sizes) == len(
            n_filters
        ), "Le nombre de kernel_sizes doit correspondre au nombre de filtres"

        self.input_channels = input_channels
        self.output_dim = output_dim

        # Couches de convolution avec différentes tailles de noyaux
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_channels if i == 0 else n_filters[i - 1],
                        out_channels=n_filters[i],
                        kernel_size=kernel_sizes[i],
                        padding=kernel_sizes[i] // 2,
                    ),
                    nn.BatchNorm1d(n_filters[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for i in range(len(kernel_sizes))
            ]
        )

        # Couche d'attention pour pondérer les features importantes
        self.attention = nn.Sequential(
            nn.Linear(n_filters[-1], n_filters[-1]),
            nn.Tanh(),
            nn.Linear(n_filters[-1], 1),
        )

        # Couches fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(n_filters[-1], n_filters[-1] * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters[-1] * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Passe avant du modèle.

        Args:
            x: Tenseur d'entrée de forme (batch_size, input_channels, seq_len)

        Returns:
            Tuple contenant:
            - Tenseur de sortie de forme (batch_size, output_dim)
            - Poids d'attention (optionnel)
        """
        # Convolutions
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Permutation pour l'attention (batch_size, seq_len, n_filters)
        x = x.transpose(1, 2)

        # Calcul des poids d'attention
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Application de l'attention
        x = torch.bmm(
            x.transpose(1, 2), attention_weights
        )  # (batch_size, n_filters, 1)
        x = x.squeeze(-1)  # (batch_size, n_filters)

        # Couches fully connected
        x = self.fc_layers(x)

        return x, attention_weights


class HybridCNNAttention(nn.Module):
    """
    Modèle hybride combinant CNN et attention temporelle.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        output_dim: int,
        cnn_kernel_sizes: list = [3, 5, 7],
        cnn_filters: list = [64, 128, 256],
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Initialise le modèle hybride.

        Args:
            input_channels: Nombre de canaux d'entrée (OHLCV = 5)
            hidden_dim: Dimension cachée
            output_dim: Dimension de sortie
            cnn_kernel_sizes: Tailles des noyaux de convolution
            cnn_filters: Nombre de filtres par couche CNN
            num_heads: Nombre de têtes d'attention
            dropout: Taux de dropout
        """
        super().__init__()

        if hidden_dim % num_heads:
            raise ValueError("hidden_dim doit être divisible par num_heads")
        # Conserve la dimension temporelle : l'ancienne version comprimait la
        # séquence en un seul vecteur avant l'attention, la rendant inutile.
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Couche de sortie
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Passe avant du modèle.

        Args:
            x: Tenseur d'entrée de forme (batch_size, input_channels, seq_len)
            mask: Masque d'attention optionnel

        Returns:
            Tuple contenant:
            - Tenseur de sortie de forme (batch_size, output_dim)
            - Dictionnaire des poids d'attention
        """
        features = self.cnn(x).transpose(1, 2)
        attended_features, temporal_attention = self.attention(
            features, features, features, key_padding_mask=mask, need_weights=True
        )

        # Dernière position de la séquence
        output = attended_features[:, -1, :]

        # Couche de sortie
        output = self.output_layer(output)

        # Retourner les deux types d'attention
        attention_weights = {"temporal": temporal_attention}

        return output, attention_weights
