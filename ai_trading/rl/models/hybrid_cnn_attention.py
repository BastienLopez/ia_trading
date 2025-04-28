import torch.nn as nn
import torch.nn.functional as F


class HybridCNNAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(HybridCNNAttention, self).__init__()
        self.cnn = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, features = x.shape

        # Normalisation des entrées
        x = F.layer_norm(x, (features,))

        # CNN processing
        cnn_out = self.cnn(x.transpose(1, 2)).transpose(1, 2)

        # Attention processing
        att_out = self.attention(cnn_out, mask=mask)

        # Combinaison des sorties avec une connexion résiduelle
        combined = cnn_out + att_out

        # Projection finale
        output = self.projection(combined)
        return output
