import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel avancé pour les séquences temporelles.
    Prend en charge des contextes temporels étendus et peut s'adapter
    à différentes longueurs de séquences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Créer l'encodage positionnel de base
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        # Format flexible pour l'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Enregistrer comme buffer pour que ce soit transféré au GPU avec le modèle
        self.register_buffer("pe", pe)
        
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ajoute l'encodage positionnel aux entrées.
        
        Args:
            x: Tenseur d'entrée de forme [batch_size, seq_len, d_model]
        
        Returns:
            Tenseur avec encodage positionnel ajouté
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            # Adapter dynamiquement à une séquence plus longue
            position = torch.arange(seq_len, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device) * 
                (-math.log(10000.0) / self.d_model)
            )
            pe_extended = torch.zeros(seq_len, self.d_model, device=x.device)
            pe_extended[:, 0::2] = torch.sin(position * div_term)
            pe_extended[:, 1::2] = torch.cos(position * div_term)
            
            # Ajouter l'encodage positionnel étendu
            x = x + pe_extended.unsqueeze(0)
        else:
            # Utiliser l'encodage pré-calculé
            x = x + self.pe[:seq_len].unsqueeze(0)
            
        return self.dropout(x)


class LongRangeTemporalAttention(nn.Module):
    """
    Module d'attention temporelle avancé qui capture efficacement les dépendances à long terme
    et intègre des mécanismes pour se concentrer sur différentes échelles temporelles.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 8, 
        dropout: float = 0.1,
        use_relative_positions: bool = True,
        max_relative_position: int = 32
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_relative_positions = use_relative_positions
        self.max_relative_position = max_relative_position
        
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        
        # Projections linéaires
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout et normalisation
        self.dropout = nn.Dropout(dropout)
        
        # Encodage de position relative si activé
        if use_relative_positions:
            self.rel_pos_encoding = nn.Parameter(
                torch.zeros(2 * max_relative_position + 1, self.head_dim)
            )
            nn.init.xavier_uniform_(self.rel_pos_encoding)
            
    def _relative_position_bucket(self, relative_position):
        """Convertit les positions relatives en indices pour l'encodage de position."""
        relative_position = relative_position.clamp(-self.max_relative_position, self.max_relative_position)
        return relative_position + self.max_relative_position
            
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calcule l'attention avec support pour les positions relatives.
        
        Args:
            query: Tenseur de requête [batch_size, seq_len_q, d_model]
            key: Tenseur de clé [batch_size, seq_len_k, d_model]
            value: Tenseur de valeur [batch_size, seq_len_v, d_model]
            mask: Masque d'attention optionnel
            return_attention: Si True, retourne les poids d'attention
            
        Returns:
            Tuple contenant:
            - Tenseur de sortie [batch_size, seq_len_q, d_model]
            - Poids d'attention (facultatif) [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Projections linéaires et reshape pour l'attention multi-têtes
        q = self.q_proj(query).view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Calcul des scores d'attention (produit scalaire)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Ajout de l'encodage de position relative si activé
        if self.use_relative_positions:
            # Générer les indices de position relative
            q_pos = torch.arange(seq_len_q, device=query.device).unsqueeze(1)
            k_pos = torch.arange(seq_len_k, device=key.device).unsqueeze(0)
            rel_pos = k_pos - q_pos
            rel_pos_indices = self._relative_position_bucket(rel_pos)
            
            # Récupérer les encodages de position relative
            rel_pos_encodings = self.rel_pos_encoding[rel_pos_indices]  # [seq_len_q, seq_len_k, head_dim]
            
            # Modification pour assurer la compatibilité des dimensions
            # Reshape q pour le produit avec les encodages relatifs
            q_reshaped = q.permute(2, 0, 1, 3).reshape(seq_len_q, batch_size * self.n_heads, self.head_dim)
            
            # Adapter les encodages relatifs pour le produit matriciel
            rel_pos_encodings_expanded = rel_pos_encodings.permute(0, 2, 1)  # [seq_len_q, head_dim, seq_len_k]
            
            # Calcul de la contribution de position relative
            rel_scores = torch.bmm(q_reshaped, rel_pos_encodings_expanded)
            rel_scores = rel_scores.reshape(seq_len_q, batch_size, self.n_heads, seq_len_k)
            rel_scores = rel_scores.permute(1, 2, 0, 3)  # [batch_size, n_heads, seq_len_q, seq_len_k]
            
            # Ajouter au score d'attention
            scores = scores + rel_scores
            
        # Appliquer le masque si fourni
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax et dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Appliquer l'attention aux valeurs
        output = torch.matmul(attention_weights, v)
        
        # Réorganiser et combiner les têtes
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Projection finale
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class MultiHorizonTemporalBlock(nn.Module):
    """
    Bloc de Transformer temporel amélioré qui peut traiter efficacement
    des dépendances à long terme et générer des prédictions multi-horizons.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_relative_positions: bool = True,
        max_relative_position: int = 32,
    ):
        super().__init__()
        
        # Couche d'attention multi-têtes avec support pour les positions relatives
        self.self_attn = LongRangeTemporalAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_relative_positions=use_relative_positions,
            max_relative_position=max_relative_position
        )
        
        # Couche feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Couches de normalisation et dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass du bloc avec connexions résiduelles et normalisation.
        
        Args:
            src: Tenseur d'entrée [batch_size, seq_len, d_model]
            src_mask: Masque d'attention [batch_size, 1, seq_len, seq_len] ou [batch_size, seq_len, seq_len]
            return_attention: Si True, retourne les poids d'attention
            
        Returns:
            Tuple contenant:
            - Tenseur de sortie [batch_size, seq_len, d_model]
            - Poids d'attention (facultatif)
        """
        # 1. Couche d'attention avec normalisation préalable
        src_norm = self.norm1(src)
        attn_output, attention_weights = self.self_attn(
            query=src_norm, key=src_norm, value=src_norm, 
            mask=src_mask, return_attention=return_attention
        )
        src = src + self.dropout1(attn_output)
        
        # 2. Couche feed-forward avec normalisation préalable
        src_norm = self.norm2(src)
        ff_output = self.feed_forward(src_norm)
        src = src + self.dropout2(ff_output)
        
        return src, attention_weights


class MultiHorizonTemporalTransformer(nn.Module):
    """
    Transformer spécialisé pour l'analyse temporelle multi-horizons des séries financières.
    Capable de générer des prédictions pour différents horizons temporels futurs
    et de capturer efficacement les dépendances à long terme.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 5000,
        forecast_horizons: List[int] = [1, 5, 10],
        use_relative_positions: bool = True,
        max_relative_position: int = 32,
        output_dim: int = 1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizons = forecast_horizons
        self.num_horizons = len(forecast_horizons)
        self.output_dim = output_dim
        
        # Couche d'entrée avec normalisation
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Encodage positionnel avec dropout
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Blocs Transformer empilés
        self.transformer_blocks = nn.ModuleList([
            MultiHorizonTemporalBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                use_relative_positions=use_relative_positions,
                max_relative_position=max_relative_position
            )
            for _ in range(num_layers)
        ])
        
        # Attention globale sur la séquence pour les prédictions multi-horizons
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout
        )
        
        # Têtes de prédiction pour chaque horizon
        self.forecast_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )
            for _ in range(self.num_horizons)
        ])
        
        # Initialisation des poids
        self._init_weights()
        
    def _init_weights(self):
        """Initialisation optimisée des poids pour la convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialisation spéciale des dernières couches de prédiction
        for head in self.forecast_heads:
            nn.init.zeros_(head[-1].bias)
            nn.init.xavier_uniform_(head[-1].weight, gain=0.01)
        
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Crée un masque causal (triangulaire inférieur) pour l'attention.
        Empêche les positions futures d'être visibles depuis les positions actuelles.
        
        Args:
            seq_len: Longueur de la séquence
            device: Périphérique sur lequel créer le masque
            
        Returns:
            Masque causal [1, seq_len, seq_len]
        """
        # Créer un masque triangulaire inférieur (1s en bas, 0s en haut)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # Inverser pour que les valeurs masquées soient 0 et les valeurs autorisées soient 1
        mask = ~mask
        # Ajouter la dimension de batch
        mask = mask.unsqueeze(0)
        return mask
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Union[
        Dict[int, torch.Tensor],
        Tuple[Dict[int, torch.Tensor], List[torch.Tensor]]
    ]:
        """
        Forward pass du modèle qui génère des prédictions pour plusieurs horizons temporels.
        
        Args:
            x: Tenseur d'entrée [batch_size, seq_len, input_dim]
            mask: Masque d'attention optionnel
            return_attentions: Si True, retourne également les poids d'attention
            
        Returns:
            - Sans return_attentions: Dict mappant les horizons à leurs prédictions
              {horizon: tensor[batch_size, output_dim]}
            - Avec return_attentions: Tuple (prédictions, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Créer un masque causal si aucun n'est fourni
        if mask is None:
            mask = self._create_causal_mask(seq_len, device)
            
        # Projection d'entrée et encodage positionnel
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Stocker les poids d'attention de chaque couche
        attention_weights = [] if return_attentions else None
        
        # Passage à travers les blocs Transformer
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask, return_attention=return_attentions)
            if return_attentions:
                attention_weights.append(attn_weights)
                
        # Extraction des derniers états cachés pour chaque position dans la séquence
        # Ces états seront utilisés pour prédire les valeurs à différents horizons
        
        # Pour les prédictions multi-horizons, nous avons besoin d'états cachés
        # qui capturent des contextes temporels différents
        
        # Attention globale pour pondérer les différentes positions temporelles
        # et créer un contexte pour chaque horizon
        query = x[:, -1, :].unsqueeze(1)  # Utiliser le dernier état comme requête
        x_permuted = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model] pour MultiheadAttention
        query_permuted = query.permute(1, 0, 2)  # [1, batch_size, d_model]
        
        global_context, _ = self.global_attention(
            query=query_permuted,
            key=x_permuted,
            value=x_permuted
        )
        global_context = global_context.permute(1, 0, 2)  # [batch_size, 1, d_model]
        
        # Créer des prédictions pour chaque horizon temporel
        predictions = {}
        for i, horizon in enumerate(self.forecast_horizons):
            # Pour chaque horizon, nous prenons le contexte global et appliquons
            # une tête de prédiction spécifique à cet horizon
            pred = self.forecast_heads[i](global_context.squeeze(1))
            predictions[horizon] = pred
            
        if return_attentions:
            return predictions, attention_weights
        else:
            return predictions
            
    def predict(
        self, 
        x: torch.Tensor,
        horizon_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Prédit les valeurs futures à un horizon spécifique ou à tous les horizons.
        
        Args:
            x: Tenseur d'entrée [batch_size, seq_len, input_dim]
            horizon_idx: Index de l'horizon à prédire (None = tous les horizons)
            
        Returns:
            Prédictions pour l'horizon spécifié ou pour tous les horizons
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            
            if horizon_idx is not None:
                # Retourner les prédictions pour un seul horizon
                horizon = self.forecast_horizons[horizon_idx]
                return predictions[horizon]
            else:
                # Combiner les prédictions de tous les horizons
                result = torch.cat(
                    [predictions[h] for h in self.forecast_horizons],
                    dim=1
                )
                return result
                
    def analyze_patterns(
        self, 
        x: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], List[torch.Tensor]]:
        """
        Analyse les motifs temporels dans les données et retourne
        les prédictions avec les poids d'attention pour interprétation.
        
        Args:
            x: Tenseur d'entrée [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple contenant:
            - Prédictions pour chaque horizon
            - Poids d'attention de chaque couche pour l'analyse des motifs
        """
        self.eval()
        with torch.no_grad():
            predictions, attention_weights = self.forward(x, return_attentions=True)
            return predictions, attention_weights 