"""
HRR (Heart Rate Recovery) Prediction Model

A hybrid architecture combining:
    - Bi-LSTM for local temporal feature extraction
    - Transformer for global temporal dependency modeling
    - Cross-Attention for dynamic-static feature fusion
    - MLP regressor (linear output for normalized regression)

Author: Your Name
License: MIT
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer.

    Args:
        d_model: Dimension of the model (embedding dimension).
        max_len: Maximum sequence length to pre-compute.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        return x + self.pe[:, : x.size(1), :]


class FinalPatentModel(nn.Module):
    """Bi-LSTM + Transformer + Cross-Attention for HRR prediction.

    This model processes dynamic temporal features through a Bi-LSTM and
    Transformer encoder, then fuses them with static features (age, gender,
    exercise mode, progress) via cross-attention, finally predicting a
    normalized HRR value via an MLP regressor with linear output.

    Args:
        dynamic_dim: Dimension of dynamic input features (default: 12).
            Typically 6 raw features + 6 diff features.
        static_dim: Dimension of static input features (default: 4).
            [age, gender, mode_id, progress]
        d_model: Hidden dimension of the model (default: 64).
        nhead: Number of attention heads (default: 8).
        num_transformer_layers: Number of Transformer encoder layers (default: 2).
        dropout: Dropout rate (default: 0.1).

    Input:
        - x_d: Dynamic features [batch, seq_len, dynamic_dim]
        - x_s: Static features [batch, static_dim]

    Output:
        - predictions: Normalized HRR values [batch] (unbounded linear output).
    """

    def __init__(
        self,
        dynamic_dim: int = 12,
        static_dim: int = 4,
        d_model: int = 64,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Temporal branch: Bi-LSTM ---
        self.lstm = nn.LSTM(
            input_size=dynamic_dim,
            hidden_size=d_model // 2,
            batch_first=True,
            bidirectional=True,
        )

        # --- Positional encoding ---
        self.pos_enc = PositionalEncoding(d_model)

        # --- Temporal branch: Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # --- Static branch: linear projection ---
        self.stat_enc = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
        )

        # --- Fusion: Cross-Attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        # --- Regression head ---
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            # Note: No activation (linear output) for normalized regression
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for better convergence."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self, x_d: torch.Tensor, x_s: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_d: Dynamic temporal features [batch, seq_len, dynamic_dim].
            x_s: Static features [batch, static_dim].

        Returns:
            Predicted normalized HRR values [batch] (unbounded).
        """
        # Step 1: Bi-LSTM extracts local temporal patterns
        x, _ = self.lstm(x_d)  # [B, T, d_model]

        # Step 2: Add positional encoding
        x = self.pos_enc(x)

        # Step 3: Transformer captures global dependencies
        x = self.transformer(x)  # [B, T, d_model]

        # Step 4: Project static features
        s = self.stat_enc(x_s).unsqueeze(1)  # [B, 1, d_model]

        # Step 5: Cross-attention fusion (static queries temporal)
        fused, attn_weights = self.cross_attn(query=s, key=x, value=x)
        # fused: [B, 1, d_model], attn_weights: [B, 1, T]

        # Step 6: Regression
        out = self.regressor(fused.squeeze(1))  # [B, 1]
        return out.squeeze(-1)  # [B]

    def get_attention_weights(
        self, x_d: torch.Tensor, x_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and return predictions + cross-attention weights.

        Useful for interpretability analysis.

        Returns:
            - predictions: [batch]
            - attn_weights: [batch, 1, seq_len] attention over time steps
        """
        x, _ = self.lstm(x_d)
        x = self.pos_enc(x)
        x = self.transformer(x)
        s = self.stat_enc(x_s).unsqueeze(1)
        fused, attn_weights = self.cross_attn(query=s, key=x, value=x)
        out = self.regressor(fused.squeeze(1)).squeeze(-1)
        return out, attn_weights
