"""Gated Cumulative Layer -- an attention replacement.

Two causal cumulative stages + a final per-channel Q*K gate.

Spec (from the user), with the obvious typo fix (stage-1 description says
"W_K is like a value matrix" -- the formula uses W_V, so W_V is the value):

  Stage 1 (global, per-channel gate):
      H_global_n = sum_{i=0..n} ( W_V(x_i) * W_G )
    W_V : linear value projection (per token).
    W_G : a *constant* learnable per-channel gate (vector of length D),
          pushed through sigmoid so each channel's gate lives in (0, 1).
          "Gates should be per channel" -> W_G has shape (D,).

  Stage 2 (main, input-dependent per-channel gate):
      H_main_n = sum_{i=0..n} ( W_V(x_i) * W_G_M(H_global_i) )
    W_G_M : a function (linear -> sigmoid) that turns the stage-1 global
            representation H_global_i into a per-channel gate. Reusing the
            same value V_i = W_V(x_i) as in stage 1 (the spec writes W_V in
            both stages).

  Final per-channel query/key gate:
      x_new_n = W_Q(x_n) * W_K(H_main_n)
    W_Q, W_K : linear projections. "*" is elementwise (Hadamard) so the
              output keeps the per-channel structure (B, T, D), which is what
              you need to drop this in where attention used to be.

Everything is causal: the cumulative sums run from 0 to n, so position n
only ever sees positions <= n. Work is O(T*D) (two cumsums), no T^2 matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCumulativeLayer(nn.Module):
    """Two-stage gated cumulative layer that replaces self-attention."""

    def __init__(
        self,
        d_model: int,
        gate_bias: bool = True,
        value_bias: bool = False,
        qk_bias: bool = False,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Value projection, shared by both stages (spec uses W_V in both).
        self.W_V = nn.Linear(d_model, d_model, bias=value_bias)

        # Stage-1 gate: a single learnable per-channel vector.
        # Sigmoid -> each channel gate in (0, 1). gate_init biases it near 0.5.
        self.W_G = nn.Parameter(torch.empty(d_model))
        nn.init.uniform_(
            self.W_G,
            -1e-2 + gate_init,
            1e-2 + gate_init,
        )

        # Stage-2 gate: function H_global -> per-channel gate logits.
        self.W_G_M = nn.Linear(d_model, d_model, bias=gate_bias)

        # Final query / key projections.
        self.W_Q = nn.Linear(d_model, d_model, bias=qk_bias)
        self.W_K = nn.Linear(d_model, d_model, bias=qk_bias)

        # Output norm keeps the cumsums from blowing up over long sequences,
        # the same role LayerNorm plays around an attention sublayer. It does
        # not change the layer's expressivity, just its training stability.
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D). Causal."""
        # Value, shared.
        v = self.W_V(x)                       # (B, T, D)

        # ---- Stage 1: global gated cumulative ----
        g1 = torch.sigmoid(self.W_G)          # (D,) per channel
        gated1 = v * g1                        # (B, T, D)
        h_global = torch.cumsum(gated1, dim=1)  # (B, T, D), causal

        # ---- Stage 2: input-dependent gated cumulative ----
        g2 = torch.sigmoid(self.W_G_M(h_global))  # (B, T, D) per channel
        gated2 = v * g2                          # (B, T, D)
        h_main = torch.cumsum(gated2, dim=1)     # (B, T, D), causal

        # ---- Final per-channel Q * K gate ----
        q = self.W_Q(x)                       # (B, T, D)
        k = self.W_K(h_main)                 # (B, T, D)
        x_new = q * k                         # (B, T, D) Hadamard
        return self.out_norm(x_new)


def make_block(d_model: int, layer: nn.Module, ffn_mult: int = 4, dropout: float = 0.1):
    """Wrap an attention-replacement sublayer into a transformer block:
    pre-norm + sublayer + residual, then pre-norm + FFN + residual.
    Identical scaffolding whether `layer` is ours or nn.MultiheadAttention,
    so the comparison is fair."""
    return TransformerBlock(d_model, layer, ffn_mult, dropout)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, attn, ffn_mult, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, D)
        h = self.n1(x)
        a = self.attn(h)
        # Make sure attention-replacement output lines up with x.
        if a.shape != x.shape:
            raise ValueError(f"attn output {a.shape} != input {x.shape}")
        x = x + a
        x = x + self.ffn(self.n2(x))
        return x


class MultiheadAttentionAdapter(nn.Module):
    """Adapter so nn.MultiheadAttention has the same (B, T, D) -> (B, T, D)
    call signature and output shape as our layer."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        # Causal mask so the comparison is apples-to-apples with our layer.
        mask = torch.triu(
            torch.full((T, T), float("-inf"), dtype=x.dtype, device=x.device),
            diagonal=1,
        )
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return out


if __name__ == "__main__":
    # Quick smoke test.
    layer = GatedCumulativeLayer(d_model=32)
    x = torch.randn(2, 10, 32)
    y = layer(x)
    print("smoke ok:", tuple(x.shape), "->", tuple(y.shape))
    print("params:", sum(p.numel() for p in layer.parameters()))
