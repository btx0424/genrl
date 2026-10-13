"""Small Diffusion Transformer over motion frames.

Each frame is one token. Time is injected with adaptive layer norm, initialized
so every block starts as the identity. ``forward(x, t)`` matches the U-Net:
``x`` is ``(batch, time, feature)`` and ``t`` is ``(batch,)`` in ``[0, 1]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from genrl.model.modules import ShapeConditioner, SinusoidalTimeEmbedding


def _sinusoidal_position(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
    )
    angles = position * freq.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class _DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden {hidden} must be divisible by heads {heads}.")
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1.0 + scale_attn[:, None]) + shift_attn[:, None]
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_attn[:, None] * attn
        h = self.norm2(x) * (1.0 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.mlp(h)


class MotionDiT(nn.Module):
    """Frame-token transformer for a fixed-length motion clip."""

    def __init__(
        self,
        data_dim: int,
        hidden: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        shape_dim: int = 1027,
        shape_drop: float = 0.1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}.")
        self.input = nn.Linear(data_dim, hidden)
        self.time_emb = SinusoidalTimeEmbedding(hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.shape_cond = ShapeConditioner(shape_dim, hidden, shape_drop)
        self.blocks = nn.ModuleList(_DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.output = nn.Linear(hidden, data_dim)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.time_mlp(self.time_emb(t)) + self.shape_cond(geom, x.shape[0])
        h = self.input(x)
        h = h + _sinusoidal_position(h.shape[1], h.shape[2], h.device, h.dtype)
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        h = self.final_norm(h) * (1.0 + scale[:, None]) + shift[:, None]
        return self.output(h)
