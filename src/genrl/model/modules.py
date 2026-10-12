"""
Minimal modules shared by Diffusion and Flow Matching models.

For fixed-length segments, a 1D U-Net over time is a fair baseline.
Both models call it as ``network(x, t)`` with ``x`` shaped ``(B, T, D)`` and a
continuous time ``t`` shaped ``(B,)`` in ``[0, 1]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _upsample_mode(name: str) -> str:
    key = name.lower().replace("-", "_")
    modes = {
        "nearest": "nearest",
        "linear": "linear",
        "convtranspose": "conv_transpose",
        "conv_transpose": "conv_transpose",
        "pixelshuffle": "pixel_shuffle",
        "pixel_shuffle": "pixel_shuffle",
    }
    if key not in modes:
        choices = "nearest, linear, conv_transpose, pixel_shuffle"
        raise ValueError(f"upsample must be one of {choices}, got {name!r}.")
    return modes[key]


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Map a continuous time in ``[0, 1]`` to a sinusoidal vector."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Scale into the same range as a 1000-step discrete schedule.
        t = t.float().reshape(-1) * 1000.0
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat((args.sin(), args.cos()), dim=-1)


class ResBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.time_proj = nn.Linear(time_dim, 2 * out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(temb)).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


def _match_length(x: torch.Tensor, target: int) -> torch.Tensor:
    length = x.shape[-1]
    if length == target:
        return x
    if length > target:
        return x[..., :target]
    return F.pad(x, (0, target - length))


class _Upsample1d(nn.Module):
    """Length upsample that keeps the channel count.

    ``conv_transpose`` is the inverse of the stride-2 downsample. ``pixel_shuffle``
    spreads channel pairs across time, which avoids the transposed-convolution
    checkerboard. Both are then cropped or padded onto the skip length.
    """

    def __init__(self, channels: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "conv_transpose":
            self.op = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
        elif mode == "pixel_shuffle":
            self.expand = nn.Conv1d(channels, channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, target: int) -> torch.Tensor:
        if self.mode == "linear":
            return F.interpolate(x, size=target, mode="linear", align_corners=False)
        if self.mode == "nearest":
            return F.interpolate(x, size=target, mode="nearest")
        if self.mode == "conv_transpose":
            return _match_length(self.op(x), target)
        expanded = self.expand(x)
        batch, channels, length = expanded.shape
        shuffled = expanded.view(batch, channels // 2, 2, length).permute(0, 1, 3, 2).contiguous()
        return _match_length(shuffled.view(batch, channels // 2, length * 2), target)


class TemporalUNet(nn.Module):
    """1D U-Net. ``x`` is ``(batch, time, channels)``."""

    def __init__(
        self,
        data_dim: int,
        model_dim: int = 128,
        multipliers: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        kernel_size: int = 3,
        upsample: str = "nearest",
    ):
        super().__init__()
        if not multipliers:
            raise ValueError("multipliers must be non-empty.")
        upsample = _upsample_mode(upsample)
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
        self.upsample = upsample
        channels = [model_dim * mult for mult in multipliers]
        padding = kernel_size // 2
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_conv = nn.Conv1d(data_dim, channels[0], kernel_size=kernel_size, padding=padding)

        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            self.down.append(
                nn.ModuleList(
                    [
                        ResBlock1d(in_ch, out_ch, time_dim, kernel_size),
                        ResBlock1d(out_ch, out_ch, time_dim, kernel_size),
                    ]
                )
            )
            in_ch = out_ch
            if level < len(channels) - 1:
                self.pool.append(nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))

        self.mid = nn.ModuleList(
            [
                ResBlock1d(in_ch, in_ch, time_dim, kernel_size),
                ResBlock1d(in_ch, in_ch, time_dim, kernel_size),
            ]
        )

        self.upsamplers = nn.ModuleList(
            _Upsample1d(channels, upsample) for channels in reversed(channels[1:])
        )
        self.up = nn.ModuleList()
        for skip_ch in reversed(channels):
            self.up.append(
                nn.ModuleList(
                    [
                        ResBlock1d(in_ch + skip_ch, skip_ch, time_dim, kernel_size),
                        ResBlock1d(skip_ch, skip_ch, time_dim, kernel_size),
                    ]
                )
            )
            in_ch = skip_ch

        self.out_norm = nn.GroupNorm(_groups(in_ch), in_ch)
        self.out_conv = nn.Conv1d(in_ch, data_dim, kernel_size=kernel_size, padding=padding)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(self.time_emb(t))
        h = self.in_conv(x.transpose(1, 2))
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            if level < len(self.pool):
                h = self.pool[level](h)
        for block in self.mid:
            h = block(h, temb)
        upsample_index = 0
        for blocks in self.up:
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = self.upsamplers[upsample_index](h, skip.shape[-1])
                upsample_index += 1
            h = torch.cat((h, skip), dim=1)
            for block in blocks:
                h = block(h, temb)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h.transpose(1, 2)
