"""
Simple unconditional diffusion model.

Epsilon-prediction DDPM with a linear noise schedule. Sampling uses DDIM.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def _noise_error(pred: torch.Tensor, noise: torch.Tensor, cond_steps: int) -> torch.Tensor:
    """Per-sample mean squared noise error. Shape ``(batch,)``."""
    error = (pred - noise).square()
    if cond_steps:
        error = error[:, cond_steps:]
    return error.mean(dim=tuple(range(1, error.ndim)))


def _extract(values: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    out = values.gather(0, t.long())
    return out.view(t.shape[0], *([1] * (like.ndim - 1)))


class GaussianDiffusion(nn.Module):
    """Epsilon-prediction DDPM. ``model(x, t)`` predicts noise; ``t`` is in ``[0, 1]``.

    The first ``cond_steps`` frames are an observed prefix. They stay equal to the
    clean clip in the loss and are written back after every DDIM step.
    """

    def __init__(self, model: nn.Module, timesteps: int = 1000):
        super().__init__()
        if timesteps < 2:
            raise ValueError(f"timesteps must be at least 2, got {timesteps}.")
        self.model = model
        self.num_timesteps = timesteps
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def corrupt(self, x0: torch.Tensor, cond_steps: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward noising step. Returns ``x_t``, the noise, and ``t`` in ``[0, 1]``."""
        if cond_steps < 0 or cond_steps >= x0.shape[1]:
            raise ValueError(f"cond_steps must be in [0, T), got {cond_steps} for T={x0.shape[1]}.")
        batch = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = (
            _extract(self.sqrt_alphas_cumprod, t, x0) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise
        )
        if cond_steps:
            x_t = x_t.clone()
            x_t[:, :cond_steps] = x0[:, :cond_steps]
        return x_t, noise, t.float() / self.num_timesteps

    def loss(self, x0: torch.Tensor, cond_steps: int = 0) -> torch.Tensor:
        """Noise loss. The first ``cond_steps`` frames stay equal to ``x0`` and are not scored."""
        x_t, noise, t_frac = self.corrupt(x0, cond_steps)
        pred = self.model(x_t, t_frac)
        return _noise_error(pred, noise, cond_steps).mean()

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        steps: int = 80,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM sample (eta = 0). ``prefix`` is a clean normalized first-frame block."""
        if steps < 2:
            raise ValueError(f"steps must be at least 2, got {steps}.")
        device = self.alphas_cumprod.device
        dtype = next(self.parameters()).dtype
        idx = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).round().long()
        keep = torch.ones_like(idx, dtype=torch.bool)
        keep[1:] = idx[1:] != idx[:-1]
        idx = idx[keep]
        x = torch.randn(shape, device=device, dtype=dtype)
        if prefix is not None:
            if prefix.ndim != 3 or prefix.shape[0] != shape[0] or prefix.shape[2] != shape[2]:
                raise ValueError(
                    f"prefix shape {tuple(prefix.shape)} does not match batch {shape}."
                )
            if prefix.shape[1] >= shape[1]:
                raise ValueError("prefix must be shorter than the generated clip.")
            prefix = prefix.to(device=device, dtype=dtype)
            x[:, : prefix.shape[1]] = prefix
        for i, t_value in enumerate(idx):
            t = torch.full((shape[0],), int(t_value), device=device, dtype=torch.long)
            pred = self.model(x, t.float() / self.num_timesteps)
            abar = _extract(self.alphas_cumprod, t, x)
            x0 = (x - torch.sqrt(1.0 - abar) * pred) / torch.sqrt(abar).clamp_min(1e-8)
            prev = int(idx[i + 1]) if i + 1 < len(idx) else -1
            if prev < 0:
                x = x0
            else:
                t_prev = torch.full((shape[0],), prev, device=device, dtype=torch.long)
                abar_prev = _extract(self.alphas_cumprod, t_prev, x)
                x = torch.sqrt(abar_prev) * x0 + torch.sqrt(1.0 - abar_prev) * pred
            if prefix is not None:
                x[:, : prefix.shape[1]] = prefix
        return x
