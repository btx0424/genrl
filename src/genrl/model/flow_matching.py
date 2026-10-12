"""
Simple unconditional flow matching model.

Conditional flow matching on the linear path from noise to data. The network
predicts the velocity ``x_data - x_noise``, and sampling is Euler integration.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FlowMatching(nn.Module):
    """Rectified-flow objective. ``model(x, t)`` predicts velocity at time ``t`` in ``[0, 1]``.

    Time is kept away from 0 and 1. The velocity target is undefined in character
    at the endpoints, and evaluating there leaves high-frequency noise in the pose.
    """

    def __init__(self, model: nn.Module, t_eps: float = 1e-2):
        super().__init__()
        if not 0.0 <= t_eps < 0.5:
            raise ValueError(f"t_eps must lie in [0, 0.5), got {t_eps}.")
        self.model = model
        self.t_eps = float(t_eps)

    def loss(self, x1: torch.Tensor, cond_steps: int = 0) -> torch.Tensor:
        """Velocity loss. The first ``cond_steps`` frames stay equal to ``x1`` and are not scored."""
        if cond_steps < 0 or cond_steps >= x1.shape[1]:
            raise ValueError(f"cond_steps must be in [0, T), got {cond_steps} for T={x1.shape[1]}.")
        x0 = torch.randn_like(x1)
        if cond_steps:
            x0 = x0.clone()
            x0[:, :cond_steps] = x1[:, :cond_steps]
        t = torch.rand(x1.shape[0], device=x1.device)
        t = t.clamp(self.t_eps, 1.0 - self.t_eps)
        t_view = t.view(-1, *([1] * (x1.ndim - 1)))
        x_t = (1.0 - t_view) * x0 + t_view * x1
        if cond_steps:
            x_t = x_t.clone()
            x_t[:, :cond_steps] = x1[:, :cond_steps]
        target = x1 - x0
        error = (self.model(x_t, t) - target).square()
        if cond_steps:
            error = error[:, cond_steps:]
        return error.mean()

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        steps: int = 80,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate ``dx/dt = v_theta`` from ``t_eps`` to ``1 - t_eps``.

        ``prefix`` is a clean normalized ``(batch, cond_steps, feature)`` block.
        It is written back after every Euler step so the first frames stay observed.
        """
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}.")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
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
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, steps, device=device, dtype=dtype)
        dt = ts[1] - ts[0] if steps > 1 else (1.0 - 2.0 * self.t_eps)
        for t_scalar in ts:
            t = torch.full((shape[0],), float(t_scalar), device=device, dtype=dtype)
            velocity = self.model(x, t)
            if prefix is not None:
                velocity = velocity.clone()
                velocity[:, : prefix.shape[1]] = 0
            x = x + dt * velocity
            if prefix is not None:
                x[:, : prefix.shape[1]] = prefix
        return x
