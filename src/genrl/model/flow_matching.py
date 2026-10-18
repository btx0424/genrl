"""
Shape-conditioned flow matching model.

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

    def __init__(self, model: nn.Module, t_eps: float = 1e-2, guidance: float = 1.25):
        super().__init__()
        if not 0.0 <= t_eps < 0.5:
            raise ValueError(f"t_eps must lie in [0, 0.5), got {t_eps}.")
        if guidance < 0.0:
            raise ValueError(f"guidance must be non-negative, got {guidance}.")
        self.model = model
        self.t_eps = float(t_eps)
        self.guidance = float(guidance)

    def time_grid(self, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Times from ``t_eps`` to ``1 - t_eps``. ``steps`` is the number of nodes.

        Euler takes ``steps - 1`` updates and lands on the last node. A single
        step is the whole interval.
        """
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}.")
        if steps == 1:
            return torch.tensor([self.t_eps, 1.0 - self.t_eps], device=device, dtype=dtype)
        return torch.linspace(self.t_eps, 1.0 - self.t_eps, steps, device=device, dtype=dtype)

    def _pinned_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        geom: torch.Tensor | None,
        cond_steps: int,
    ) -> torch.Tensor:
        velocity = self.predict(x, t, geom)
        if cond_steps:
            velocity = velocity.clone()
            velocity[:, :cond_steps] = 0
        return velocity

    def predict(self, x: torch.Tensor, t: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        """Velocity. ``geom is None`` is the unconditional branch.

        Classifier-free guidance mixes the two when a shape is present and
        ``guidance`` is not 1. The pinned prefix is left to the caller.
        """
        if geom is None or self.guidance == 1.0:
            return self.model(x, t, geom)
        conditional = self.model(x, t, geom)
        unconditional = self.model(x, t, None)
        return unconditional + self.guidance * (conditional - unconditional)

    def loss(self, x1: torch.Tensor, cond_steps: int = 0, geom: torch.Tensor | None = None) -> torch.Tensor:
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
        error = (self.model(x_t, t, geom) - target).square()
        if cond_steps:
            error = error[:, cond_steps:]
        return error.mean()

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        steps: int = 80,
        prefix: torch.Tensor | None = None,
        geom: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate ``dx/dt = v_theta`` from ``t_eps`` to ``1 - t_eps``.

        ``steps`` is the number of time nodes, so the integrator takes
        ``steps - 1`` Euler updates and stops on the last node. ``prefix`` is a
        clean normalized ``(batch, cond_steps, feature)`` block, written back
        after every update. ``source`` is the state at ``t_eps``; a missing
        source is fresh Gaussian noise.
        """
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}.")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if source is None:
            x = torch.randn(shape, device=device, dtype=dtype)
        else:
            if tuple(source.shape) != tuple(shape):
                raise ValueError(f"source shape {tuple(source.shape)} does not match {shape}.")
            x = source.to(device=device, dtype=dtype).clone()
        cond = 0
        if prefix is not None:
            if prefix.ndim != 3 or prefix.shape[0] != shape[0] or prefix.shape[2] != shape[2]:
                raise ValueError(
                    f"prefix shape {tuple(prefix.shape)} does not match batch {shape}."
                )
            if prefix.shape[1] >= shape[1]:
                raise ValueError("prefix must be shorter than the generated clip.")
            prefix = prefix.to(device=device, dtype=dtype)
            cond = int(prefix.shape[1])
            x[:, :cond] = prefix
        if geom is not None:
            geom = geom.to(device=device, dtype=dtype)
            if geom.shape[0] != shape[0]:
                raise ValueError(f"geom batch {geom.shape[0]} does not match {shape[0]}.")
        ts = self.time_grid(steps, device, dtype)
        for i in range(ts.shape[0] - 1):
            t = torch.full((shape[0],), float(ts[i]), device=device, dtype=dtype)
            velocity = self._pinned_velocity(x, t, geom, cond)
            x = x + (ts[i + 1] - ts[i]) * velocity
            if cond:
                x[:, :cond] = prefix
        return x

    @torch.no_grad()
    def invert(
        self,
        clip: torch.Tensor,
        steps: int = 80,
        prefix: torch.Tensor | None = None,
        geom: torch.Tensor | None = None,
        iters: int = 12,
    ) -> torch.Tensor:
        """Recover the ``t_eps`` state that ``sample`` would integrate to ``clip``.

        Each Euler update ``x_next = x + dt v(x, t)`` is solved for ``x`` by
        fixed-point iteration. Twelve evaluations per step recover the training
        sampler's noise to about 1e-5.
        """
        if iters < 1:
            raise ValueError(f"iters must be positive, got {iters}.")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = clip.to(device=device, dtype=dtype).clone()
        cond = 0
        if prefix is not None:
            if prefix.shape[0] != x.shape[0] or prefix.shape[2] != x.shape[2]:
                raise ValueError(
                    f"prefix shape {tuple(prefix.shape)} does not match clip {tuple(x.shape)}."
                )
            prefix = prefix.to(device=device, dtype=dtype)
            cond = int(prefix.shape[1])
            x[:, :cond] = prefix
        if geom is not None:
            geom = geom.to(device=device, dtype=dtype)
            if geom.shape[0] != x.shape[0]:
                raise ValueError(f"geom batch {geom.shape[0]} does not match {x.shape[0]}.")
        ts = self.time_grid(steps, device, dtype)
        for i in range(ts.shape[0] - 2, -1, -1):
            t = torch.full((x.shape[0],), float(ts[i]), device=device, dtype=dtype)
            dt = ts[i + 1] - ts[i]
            x_next = x
            for _ in range(iters):
                velocity = self._pinned_velocity(x, t, geom, cond)
                x = x_next - dt * velocity
                if cond:
                    x[:, :cond] = prefix
        return x
