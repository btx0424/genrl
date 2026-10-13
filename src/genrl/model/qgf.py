"""Q-guided flow (QGF) on top of a behavior flow.

The reference policy is the usual flow-matching model. A separate IQL critic
scores a whole clip, and at sampling time its gradient steers the velocity.
State and action are the same tensor: the full feature, not a split of it.
Guidance follows Algorithm 1 of Zhou et al., arXiv:2606.11087. The one-step
estimate of the clean clip is ``a_t + (1 - t) v``, and the critic gradient is
taken there. The Jacobian of that estimate is replaced by the identity, which
is the variant the paper finds more stable.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from genrl.model.flow_matching import FlowMatching


def expectile_loss(residual: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric square loss. ``expectile`` in (0.5, 1] weights positive residuals more."""
    if not 0.5 < expectile <= 1.0:
        raise ValueError(f"expectile must lie in (0.5, 1], got {expectile}.")
    weight = torch.where(residual < 0, 1.0 - expectile, expectile)
    return weight * residual.square()


class _ClipEncoder(nn.Module):
    """Per-frame conv stack, pooled over time. Input is ``(batch, time, feature)``."""

    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.transpose(1, 2)).mean(dim=-1)


class ClipQ(nn.Module):
    """Q(s, a) with one shared encoder. Both arguments are the full feature."""

    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.encoder = _ClipEncoder(feature_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(state) + self.encoder(action)
        return self.head(hidden).squeeze(-1)


class ClipV(nn.Module):
    """State value. The state is the full feature."""

    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.encoder = _ClipEncoder(feature_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state)).squeeze(-1)


class IQL(nn.Module):
    """In-sample IQL critic. An ensemble of two Qs is aggregated by the minimum.

    Each clip is one terminal transition: there is no next segment, so the Q
    target is the reward and ``discount`` does not enter. V still regresses to
    the upper expectile of Q, which is the other half of the IQL update.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 256,
        ensemble: int = 2,
        expectile: float = 0.9,
        discount: float = 0.0,
    ):
        super().__init__()
        if ensemble < 1:
            raise ValueError(f"ensemble must be positive, got {ensemble}.")
        if not 0.0 <= discount < 1.0:
            raise ValueError(f"discount must lie in [0, 1), got {discount}.")
        self.expectile = float(expectile)
        self.discount = float(discount)
        self.q = nn.ModuleList(ClipQ(feature_dim, hidden) for _ in range(ensemble))
        self.v = ClipV(feature_dim, hidden)
        self.q_target = copy.deepcopy(self.q)
        self.v_target = copy.deepcopy(self.v)
        self.q_target.requires_grad_(False)
        self.v_target.requires_grad_(False)

    def q_values(self, state: torch.Tensor, action: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        """Stacked ensemble values, shape ``(ensemble, batch)``."""
        critics = self.q_target if target else self.q
        return torch.stack([critic(state, action) for critic in critics], dim=0)

    def q_min(self, state: torch.Tensor, action: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        return self.q_values(state, action, target=target).min(dim=0).values

    def losses(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Q regression onto ``reward``, and expectile regression of V onto Q.

        ``reward`` is ``(batch,)`` and already scaled. State and action are the
        same clip at training time; they stay separate arguments so sampling can
        differentiate the action only.
        """
        if self.discount != 0.0:
            raise ValueError(
                "Each clip is a terminal transition, so discount must be 0. "
                f"Got {self.discount}."
            )
        with torch.no_grad():
            target_q = self.q_min(state, action, target=True)
        backup = reward
        q_loss = (self.q_values(state, action) - backup).square().mean()
        value_error = target_q - self.v(state)
        v_loss = expectile_loss(value_error, self.expectile).mean()
        return q_loss, v_loss

    def reward_loss(
        self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        """Squared error of every ensemble member against a known reward.

        Used for clips whose reward can be evaluated directly, including clips
        drawn from the flow. V stays on the dataset expectile.
        """
        return (self.q_values(state, action) - reward).square().mean()


def qgf_sample(
    flow: FlowMatching,
    critic,
    shape: tuple[int, ...],
    steps: int = 80,
    guidance: float = 0.1,
    prefix: torch.Tensor | None = None,
    geom: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler integration with the QGF velocity ``v + guidance * grad Q``.

    ``critic(state, action)`` returns one value per batch row. The state passed
    in is the stop-gradient clean estimate, so the gradient is only through the
    action, matching ``grad_{a_hat} Q(s, a_hat)``. ``guidance`` is ``1 / beta``.
    Gradients are enabled inside the call so it can run under ``torch.no_grad``.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}.")
    device = next(flow.parameters()).device
    dtype = next(flow.parameters()).dtype
    x = torch.randn(shape, device=device, dtype=dtype)
    cond = 0
    if prefix is not None:
        if prefix.ndim != 3 or prefix.shape[0] != shape[0] or prefix.shape[2] != shape[2]:
            raise ValueError(f"prefix shape {tuple(prefix.shape)} does not match batch {shape}.")
        if prefix.shape[1] >= shape[1]:
            raise ValueError("prefix must be shorter than the generated clip.")
        prefix = prefix.to(device=device, dtype=dtype)
        cond = int(prefix.shape[1])
        x[:, :cond] = prefix
    if geom is not None:
        geom = geom.to(device=device, dtype=dtype)
    ts = torch.linspace(flow.t_eps, 1.0 - flow.t_eps, steps, device=device, dtype=dtype)
    dt = ts[1] - ts[0] if steps > 1 else (1.0 - 2.0 * flow.t_eps)
    with torch.enable_grad():
        for t_scalar in ts:
            t_value = float(t_scalar)
            t = torch.full((shape[0],), t_value, device=device, dtype=dtype)
            with torch.no_grad():
                velocity = flow.predict(x, t, geom)
                if cond:
                    velocity = velocity.clone()
                    velocity[:, :cond] = 0
            # One Euler step all the way to t = 1. Identity Jacobian: do not
            # differentiate through the velocity network.
            clean = (x + (1.0 - t_value) * velocity).detach().requires_grad_(True)
            value = critic(clean.detach(), clean)
            gradient, = torch.autograd.grad(value.sum(), clean)
            if cond:
                gradient = gradient.clone()
                gradient[:, :cond] = 0
            velocity = velocity + guidance * gradient.detach()
            x = (x + dt * velocity).detach()
            if cond:
                x[:, :cond] = prefix
    return x
