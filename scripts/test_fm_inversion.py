"""Check that flow-matching sampling inverts.

A constant-velocity field must round-trip exactly. The trained checkpoint is
then sampled from a known noise, inverted, and sampled again. Both the noise
and the clip should come back.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from genrl.data.geometry import object_shape_table
from genrl.model.flow_matching import FlowMatching
from genrl.train_common import build_model, load_raw_segments

ROOT = Path(__file__).resolve().parents[1]


class _ConstantVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor, t: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        del t, geom
        return torch.ones_like(x) * self.scale


def _latest_checkpoint() -> Path:
    candidates = sorted((ROOT / "outputs" / "flow_matching-dit-geom").glob("*/checkpoints/last.pt"))
    if not candidates:
        raise FileNotFoundError("No flow_matching-dit-geom checkpoint under outputs/.")
    return candidates[-1]


def _report(name: str, recovered: torch.Tensor, original: torch.Tensor) -> float:
    gap = (recovered - original).abs()
    worst = float(gap.max())
    print(f"{name}: max {worst:.3e}  mean {float(gap.mean()):.3e}")
    return worst


def _constant_field(steps: int, iters: int) -> None:
    flow = FlowMatching(_ConstantVelocity(), t_eps=1e-2, guidance=1.0)
    prefix = torch.randn(2, 1, 4)
    source = torch.randn(2, 8, 4)
    source[:, :1] = prefix
    clip = flow.sample((2, 8, 4), steps=steps, prefix=prefix, source=source)
    recovered = flow.invert(clip, steps=steps, prefix=prefix, iters=iters)
    again = flow.sample((2, 8, 4), steps=steps, prefix=prefix, source=recovered)
    print("constant velocity")
    _report("  noise", recovered, source)
    _report("  clip", again, clip)


def _checkpoint(path: Path, count: int, steps: int, iters: int, device: torch.device) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(saved["config"])
    flow = build_model(cfg)
    flow.load_state_dict(saved["ema"])
    flow.to(device).eval()
    raw, _fps, _anchors, _skip, names = load_raw_segments(cfg)
    features = (raw - saved["mean"]) / saved["std"]
    shapes = object_shape_table(Path(cfg.data.root), names)
    index = torch.arange(count)
    prefix = features[index, :1].to(device)
    geom = shapes[index].to(device)
    source = torch.randn(count, features.shape[1], features.shape[2], device=device)
    source[:, :1] = prefix
    with torch.no_grad():
        clip = flow.sample(tuple(source.shape), steps=steps, prefix=prefix, geom=geom, source=source)
        recovered = flow.invert(clip, steps=steps, prefix=prefix, geom=geom, iters=iters)
        again = flow.sample(tuple(source.shape), steps=steps, prefix=prefix, geom=geom, source=recovered)
    print(f"checkpoint {path}")
    noise = _report("  noise", recovered, source)
    clip_gap = _report("  clip", again, clip)
    if noise > 1e-3 or clip_gap > 1e-3:
        raise SystemExit("round trip is above 1e-3; increase --iters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _constant_field(args.steps, args.iters)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    _checkpoint(args.checkpoint or _latest_checkpoint(), args.count, args.steps, args.iters, device)


if __name__ == "__main__":
    main()
