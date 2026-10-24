"""Held-out first frames through the flow-matching policy.

Pins the real frame 0 of clips from a chosen split (default: `train`, which the
current checkpoint never saw) and integrates the EMA flow from there. Writes one
GIF per prefix and reports the same consistency / rot6d metrics the training log
uses, plus the error against the clip's real continuation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from genrl.data.geometry import object_shape_table
from genrl.data.segments import (
    FEATURE_DIM,
    compose_rotation_anchor,
    object_motion_before_touch,
    position_velocity_consistency,
    project_rot6d,
    rot6d_gram_error,
)
from genrl.model.flow_matching import FlowMatching
from genrl.train_common import _guidance, build_backbone, load_raw_segments
from genrl.viz.render import expand_time
from genrl.viz.warp_clips import WarpSampleRenderer

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "flow_eval")
    parser.add_argument("--render", action="store_true", help="write GIFs with mujoco_warp")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def load_flow(checkpoint: Path, device: torch.device) -> tuple[FlowMatching, OmegaConf]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(saved["config"])
    flow = FlowMatching(
        build_backbone(config),
        t_eps=float(config.t_eps),
        guidance=_guidance(config),
    )
    state = saved.get("ema", saved["model"])
    flow.load_state_dict(state)
    flow.to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    print(f"loaded {'EMA' if 'ema' in saved else 'raw'} weights from {checkpoint} at step {saved['step']}")
    return flow, config, saved


def pick_indices(names: np.ndarray, count: int) -> np.ndarray:
    """A deterministic spread over clips, one at a time per object in turn."""
    order = np.argsort(names, kind="stable")
    per_object: list[np.ndarray] = []
    for name in dict.fromkeys(str(n) for n in names[order]):
        rows = order[names[order] == name]
        take = max(1, count // len(dict.fromkeys(str(n) for n in names)))
        stride = max(1, len(rows) // take)
        per_object.append(rows[::stride][:take])
    chosen = np.concatenate(per_object)
    return chosen[:count]


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    flow, config, saved = load_flow(args.checkpoint, device)

    mean, std = saved["mean"].to(device), saved["std"].to(device)
    data = OmegaConf.create(OmegaConf.to_container(config.data, resolve=True))
    data.split = args.split
    data.objects = list(args.objects)
    cfg = OmegaConf.create({"data": data})

    raw, fps, anchors, skip_tail, names = load_raw_segments(cfg)
    features = ((raw - mean) / std).to(device)
    shapes = object_shape_table(Path(config.data.root), names).to(device)
    print(f"held-out split {args.split!r}: {features.shape[0]} clips of {features.shape[1]} frames at {fps} Hz")

    index = torch.as_tensor(pick_indices(names, args.num_clips), device=device)
    prefix = features[index, :1]
    geom = shapes[index]
    selection = names[index.detach().cpu().numpy()]
    print(f"pinned first frames: {list(selection)}")

    torch.manual_seed(args.seed)
    with torch.no_grad():
        generated = flow.sample(
            (index.shape[0], features.shape[1], FEATURE_DIM),
            steps=args.steps,
            prefix=prefix,
            geom=geom,
        )
    physical = generated * std + mean
    real = raw[index.detach().cpu().numpy()]

    # The pinned frame is copied verbatim, so score only the frames after it.
    joint = slice(0, 53)
    gap = (physical[:, 1:, joint] - torch.as_tensor(real[:, 1:, joint])).abs().mean().item()
    spread = torch.as_tensor(real[:, 1:, joint]).std().item()
    print(f"policy frame-1.. : joint error {gap:.4f} rad   held-out spread {spread:.4f} rad")
    print(f"  imitates as well as a constant predictor? {'no' if gap < spread else 'yes'}")

    for label, clips in (("generated", physical), ("real", torch.as_tensor(real))):
        consistency = position_velocity_consistency(clips, fps)
        rotation = rot6d_gram_error(clips, skip_frames=1)
        motion = object_motion_before_touch(clips, skip_tail=skip_tail)
        print(
            f"{label:9s} consistency "
            + ", ".join(f"{k} {v:.4f}" for k, v in consistency.items())
        )
        print(
            f"{label:9s} rot6d       "
            + ", ".join(f"{k} {v:.4f}" for k, v in rotation.items())
        )
        print(f"{label:9s} object speed before touch {motion['speed_before_touch']:.4f} m/s")

    if not args.render:
        return
    shown = project_rot6d(physical)
    shown = compose_rotation_anchor(shown, anchors[index].to(device))
    clips = expand_time(shown.detach().cpu().numpy(), fps, 50)
    renderer = WarpSampleRenderer(config.data.root, config.robot.urdf, width=args.width, height=args.height)
    paths = renderer.write_gifs(clips, selection, args.out / args.split, 50)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
