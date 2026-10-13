"""Compare QGF guidance weights on one frozen flow and one critic checkpoint.

The same noise and the same pinned first frames are used for every weight, so
the speed differences are paired. Guidance 0 is the frozen flow by itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from genrl.data.geometry import object_shape_table
from genrl.data.segments import FEATURE_DIM, clip_penalties
from genrl.model.flow_matching import FlowMatching
from genrl.model.qgf import IQL, qgf_sample
from genrl.train_common import _guidance, build_backbone, load_raw_segments

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUIDANCE = (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
PENALTY_NAMES = (
    "object_vel",
    "object_pos",
    "object_spin",
    "consistency_object",
    "consistency_pelvis",
    "consistency_joint",
)


def _latest_checkpoint() -> Path:
    candidates = list((ROOT / "outputs" / "qgf").glob("*/checkpoints/last.pt"))
    if not candidates:
        raise FileNotFoundError(f"No QGF checkpoint under {ROOT / 'outputs' / 'qgf'}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "critic" not in saved or "flow" not in saved:
        raise ValueError(f"{checkpoint} is not a QGF checkpoint.")
    config = OmegaConf.create(saved["config"])
    flow_path = Path(str(config.flow))
    if not flow_path.is_file():
        raise FileNotFoundError(f"Flow checkpoint recorded in the QGF run is missing: {flow_path}")
    flow_saved = torch.load(flow_path, map_location="cpu", weights_only=False)
    flow_config = OmegaConf.create(flow_saved["config"])
    flow = FlowMatching(
        build_backbone(flow_config),
        t_eps=float(flow_config.t_eps),
        guidance=_guidance(flow_config),
    )
    flow.load_state_dict(saved["flow"])
    flow.to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    critic = IQL(
        FEATURE_DIM,
        hidden=int(config.iql.hidden),
        ensemble=int(config.iql.ensemble),
        expectile=float(config.iql.expectile),
        discount=float(config.iql.discount),
    )
    critic.load_state_dict(saved["critic"])
    critic.to(device).eval()
    return saved, config, flow_config, flow, critic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="QGF last.pt. Defaults to the newest run.")
    parser.add_argument("--guidance", type=float, nargs="+", default=list(DEFAULT_GUIDANCE))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(weight < 0 for weight in args.guidance):
        raise ValueError(f"Guidance weights must be non-negative, got {args.guidance}.")
    if 0.0 not in args.guidance:
        args.guidance = [0.0, *args.guidance]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or _latest_checkpoint()
    saved, config, flow_config, flow, critic = _load(checkpoint, device)
    raw, fps, _anchors, skip_tail, object_names = load_raw_segments(
        OmegaConf.create({"data": config.data})
    )
    shapes = object_shape_table(Path(config.data.root), object_names)
    mean = saved["mean"].to(device)
    std = saved["std"].to(device)
    features = (raw - saved["mean"]) / saved["std"]
    if "penalty_scales" not in saved:
        raise ValueError(
            f"{checkpoint} was trained on object speed only. "
            "Train a new critic after the motion and consistency rewards were added."
        )
    real = {name: float(value.mean()) for name, value in clip_penalties(raw, fps, skip_tail).items()}
    print(f"checkpoint {checkpoint} step {int(saved['step'])}", flush=True)

    cond_steps = int(flow_config.cond_steps)
    length = int(features.shape[1])
    count = features.shape[0]
    scores = {weight: {name: [] for name in PENALTY_NAMES} for weight in args.guidance}
    values = {weight: [] for weight in args.guidance}
    for batch_id in range(args.batches):
        start = (batch_id * args.batch_size) % count
        index = torch.arange(start, start + args.batch_size) % count
        prefix = features[index, :cond_steps].to(device) if cond_steps else None
        for weight in args.guidance:
            torch.manual_seed(args.seed + batch_id)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + batch_id)
            sample = qgf_sample(
                flow,
                critic.q_min,
                (args.batch_size, length, FEATURE_DIM),
                steps=args.steps,
                guidance=weight,
                prefix=prefix,
                geom=shapes[index].to(device),
            )
            physical = sample * std + mean
            penalties = clip_penalties(physical.detach(), fps, skip_tail)
            for name in PENALTY_NAMES:
                scores[weight][name].append(penalties[name].detach().cpu())
            with torch.no_grad():
                values[weight].append(critic.q_min(sample, sample).detach().cpu())
        print(f"batch {batch_id + 1}/{args.batches}", flush=True)

    header = (
        f"{'guidance':>10} {'obj_vel':>8} {'obj_pos':>8} {'obj_spin':>8} "
        f"{'c_obj':>8} {'c_pelvis':>8} {'c_joint':>8} {'Q':>8}"
    )
    print(f"\n{args.batches * args.batch_size} paired clips, {args.steps} Euler steps")
    print("obj_vel and obj_pos are m/s, obj_spin is rad/s. c_* are position-velocity gaps.")
    print(header)
    real_line = (
        f"{'real':>10} {real['object_vel']:8.4f} {real['object_pos']:8.4f} {real['object_spin']:8.4f} "
        f"{real['consistency_object']:8.4f} {real['consistency_pelvis']:8.4f} "
        f"{real['consistency_joint']:8.4f} {'':>8}"
    )
    print(real_line)
    base = {name: torch.cat(scores[0.0][name]).mean() for name in PENALTY_NAMES}
    for weight in args.guidance:
        means = {name: torch.cat(scores[weight][name]).mean() for name in PENALTY_NAMES}
        value = torch.cat(values[weight]).mean()
        print(
            f"{weight:10.1f} {float(means['object_vel']):8.4f} {float(means['object_pos']):8.4f} "
            f"{float(means['object_spin']):8.4f} {float(means['consistency_object']):8.4f} "
            f"{float(means['consistency_pelvis']):8.4f} {float(means['consistency_joint']):8.4f} "
            f"{float(value):8.3f}"
        )
    print("change from guidance 0")
    print(header)
    for weight in args.guidance:
        if weight == 0.0:
            continue
        means = {name: torch.cat(scores[weight][name]).mean() for name in PENALTY_NAMES}
        print(
            f"{weight:10.1f} {float(means['object_vel'] - base['object_vel']):+8.4f} "
            f"{float(means['object_pos'] - base['object_pos']):+8.4f} "
            f"{float(means['object_spin'] - base['object_spin']):+8.4f} "
            f"{float(means['consistency_object'] - base['consistency_object']):+8.4f} "
            f"{float(means['consistency_pelvis'] - base['consistency_pelvis']):+8.4f} "
            f"{float(means['consistency_joint'] - base['consistency_joint']):+8.4f}"
        )


if __name__ == "__main__":
    main()
