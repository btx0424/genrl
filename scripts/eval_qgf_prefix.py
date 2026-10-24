"""Novel first frames through a QGF-guided policy.

Same prefix construction as ``eval_flow_prefix.py`` -- real, mid-approach,
interpolated, jittered, object-shifted, or pure noise -- but the sampler is
``qgf_sample``: the frozen flow plus ``guidance * grad Q``. Both the flow and the
critic come from the QGF checkpoint.

The flow and critic are stored together, and the guidance weight defaults to the
one the run trained with. ``--compare`` also samples the frozen flow alone, so the
guided and unguided motions can be placed side by side.

    python scripts/eval_qgf_prefix.py --checkpoint outputs/qgf/<run>/checkpoints/last.pt \
        --mode mid --render --render-gt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from genrl.data.geometry import object_shape_table
from genrl.data.segments import FEATURE_DIM
from genrl.model.flow_matching import FlowMatching
from genrl.model.qgf import IQL, qgf_sample
from genrl.train_common import _guidance, build_backbone
from genrl.viz.render import expand_time
from genrl.viz.warp_clips import WarpSampleRenderer

from qgf_prefix_common import add_prefix_args, build_clip, build_prefix, load_data, novelty

ROOT = Path(__file__).resolve().parents[1]


def load_qgf(checkpoint: Path, device: torch.device) -> tuple[FlowMatching, IQL, OmegaConf, dict]:
    """The frozen flow and the critic, both stored in the QGF checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "critic" not in saved or "flow" not in saved:
        raise ValueError(f"{checkpoint} is not a QGF checkpoint.")
    config = OmegaConf.create(saved["config"])
    flow_saved = torch.load(str(config.flow), map_location="cpu", weights_only=False)
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
    print(
        f"loaded QGF step {int(saved['step'])}: frozen flow {config.flow}, "
        f"critic ensemble {int(config.iql.ensemble)}",
        flush=True,
    )
    return flow, critic, flow_config, saved


def parse_args() -> argparse.Namespace:
    parser = add_prefix_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(out=ROOT / "outputs" / "qgf_prefix")
    parser.add_argument(
        "--guidance",
        type=float,
        default=-1.0,
        help="QGF guidance weight, <0 uses the value the checkpoint trained with",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also sample the frozen flow alone (guidance 0) as a reference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    flow, critic, flow_config, saved = load_qgf(args.checkpoint, device)
    mean, std = saved["mean"].to(device), saved["std"].to(device)
    trained_guidance = float(OmegaConf.create(saved["config"]).guidance)
    guidance = trained_guidance if args.guidance < 0.0 else args.guidance

    raw, fps, _anchors, _skip_tail, names = load_data(flow_config, args)
    raw = torch.as_tensor(raw)
    length = raw.shape[1]
    index = torch.arange(min(args.num_clips, raw.shape[0]))
    generator = torch.Generator().manual_seed(args.seeds[0])

    offsets, clip = build_clip(raw, args.mode, args, generator)
    features = (clip - mean) / std
    prefix = build_prefix(args.mode, features, raw, offsets, args, generator)
    selection = names[index.detach().cpu().numpy()]
    shapes = object_shape_table(Path(flow_config.data.root), selection).to(device)

    print(
        f"mode {args.mode!r} on {list(selection)} with {args.objects!r} shapes, "
        f"{length} frames at {fps} Hz, start frames {offsets[index].tolist()}, guidance {guidance}"
    )
    print(
        "prefix novelty (std units to the nearest real frame 0): "
        + ", ".join(f"{float(v):.2f}" for v in novelty(prefix, index, raw, mean, std, args.mode))
    )

    weights = [guidance, 0.0] if args.compare else [guidance]
    groups = {}
    for weight in weights:
        clips = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            with torch.no_grad():
                generated = qgf_sample(
                    flow,
                    critic.q_min,
                    (index.shape[0], length, FEATURE_DIM),
                    steps=args.steps,
                    guidance=weight,
                    prefix=prefix[index, None],
                    geom=shapes,
                )
            clips.append((generated * std + mean).detach().cpu().numpy())
        groups[weight] = np.concatenate(clips, axis=0)

    if not (args.render or args.render_gt):
        return
    renderer = WarpSampleRenderer(flow_config.data.root, flow_config.robot.urdf, args.width, args.height)
    if args.render:
        meshes = np.asarray(selection)[np.tile(np.arange(index.shape[0]), len(args.seeds))]
        for weight, clips in groups.items():
            label = f"qgf_g{weight:g}"
            directory = args.out / f"{args.mode}_{label}"
            played = expand_time(clips, fps, 50)
            labels = [
                f"{position:02d}_{name}_s{seed}"
                for seed in args.seeds
                for position, name in enumerate(selection)
            ]
            for path, name_out in zip(renderer.write_gifs(played, meshes, directory, 50), labels):
                renamed = path.with_name(f"{name_out}.gif")
                path.replace(renamed)
                print(f"wrote {renamed}")
    if args.render_gt:
        played = expand_time(clip[index].detach().cpu().numpy(), fps, 50)
        for path in renderer.write_gifs(played, selection, args.out / f"{args.mode}_gt", 50):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
