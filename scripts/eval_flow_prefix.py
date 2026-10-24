"""Novel first frames through the frozen flow-matching policy.

Training always pins a real frame 0 of a real clip. This script instead builds
first frames that no clip in the data starts from and shows what the policy does
from there, with ``smalltable`` (test split) shapes throughout -- so only the
prefix is novel, not the object:

  real      the real frame 0 (control)
  mid       a later frame of the clip, re-anchored to look like a start
  lerp      a time-interpolation of one clip from frame 0 toward a later frame
  interp    a blend of two different clips' frame 0
  jitter    the real frame 0 plus Gaussian noise
  objshift  the real frame 0 with the object pushed along the reach direction
  randn     pure noise (sanity check: a nonsense start)

The policy conditions on the whole 130-d frame -- ``FlowMatching.sample`` writes
that frame in verbatim and pins it after every Euler step, so it is not a latent.
Only the initial noise is random, so ``--seeds`` varies the sample while the
prefix stays fixed. ``--render-gt`` writes the real continuation of the same clip
beside the generated GIF.

See ``eval_qgf_prefix.py`` for the same experiment through the QGF sampler.
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
from genrl.train_common import _guidance, build_backbone
from genrl.viz.render import expand_time
from genrl.viz.warp_clips import WarpSampleRenderer

from qgf_prefix_common import (
    add_prefix_args,
    build_clip,
    build_prefix,
    load_data,
    novelty,
)

ROOT = Path(__file__).resolve().parents[1]


def load_flow(checkpoint: Path, device: torch.device) -> tuple[FlowMatching, OmegaConf, dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(saved["config"])
    flow = FlowMatching(build_backbone(config), t_eps=float(config.t_eps), guidance=_guidance(config))
    state = saved["ema"] if "ema" in saved else saved["model"]
    flow.load_state_dict(state)
    flow.to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    print(f"loaded {'EMA' if 'ema' in saved else 'raw'} weights from {checkpoint} at step {saved['step']}")
    return flow, config, saved


def parse_args() -> argparse.Namespace:
    parser = add_prefix_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(out=ROOT / "outputs" / "flow_prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    flow, config, saved = load_flow(args.checkpoint, device)
    mean, std = saved["mean"].to(device), saved["std"].to(device)

    raw, fps, _anchors, _skip_tail, names = load_data(config, args)
    raw = torch.as_tensor(raw)
    length = raw.shape[1]
    index = torch.arange(min(args.num_clips, raw.shape[0]))
    generator = torch.Generator().manual_seed(args.seeds[0])

    offsets, clip = build_clip(raw, args.mode, args, generator)
    features = (clip - mean) / std
    prefix = build_prefix(args.mode, features, raw, offsets, args, generator)
    selection = names[index.detach().cpu().numpy()]
    shapes = object_shape_table(Path(config.data.root), selection).to(device)

    print(
        f"mode {args.mode!r} on {list(selection)} with {args.objects!r} shapes, "
        f"{length} frames at {fps} Hz, start frames {offsets[index].tolist()}"
    )
    print(
        "prefix novelty (std units to the nearest real frame 0): "
        + ", ".join(f"{float(v):.2f}" for v in novelty(prefix, index, raw, mean, std, args.mode))
    )

    clips, labels = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        with torch.no_grad():
            generated = flow.sample(
                (index.shape[0], length, FEATURE_DIM),
                steps=args.steps,
                prefix=prefix[index, None],
                geom=shapes,
            )
        clips.append((generated * std + mean).detach().cpu().numpy())
        labels.extend(f"{position:02d}_{name}_s{seed}" for position, name in enumerate(selection))
    clips = np.concatenate(clips, axis=0)

    if not (args.render or args.render_gt):
        return
    renderer = WarpSampleRenderer(config.data.root, config.robot.urdf, width=args.width, height=args.height)
    if args.render:
        meshes = np.asarray(selection)[np.tile(np.arange(index.shape[0]), len(args.seeds))]
        directory = args.out / args.mode
        for path, label in zip(
            renderer.write_gifs(expand_time(clips, fps, 50), meshes, directory, 50), labels
        ):
            renamed = path.with_name(f"{label}.gif")
            path.replace(renamed)
            print(f"wrote {renamed}")
    if args.render_gt:
        played = expand_time(clip[index].detach().cpu().numpy(), fps, 50)
        for path in renderer.write_gifs(played, selection, args.out / f"{args.mode}_gt", 50):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
