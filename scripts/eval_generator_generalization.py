"""Generalization: how far off-dataset do the generators reach?

Coverage (``eval_generator_coverage.py``) starts from real first frames and
checks the generator lands back on the data. This script starts somewhere the
data never does, and asks the opposite question: the generated windows should
*not* collapse onto the nearest training window.

Two families of novel first frames, both scored with the same motion distance
(mean per-frame L2 in the checkpoint's normalized feature space, reported per
feature std by ``per_dimension``):

  * held-out data -- real first frames from ``reference/train`` (motions the
    checkpoints never trained on), so generalization can be read against the
    data distribution directly.
  * synthetic -- ``jitter`` (real frame plus noise), ``interp`` (average of two
    real frames) and ``randn`` (noise), so the reach can be pushed past the data.

For every generated window the script prints the distance to the window it was
seeded from and the distance to the nearest training window. Far from the seed
and far from the nearest training window is the generalization being asked for;
printing both keeps a collapse onto a training window visible. The GIF panels
are the novel first frame's seed clip (when the prefix is real) or nothing, plus
one panel per checkpoint.

    uv run python scripts/eval_generator_generalization.py \
        --checkpoints flow-dit=.../last.pt flow-unet=.../last.pt diffusion=.../last.pt \
        --mode real --seed-root reference --seed-split train --objects smalltable
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from generator_eval_common import (
    DEFAULT_WEAVE_ROOT,
    denormalize,
    different_motion_distance,
    label_frames,
    load_generator,
    load_windows,
    motion_bases,
    nearest_motion_distance,
    novel_prefixes,
    parse_checkpoints,
    per_dimension,
    prefix_novelty,
    reference_shim,
    render_clips,
    sample_windows,
    shapes_for,
    shown,
    window_distance,
    write_side_by_side,
)
from genrl.viz.render import expand_time
from genrl.viz.warp_clips import WarpSampleRenderer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("real", "jitter", "interp", "randn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="label=path pairs")
    parser.add_argument("--weave-root", type=Path, default=DEFAULT_WEAVE_ROOT)
    parser.add_argument("--objects", nargs="+", default=["smalltable"])
    parser.add_argument(
        "--train-root",
        default="rollout",
        choices=("rollout", "reference"),
        help="where the checkpoints' training windows live, for the nearest-window score",
    )
    parser.add_argument("--train-split", default="test")
    parser.add_argument(
        "--seed-root",
        default="reference",
        choices=("rollout", "reference"),
        help="where the novel first frames are read from",
    )
    parser.add_argument("--seed-split", default="train")
    parser.add_argument("--count", type=int, default=8, help="novel prefixes to draw")
    parser.add_argument("--mode", default="real", choices=MODES)
    parser.add_argument("--alpha", type=float, default=0.35, help="jitter noise scale, in std")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--playback-hz", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "generator_ood")
    return parser.parse_args()


def _root(args, kind: str, split: str) -> Path:
    choice = getattr(args, f"{kind}_root")
    if choice == "reference":
        return reference_shim(args.weave_root, split, list(args.objects))
    return args.weave_root


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints = parse_checkpoints(args.checkpoints)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    train_root = _root(args, "train", args.train_split)
    train_raw, hz, _, train_objects, train_motions = load_windows(
        train_root, args.train_split, list(args.objects)
    )
    train_bases = motion_bases(train_motions)
    seed_root = _root(args, "seed", args.seed_split)
    if seed_root == train_root and args.seed_split == args.train_split:
        seed_raw, seed_objects, seed_motions = train_raw, train_objects, train_motions
    else:
        seed_raw, _hz2, _, seed_objects, seed_motions = load_windows(
            seed_root, args.seed_split, list(args.objects)
        )
    print(
        f"training windows {train_raw.shape[0]} ({args.train_root}/{args.train_split}); "
        f"novel-prefix pool {seed_raw.shape[0]} ({args.seed_root}/{args.seed_split})"
    )

    count = min(args.count, seed_raw.shape[0])
    order = torch.from_numpy(np.linspace(0, seed_raw.shape[0] - 1, count).round().astype(np.int64))
    picked = seed_raw[order]

    # The checkpoints share the same training data, so one normalization defines
    # the metric space used for every panel.
    _, first_config, first_saved = load_generator(checkpoints[0][1], device)
    mean, std = first_saved["mean"].to(device), first_saved["std"].to(device)
    train_features = (train_raw.to(device) - mean) / std
    train_first = train_features[:, 0]
    seed_features = (picked.to(device) - mean) / std
    prefix = novel_prefixes(
        args.mode, seed_features[:, 0], generator, alpha=args.alpha, partner=torch.roll(seed_features[:, 0], 1, 0)
    )
    novelty = prefix_novelty(prefix, train_first)
    print(
        f"mode {args.mode!r}: prefix novelty (per-feature std to nearest training first frame) "
        f"mean {novelty.mean():.3f} median {novelty.median():.3f}"
    )

    # Scale for the distance numbers: how far a training window sits from the
    # nearest *different* training motion. Same-motion variants sit almost on
    # top of each other, so they are masked out. A generated window at least
    # this far from every training window is as novel as one training clip is
    # from the next, i.e. it leaves the data instead of collapsing onto a
    # memorized clip.
    cross = different_motion_distance(train_features, train_features, train_bases, train_bases)
    cross_scale = float(per_dimension(cross).mean())
    print(f"reference scale: a window sits {cross_scale:.3f} from the nearest different training motion")

    object_name = str(seed_objects[order.numpy()][0])
    renderer = WarpSampleRenderer(args.weave_root, width=args.width, height=args.height)
    directory = args.out / f"{args.mode}_{args.seed_root}{args.seed_split}"

    # A real seed has a ground-truth continuation to show; synthetic seeds do not.
    seed_panel = None
    if args.mode == "real":
        seed_panel = render_clips(
            renderer, expand_time(shown(picked).numpy(), hz, args.playback_hz), object_name
        )

    rows: list[dict[str, object]] = []
    panels: list[list[np.ndarray]] = [[] for _ in range(count)]
    # In-distribution floor: each checkpoint sampled from a real training first
    # frame should land back on the data, so its to-nearest-training score is what
    # "on the data" looks like. A novel prefix has to clear this to count as
    # leaving the data.
    floor_index = torch.arange(min(count, train_features.shape[0]))
    floors: list[tuple[str, float]] = []
    for label, path in checkpoints:
        module, config, saved = load_generator(path, device)
        local_mean, local_std = saved["mean"].to(device), saved["std"].to(device)
        shape = shapes_for(config, np.asarray([object_name] * count)).to(device)
        normalized = sample_windows(module, prefix[:, None], shape, train_raw.shape[1], args.steps, args.seed)
        physical = denormalize(normalized, local_mean, local_std)

        floor_shape = shapes_for(config, train_objects[floor_index.numpy()]).to(device)
        floor_physical = denormalize(
            sample_windows(module, train_features[floor_index, :1], floor_shape, train_raw.shape[1], args.steps, args.seed),
            local_mean,
            local_std,
        )
        floor = nearest_motion_distance(floor_physical[:, 1:], train_features[:, 1:])
        floors.append((label, float(per_dimension(floor).mean())))

        # Distance to the seed's own continuation, when it exists.
        seeded = (
            window_distance(physical[:, 1:], picked.to(device)[:, 1:])
            if args.mode == "real"
            else None
        )
        nearest = nearest_motion_distance(physical[:, 1:], train_features[:, 1:])
        print(
            f"{label:10s} to-nearest-training {per_dimension(nearest).mean():6.3f} "
            f"(real-prefix floor {floors[-1][1]:5.3f}, data spread {cross_scale:.3f}) "
            + (f"to-seed-continuation {per_dimension(seeded).mean():6.3f}" if seeded is not None else "(no seed continuation)")
        )

        shown_generated = shown(physical)
        frames = render_clips(
            renderer, expand_time(shown_generated.cpu().numpy(), hz, args.playback_hz), object_name
        )
        for index in range(count):
            panels[index].append(label_frames(frames[index], f"gen {label}"))
            rows.append(
                {
                    "mode": args.mode,
                    "checkpoint": label,
                    "index": index,
                    "seed": str(seed_motions[order[index]]).replace("__policy_rollout", ""),
                    "prefix_novelty": float(novelty[index]),
                    "distance_to_nearest_training": float(per_dimension(nearest[index])),
                    "distance_to_seed_continuation": float(per_dimension(seeded[index])) if seeded is not None else float("nan"),
                    "real_prefix_floor": floors[-1][1],
                    "other_motion_scale": cross_scale,
                }
            )

    for index in range(count):
        strip = ([label_frames(seed_panel[index], "seed real")] if seed_panel is not None else []) + panels[index]
        write_side_by_side(strip, directory / f"{index:02d}_{args.mode}.gif", args.playback_hz)

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "odometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {count} GIFs and odometry.csv to {directory}")


if __name__ == "__main__":
    main()
