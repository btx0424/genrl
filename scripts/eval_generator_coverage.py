"""Coverage: does each generator reproduce every window it was trained on?

One GIF per motion. The leftmost panel is the real pre-interaction window; the
panels to its right are windows sampled from each checkpoint with that real
first frame pinned. A generator that covers the data puts its panel on top of
the leftmost one; a bar chart and a CSV record the motion distance between them.

The dataset the checkpoints were trained on is ``rollout/<split>`` under the
snapshot root (the policy trajectories). ``--data-root reference`` reads the
``reference/<split>`` motions instead -- the clean retargeted clips with the same
motion names, which the generators never saw as training windows.

Motion distance is the mean per-frame L2 in the checkpoint's normalized feature
space (see ``generator_eval_common``); 0 means the panels coincide. The printed
``real self-NN`` is the scale: the distance from every real window to its
nearest other real window, i.e. the spread of the data itself.

    uv run python scripts/eval_generator_coverage.py \
        --checkpoints flow-dit=outputs/flow_matching-dit-geom/<run>/checkpoints/last.pt \
                      flow-unet=outputs/flow_matching-unet-geom/<run>/checkpoints/last.pt \
                      diffusion=outputs/diffusion-dit-geom/<run>/checkpoints/last.pt \
        --split test --objects smalltable
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image, ImageDraw

from generator_eval_common import (
    DEFAULT_WEAVE_ROOT,
    denormalize,
    distance_groups,
    label_frames,
    load_generator,
    load_windows,
    pairwise_frame_distance,
    per_dimension,
    parse_checkpoints,
    pick_one_per_motion,
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


def write_summary_chart(
    rows: list[dict[str, object]], labels: list[str], other_reference: float, path: Path
) -> None:
    """Mean nearest-real distance, with the real-motion self-match as zero control."""
    chart_labels = ["reference", *labels]
    values = [0.0] + [
        float(np.mean([float(row["nearest_dataset_distance"]) for row in rows if row["checkpoint"] == label]))
        for label in labels
    ]
    colours = [(100, 100, 100), (55, 108, 177), (221, 126, 47), (55, 146, 82)]
    width, height = 760, 440
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, bottom = 72, 45, 350
    peak = max(values) * 1.2 or 1.0
    draw.text((left, 14), "Mean shortest motion distance to real coverage data", fill=(20, 20, 20))
    draw.text(
        (left + 350, 14),
        f"reference nearest-other mean: {other_reference:.4f}",
        fill=(80, 80, 80),
    )
    draw.line((left, top, left, bottom), fill=(30, 30, 30), width=2)
    draw.line((left, bottom, width - 30, bottom), fill=(30, 30, 30), width=2)
    slot = (width - left - 50) / len(chart_labels)
    for index, (label, value) in enumerate(zip(chart_labels, values, strict=True)):
        x0 = left + 20 + index * slot
        x1 = x0 + slot * 0.62
        y = bottom - (bottom - top) * value / peak
        draw.rectangle((x0, y, x1, bottom), fill=colours[index % len(colours)])
        draw.text((x0, y - 18), f"{value:.4f}", fill=(25, 25, 25))
        draw.text((x0, bottom + 12), label, fill=(25, 25, 25))
    for tick in range(5):
        value = peak * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.text((8, y - 7), f"{value:.3f}", fill=(50, 50, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="label=path pairs")
    parser.add_argument(
        "--data-root",
        default="rollout",
        choices=("rollout", "reference"),
        help="read rollout/<split> (the training data) or reference/<split>",
    )
    parser.add_argument("--weave-root", type=Path, default=DEFAULT_WEAVE_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--objects", nargs="+", default=["smalltable"])
    parser.add_argument("--count", type=int, default=0, help="motions to keep; 0 is all")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--playback-hz", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "generator_coverage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints = parse_checkpoints(args.checkpoints)
    labels = [label for label, _path in checkpoints]

    if args.data_root == "reference":
        root = reference_shim(args.weave_root, args.split, list(args.objects))
    else:
        root = args.weave_root
    raw, hz, anchors, object_names, motion_names = load_windows(root, args.split, list(args.objects))
    chosen = torch.as_tensor(pick_one_per_motion(motion_names, args.count or None))
    print(f"{args.data_root}/{args.split}: {raw.shape[0]} windows, {len(chosen)} motions @ {hz} Hz")

    unique_objects = list(dict.fromkeys(object_names[chosen.numpy()].tolist()))
    if len(unique_objects) != 1:
        raise ValueError(f"Coverage renders one object at a time, got {unique_objects}.")
    object_name = unique_objects[0]
    renderer = WarpSampleRenderer(args.weave_root, width=args.width, height=args.height)

    real_physical = raw[chosen]
    real_anchor = anchors[chosen]
    real_shown = shown(real_physical, real_anchor)
    real_frames = render_clips(
        renderer, expand_time(real_shown.numpy(), hz, args.playback_hz), object_name
    )
    names = [str(motion_names[i]).replace("__policy_rollout", "") for i in chosen]

    # One normalization defines a comparable metric for all checkpoints.
    metric_module, _metric_config, metric_saved = load_generator(checkpoints[0][1], device)
    del metric_module
    metric_mean = metric_saved["mean"].to(device)
    metric_std = metric_saved["std"].to(device)
    dataset_metric = (raw.to(device) - metric_mean) / metric_std
    real_distances = pairwise_frame_distance(dataset_metric[:, 1:], dataset_metric[:, 1:])
    real_distances.fill_diagonal_(0.0)
    real_self = per_dimension(real_distances.min(dim=1).values)
    real_distances.fill_diagonal_(float("inf"))
    real_other = per_dimension(real_distances.min(dim=1).values)
    print(
        f"reference baseline: self {real_self.mean():.6f}, "
        f"nearest other motion {real_other.mean():.4f}"
    )

    rows: list[tuple[str, str, float]] = []
    csv_rows: list[dict[str, object]] = []
    # panels[motion] collects the generated panel per checkpoint, in checkpoint order.
    panels: list[list[np.ndarray]] = [[] for _ in names]
    directory = args.out / f"{args.data_root}_{args.split}"
    for label, path in checkpoints:
        module, config, saved = load_generator(path, device)
        mean, std = saved["mean"].to(device), saved["std"].to(device)
        features = (raw.to(device) - mean) / std
        geom = shapes_for(config, object_names[chosen.numpy()]).to(device)

        normalized = sample_windows(module, features[chosen, :1], geom, raw.shape[1], args.steps, args.seed)
        physical = denormalize(normalized, mean, std)
        real_device = real_physical.to(device)
        generated_metric = (physical - metric_mean) / metric_std
        paired = per_dimension(window_distance(generated_metric[:, 1:], dataset_metric[chosen, 1:]))
        nearest = per_dimension(
            pairwise_frame_distance(generated_metric[:, 1:], dataset_metric[:, 1:]).min(dim=1).values
        )
        print(
            f"{label:10s} paired {paired.mean():7.3f}  nearest-dataset {nearest.mean():7.3f}  "
            f"real other-NN {real_other.mean():7.3f}"
        )

        generated_shown = shown(physical, real_anchor.to(device))
        frames = render_clips(
            renderer, expand_time(generated_shown.cpu().numpy(), hz, args.playback_hz), object_name
        )
        for index, name in enumerate(names):
            panels[index].append(label_frames(frames[index], f"gen {label}"))
            rows.append((name, label, float(paired[index])))
            csv_rows.append(
                {
                    "motion": name,
                    "checkpoint": label,
                    "paired_distance": float(paired[index]),
                    "nearest_dataset_distance": float(nearest[index]),
                    **{
                        f"group_{key}": value
                        for key, value in distance_groups(
                            physical[index : index + 1, 1:], real_device[index : index + 1, 1:]
                        ).items()
                    },
                }
            )

    # One GIF per motion: the real window, then each checkpoint's window.
    for index, name in enumerate(names):
        strip = [label_frames(real_frames[index], f"real {name}"), *panels[index]]
        write_side_by_side(strip, directory / f"{index:02d}_{name}.gif", args.playback_hz)

    args.out.mkdir(parents=True, exist_ok=True)
    write_summary_chart(csv_rows, labels, float(real_other.mean()), directory / "distance.png")
    with (directory / "distance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {len(names)} paired GIFs, distance.png and distance.csv to {directory}")


if __name__ == "__main__":
    main()
