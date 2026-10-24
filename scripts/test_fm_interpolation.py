"""Interpolate reference motions in noise space and compare three generators.

Each GIF is: reference 1, flow-DiT, flow-UNet, diffusion, reference 2.
Each middle panel inverts the same references with that model, blends the
recovered noise, and samples the blended noise at ``alpha``.
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
    load_generator,
    load_windows,
    pairwise_frame_distance,
    per_dimension,
    reference_shim,
)
from genrl.data.geometry import object_shape_table
from genrl.data.segments import compose_rotation_anchor, project_rot6d
from genrl.viz.render import expand_time
from genrl.viz.warp_clips import features_to_qpos

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    "fm_dit=outputs/flow_matching-dit-geom/2026-10-17_00-23-06/checkpoints/last.pt",
    "fm_unet=outputs/flow_matching-unet-geom/2026-10-19_11-45-33/checkpoints/last.pt",
    "diffusion=outputs/diffusion-dit-geom/2026-10-22_08-51-35/checkpoints/last.pt",
)


def _pairs(features: torch.Tensor, names: np.ndarray, count: int) -> list[tuple[int, int, str]]:
    """Greedily choose far-apart pairs without reusing trajectories."""
    chosen: list[tuple[int, int, str]] = []
    flat = features.detach().flatten(1).float().cpu()
    for name in dict.fromkeys(str(item) for item in names.tolist()):
        index = np.flatnonzero(names == name)
        distance = torch.cdist(flat[index], flat[index])
        candidates = torch.triu_indices(index.size, index.size, offset=1)
        order = distance[candidates[0], candidates[1]].argsort(descending=True)
        used: set[int] = set()
        for position in order.tolist():
            left = int(index[int(candidates[0, position])])
            right = int(index[int(candidates[1, position])])
            if left in used or right in used:
                continue
            chosen.append((left, right, name))
            used.update((left, right))
            if len(chosen) >= count:
                return chosen
    raise ValueError(f"Need {count} non-overlapping reference pairs, found {len(chosen)}.")


def _blend(model, left, right, prefix, geom, steps: int, iters: int, alpha: float):
    left_args = {"steps": steps, "prefix": prefix[:1], "geom": geom[:1]}
    right_args = {"steps": steps, "prefix": prefix[1:], "geom": geom[1:]}
    if model.__class__.__name__ == "FlowMatching":
        left_args["iters"] = iters
        right_args["iters"] = iters
    noise_left = model.invert(left, **left_args)
    noise_right = model.invert(right, **right_args)
    mixed = (1.0 - alpha) * noise_left + alpha * noise_right
    cond = prefix.shape[1]
    free = mixed[:, cond:]
    target_norm = free[0].numel() ** 0.5
    norm = free.flatten(1).norm(dim=1).clamp_min(1e-8)
    mixed[:, cond:] = free * (target_norm / norm).view(-1, 1, 1)
    prefix_mid = (1.0 - alpha) * prefix[:1] + alpha * prefix[1:]
    geom_mid = (1.0 - alpha) * geom[:1] + alpha * geom[1:]
    mixed[:, :cond] = prefix_mid
    return model.sample(
        tuple(mixed.shape), steps=steps, prefix=prefix_mid, geom=geom_mid, source=mixed
    )


def _panel(frames: np.ndarray, title: str) -> np.ndarray:
    labeled = []
    for frame in frames:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 22), fill=(20, 20, 20))
        draw.text((8, 4), title, fill=(240, 240, 240))
        labeled.append(np.asarray(image))
    return np.stack(labeled)


def _write_side_by_side(panels: list[np.ndarray], path: Path, fps: int) -> None:
    gap = 8
    height, width = panels[0].shape[1], panels[0].shape[2]
    canvas = np.full(
        (panels[0].shape[0], height, width * len(panels) + gap * (len(panels) - 1), 3),
        255,
        np.uint8,
    )
    for index, panel in enumerate(panels):
        start = index * (width + gap)
        canvas[:, :, start : start + width] = panel
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in canvas]
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=max(20, int(round(1000 / fps))), loop=0,
    )


def _write_chart(rows: list[dict[str, object]], labels: list[str], path: Path) -> None:
    values = {
        label: [float(row["min_reference_distance"]) for row in rows if row["model"] == label]
        for label in labels
    }
    means = [float(np.mean(values[label])) for label in labels]
    colours = [(100, 100, 100), (55, 108, 177), (221, 126, 47), (55, 146, 82)]
    width, height = 760, 440
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, bottom = 72, 45, 350
    peak = max(means) * 1.2 or 1.0
    draw.text((left, 14), "Mean shortest motion distance to reference/test", fill=(20, 20, 20))
    draw.line((left, top, left, bottom), fill=(30, 30, 30), width=2)
    draw.line((left, bottom, width - 30, bottom), fill=(30, 30, 30), width=2)
    slot = (width - left - 50) / len(labels)
    for index, (label, mean) in enumerate(zip(labels, means, strict=True)):
        x0 = left + 20 + index * slot
        x1 = x0 + slot * 0.62
        y = bottom - (bottom - top) * mean / peak
        draw.rectangle((x0, y, x1, bottom), fill=colours[index % len(colours)])
        draw.text((x0, y - 18), f"{mean:.4f}", fill=(25, 25, 25))
        draw.text((x0, bottom + 12), label, fill=(25, 25, 25))
    for tick in range(5):
        value = peak * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.text((8, y - 7), f"{value:.3f}", fill=(50, 50, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=3, default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--weave-root", type=Path, default=DEFAULT_WEAVE_ROOT)
    parser.add_argument("--object", default="smalltable")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "fm_interpolation")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must lie in [0, 1].")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoints = []
    for value in args.checkpoints:
        label, path = value.split("=", 1)
        checkpoints.append((label, Path(path)))

    root = reference_shim(args.weave_root, "test", [args.object])
    raw, fps, anchors, names, _motions = load_windows(root, "test", [args.object])
    selected_pairs = _pairs(raw, names, args.pairs)
    playback = 50

    metric_model, _metric_cfg, metric_saved = load_generator(checkpoints[0][1], device)
    del metric_model
    metric_mean = metric_saved["mean"].to(device)
    metric_std = metric_saved["std"].to(device)
    reference_metric = (raw.to(device) - metric_mean) / metric_std
    reference_distances = pairwise_frame_distance(reference_metric[:, 1:], reference_metric[:, 1:])
    reference_distances.fill_diagonal_(0.0)
    self_distance = per_dimension(reference_distances.min(1).values)
    reference_distances.fill_diagonal_(float("inf"))
    leave_one_out = per_dimension(reference_distances.min(1).values)
    print(
        f"reference baseline: self {self_distance.mean():.6f}, "
        f"nearest other motion {leave_one_out.mean():.4f}"
    )

    from genrl.viz.warp_clips import WarpSampleRenderer
    renderer = WarpSampleRenderer(args.weave_root, width=args.width, height=args.height)
    rows: list[dict[str, object]] = []
    labels = ["reference", *[label for label, _path in checkpoints]]

    for pair_number, (left, right, name) in enumerate(selected_pairs):
        index = torch.tensor([left, right])
        midpoint_anchor = (1.0 - args.alpha) * anchors[left:left + 1] + args.alpha * anchors[right:right + 1]
        generated: list[tuple[str, torch.Tensor, float]] = []
        for label, path in checkpoints:
            model, cfg, saved = load_generator(path, device)
            mean, std = saved["mean"].to(device), saved["std"].to(device)
            features = (raw[index].to(device) - mean) / std
            geom = object_shape_table(Path(cfg.data.root), names[index.numpy()]).to(device)
            with torch.no_grad():
                middle = _blend(
                    model, features[:1], features[1:], features[:, :1], geom,
                    args.steps, args.iters, args.alpha,
                )
            physical = middle * std + mean
            middle_metric = (physical - metric_mean) / metric_std
            distance = per_dimension(
                pairwise_frame_distance(middle_metric[:, 1:], reference_metric[:, 1:]).min(1).values
            ).item()
            generated.append((label, physical, distance))
            rows.append({
                "pair": pair_number, "left_reference": left, "right_reference": right,
                "model": label, "min_reference_distance": distance,
            })
            print(f"pair {pair_number:02d} {label:10s} min reference distance {distance:.4f}")
            del model

        rows.append({
            "pair": pair_number, "left_reference": left, "right_reference": right,
            "model": "reference", "min_reference_distance": float(self_distance[left]),
        })
        physical = torch.cat(
            (raw[left:left + 1].to(device), *[item[1] for item in generated], raw[right:right + 1].to(device)),
            dim=0,
        )
        panel_anchors = torch.cat(
            (anchors[left:left + 1], midpoint_anchor.repeat(3, 1), anchors[right:right + 1]), dim=0,
        ).to(device)
        shown = compose_rotation_anchor(project_rot6d(physical), panel_anchors).cpu().numpy()
        scene = renderer._scene(name)
        frames = scene.render(features_to_qpos(expand_time(shown, int(fps), playback), scene.layout))
        titles = [
            f"reference 1 ({left})",
            *[f"{label}  d={distance:.3f}" for label, _physical, distance in generated],
            f"reference 2 ({right})",
        ]
        gif = args.out / f"{pair_number:02d}_{name}_{left:04d}_{right:04d}.gif"
        _write_side_by_side([_panel(frames[i], titles[i]) for i in range(5)], gif, playback)
        print(gif)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "distance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_chart(rows, labels, args.out / "distance.png")
    print(f"wrote {len(selected_pairs)} GIFs, distance.csv, and distance.png to {args.out}")


if __name__ == "__main__":
    main()
