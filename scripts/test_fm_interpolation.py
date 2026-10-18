"""Interpolate two flow-matching samples and write a side-by-side GIF.

Each GIF is the first clip, the alpha=0.5 interpolant, and the second clip.
The interpolant is sampled from the blended t_eps noise. The panels on the
ends are the original samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from genrl.data.geometry import object_shape_table
from genrl.data.segments import compose_rotation_anchor, project_rot6d
from genrl.train_common import build_model, load_raw_segments
from genrl.viz.render import expand_time
from genrl.viz.warp_clips import features_to_qpos

ROOT = Path(__file__).resolve().parents[1]


def _latest_checkpoint() -> Path:
    candidates = sorted((ROOT / "outputs" / "flow_matching-dit-geom").glob("*/checkpoints/last.pt"))
    if not candidates:
        raise FileNotFoundError("No flow_matching-dit-geom checkpoint under outputs/.")
    return candidates[-1]


def _pairs(features: torch.Tensor, names: np.ndarray, clips_per_object: int) -> list[tuple[int, int, str]]:
    """Two clips of one object whose motions differ the most.

    Adjacent dataset rows are nearly the same approach, so the midpoint of
    those two has nothing to show.
    """
    chosen = []
    flat = features.detach().flatten(1).float().cpu()
    for name in dict.fromkeys(str(item) for item in names.tolist()):
        index = np.flatnonzero(names == name)
        if index.size < 2:
            continue
        rows = flat[index]
        distance = torch.cdist(rows, rows)
        distance.fill_diagonal_(-1.0)
        left, right = divmod(int(distance.argmax()), index.size)
        chosen.append((int(index[left]), int(index[right]), name))
        if len(chosen) >= clips_per_object:
            break
    if len(chosen) < clips_per_object:
        raise ValueError(f"Need {clips_per_object} objects with two clips, found {len(chosen)}.")
    return chosen


def _blend(flow, left, right, prefix_left, prefix_right, geom, steps: int, iters: int, alpha: float):
    noise_left = flow.invert(left, steps=steps, prefix=prefix_left, geom=geom[:1], iters=iters)
    noise_right = flow.invert(right, steps=steps, prefix=prefix_right, geom=geom[1:], iters=iters)
    mixed = (1.0 - alpha) * noise_left + alpha * noise_right
    cond = prefix_left.shape[1]
    free = mixed[:, cond:]
    width = free[0].numel() ** 0.5
    norm = free.flatten(1).norm(dim=1).clamp_min(1e-8)
    mixed = mixed.clone()
    mixed[:, cond:] = free * (width / norm).view(-1, 1, 1)
    prefix = mixed[:, :cond]
    geom_mid = (1.0 - alpha) * geom[:1] + alpha * geom[1:]
    mid = flow.sample(tuple(mixed.shape), steps=steps, prefix=prefix, geom=geom_mid, source=mixed)
    return mid, prefix


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
    canvas = np.full((panels[0].shape[0], height, width * len(panels) + gap * (len(panels) - 1), 3), 255, np.uint8)
    for index, panel in enumerate(panels):
        start = index * (width + gap)
        canvas[:, :, start : start + width] = panel
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in canvas]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(20, int(round(1000 / fps))),
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "fm_interpolation")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    path = args.checkpoint or _latest_checkpoint()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(saved["config"])
    flow = build_model(cfg)
    flow.load_state_dict(saved["ema"])
    flow.to(device).eval()
    raw, fps, anchors, _skip, names = load_raw_segments(cfg)
    mean = saved["mean"].to(device)
    std = saved["std"].to(device)
    features = ((raw - saved["mean"]) / saved["std"]).to(device)
    shapes = object_shape_table(Path(cfg.data.root), names).to(device)
    steps = args.steps
    playback = int(cfg.sample.playback_hz) if "playback_hz" in cfg.sample else 50

    from genrl.viz.warp_clips import WarpSampleRenderer

    renderer = WarpSampleRenderer(cfg.data.root, width=args.width, height=args.height)
    print(f"checkpoint {path}")
    with torch.no_grad():
        for left, right, name in _pairs(raw, names, args.pairs):
            index = torch.tensor([left, right])
            prefix = features[index, :1]
            geom = shapes[index]
            source = torch.randn(2, features.shape[1], features.shape[2], device=device)
            ends = flow.sample(tuple(source.shape), steps=steps, prefix=prefix, geom=geom, source=source)
            mid, _prefix_mid = _blend(
                flow, ends[:1], ends[1:], prefix[:1], prefix[1:], geom, steps, args.iters, args.alpha
            )
            physical = torch.cat((ends[:1], mid, ends[1:]), dim=0) * std + mean
            anchor = anchors[index].to(device)
            mid_anchor = (1.0 - args.alpha) * anchor[:1] + args.alpha * anchor[1:]
            anchor = torch.cat((anchor[:1], mid_anchor, anchor[1:]), dim=0)
            shown = compose_rotation_anchor(project_rot6d(physical), anchor).cpu().numpy()
            clips = expand_time(shown, int(fps), playback)
            scene = renderer._scene(name)
            frames = scene.render(features_to_qpos(clips, scene.layout))
            titles = ("A", f"alpha {args.alpha:g}", "B")
            gif = args.out / f"{name}_{left:04d}_{right:04d}.gif"
            _write_side_by_side([_panel(frames[i], titles[i]) for i in range(3)], gif, playback)
            print(gif)


if __name__ == "__main__":
    main()
