"""Shared loading and rendering for generator coverage and generalization evals.

A checkpoint is a flow-matching or diffusion run written by
``train_common.train``. Both wrappers expose ``sample(shape, steps, prefix,
geom)`` and pin the first frame, so one loader covers the two families. The EMA
weights are used, matching what training renders.

Windows are built the way training builds them: the last ``duration`` seconds
before the interaction onset, subsampled to ``hz``, with the pelvis and object
rotations made relative to the pinned first frame. The real first frame is the
inpainting prefix at sample time.

The Weave ``reference`` split is laid out as
``reference/<split>/<object>_<n>clip*.npz``, not
``rollout/<split>/<object>_rollout.npz``. ``reference_shim`` builds the symlink
farm the dataset loader expects so both splits load through one code path.

"Motion distance" between two ``(time, feature)`` windows is the mean over
frames of the L2 distance in the checkpoint's normalized feature space: each
frame is one 130-d vector scaled by the training mean and std, so the number is
dimensionless and comparable across clips. Rotations stay in their raw 6D
columns, as in training. ``window_distance`` scores a pair, and
``pairwise_frame_distance`` scores every pair of two sets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from genrl.data.geometry import object_shape_table
from genrl.data.segments import (
    FEATURE_DIM,
    JOINT_POS,
    OBJECT_POS,
    OBJECT_ROT,
    PELVIS_POS,
    PELVIS_ROT,
    clips_to_features,
    compose_rotation_anchor,
    make_rotations_relative,
    project_rot6d,
)
from genrl.data.weave import WeavePreInteractionDataset
from genrl.train_common import build_model

DEFAULT_WEAVE_ROOT = Path(
    "/media/zhangjin/Elements/hub/datasets--appolyn--Weave/snapshots/"
    "db4a90f705f06667bd6b200053237e69644f0064"
)
DEFAULT_SHIM = Path("/tmp/genrl_eval_shim")

_GROUPS = {
    "joint": (JOINT_POS,),
    "pelvis_pos": (PELVIS_POS,),
    "object_pos": (OBJECT_POS,),
    "rotation": (PELVIS_ROT, OBJECT_ROT),
}


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    """``label=path`` pairs. A bare path is labelled by its output directory."""
    parsed: list[tuple[str, Path]] = []
    for item in values:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).parent.parent.name
        parsed.append((label, Path(path)))
    return parsed


def reference_shim(
    weave_root: str | Path,
    split: str,
    objects: list[str],
    workdir: str | Path = DEFAULT_SHIM,
) -> Path:
    """Symlink ``reference/<split>/*.npz`` into the rollout layout the loader reads."""
    source = Path(weave_root) / "reference" / split
    root = Path(workdir) / f"reference_{split}"
    target = root / "rollout" / split
    target.mkdir(parents=True, exist_ok=True)
    for name in objects:
        candidates = sorted(source.glob(f"{name}_*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No reference file for {name!r} under {source}.")
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous reference files for {name!r}: {candidates}.")
        link = target / f"{name}_rollout.npz"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(candidates[0])
    return root


def load_windows(
    root: str | Path,
    split: str,
    objects: list[str],
    hz: int = 10,
    duration: float = 1.0,
    speed_threshold: float = 0.2,
):
    """Real pre-interaction windows plus their rotation anchors and names.

    Returns ``(raw, hz, anchors, object_names, motion_names)``; ``raw`` is
    ``(clips, frames, FEATURE_DIM)`` with relative rotations.
    """
    dataset = WeavePreInteractionDataset(
        list(objects),
        root=root,
        split=split,
        duration_s=duration,
        speed_threshold=speed_threshold,
    )
    raw = torch.from_numpy(clips_to_features(dataset))
    stride = int(dataset.fps) // int(hz)
    if stride < 1 or int(dataset.fps) % int(hz) != 0:
        raise ValueError(f"{dataset.fps} Hz is not divisible by {hz} Hz.")
    raw = raw[:, ::stride].contiguous()
    raw, anchors = make_rotations_relative(raw)
    motion_names = np.asarray([str(name) for name in dataset.motion_names])
    object_names = np.asarray([str(name) for name in dataset.object_names])
    return raw, int(hz), anchors, object_names, motion_names


def load_generator(checkpoint: str | Path, device: torch.device):
    """The EMA generator from a checkpoint, in eval mode with gradients off."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(saved["config"])
    module = build_model(config)
    module.load_state_dict(saved["ema"] if "ema" in saved else saved["model"])
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    print(f"loaded {Path(checkpoint)} at step {saved['step']} ({config.name}, {config.model.name})")
    return module, config, saved


def shapes_for(config, object_names) -> torch.Tensor:
    """Object shape vectors from the mesh cache, using the training mesh root."""
    return object_shape_table(Path(config.data.root), np.asarray(object_names))


def sample_windows(module, prefixes: torch.Tensor, geom: torch.Tensor, length: int, steps: int, seed: int):
    """Sample ``length``-frame windows with the given (normalized) first frames pinned."""
    torch.manual_seed(seed)
    with torch.no_grad():
        return module.sample(
            (prefixes.shape[0], length, FEATURE_DIM),
            steps=steps,
            prefix=prefixes,
            geom=geom,
        )


def denormalize(normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalized clips back to physical units."""
    return normalized * std + mean


def shown(physical: torch.Tensor, anchors: torch.Tensor | None = None) -> torch.Tensor:
    """Physical clips to renderable clips: nearest proper rotation, re-anchored.

    ``project_rot6d`` takes the nearest proper rotation (SVD, not Gram-Schmidt)
    and ``compose_rotation_anchor`` puts the relative rotations back in the
    world, exactly as the training GIFs do.
    """
    out = project_rot6d(physical)
    if anchors is not None:
        out = compose_rotation_anchor(out, anchors)
    return out


def window_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Mean per-frame L2 between paired windows. ``(batch, frames, feature)`` -> ``(batch,)``."""
    return (first - second).norm(dim=-1).mean(dim=-1)


def pairwise_frame_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Mean per-frame L2 for every pair. ``(n, T, D)`` x ``(m, T, D)`` -> ``(n, m)``."""
    total = None
    for frame in range(first.shape[1]):
        step = torch.cdist(first[:, frame], second[:, frame])
        total = step if total is None else total + step
    return total / first.shape[1]


def per_dimension(distance: torch.Tensor) -> torch.Tensor:
    """Total feature-space L2 divided by sqrt(dim), i.e. the RMS per-dimension error.

    Every dimension is already in training-std units, so the result reads like a
    z-score: 1.0 is one std per dimension on average.
    """
    return distance / FEATURE_DIM**0.5


def nearest_motion_distance(generated: torch.Tensor, dataset: torch.Tensor) -> torch.Tensor:
    """For each generated clip (frames 1:), the closest real window by motion distance."""
    return pairwise_frame_distance(generated, dataset).min(dim=1).values


def motion_bases(motion_names: np.ndarray) -> np.ndarray:
    """Strip the trailing ``_v<nn>`` variant tag, leaving one string per motion."""
    return np.asarray(
        [str(name).replace("__policy_rollout", "").rsplit("_v", 1)[0] for name in motion_names]
    )


def different_motion_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    first_bases: np.ndarray,
    second_bases: np.ndarray,
) -> torch.Tensor:
    """For each window in ``first``, the closest window in ``second`` from another motion.

    Pairs that share a motion base are masked out. The nearest neighbour of a
    real window is nearly always a variant of its own motion, so the unmasked
    minimum measures duplicates; masking the base measures the distance to the
    nearest genuinely different motion, which is the scale a novel generated
    clip should be read against.
    """
    distance = pairwise_frame_distance(first[:, 1:], second[:, 1:])
    same = torch.from_numpy(first_bases[:, None] == second_bases[None, :]).to(distance.device)
    return distance.masked_fill(same, float("inf")).min(dim=1).values


def novel_prefixes(
    mode: str,
    first: torch.Tensor,
    generator: torch.Generator,
    alpha: float = 0.35,
    partner: torch.Tensor | None = None,
) -> torch.Tensor:
    """First frames no training clip starts from.

    ``real`` returns the frame unchanged; it is the control that keeps the
    prefix on held-out data (from another split) rather than off the data.
    ``jitter`` adds Gaussian noise to the joints/positions and leaves the 6D
    rotations alone. ``interp`` averages two real first frames, projecting the
    rotation average back onto a proper rotation with an SVD. ``randn`` is a
    pure noise frame, the sanity check that the sampler needs no real prefix.
    """
    if mode == "real":
        return first.clone()
    if mode == "jitter":
        noise = torch.randn(first.shape, generator=generator).to(first.device) * alpha
        for rotation in (PELVIS_ROT, OBJECT_ROT):
            noise[:, rotation] = 0.0
        return first + noise
    if mode == "interp":
        if partner is None:
            raise ValueError("interp needs a partner frame set.")
        mixed = 0.5 * (first + partner)
        for rotation in (PELVIS_ROT, OBJECT_ROT):
            columns = torch.stack(
                (0.5 * (first[:, rotation][..., 0:3] + partner[:, rotation][..., 0:3]),
                 0.5 * (first[:, rotation][..., 3:6] + partner[:, rotation][..., 3:6])),
                dim=-1,
            )
            left, _singular, right = torch.linalg.svd(columns, full_matrices=False)
            nearest = left @ right
            mixed[:, rotation] = torch.cat((nearest[..., 0], nearest[..., 1]), dim=-1)
        return mixed
    if mode == "randn":
        return torch.randn(first.shape, generator=generator).to(first.device)
    raise ValueError(f"Unknown novel prefix mode {mode!r}.")


def prefix_novelty(prefix: torch.Tensor, dataset_first: torch.Tensor) -> torch.Tensor:
    """Distance of each prefix to the nearest real first frame, per-feature std."""
    return per_dimension(torch.cdist(prefix, dataset_first).min(dim=1).values)


def distance_groups(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Per-frame L2 distance split by feature group, for interpretation."""
    out = {"total": float(window_distance(first, second).mean())}
    for name, slices in _GROUPS.items():
        parts = [float((first[..., sl] - second[..., sl]).norm(dim=-1).mean()) for sl in slices]
        out[name] = sum(parts) / len(parts)
    return out


def pick_one_per_motion(motion_names: np.ndarray, count: int | None = None) -> np.ndarray:
    """First variant of each distinct motion (strip a trailing ``_v<nn>``)."""
    seen: dict[str, int] = {}
    for index, name in enumerate(motion_names):
        base = str(name).rsplit("_v", 1)[0]
        seen.setdefault(base, index)
    chosen = np.asarray(sorted(seen.values()), dtype=np.int64)
    return chosen if count is None else chosen[:count]


def label_frames(frames: np.ndarray, title: str) -> np.ndarray:
    """Draw a title bar on each frame of one panel. ``frames`` is ``(T, H, W, 3)``."""
    labeled = []
    for frame in frames:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 20), fill=(20, 20, 20))
        draw.text((6, 4), title, fill=(240, 240, 240))
        labeled.append(np.asarray(image))
    return np.stack(labeled)


def write_side_by_side(panels: list[np.ndarray], path: Path, fps: int) -> None:
    """Stitch equal-length panels horizontally into one GIF."""
    gap = 8
    length, height, width = panels[0].shape[:3]
    canvas = np.full(
        (length, height, width * len(panels) + gap * (len(panels) - 1), 3), 255, np.uint8
    )
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


def write_distance_chart(rows: list[tuple[str, str, float]], path: Path, labels: list[str]) -> None:
    """A bar per (motion, checkpoint), grouped by motion, coloured per checkpoint."""
    colours = [(66, 118, 189), (222, 124, 40), (60, 150, 90), (170, 70, 150), (120, 120, 120)]
    motions = list(dict.fromkeys(motion for motion, _label, _value in rows))
    values = {(motion, label): value for motion, label, value in rows}
    width, height = max(900, 26 * len(motions) + 160), 420
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    top, bottom = 30, height - 30
    peak = max((value for _m, _l, value in rows), default=1.0) or 1.0
    draw.line((40, top, 40, bottom), fill=(40, 40, 40))
    draw.text((6, 8), "motion distance real vs generated (normalized per-frame L2)", fill=(20, 20, 20))
    slot = (width - 60) / max(len(motions), 1)
    bar = slot * 0.8 / max(len(labels), 1)
    for row, motion in enumerate(motions):
        for column, label in enumerate(labels):
            value = values.get((motion, label), 0.0)
            x0 = 44 + row * slot + column * bar
            y0 = bottom - (bottom - top) * (value / peak)
            draw.rectangle((x0, y0, x0 + bar * 0.9, bottom), fill=colours[column % len(colours)])
    for column, label in enumerate(labels):
        draw.rectangle((60 + column * 190, height - 22, 74 + column * 190, height - 8), fill=colours[column % len(colours)])
        draw.text((80 + column * 190, height - 22), label, fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_clips(renderer, clips: np.ndarray, object_name: str) -> np.ndarray:
    """Render ``(count, frames, feature)`` clips for one object. Returns ``(count, T, H, W, 3)``."""
    from genrl.viz.warp_clips import features_to_qpos

    scene = renderer._scene(object_name)
    return scene.render(features_to_qpos(clips, scene.layout))
