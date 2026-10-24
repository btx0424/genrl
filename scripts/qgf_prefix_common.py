"""Shared prefix construction for the flow-matching and QGF prefix evals.

Keeping this in one place means the two scripts build identical ``real``, ``mid``,
``lerp``, ``interp``, ``jitter``, ``objshift`` and ``randn`` first frames, so only
the sampler differs between them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from genrl.data.segments import (
    OBJECT_POS,
    OBJECT_ROT,
    OBJECT_VEL,
    PELVIS_POS,
    PELVIS_ROT,
    PELVIS_VEL,
    _matrix_to_rot6d_torch,
    _rot6d_to_matrix_torch,
)
from genrl.train_common import load_raw_segments

ROTATIONS = (PELVIS_ROT, OBJECT_ROT)
POSITIONS = (PELVIS_POS, OBJECT_POS)
VELOCITIES = (PELVIS_VEL, OBJECT_VEL)
MODES = ("real", "mid", "lerp", "interp", "jitter", "objshift", "randn")


def add_prefix_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--objects", nargs="+", default=["smalltable"])
    parser.add_argument("--mode", default="mid", choices=MODES)
    parser.add_argument("--min-offset", type=int, default=1, help="mid/lerp: smallest later frame used")
    parser.add_argument("--max-offset", type=int, default=4, help="mid/lerp: largest later frame used")
    parser.add_argument("--alpha", type=float, default=0.5, help="jitter: frame-0 noise in std units")
    parser.add_argument("--weight", type=float, default=-1.0, help="lerp/interp: fixed blend, <0 draws")
    parser.add_argument("--shift", type=float, default=0.15, help="objshift: metres along the reach")
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("outputs/flow_prefix"))
    parser.add_argument("--render", action="store_true", help="write generated GIFs")
    parser.add_argument("--render-gt", action="store_true", help="also write the real clip")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    return parser


def load_data(flow_config: DictConfig, args) -> tuple[np.ndarray, int, np.ndarray, int, np.ndarray]:
    data = OmegaConf.create(OmegaConf.to_container(flow_config.data, resolve=True))
    data.split = args.split
    data.objects = list(args.objects)
    return load_raw_segments(OmegaConf.create({"data": data}))


def _rotate_z(vector: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out = vector.clone()
    out[..., 0] = cos[:, None] * vector[..., 0] + sin[:, None] * vector[..., 1]
    out[..., 1] = -sin[:, None] * vector[..., 0] + cos[:, None] * vector[..., 1]
    return out


def recenter(clips: torch.Tensor) -> torch.Tensor:
    """Re-anchor each clip on its own frame 0, as ``clips_to_features`` does."""
    frame = clips[:, 0]
    pelvis_rotation = _rot6d_to_matrix_torch(frame[:, PELVIS_ROT])
    cos, sin = pelvis_rotation[:, 0, 0], pelvis_rotation[:, 1, 0]
    origin = frame[:, PELVIS_POS].clone()
    origin[:, 2] = 0.0

    out = clips.clone()
    for position in POSITIONS:
        out[:, :, position] = _rotate_z(out[:, :, position] - origin[:, None], cos, sin)
    for velocity in VELOCITIES:
        out[:, :, velocity] = _rotate_z(out[:, :, velocity], cos, sin)
    for rotation in ROTATIONS:
        anchor = _rot6d_to_matrix_torch(frame[:, rotation])
        out[:, :, rotation] = _matrix_to_rot6d_torch(
            anchor.transpose(-1, -2)[:, None] @ _rot6d_to_matrix_torch(clips[:, :, rotation])
        )
    return out


def take_window(clips: torch.Tensor, offsets: torch.Tensor, length: int) -> torch.Tensor:
    rows = []
    for index, offset in enumerate(offsets.tolist()):
        frame = clips[index, offset : offset + length]
        if frame.shape[0] < length:
            frame = torch.cat([frame, frame[-1:].expand(length - frame.shape[0], -1)])
        rows.append(frame)
    return torch.stack(rows)


def blend(first: torch.Tensor, second: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    out = (1.0 - weights[:, None]) * first + weights[:, None] * second
    matrix_weight = weights[:, None, None]
    for rotation in ROTATIONS:
        left, _singular, right = torch.linalg.svd(
            (1.0 - matrix_weight) * _rot6d_to_matrix_torch(first[:, rotation])
            + matrix_weight * _rot6d_to_matrix_torch(second[:, rotation]),
            full_matrices=False,
        )
        out[:, rotation] = _matrix_to_rot6d_torch(left @ right)
    return out


def build_clip(raw: torch.Tensor, mode: str, args, generator) -> tuple[torch.Tensor, torch.Tensor]:
    """The window the prefix is drawn from, and the per-clip start frame."""
    length = raw.shape[1]
    offsets = torch.zeros(raw.shape[0], dtype=torch.long)
    clip = raw
    if mode in ("mid", "lerp"):
        offsets = torch.randint(args.min_offset, args.max_offset + 1, (raw.shape[0],), generator=generator)
        clip = take_window(raw, offsets, length)
        if mode == "mid":
            clip = recenter(clip)
    return offsets, clip


def draw_weights(count: int, args, generator) -> torch.Tensor:
    if args.weight >= 0.0:
        return torch.full((count,), args.weight)
    return torch.rand(count, generator=generator) * 0.6 + 0.2


def build_prefix(
    mode: str, features: torch.Tensor, raw: torch.Tensor, offsets: torch.Tensor, args, generator
) -> torch.Tensor:
    first = features[:, 0]
    if mode in ("real", "mid"):
        return first.clone()
    if mode == "lerp":
        later = features[torch.arange(raw.shape[0]), offsets]
        return blend(first, later, draw_weights(raw.shape[0], args, generator))
    if mode == "interp":
        count = raw.shape[0]
        partner = (torch.arange(count) + count // 2) % count
        return blend(first, first[partner], draw_weights(count, args, generator))
    if mode == "jitter":
        noise = torch.randn(first.shape, generator=generator) * args.alpha
        for rotation in ROTATIONS:
            noise[:, rotation] = 0.0
        return first + noise
    if mode == "objshift":
        shifted = first.clone()
        direction = shifted[:, OBJECT_POS] - shifted[:, PELVIS_POS]
        direction[:, 2] = 0.0
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        shifted[:, OBJECT_POS] = first[:, OBJECT_POS] + args.shift * direction
        return shifted
    if mode == "randn":
        return torch.randn(first.shape, generator=generator)
    raise ValueError(mode)


def novelty(prefix: torch.Tensor, index: torch.Tensor, raw: torch.Tensor, mean, std, mode: str) -> torch.Tensor:
    """Distance of each prefix to the nearest real frame 0, in std units."""
    if mode == "randn":
        return torch.full((index.shape[0],), float("nan"))
    real_first = ((raw[:, 0:1] - mean) / std)[:, 0]
    return torch.cdist(prefix[index], real_first).min(dim=1).values
