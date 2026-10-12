"""Fixed-length feature windows for the pre-interaction generator.

Each clip is written in a heading-and-position frame defined by the first
pelvis pose: yaw is removed and the pelvis xy at frame 0 is the origin.
Heights stay in world z. Rotations are stored as the first two columns of
the rotation matrix.
"""

from __future__ import annotations

import numpy as np

from genrl.data.weave import BODY_NAMES, WeavePreInteractionDataset

JOINT_DIM = 53
# Layout: joint position, pelvis xyz, pelvis 6D rotation, object xyz, object 6D
# rotation, then the velocities paired with those positions. Velocities are
# generated with the poses and are not tied to them by the training loss.
JOINT_POS = slice(0, JOINT_DIM)
PELVIS_POS = slice(JOINT_DIM, JOINT_DIM + 3)
PELVIS_ROT = slice(JOINT_DIM + 3, JOINT_DIM + 9)
OBJECT_POS = slice(JOINT_DIM + 9, JOINT_DIM + 12)
OBJECT_ROT = slice(JOINT_DIM + 12, JOINT_DIM + 18)
JOINT_VEL = slice(JOINT_DIM + 18, 2 * JOINT_DIM + 18)
PELVIS_VEL = slice(2 * JOINT_DIM + 18, 2 * JOINT_DIM + 21)
OBJECT_VEL = slice(2 * JOINT_DIM + 21, 2 * JOINT_DIM + 24)
FEATURE_DIM = OBJECT_VEL.stop


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from wxyz quaternions. ``R`` applies to column vectors."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1).astype(np.float32)


def _yaw(matrix: np.ndarray) -> np.ndarray:
    return np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    ).astype(np.float32)


def _yaw_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _rotate_xy(points: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate xyz points about z by ``yaw`` radians."""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    x = points[..., 0]
    y = points[..., 1]
    out = np.empty_like(points)
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    out[..., 2] = points[..., 2]
    return out


def clips_to_features(dataset: WeavePreInteractionDataset) -> np.ndarray:
    """Stack kept clips into ``(num_clips, time, FEATURE_DIM)`` float32 features."""
    if BODY_NAMES[0] != "pelvis":
        raise ValueError("Expected the pelvis to be body 0.")
    length = int(dataset.motion_lengths[0])
    if not np.all(dataset.motion_lengths == length):
        raise ValueError(
            "Pre-interaction clips must share one length, "
            f"got {sorted(set(int(v) for v in dataset.motion_lengths.tolist()))}."
        )
    if dataset.joint_pos.shape[1] != JOINT_DIM:
        raise ValueError(f"Expected {JOINT_DIM} joints, got {dataset.joint_pos.shape[1]}.")

    features = np.empty((dataset.num_clips, length, FEATURE_DIM), dtype=np.float32)
    for clip in range(dataset.num_clips):
        frames = dataset.clip_slice(clip)
        joints = np.asarray(dataset.joint_pos[frames], dtype=np.float32)
        pelvis = np.asarray(dataset.body_pos_w[frames, 0], dtype=np.float32)
        obj_pos = np.asarray(dataset.object_pos_w[frames], dtype=np.float32)
        pelvis_quat = np.asarray(dataset.body_quat_w[frames, 0], dtype=np.float32)
        obj_quat = np.asarray(dataset.object_quat_w[frames], dtype=np.float32)

        yaw0 = float(_yaw(_quat_to_matrix(pelvis_quat[0])))
        delta = _yaw_quat(-yaw0)
        origin = np.array([pelvis[0, 0], pelvis[0, 1], 0.0], dtype=np.float32)
        pelvis = _rotate_xy(pelvis - origin, -yaw0)
        obj_pos = _rotate_xy(obj_pos - origin, -yaw0)
        pelvis_rot = _matrix_to_rot6d(_quat_to_matrix(_quat_mul(delta, pelvis_quat)))
        obj_rot = _matrix_to_rot6d(_quat_to_matrix(_quat_mul(delta, obj_quat)))
        joint_vel = np.asarray(dataset.joint_vel[frames], dtype=np.float32)
        pelvis_vel = _rotate_xy(
            np.asarray(dataset.body_lin_vel_w[frames, 0], dtype=np.float32),
            -yaw0,
        )
        obj_vel = _rotate_xy(
            np.asarray(dataset.object_lin_vel_w[frames], dtype=np.float32),
            -yaw0,
        )
        features[clip] = np.concatenate(
            (joints, pelvis, pelvis_rot, obj_pos, obj_rot, joint_vel, pelvis_vel, obj_vel),
            axis=-1,
        )
    return features


def _as_feature_batch(features) -> "torch.Tensor":
    import torch

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"Expected features (batch, time, {FEATURE_DIM}), got {tuple(features.shape)}."
        )
    if features.shape[1] < 2:
        raise ValueError("Object-motion reward needs a clip of at least two frames.")
    return features


def _speed_before_touch(features: "torch.Tensor", skip_tail: int) -> "torch.Tensor":
    """Per-clip mean object speed, dropping ``skip_tail`` frames at the end.

    ``skip_tail`` is 1 when the stored clip still ends on the interaction onset,
    which is the touch and is allowed to move. It is 0 when subsampling has
    already dropped that frame, so the whole clip is before the touch.
    """
    if skip_tail < 0 or skip_tail >= features.shape[1]:
        raise ValueError(
            f"skip_tail must be in [0, time), got {skip_tail} for time {features.shape[1]}."
        )
    velocity = features[:, :, OBJECT_VEL] if skip_tail == 0 else features[:, :-skip_tail, OBJECT_VEL]
    return velocity.norm(dim=-1).mean(dim=-1)


def object_still_reward(features, skip_tail: int = 0) -> "torch.Tensor":
    """Negative mean object speed before the touch.

    A still object scores zero. Motion before contact scores below zero.
    ``features`` is ``(batch, time, FEATURE_DIM)`` in physical units, and the
    result is ``(batch,)`` in m/s. See ``_speed_before_touch`` for ``skip_tail``.
    """
    features = _as_feature_batch(features)
    return -_speed_before_touch(features, skip_tail)


def _scored_clip(features, skip_tail: int) -> "torch.Tensor":
    features = _as_feature_batch(features)
    if skip_tail < 0 or skip_tail >= features.shape[1]:
        raise ValueError(
            f"skip_tail must be in [0, time), got {skip_tail} for time {features.shape[1]}."
        )
    if skip_tail:
        features = features[:, :-skip_tail]
    if features.shape[1] < 2:
        raise ValueError("Reward terms need at least two scored frames.")
    return features


def clip_penalties(features, fps: float, skip_tail: int = 0) -> dict[str, "torch.Tensor"]:
    """Per-clip penalties in physical units. Each value is ``(batch,)``, lower better.

    Object motion is the linear velocity, the finite difference of position, and
    the frame-to-frame spin. Consistency is the gap between each finite
    difference and the paired velocity channel. The touch frame is dropped when
    ``skip_tail`` is 1.
    """
    import torch

    features = _scored_clip(features, skip_tail)
    object_vel = features[:, :, OBJECT_VEL].norm(dim=-1).mean(dim=-1)
    object_pos = (features[:, 1:, OBJECT_POS] - features[:, :-1, OBJECT_POS]).norm(dim=-1).mean(dim=-1)
    object_pos = object_pos * fps

    rotation = _rot6d_to_matrix_torch(features[:, :, OBJECT_ROT])
    relative = rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    object_spin = torch.acos(cosine).mean(dim=-1) * fps

    def gap(position: slice, velocity: slice) -> "torch.Tensor":
        finite = (features[:, 1:, position] - features[:, :-1, position]) * fps
        return finite - features[:, 1:, velocity]

    return {
        "object_vel": object_vel,
        "object_pos": object_pos,
        "object_spin": object_spin,
        "consistency_object": gap(OBJECT_POS, OBJECT_VEL).norm(dim=-1).mean(dim=-1),
        "consistency_pelvis": gap(PELVIS_POS, PELVIS_VEL).norm(dim=-1).mean(dim=-1),
        "consistency_joint": gap(JOINT_POS, JOINT_VEL).abs().mean(dim=(-2, -1)),
    }


def penalty_scales(penalties: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
    """Dataset mean of each penalty, so a real clip scores about one per term."""
    return {name: value.detach().mean().clamp_min(1e-6) for name, value in penalties.items()}


def combined_reward(
    penalties: dict[str, "torch.Tensor"], scales: dict[str, "torch.Tensor"]
) -> "torch.Tensor":
    """Negative sum of penalties divided by their dataset scales. Shape ``(batch,)``."""
    total = None
    for name, value in penalties.items():
        term = value / scales[name]
        total = term if total is None else total + term
    return -total


def object_motion_before_touch(features, skip_tail: int = 0) -> dict[str, float]:
    """Mean object speed before the touch, and on the last stored frame.

    The last-frame speed is only a monitor. It is the onset when ``skip_tail``
    is 1, and an earlier pre-touch frame when subsampling dropped the onset.
    """
    features = _as_feature_batch(features)
    return {
        "speed_before_touch": float(_speed_before_touch(features, skip_tail).mean()),
        "speed_at_end": float(features[:, -1, OBJECT_VEL].norm(dim=-1).mean()),
    }


def position_velocity_consistency(features, fps: float) -> dict[str, float]:
    """Mean gap between a finite difference of position and the paired velocity.

    ``features`` is ``(batch, time, FEATURE_DIM)`` in physical units. Object and
    pelvis gaps are the L2 norm in m/s. The joint gap is the mean absolute
    residual in rad/s. This is a monitor, not a training objective.
    """
    import torch

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)

    def residual(pos: slice, vel: slice) -> torch.Tensor:
        position = features[..., pos]
        finite = (position[..., 1:, :] - position[..., :-1, :]) * fps
        return finite - features[..., 1:, vel]

    object_gap = residual(OBJECT_POS, OBJECT_VEL).norm(dim=-1).mean()
    pelvis_gap = residual(PELVIS_POS, PELVIS_VEL).norm(dim=-1).mean()
    joint_gap = residual(JOINT_POS, JOINT_VEL).abs().mean()
    return {
        "object": float(object_gap),
        "pelvis": float(pelvis_gap),
        "joint": float(joint_gap),
    }


def _rot6d_to_matrix_torch(rot6d: "torch.Tensor") -> "torch.Tensor":
    import torch

    first = rot6d[..., 0:3]
    second = rot6d[..., 3:6]
    b1 = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b2 = second - (b1 * second).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _matrix_to_rot6d_torch(matrix: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def make_rotations_relative(features: "torch.Tensor"):
    """Express pelvis and object rotations relative to frame 0.

    Frame 0 becomes the identity 6D value ``[1, 0, 0, 0, 1, 0]``. The returned
    anchor is the absolute frame-0 rotation, ``(batch, 12)`` as pelvis then object,
    so a viewer can put the motion back in the world.
    """
    import torch

    out = features.clone()
    pieces = []
    for rotation in (PELVIS_ROT, OBJECT_ROT):
        matrix = _rot6d_to_matrix_torch(features[..., rotation])
        anchor = matrix[:, 0]
        relative = anchor.transpose(-1, -2).unsqueeze(1) @ matrix
        out[..., rotation] = _matrix_to_rot6d_torch(relative)
        pieces.append(_matrix_to_rot6d_torch(anchor))
    return out, torch.cat(pieces, dim=-1)


def compose_rotation_anchor(clips: "torch.Tensor", anchor: "torch.Tensor") -> "torch.Tensor":
    """Map relative rotations back to absolute ones. ``anchor`` is ``(batch, 12)``."""
    import torch

    out = clips.clone()
    for rotation, offset in ((PELVIS_ROT, 0), (OBJECT_ROT, 6)):
        anchor_matrix = _rot6d_to_matrix_torch(anchor[:, offset : offset + 6])
        relative = _rot6d_to_matrix_torch(clips[..., rotation])
        absolute = anchor_matrix.unsqueeze(1) @ relative
        out[..., rotation] = _matrix_to_rot6d_torch(absolute)
    return out


def project_rot6d(features: "torch.Tensor") -> "torch.Tensor":
    """Replace both 6D rotations by the nearest proper rotation's columns."""
    import torch

    out = features.clone()
    for rotation in (PELVIS_ROT, OBJECT_ROT):
        raw = features[..., rotation]
        columns = torch.stack((raw[..., 0:3], raw[..., 3:6]), dim=-1)
        left, _singular, right = torch.linalg.svd(columns, full_matrices=False)
        nearest = left @ right
        first, second = nearest[..., 0], nearest[..., 1]
        third = torch.cross(first, second, dim=-1)
        det = (
            first[..., 0] * (second[..., 1] * third[..., 2] - second[..., 2] * third[..., 1])
            - first[..., 1] * (second[..., 0] * third[..., 2] - second[..., 2] * third[..., 0])
            + first[..., 2] * (second[..., 0] * third[..., 1] - second[..., 1] * third[..., 0])
        )
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        second = second * sign.unsqueeze(-1)
        out[..., rotation] = torch.cat((first, second), dim=-1)
    return out


def rot6d_gram_error(features, skip_frames: int = 0) -> dict[str, float]:
    """How far each 6D rotation's two columns are from an orthonormal pair.

    The score is the Frobenius distance between the columns' Gram matrix and
    the 2x2 identity, averaged over frames. A true rotation scores zero.
    ``skip_frames`` drops a pinned prefix so the score is only on generated frames.
    """
    import torch

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if skip_frames:
        features = features[:, skip_frames:]
    if features.shape[-2] == 0:
        return {"pelvis": 0.0, "object": 0.0}

    def gram_error(rotation: slice) -> torch.Tensor:
        columns = features[..., rotation]
        first = columns[..., 0:3]
        second = columns[..., 3:6]
        gram_00 = first.square().sum(dim=-1) - 1
        gram_11 = second.square().sum(dim=-1) - 1
        gram_01 = (first * second).sum(dim=-1)
        return torch.sqrt(gram_00.square() + gram_11.square() + 2 * gram_01.square()).mean()

    return {
        "pelvis": float(gram_error(PELVIS_ROT)),
        "object": float(gram_error(OBJECT_ROT)),
    }
