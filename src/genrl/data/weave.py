"""Motion clips from the Weave rollout dataset.

Clips from the selected objects are concatenated on the time axis and kept
in CPU memory. Body and joint names follow the dataset order documented in
the Weave dataset card. A robot URDF can list those same names in a different
order; compare names explicitly and permute before driving the asset.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("/mnt/workspace/btx0424/Weave")

# Dataset card order. The npz files do not store these names.
BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "L_index_proximal",
    "L_middle_proximal",
    "L_pinky_proximal",
    "L_ring_proximal",
    "L_thumb_proximal_base",
    "R_index_proximal",
    "R_middle_proximal",
    "R_pinky_proximal",
    "R_ring_proximal",
    "R_thumb_proximal_base",
    "L_index_intermediate",
    "L_middle_intermediate",
    "L_pinky_intermediate",
    "L_ring_intermediate",
    "L_thumb_proximal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_pinky_intermediate",
    "R_ring_intermediate",
    "R_thumb_proximal",
    "L_thumb_intermediate",
    "R_thumb_intermediate",
    "L_thumb_distal",
    "R_thumb_distal",
)

JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
)

_ARRAY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
    "contact_label",
)

# Knees and ankles touch the ground. Frame 0 also tags both knees in contact
# even though that contact is gone a frame later, so they are not object contact.
_GROUND_CONTACT_BODIES = frozenset(
    {
        "left_knee_link",
        "right_knee_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    }
)

_AVAILABLE_OBJECTS = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "smallbox",
    "smalltable",
    "tripod",
    "whitechair",
    "woodchair",
)


def _concat(chunks: list[np.ndarray]) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


class WeaveDataset:
    """Flat, in-memory Weave rollout clips for a chosen object subset."""

    def __init__(
        self,
        objects: Sequence[str] = ("smalltable", "smallbox"),
        root: str | Path = DEFAULT_ROOT,
        split: str = "test",
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
        unknown = [name for name in objects if name not in _AVAILABLE_OBJECTS]
        if unknown:
            raise ValueError(
                f"Unknown objects {unknown}. Available: {list(_AVAILABLE_OBJECTS)}."
            )
        if len(objects) == 0:
            raise ValueError("Select at least one object.")

        self.root = Path(root)
        self.split = split
        self.objects = tuple(objects)

        chunks: dict[str, list[np.ndarray]] = {key: [] for key in _ARRAY_KEYS}
        name_chunks: list[np.ndarray] = []
        object_chunks: list[np.ndarray] = []
        length_chunks: list[np.ndarray] = []
        fps: int | None = None

        for name in self.objects:
            path = self.root / "rollout" / split / f"{name}_rollout.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as data:
                file_fps = int(np.asarray(data["fps"]).reshape(-1)[0])
                if fps is None:
                    fps = file_fps
                elif file_fps != fps:
                    raise ValueError(f"{path} is {file_fps} fps; expected {fps}.")
                lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
                print(
                    f"loading {path.name}: {len(lengths)} clips, {int(lengths.sum())} frames",
                    flush=True,
                )
                for key in _ARRAY_KEYS:
                    chunks[key].append(np.asarray(data[key]))
                name_chunks.append(np.asarray(data["motion_names"]))
                object_chunks.append(np.asarray(data["object_names"]))
                length_chunks.append(lengths)

        assert fps is not None
        self._fps = fps
        self.joint_pos = _concat(chunks["joint_pos"])
        self.joint_vel = _concat(chunks["joint_vel"])
        self.body_pos_w = _concat(chunks["body_pos_w"])
        self.body_quat_w = _concat(chunks["body_quat_w"])
        self.body_lin_vel_w = _concat(chunks["body_lin_vel_w"])
        self.body_ang_vel_w = _concat(chunks["body_ang_vel_w"])
        self.object_pos_w = _concat(chunks["object_pos_w"])
        self.object_quat_w = _concat(chunks["object_quat_w"])
        self.object_lin_vel_w = _concat(chunks["object_lin_vel_w"])
        self.object_ang_vel_w = _concat(chunks["object_ang_vel_w"])
        self.contact_label = _concat(chunks["contact_label"])
        self.motion_names = _concat(name_chunks)
        self.object_names = _concat(object_chunks)
        self.motion_lengths = _concat(length_chunks)

        if self.joint_pos.shape[1] != len(JOINT_NAMES):
            raise ValueError(
                f"joint_pos has {self.joint_pos.shape[1]} joints; "
                f"dataset order lists {len(JOINT_NAMES)}."
            )
        if self.body_pos_w.shape[1] != len(BODY_NAMES):
            raise ValueError(
                f"body_pos_w has {self.body_pos_w.shape[1]} bodies; "
                f"dataset order lists {len(BODY_NAMES)}."
            )
        if int(self.motion_lengths.sum()) != self.joint_pos.shape[0]:
            raise ValueError("motion_lengths do not sum to the flat frame count.")

        boundaries = np.empty(len(self.motion_lengths) + 1, dtype=np.int64)
        boundaries[0] = 0
        np.cumsum(self.motion_lengths, out=boundaries[1:])
        # clip_end_idx is the inclusive index of each clip's last frame.
        self._clip_start = boundaries[:-1]
        self._clip_end = boundaries[1:] - 1
        self._joint_index = {name: i for i, name in enumerate(JOINT_NAMES)}
        self._body_index = {name: i for i, name in enumerate(BODY_NAMES)}

    @property
    def body_names(self) -> tuple[str, ...]:
        """Robot body names in dataset order."""
        return BODY_NAMES

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Robot joint names in dataset order."""
        return JOINT_NAMES

    @property
    def clip_start_idx(self) -> np.ndarray:
        """Index of the first frame of each clip. Shape (num_clips,)."""
        return self._clip_start

    @property
    def clip_end_idx(self) -> np.ndarray:
        """Inclusive index of the last frame of each clip. Shape (num_clips,)."""
        return self._clip_end

    @property
    def num_clips(self) -> int:
        return int(self.motion_lengths.shape[0])

    @property
    def total_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def fps(self) -> int:
        return self._fps

    def clip_slice(self, index: int) -> slice:
        """Half-open frame slice for one clip."""
        if index < 0 or index >= self.num_clips:
            raise IndexError(index)
        return slice(int(self._clip_start[index]), int(self._clip_end[index]) + 1)

    def joint_permutation(self, joint_names: Sequence[str]) -> np.ndarray:
        """Columns of ``joint_pos`` that reproduce ``joint_names`` in that order.

        The name set must match the dataset. Order is allowed to differ; that
        is the usual case for a URDF that is not stored in Isaac articulation order.
        """
        requested = list(joint_names)
        missing = [name for name in JOINT_NAMES if name not in requested]
        extra = [name for name in requested if name not in self._joint_index]
        if missing or extra or len(requested) != len(JOINT_NAMES):
            raise ValueError(
                "Robot joint names do not match the dataset. "
                f"Missing {missing}. Extra {extra}."
            )
        return np.asarray([self._joint_index[name] for name in requested], dtype=np.int64)

    def check_body_names(self, body_names: Sequence[str]) -> None:
        """Require every dataset body to exist on the robot asset.

        Extra asset bodies (head, logos, fixed hand bases) are expected.
        """
        present = set(body_names)
        missing = [name for name in BODY_NAMES if name not in present]
        if missing:
            raise ValueError(f"Robot asset is missing dataset bodies: {missing}.")


def _interaction_onset(
    contact: np.ndarray,
    speed: np.ndarray,
    threshold: float,
    quiet_frames: int,
    grace_frames: int,
) -> int:
    """First local frame where object contact and a fresh speed crossing coincide.

    A crossing is a frame whose speed exceeds ``threshold`` after ``quiet_frames``
    at or below it. Contact may arrive up to ``grace_frames`` later, which covers
    clips where the object is already moving on the frame contact closes.
    Returns -1 when the clip never does this.
    """
    above = speed > threshold
    for t in range(speed.shape[0]):
        history = above[max(0, t - quiet_frames) : t]
        if not above[t] or history.any():
            continue
        last = min(speed.shape[0], t + grace_frames + 1)
        hits = np.flatnonzero(contact[t:last])
        if hits.size == 0:
            continue
        onset = t + int(hits[0])
        if speed[onset] > threshold:
            return onset
    return -1


class WeavePreInteractionDataset(WeaveDataset):
    """The last ``duration_s`` of each clip before the object starts moving.

    Interaction onset is the first frame where a non-ground body is in contact
    and the object's linear speed has just crossed ``speed_threshold``. Each
    kept segment ends on that frame and is exactly ``duration_s`` long. Clips
    with no onset, or whose onset is too early to fill that window, are dropped.
    """

    def __init__(
        self,
        objects: Sequence[str] = ("smalltable", "smallbox"),
        root: str | Path = DEFAULT_ROOT,
        split: str = "train",
        duration_s: float = 1.0,
        speed_threshold: float = 0.2,
        quiet_s: float = 0.2,
        grace_s: float = 0.1,
    ):
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}.")
        if speed_threshold <= 0:
            raise ValueError(f"speed_threshold must be positive, got {speed_threshold}.")
        super().__init__(objects, root=root, split=split)
        self.duration_s = float(duration_s)
        self.speed_threshold = float(speed_threshold)

        object_bodies = np.array(
            [name not in _GROUND_CONTACT_BODIES for name in BODY_NAMES]
        )
        in_contact = (self.contact_label[:, object_bodies] > 0.5).any(axis=1)
        speed = np.linalg.norm(self.object_lin_vel_w, axis=-1)
        quiet_frames = max(1, int(round(quiet_s * self.fps)))
        grace_frames = max(0, int(round(grace_s * self.fps)))
        window = max(1, int(round(self.duration_s * self.fps)))

        slices: list[slice] = []
        kept: list[int] = []
        onset_speeds: list[float] = []
        short_names: list[str] = []
        missing_onset = 0
        for clip in range(self.num_clips):
            frames = self.clip_slice(clip)
            onset = _interaction_onset(
                in_contact[frames],
                speed[frames],
                self.speed_threshold,
                quiet_frames,
                grace_frames,
            )
            if onset < 0:
                missing_onset += 1
                continue
            end = int(frames.start) + onset
            start = end - window + 1
            if start < int(frames.start):
                short_names.append(str(self.motion_names[clip]))
                continue
            slices.append(slice(start, end + 1))
            kept.append(clip)
            onset_speeds.append(float(speed[end]))

        if short_names:
            preview = ", ".join(short_names[:5])
            if len(short_names) > 5:
                preview += f", +{len(short_names) - 5} more"
            warnings.warn(
                f"Discarded {len(short_names)} clips shorter than "
                f"{self.duration_s:.2f}s ({window} frames) because interaction "
                f"starts too early: {preview}.",
                stacklevel=2,
            )

        if not slices:
            raise ValueError(
                "No clip has a full "
                f"{self.duration_s:.2f}s before object contact and a speed "
                f"crossing above {self.speed_threshold:.2f} m/s. "
                f"{missing_onset} had no onset, {len(short_names)} were too short."
            )

        kept_index = np.asarray(kept, dtype=np.int64)
        self.motion_names = self.motion_names[kept_index]
        self.object_names = self.object_names[kept_index]
        self.onset_speed = np.asarray(onset_speeds, dtype=np.float32)
        for key in _ARRAY_KEYS:
            parts = [getattr(self, key)[frame_slice] for frame_slice in slices]
            setattr(self, key, _concat(parts))
        self.motion_lengths = np.asarray([s.stop - s.start for s in slices], dtype=np.int64)

        boundaries = np.empty(len(self.motion_lengths) + 1, dtype=np.int64)
        boundaries[0] = 0
        np.cumsum(self.motion_lengths, out=boundaries[1:])
        self._clip_start = boundaries[:-1]
        self._clip_end = boundaries[1:] - 1
        print(
            f"pre-interaction: kept {self.num_clips} clips, "
            f"dropped {missing_onset} with no onset and {len(short_names)} shorter "
            f"than {window} frames, ending at {self.speed_threshold:.2f} m/s in contact",
            flush=True,
        )
