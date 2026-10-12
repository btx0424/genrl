"""Rasterize a generated segment to a GIF.

The robot is the URDF kinematic tree driven by the generated joints and pelvis
pose. The object is a box at the generated object pose, in the same canonical
frame the features use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yourdfpy
from PIL import Image, ImageDraw

from genrl.data.segments import JOINT_POS, OBJECT_POS, OBJECT_ROT, PELVIS_POS, PELVIS_ROT
from genrl.data.view import DEFAULT_URDF, load_robot
from genrl.data.weave import JOINT_NAMES

_BOX = 0.12
_WIDTH = 480
_HEIGHT = 480
_PIXELS_PER_METER = 150


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Nearest proper rotation to a 6D pair, as a matrix of columns.

    Gram-Schmidt flips the body when the two columns are nearly parallel.
    The SVD projection stays on the closest rotation instead.
    """
    first = rot6d[..., 0:3]
    second = rot6d[..., 3:6]
    columns = np.stack((first, second), axis=-1)
    left, _singular, right = np.linalg.svd(columns, full_matrices=False)
    nearest = left @ right
    col0 = nearest[..., 0]
    col1 = nearest[..., 1]
    col2 = np.cross(col0, col1)
    det = (
        col0[..., 0] * (col1[..., 1] * col2[..., 2] - col1[..., 2] * col2[..., 1])
        - col0[..., 1] * (col1[..., 0] * col2[..., 2] - col1[..., 2] * col2[..., 0])
        + col0[..., 2] * (col1[..., 0] * col2[..., 1] - col1[..., 1] * col2[..., 0])
    )
    sign = np.where(det < 0, -1.0, 1.0)
    col2 = col2 * sign[..., None]
    return np.stack((col0, col1, col2), axis=-1)


def _box_edges() -> list[tuple[int, int]]:
    edges = []
    for axis in range(3):
        for i in range(8):
            j = i ^ (1 << axis)
            if i < j:
                edges.append((i, j))
    return edges


class ClipRenderer:
    """Project one canonical-frame clip to GIF frames."""

    def __init__(self, urdf_path: str | Path | None = None):
        path = Path(urdf_path) if urdf_path else DEFAULT_URDF
        self.urdf = load_robot(path, load_meshes=False)
        dataset_index = {name: i for i, name in enumerate(JOINT_NAMES)}
        actuated = tuple(self.urdf.actuated_joint_names)
        missing = [name for name in actuated if name not in dataset_index]
        if missing or len(actuated) != len(JOINT_NAMES):
            raise ValueError(f"URDF joints do not match the dataset: {missing}.")
        self.permutation = np.asarray([dataset_index[name] for name in actuated], dtype=np.int64)
        self.edges: list[tuple[str, str]] = []
        for joint in self.urdf.joint_map.values():
            parent = joint.parent
            child = joint.child
            if parent in (None, "world", self.urdf.base_link):
                parent = self.urdf.base_link
            self.edges.append((parent, child))
        self.box_edges = _box_edges()
        forward = _unit(np.array([-1.0, 1.0, -0.35], dtype=np.float64))
        right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0])))
        up = np.cross(right, forward)
        self.right = right
        self.up = up

    def _link_points(self, joint_pos: np.ndarray) -> dict[str, np.ndarray]:
        self.urdf.update_cfg(joint_pos[self.permutation])
        points = {self.urdf.base_link: np.zeros(3, dtype=np.float64)}
        for _parent, child in self.edges:
            local = self.urdf.get_transform(child, self.urdf.base_link)[:3, 3]
            points[child] = np.asarray(local, dtype=np.float64)
        return points

    def _project(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        relative = points - center
        u = relative @ self.right * _PIXELS_PER_METER + _WIDTH / 2
        v = _HEIGHT / 2 - relative @ self.up * _PIXELS_PER_METER
        return np.stack((u, v), axis=-1)

    def write_gif(self, clip: np.ndarray, path: Path, fps: int) -> None:
        """Write one ``(time, feature)`` clip. ``path`` is the GIF file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        center = np.median(clip[:, PELVIS_POS], axis=0).astype(np.float64)
        center[2] = 0.7
        frames: list[Image.Image] = []
        box = _box_corners()
        for frame in range(clip.shape[0]):
            pelvis_rot = _rot6d_to_matrix(clip[frame, PELVIS_ROT])
            pelvis_pos = clip[frame, PELVIS_POS].astype(np.float64)
            local = self._link_points(clip[frame, JOINT_POS])
            world = {
                name: pelvis_rot @ point + pelvis_pos for name, point in local.items()
            }
            image = Image.new("RGB", (_WIDTH, _HEIGHT), (248, 248, 248))
            draw = ImageDraw.Draw(image)
            self._draw_grid(draw, center)
            for parent, child in self.edges:
                self._draw_segment(draw, world[parent], world[child], center, (50, 50, 55))
            object_rot = _rot6d_to_matrix(clip[frame, OBJECT_ROT])
            object_pos = clip[frame, OBJECT_POS].astype(np.float64)
            corners = (object_rot @ box.T).T + object_pos
            for i, j in self.box_edges:
                self._draw_segment(draw, corners[i], corners[j], center, (210, 110, 40))
            frames.append(image)
        duration = max(20, int(round(1000 / fps)))
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

    def _draw_grid(self, draw: ImageDraw.ImageDraw, center: np.ndarray) -> None:
        for value in np.linspace(-1.5, 1.5, 7):
            self._draw_segment(
                draw,
                np.array([value, -1.5, 0.0]),
                np.array([value, 1.5, 0.0]),
                center,
                (220, 220, 220),
            )
            self._draw_segment(
                draw,
                np.array([-1.5, value, 0.0]),
                np.array([1.5, value, 0.0]),
                center,
                (220, 220, 220),
            )

    def _draw_segment(
        self,
        draw: ImageDraw.ImageDraw,
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        projected = self._project(np.stack((start, end)), center)
        draw.line(
            [tuple(projected[0]), tuple(projected[1])],
            fill=color,
            width=2,
        )


def _box_corners() -> np.ndarray:
    signs = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    return signs * _BOX


def expand_time(clips: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Linearly stretch ``(count, time, feature)`` clips from ``src_hz`` to ``dst_hz``.

    Endpoints stay put. A 10 Hz second becomes 46 frames played at 50 Hz, which
    covers the same 0.9 s span as the kept samples.
    """
    if dst_hz == src_hz:
        return clips
    if dst_hz < src_hz or dst_hz % src_hz != 0:
        raise ValueError(f"Cannot expand {src_hz} Hz to {dst_hz} Hz.")
    factor = dst_hz // src_hz
    count, time, dim = clips.shape
    out_time = (time - 1) * factor + 1
    stretched = np.empty((count, out_time, dim), dtype=clips.dtype)
    src = np.linspace(0.0, out_time - 1, time)
    dst = np.arange(out_time, dtype=np.float64)
    for index in range(count):
        for channel in range(dim):
            stretched[index, :, channel] = np.interp(dst, src, clips[index, :, channel])
    return stretched


def write_sample_gifs(
    clips: np.ndarray,
    directory: Path,
    fps: int,
    urdf_path: str | Path | None = None,
    renderer: ClipRenderer | None = None,
) -> list[Path]:
    """Write ``clips`` of shape ``(count, time, feature)`` and return the GIF paths."""
    if renderer is None:
        renderer = ClipRenderer(urdf_path)
    paths = []
    for index in range(clips.shape[0]):
        path = directory / f"{index:02d}.gif"
        renderer.write_gif(clips[index], path, fps)
        paths.append(path)
    return paths
