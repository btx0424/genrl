"""Render Weave clips with mujoco_warp, one world per clip.

Each clip is a world in one batched scene: the G1 visual meshes plus that
clip's object mesh. A frame is one kinematics update and one render across
every clip, so the GIFs stay in lockstep. Training writes its samples through
``WarpSampleRenderer``.

mujoco_warp is the local checkout at ``references/mujoco_warp``. warp-lang
1.15 is required: 1.18 asks for a CUDA 13 driver and falls back to CPU here.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from PIL import Image

from genrl.data.segments import (
    JOINT_POS,
    OBJECT_POS,
    OBJECT_ROT,
    PELVIS_POS,
    PELVIS_ROT,
)
from genrl.data.view import DEFAULT_URDF
from genrl.data.weave import (
    DEFAULT_ROOT,
    JOINT_NAMES,
    WeavePreInteractionDataset,
)

_MIMIC = re.compile(r"\s*<mimic [^>]*/>")
_PELVIS_JOINT = "pelvis_free"
_OBJECT_JOINT = "object_free"
_CAMERA = "view"


@dataclass(frozen=True)
class SceneLayout:
    """qpos addresses for the free pelvis, the 53 actuated joints, and the object."""

    nq: int
    joint_qpos: np.ndarray  # (53,) dataset joint order
    object_pos: slice
    object_quat: slice


def _object_mesh(root: Path, name: str) -> Path:
    path = root / "objects" / name / f"{name}.obj"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_scene(urdf_path: Path, object_name: str, object_file: Path) -> tuple[mujoco.MjModel, SceneLayout]:
    """G1 with a free pelvis, plus one free-floating object mesh and a camera.

    Mimic tags are stripped first. Weave records the Inspire finger joints as
    independent angles, and they do not follow the URDF mimic equation.
    """
    text = _MIMIC.sub("", Path(urdf_path).read_text())
    spec = mujoco.MjSpec.from_string(text)
    spec.compiler.meshdir = str(Path(urdf_path).parent)
    spec.body("pelvis").add_freejoint(name=_PELVIS_JOINT)

    spec.add_mesh(name=object_name, file=str(object_file))
    obj = spec.worldbody.add_body(name="object")
    obj.add_freejoint(name=_OBJECT_JOINT)
    obj.add_geom(
        name="object_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=object_name,
        rgba=[0.92, 0.48, 0.16, 1.0],
    )
    # 0.25 m checker squares. texuniform maps one repeat to one meter, so
    # texrepeat 4 puts a line every 25 cm and a sunk foot crosses a square.
    spec.add_texture(
        name="ground",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=128,
        height=128,
        rgb1=[0.86, 0.87, 0.84],
        rgb2=[0.45, 0.48, 0.44],
    )
    ground = spec.add_material(name="ground", texuniform=True, texrepeat=[4.0, 4.0])
    ground.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "ground"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="ground",
        rgba=[0.8, 0.8, 0.8, 1.0],
    )
    spec.worldbody.add_light(
        name="sun",
        pos=[0, 0, 4],
        dir=[0.3, 0.4, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.8, 0.8, 0.8],
        specular=[0.2, 0.2, 0.2],
    )
    # Placeholder pose. Each frame overwrites cam_xpos and cam_xmat per world.
    spec.worldbody.add_camera(name=_CAMERA, pos=[2.5, -2.5, 1.6], fovy=40)
    model = spec.compile()

    addresses = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        addresses[name] = int(model.jnt_qposadr[joint_id])
    missing = [name for name in JOINT_NAMES if name not in addresses]
    if missing or _PELVIS_JOINT not in addresses or _OBJECT_JOINT not in addresses:
        raise ValueError(f"Compiled scene is missing joints: {missing}.")
    if addresses[_PELVIS_JOINT] != 0:
        raise ValueError("The pelvis free joint is not the first qpos block.")
    object_adr = addresses[_OBJECT_JOINT]
    layout = SceneLayout(
        nq=int(model.nq),
        joint_qpos=np.asarray([addresses[name] for name in JOINT_NAMES], dtype=np.int64),
        object_pos=slice(object_adr, object_adr + 3),
        object_quat=slice(object_adr + 3, object_adr + 7),
    )
    if layout.nq != 7 + len(JOINT_NAMES) + 7:
        raise ValueError(f"Expected 67 qpos, found {layout.nq}.")
    return model, layout


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Nearest proper rotation to a 6D pair, as a matrix of columns."""
    columns = np.stack((rot6d[..., 0:3], rot6d[..., 3:6]), axis=-1)
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
    col2 = col2 * np.where(det < 0, -1.0, 1.0)[..., None]
    return np.stack((col0, col1, col2), axis=-1)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """wxyz quaternions. ``matrix`` columns are the rotated basis."""
    flat = np.reshape(np.ascontiguousarray(matrix, dtype=np.float64), (-1, 3, 3))
    quat = np.empty((flat.shape[0], 4), dtype=np.float64)
    one = np.zeros(4, dtype=np.float64)
    for index, item in enumerate(flat):
        mujoco.mju_mat2Quat(one, item.reshape(-1))
        quat[index] = one
    return quat.reshape(matrix.shape[:-2] + (4,)).astype(np.float32)


def features_to_qpos(clips: np.ndarray, layout: SceneLayout) -> np.ndarray:
    """Canonical-frame feature clips to scene qpos. Shape (nworld, time, nq)."""
    if clips.ndim != 3 or clips.shape[-1] < OBJECT_ROT.stop:
        raise ValueError(f"Expected (nworld, time, features), got {clips.shape}.")
    qpos = np.zeros((clips.shape[0], clips.shape[1], layout.nq), dtype=np.float32)
    qpos[:, :, 0:3] = clips[:, :, PELVIS_POS]
    qpos[:, :, 3:7] = _matrix_to_quat(_rot6d_to_matrix(clips[:, :, PELVIS_ROT]))
    qpos[:, :, layout.joint_qpos] = clips[:, :, JOINT_POS]
    qpos[:, :, layout.object_pos] = clips[:, :, OBJECT_POS]
    qpos[:, :, layout.object_quat] = _matrix_to_quat(_rot6d_to_matrix(clips[:, :, OBJECT_ROT]))
    return qpos


def clip_qpos(dataset: WeavePreInteractionDataset, index: int, layout: SceneLayout) -> np.ndarray:
    """World-frame qpos for one clip. Shape (T, nq). Quaternions are wxyz."""
    frames = dataset.clip_slice(index)
    length = frames.stop - frames.start
    qpos = np.zeros((length, layout.nq), dtype=np.float32)
    qpos[:, 0:3] = dataset.body_pos_w[frames, 0]
    qpos[:, 3:7] = dataset.body_quat_w[frames, 0]
    qpos[:, layout.joint_qpos] = dataset.joint_pos[frames]
    qpos[:, layout.object_pos] = dataset.object_pos_w[frames]
    qpos[:, layout.object_quat] = dataset.object_quat_w[frames]
    return qpos


def _camera_frame(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera orientation whose columns are (right, up, back).

    MuJoCo looks along -Z. Warp stores ``mat33`` with those axes as columns,
    and ``cam_xmat @ ray`` maps a camera-frame direction into the world.
    """
    back = eye - target
    back = back / np.clip(np.linalg.norm(back, axis=-1, keepdims=True), 1e-8, None)
    world_up = np.zeros_like(back)
    world_up[..., 2] = 1.0
    right = np.cross(world_up, back)
    right_norm = np.linalg.norm(right, axis=-1, keepdims=True)
    fallback = np.zeros_like(back)
    fallback[..., 1] = 1.0
    right = np.where(right_norm < 1e-6, np.cross(fallback, back), right)
    right = right / np.clip(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8, None)
    up = np.cross(back, right)
    return np.stack((right, up, back), axis=-1)


class ParallelClipRenderer:
    """Batched mujoco_warp renderer. Worlds share one object mesh."""

    def __init__(self, model: mujoco.MjModel, layout: SceneLayout, width: int = 480, height: int = 480):
        self.model = model
        self.layout = layout
        self.width = int(width)
        self.height = int(height)
        self._device_model = mjw.put_model(model)
        self._nworld = 0
        self._data = None
        self._context = None
        self._rgb = None

    def _ensure(self, nworld: int) -> None:
        if self._nworld == nworld:
            return
        self._data = mjw.make_data(self.model, nworld=nworld)
        self._context = mjw.create_render_context(
            self.model,
            nworld=nworld,
            cam_res=(self.width, self.height),
            render_rgb=True,
            render_depth=False,
            render_seg=False,
            use_shadows=False,
            use_ambient_lighting=True,
            background_color=(0.93, 0.94, 0.96, 1.0),
        )
        self._rgb = wp.zeros((nworld, self.height, self.width), dtype=wp.vec3)
        self._nworld = nworld

    def _aim(self, qpos: np.ndarray) -> None:
        """Point each world's camera at the midpoint of the pelvis and the object."""
        pelvis = qpos[:, 0:3]
        obj = qpos[:, self.layout.object_pos]
        target = 0.5 * (pelvis + obj)
        target[:, 2] = 0.7
        direction = np.array([1.15, -1.35, 0.85], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        eye = target + direction * 2.6
        position = np.asarray(self._data.cam_xpos.numpy())
        orientation = np.asarray(self._data.cam_xmat.numpy())
        position[:, 0] = eye
        orientation[:, 0] = _camera_frame(eye, target)
        self._data.cam_xpos.assign(position)
        self._data.cam_xmat.assign(orientation)

    def render(self, qpos: np.ndarray) -> np.ndarray:
        """Render a batch of clips.

        Args:
            qpos: (nworld, time, nq) in the compiled scene order.

        Returns:
            uint8 RGB images, shape (nworld, time, height, width, 3).
        """
        if qpos.ndim != 3 or qpos.shape[-1] != self.layout.nq:
            raise ValueError(f"qpos must be (nworld, time, {self.layout.nq}), got {qpos.shape}.")
        nworld, length, _ = qpos.shape
        self._ensure(nworld)
        frames = np.empty((nworld, length, self.height, self.width, 3), dtype=np.uint8)
        posed = np.ascontiguousarray(qpos, dtype=np.float32)
        for t in range(length):
            self._data.qpos.assign(np.ascontiguousarray(posed[:, t]))
            mjw.fwd_kinematics(self._device_model, self._data)
            self._aim(posed[:, t])
            mjw.refit_bvh(self._device_model, self._data, self._context)
            mjw.render(self._device_model, self._data, self._context)
            mjw.get_rgb(self._context, 0, self._rgb)
            frames[:, t] = (np.clip(self._rgb.numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
        return frames


class WarpSampleRenderer:
    """Training GIF writer. One cached scene per object mesh."""

    def __init__(
        self,
        root: str | Path,
        urdf_path: str | Path | None = None,
        width: int = 480,
        height: int = 480,
    ):
        wp.init()
        self.root = Path(root)
        self.urdf_path = Path(urdf_path) if urdf_path else DEFAULT_URDF
        self.width = int(width)
        self.height = int(height)
        self._scenes: dict[str, ParallelClipRenderer] = {}

    def _scene(self, object_name: str) -> ParallelClipRenderer:
        cached = self._scenes.get(object_name)
        if cached is not None:
            return cached
        print(f"mujoco_warp scene for {object_name}", flush=True)
        model, layout = build_scene(
            self.urdf_path,
            object_name,
            _object_mesh(self.root, object_name),
        )
        renderer = ParallelClipRenderer(model, layout, width=self.width, height=self.height)
        self._scenes[object_name] = renderer
        return renderer

    def write_gifs(
        self,
        clips: np.ndarray,
        object_names: np.ndarray,
        directory: Path,
        fps: int,
    ) -> list[Path]:
        """Write ``clips`` of shape ``(count, time, feature)``.

        ``object_names`` names the mesh for each clip, in the same order.
        Poses are the canonical training frame: pelvis xy starts at the origin
        and the first-frame yaw has been removed.
        """
        names = [str(name) for name in np.asarray(object_names).tolist()]
        if len(names) != clips.shape[0]:
            raise ValueError(f"{len(names)} object names for {clips.shape[0]} clips.")
        directory = Path(directory)
        paths = [directory / f"{index:02d}_{name}.gif" for index, name in enumerate(names)]
        groups: dict[str, list[int]] = {}
        for index, name in enumerate(names):
            groups.setdefault(name, []).append(index)
        for name, indices in groups.items():
            renderer = self._scene(name)
            images = renderer.render(features_to_qpos(clips[indices], renderer.layout))
            for local, index in enumerate(indices):
                write_gif(images[local], paths[index], fps)
        return paths


def write_gif(frames: np.ndarray, path: Path, fps: int) -> None:
    """Save one clip. ``frames`` is (time, height, width, 3) uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / fps)),
        loop=0,
    )


def write_dataset_gifs(
    dataset: WeavePreInteractionDataset,
    clip_indices: np.ndarray,
    out_dir: Path,
    urdf_path: Path = DEFAULT_URDF,
    width: int = 480,
    height: int = 480,
) -> list[Path]:
    """Render the given clips, grouped by object so each scene shares one mesh."""
    if len(clip_indices) == 0:
        raise ValueError("No clips to render.")
    out_dir = Path(out_dir)
    written: list[Path] = []
    order = np.asarray(clip_indices, dtype=np.int64)
    names = dataset.object_names[order]
    for object_name in dict.fromkeys(names.tolist()):
        group = order[names == object_name]
        model, layout = build_scene(urdf_path, str(object_name), _object_mesh(Path(dataset.root), str(object_name)))
        stacked = [clip_qpos(dataset, int(index), layout) for index in group]
        lengths = {clip.shape[0] for clip in stacked}
        if len(lengths) != 1:
            raise ValueError(f"{object_name} clips have different lengths: {sorted(lengths)}.")
        renderer = ParallelClipRenderer(model, layout, width=width, height=height)
        images = renderer.render(np.stack(stacked, axis=0))
        for local, index in enumerate(group):
            path = out_dir / f"{object_name}_{int(index):04d}.gif"
            write_gif(images[local], path, dataset.fps)
            written.append(path)
    return written


def _sample_indices(num_clips: int, count: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError(f"num_clips must be positive, got {count}.")
    take = min(count, num_clips)
    if take < count:
        print(f"only {num_clips} clips are available; rendering {take}.")
    return np.random.default_rng(seed).choice(num_clips, size=take, replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render dataset clips with mujoco_warp.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--objects", nargs="+", default=["smallbox"])
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/warp_clips"),
    )
    args = parser.parse_args()

    wp.init()
    dataset = WeavePreInteractionDataset(objects=args.objects, root=args.root, split=args.split)
    indices = _sample_indices(dataset.num_clips, args.num_clips, args.seed)
    paths = write_dataset_gifs(
        dataset,
        indices,
        args.out,
        width=args.width,
        height=args.height,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
