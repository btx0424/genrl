"""Viser playback of Weave clips: robot URDF plus the interaction object."""

from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

from genrl.data.weave import (
    DEFAULT_ROOT,
    BODY_NAMES,
    WeaveDataset,
    WeavePreInteractionDataset,
)

# Official Weave URDF. Meshes are the matching Unitree G1 + Inspire STLs.
DEFAULT_URDF = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "unitree_g1"
    / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
)


def load_robot(urdf_path: Path, load_meshes: bool = True) -> yourdfpy.URDF:
    """Load the G1 URDF with every recorded joint independently actuated.

    The Inspire URDF marks the intermediate and distal finger joints as mimics.
    Weave stores those joints as their own angles, and they do not follow the
    mimic equation, so the tags are dropped before parsing.
    """
    text = re.sub(r"\s*<mimic [^>]*/>", "", urdf_path.read_text())
    return yourdfpy.URDF.load(
        io.StringIO(text),
        build_scene_graph=True,
        load_meshes=load_meshes,
        filename_handler=lambda fname: yourdfpy.filename_handler_magic(
            fname, dir=urdf_path.parent
        ),
    )


def _load_object_mesh(root: Path, name: str) -> trimesh.Trimesh:
    path = root / "objects" / name / f"{name}.obj"
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a triangle mesh.")
    return mesh


def _fk_matches_dataset(
    urdf: yourdfpy.URDF,
    dataset: WeaveDataset,
    permutation: np.ndarray,
    frame: int,
) -> float:
    """Max distance, in meters, between URDF link origins and recorded body poses.

    The floating base is the recorded pelvis pose. Joint angles are reordered
    from dataset order into the URDF's actuated order.
    """
    urdf.update_cfg(dataset.joint_pos[frame, permutation])
    pelvis_pos = dataset.body_pos_w[frame, 0]
    pelvis_rot = viser.transforms.SO3(dataset.body_quat_w[frame, 0]).as_matrix()
    error = 0.0
    for body_index, name in enumerate(BODY_NAMES):
        local = urdf.get_transform(name, "pelvis")[:3, 3]
        predicted = pelvis_rot @ local + pelvis_pos
        error = max(error, float(np.linalg.norm(predicted - dataset.body_pos_w[frame, body_index])))
    return error


def _clip_caption(dataset: WeaveDataset, clip: int, length: int) -> str:
    seconds = length / dataset.fps
    text = (
        f"**{dataset.motion_names[clip]}**  \n"
        f"{dataset.object_names[clip]} · clip {clip} · {length} frames · {seconds:.2f} s"
    )
    if isinstance(dataset, WeavePreInteractionDataset):
        text += (
            f"  \nends at interaction: contact and object speed "
            f"{dataset.onset_speed[clip]:.2f} m/s "
            f"(threshold {dataset.speed_threshold:.2f})"
        )
    return text


def view(
    dataset: WeaveDataset,
    urdf_path: str | Path = DEFAULT_URDF,
    port: int = 8080,
) -> None:
    """Open a Viser server and play clips until the process is interrupted."""
    urdf_path = Path(urdf_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"Robot URDF not found at {urdf_path}. "
            "Expected the Weave G1 + Inspire asset."
        )

    urdf = load_robot(urdf_path)
    actuated = tuple(urdf.actuated_joint_names)
    permutation = dataset.joint_permutation(actuated)
    dataset.check_body_names(urdf.link_map)
    if not np.array_equal(permutation, np.arange(len(permutation))):
        print("URDF joint order differs from the dataset; remapping by name.", flush=True)

    fk_error = _fk_matches_dataset(urdf, dataset, permutation, frame=int(dataset.clip_start_idx[0]))
    print(f"URDF vs recorded body origins: max error {fk_error * 1000:.2f} mm", flush=True)
    if fk_error > 0.02:
        raise RuntimeError(
            f"URDF forward kinematics disagree with body_pos_w by {fk_error:.3f} m. "
            "The asset joint frames do not match this dataset."
        )

    server = viser.ViserServer(port=port, label="Weave")
    server.scene.set_up_direction("+z")
    pelvis0 = np.asarray(dataset.body_pos_w[int(dataset.clip_start_idx[0]), 0], dtype=np.float64)
    server.initial_camera.position = pelvis0 + np.array([2.2, -2.2, 1.4])
    server.initial_camera.look_at = pelvis0
    server.scene.add_grid("/ground", plane="xy", cell_size=0.5, section_size=1.0)

    robot_base = server.scene.add_frame("/robot", show_axes=False)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    object_root = server.scene.add_frame("/object", show_axes=False)
    object_meshes: dict[str, viser.SceneNodeHandle] = {}
    for name in dataset.objects:
        object_meshes[name] = server.scene.add_mesh_trimesh(
            f"/object/{name}",
            _load_object_mesh(dataset.root, name),
            visible=False,
        )
    contact = server.scene.add_point_cloud(
        "/contact",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 120, 40),
        point_size=0.03,
        point_shape="circle",
        visible=False,
    )

    with server.gui.add_folder("Clip"):
        object_filter = server.gui.add_dropdown(
            "Object",
            options=("all", *dataset.objects),
            initial_value="all",
        )
        clip_slider = server.gui.add_slider(
            "Clip",
            min=0,
            max=max(dataset.num_clips - 1, 0),
            step=1,
            initial_value=0,
        )
        info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Play", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        speed = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=4.0,
            step=0.1,
            initial_value=1.0,
        )
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(int(dataset.motion_lengths[0]) - 1, 0),
            step=1,
            initial_value=0,
        )
        show_contact = server.gui.add_checkbox("Contact", initial_value=True)

    def clips_for_object() -> np.ndarray:
        if object_filter.value == "all":
            return np.arange(dataset.num_clips)
        return np.flatnonzero(dataset.object_names == object_filter.value)

    state = {"clip_list": clips_for_object(), "shown_clip": -1}

    def current_clip() -> int:
        clips = state["clip_list"]
        local = int(np.clip(clip_slider.value, 0, len(clips) - 1))
        return int(clips[local])

    def show_frame(clip: int, local_frame: int) -> None:
        frames = dataset.clip_slice(clip)
        frame = int(frames.start) + local_frame
        robot_base.position = dataset.body_pos_w[frame, 0]
        robot_base.wxyz = dataset.body_quat_w[frame, 0]
        viser_urdf.update_cfg(dataset.joint_pos[frame, permutation])
        object_root.position = dataset.object_pos_w[frame]
        object_root.wxyz = dataset.object_quat_w[frame]
        object_name = str(dataset.object_names[clip])
        for name, handle in object_meshes.items():
            handle.visible = name == object_name
        if show_contact.value:
            touching = dataset.contact_label[frame] > 0.5
            if np.any(touching):
                contact.points = dataset.body_pos_w[frame, touching]
                contact.visible = True
            else:
                contact.visible = False
        else:
            contact.visible = False

    def refresh_clip_bounds() -> None:
        clips = clips_for_object()
        state["clip_list"] = clips
        clip_slider.value = 0
        clip_slider.max = max(len(clips) - 1, 0)
        clip = current_clip()
        length = int(dataset.motion_lengths[clip])
        frame_slider.value = 0
        frame_slider.max = max(length - 1, 0)
        state["shown_clip"] = clip
        info.content = _clip_caption(dataset, clip, length)

    @object_filter.on_update
    def _(_event: viser.GuiEvent) -> None:
        refresh_clip_bounds()

    @clip_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        clip = current_clip()
        if clip != state["shown_clip"]:
            length = int(dataset.motion_lengths[clip])
            frame_slider.value = 0
            frame_slider.max = max(length - 1, 0)
            state["shown_clip"] = clip
            info.content = _clip_caption(dataset, clip, length)

    refresh_clip_bounds()
    show_frame(current_clip(), 0)
    print(f"Viewer at http://localhost:{port}", flush=True)

    accumulator = 0.0
    last = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            clip = current_clip()
            length = int(dataset.motion_lengths[clip])
            if playing.value and length > 1:
                accumulator += dt * float(speed.value) * dataset.fps
                steps = int(accumulator)
                if steps:
                    accumulator -= steps
                    nxt = int(frame_slider.value) + steps
                    last_frame = length - 1
                    if nxt > last_frame:
                        if loop.value:
                            nxt = nxt % length
                        else:
                            nxt = last_frame
                            playing.value = False
                    frame_slider.value = nxt
            else:
                accumulator = 0.0
            local_frame = int(np.clip(frame_slider.value, 0, length - 1))
            show_frame(clip, local_frame)
            time.sleep(0.02)
    except KeyboardInterrupt:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Weave rollout clips in Viser.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["smalltable", "smallbox"],
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--pre-interaction",
        action="store_true",
        help="Keep only the short approach before the object starts moving in contact.",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds kept before onset.")
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=0.2,
        help="Object linear speed, in m/s, that marks interaction onset.",
    )
    args = parser.parse_args()
    if args.pre_interaction:
        dataset = WeavePreInteractionDataset(
            args.objects,
            root=args.root,
            split=args.split,
            duration_s=args.duration,
            speed_threshold=args.speed_threshold,
        )
    else:
        dataset = WeaveDataset(args.objects, root=args.root, split=args.split)
    view(dataset, urdf_path=args.urdf, port=args.port)


if __name__ == "__main__":
    main()
