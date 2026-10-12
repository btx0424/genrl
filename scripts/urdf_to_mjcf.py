"""Convert the G1 + Inspire hand URDF into an MJCF with a floating base.

URDF <mimic> tags become MuJoCo joint equality constraints. They only act during
dynamics; kinematic replay (mj_kinematics / mj_forward with qpos set) ignores them.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import mujoco

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1"
DEFAULT_URDF = ASSET_DIR / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
STANDING_HEIGHT = 0.793


def convert(urdf_path: Path, out_path: Path) -> mujoco.MjModel:
    src = urdf_path.read_text()
    # Mesh filenames already carry the "meshes/" prefix; keep every link as its own body.
    src = re.sub(
        r"<compiler[^>]*/>",
        '<compiler meshdir="." discardvisual="false" fusestatic="false" balanceinertia="true"/>',
        src,
    )
    tmp = urdf_path.with_name(f".{urdf_path.stem}.tmp.urdf")
    tmp.write_text(src)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        os.remove(tmp)

    mujoco.mj_saveLastXML(str(out_path), model)
    xml = out_path.read_text()
    xml = xml.replace(
        '<body name="pelvis">',
        f'<body name="pelvis" pos="0 0 {STANDING_HEIGHT}">\n      <freejoint name="floating_base_joint"/>',
        1,
    )
    out_path.write_text(xml)
    return mujoco.MjModel.from_xml_path(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.urdf.with_suffix(".xml")
    model = convert(args.urdf, out)
    print(f"wrote {out}: nbody={model.nbody} njnt={model.njnt} nq={model.nq} neq={model.neq}")


if __name__ == "__main__":
    main()
