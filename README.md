# genrl

Generative models for fixed-length pre-interaction clips from the [Weave](https://huggingface.co/datasets/appolyn/Weave) rollout dataset. A Unitree G1 with Inspire hands approaches an object. Each clip ends when a non-ground body is in contact and the object's linear speed crosses 0.2 m/s.

The active model is flow matching. A diffusion baseline is in the same repo.

## Setup

```bash
uv sync
```

Torch is installed from the CUDA 12.8 index declared in `pyproject.toml`. The G1 URDF lives in `assets/unitree_g1/`, along with a matching MJCF (`.xml`, regenerate with `uv run python scripts/urdf_to_mjcf.py`). Meshes are a local symlink and are not part of the git tree:

```bash
ln -s /path/to/unitree_g1/meshes assets/unitree_g1/meshes
```

The Weave rollout files are expected at the `root` in `configs/data/weave.yaml` (currently `/mnt/workspace/btx0424/Weave`). SwanLab reads the API key from `swanlab.key` in the training config. That file is outside this repo.

## Train

```bash
uv run python scripts/train_flow_matching.py
uv run python scripts/train_diffusion.py
uv run python scripts/train_awd.py \
  diffusion=outputs/diffusion-<backbone>/<timestamp>/checkpoints/last.pt
uv run python scripts/train_qgf.py \
  flow=outputs/flow_matching/<timestamp>/checkpoints/last.pt
```

Hydra overrides change a run without editing the yaml. Resume from a checkpoint written by the same config:

```bash
uv run python scripts/train_flow_matching.py \
  checkpoint.resume=outputs/flow_matching/<timestamp>/checkpoints/last.pt
```

Do not resume a checkpoint from an older representation. Feature scaling, the noise schedule, and the rotation layout have changed.

## Look at samples

Each run writes to `outputs/<model>/<timestamp>/`.

- `samples/step_<step>/*.gif` are generated clips. Flow-matching clips are trained at 10 Hz and played back at 50 Hz. The network is conditioned on a frozen PointNet++ vector of the object mesh plus its extents. mujoco_warp draws the G1 meshes and that object mesh.
- `checkpoints/last.pt` is the latest full state (weights, EMA, optimizer, normalization, step). `checkpoints/step_*.pt` keeps the last three periodic copies.
- The same GIFs are uploaded to the SwanLab run under the `samples` key. The project page is https://swanlab.cn/@btx0424/genrl.

To inspect the real dataset instead of a generated clip:

```bash
uv run python -m genrl.data.view --split test --objects smallbox
```

## Layout

| Path | Role |
|---|---|
| `configs/` | Hydra configs. `data/weave.yaml` selects the rollout split and objects. |
| `scripts/train_flow_matching.py` | Flow matching entry point. |
| `scripts/train_diffusion.py` | Diffusion entry point. |
| `scripts/train_awd.py` | Advantage-weighted fine-tune of a diffusion checkpoint. |
| `scripts/train_qgf.py` | QGF critic on a frozen flow-matching checkpoint. Sampling is guided. |
| `scripts/eval_qgf.py` | Sweep guidance weights for one QGF checkpoint. |
| `scripts/eval_generator_coverage.py` | Real vs generated windows for every training motion, one clip per checkpoint. |
| `scripts/eval_generator_generalization.py` | The same windows from novel first frames, scored by distance to the training data. |
| `src/genrl/data/` | Dataset loader and the fixed-length feature window. |
| `src/genrl/model/` | 1D U-Net, flow matching, diffusion. |
| `src/genrl/viz/` | GIF renderer. |
| `src/genrl/train_common.py` | Training loop, checkpoints, SwanLab. |
