"""Fine-tune a diffusion model with advantage-weighted denoising.

Each step draws clips from the current EMA model and uses those clips as the
denoising targets. The noise loss is multiplied by ``exp(advantage / beta)``,
where the advantage is the clip's reward minus the mean of that sampled batch,
in units of the batch standard deviation. A frozen copy of the pretrained
checkpoint is the behavior-cloning target.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import swanlab
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from genrl.data.segments import (
    FEATURE_DIM,
    clip_penalties,
    combined_reward,
    object_motion_before_touch,
    penalty_scales,
    position_velocity_consistency,
    rot6d_gram_error,
)
from genrl.model.diffusion import GaussianDiffusion, _noise_error
from genrl.train_common import (
    _adamw_param_groups,
    _device,
    _global_grad_norm,
    _log_samples,
    _public_config,
    _sample_condition,
    _start_swanlab,
    _update_ema,
    _weight_norms,
    _guidance,
    build_backbone,
    load_raw_segments,
    set_seed,
)
from genrl.data.geometry import object_shape_table
from genrl.viz.warp_clips import WarpSampleRenderer


def advantage_weights(reward: torch.Tensor, beta: float, weight_max: float) -> torch.Tensor:
    """Normalized ``exp(advantage / beta)`` using this batch's own mean and spread.

    Mean weight is one. A batch whose rewards barely differ gets uniform weights.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}.")
    advantage = reward - reward.mean()
    scale = advantage.square().mean().sqrt()
    if float(scale) < 1e-4:
        return torch.ones_like(reward)
    weights = (advantage / scale / beta).exp()
    if weight_max > 0:
        weights = weights.clamp(max=weight_max)
    return weights / weights.mean().clamp_min(1e-6)


def _load_diffusion(
    path: str | None, device: torch.device
) -> tuple[GaussianDiffusion, dict, DictConfig]:
    if path is None or str(path).lower() in ("", "null", "none"):
        raise ValueError(
            "Advantage-weighted diffusion fine-tunes a trained model. Pass "
            "diffusion=outputs/diffusion-<backbone>/<run>/checkpoints/last.pt."
        )
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if loaded.get("config", {}).get("name") != "diffusion" or "ema" not in loaded:
        raise ValueError(
            f"{checkpoint_path} is not a diffusion checkpoint. "
            "Train one with scripts/train_diffusion.py and pass its last.pt."
        )
    saved = OmegaConf.create(loaded["config"])
    model = GaussianDiffusion(
        build_backbone(saved),
        timesteps=int(saved.timesteps),
        guidance=_guidance(saved),
    )
    model.load_state_dict(loaded["ema"])
    model.to(device)
    print(
        f"diffusion init from {checkpoint_path} (step {int(loaded['step'])})",
        flush=True,
    )
    return model, loaded, saved


def _check_data(saved: DictConfig, cfg: DictConfig, raw: torch.Tensor, fps: int) -> None:
    mismatches = []
    for key in ("sample_hz", "relative_rotation", "duration"):
        trained = saved.data[key]
        current = cfg.data[key]
        if trained != current:
            mismatches.append(f"{key}: checkpoint has {trained}, config has {current}")
    if int(saved.feature_dim) != int(raw.shape[-1]):
        mismatches.append(
            f"feature dim: checkpoint has {int(saved.feature_dim)}, data has {int(raw.shape[-1])}"
        )
    if int(saved.segment_length) != int(raw.shape[1]):
        mismatches.append(
            f"segment length: checkpoint has {int(saved.segment_length)}, data has {int(raw.shape[1])}"
        )
    if int(saved.fps) != int(fps):
        mismatches.append(f"fps: checkpoint has {int(saved.fps)}, data has {int(fps)}")
    if mismatches:
        raise ValueError(
            "Advantage-weighted data does not match the diffusion checkpoint. "
            + "; ".join(mismatches)
            + "."
        )


def _save(
    directory: Path,
    step: int,
    module: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    ema: GaussianDiffusion,
    mean: torch.Tensor,
    std: torch.Tensor,
    scales: dict[str, torch.Tensor],
    fps: int,
    config: dict,
    *,
    named: bool,
    keep_last: int,
) -> None:
    payload = {
        "step": step,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "penalty_scales": {name: float(value) for name, value in scales.items()},
        "fps": fps,
        "config": config,
    }
    directory.mkdir(parents=True, exist_ok=True)
    last = directory / "last.pt"
    temporary = directory / "last.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(last)
    if named:
        path = directory / f"step_{step:07d}.pt"
        shutil.copy2(last, path)
        history = sorted(directory.glob("step_*.pt"))
        for old in history[:-keep_last]:
            old.unlink()
    print(f"saved {last}", flush=True)


def train(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = _device(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    module, loaded_init, saved = _load_diffusion(cfg.diffusion, device)
    raw, fps, rotation_anchors, skip_tail, object_names = load_raw_segments(cfg)
    _check_data(saved, cfg, raw, fps)
    config = _public_config(cfg)
    config["fps"] = fps
    config["feature_dim"] = FEATURE_DIM
    config["segment_length"] = int(raw.shape[1])
    config["diffusion"] = str(cfg.diffusion)

    mean = loaded_init["mean"]
    std = loaded_init["std"]
    features = (raw - mean) / std
    dataset_penalties = clip_penalties(raw, fps, skip_tail)
    resume = cfg.checkpoint.resume
    if resume:
        resumed = torch.load(resume, map_location="cpu", weights_only=False)
        if resumed.get("config", {}).get("name") != "awd" or "penalty_scales" not in resumed:
            raise ValueError(
                f"{resume} is not an advantage-weighted checkpoint. "
                "Pass the pretrained diffusion model as diffusion=..."
            )
        scales = {name: torch.tensor(value) for name, value in resumed["penalty_scales"].items()}
        print(f"resuming fine-tune from {resume}", flush=True)
    else:
        resumed = None
        scales = penalty_scales(dataset_penalties)

    teacher = copy.deepcopy(module).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        _adamw_param_groups(module, float(cfg.weight_decay)),
        lr=float(cfg.lr),
    )
    ema = copy.deepcopy(module).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    start = 0
    if resumed is not None:
        module.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        ema.load_state_dict(resumed["ema"])
        start = int(resumed["step"])
        print(f"resumed at step {start}", flush=True)
    mean = mean.to(device)
    std = std.to(device)
    features = features.to(device)
    shapes = object_shape_table(Path(cfg.data.root), object_names).to(device)
    scales = {name: value.to(device) for name, value in scales.items()}
    beta = float(cfg.beta)
    bc_weight = float(cfg.bc_weight)
    weight_max = float(cfg.weight_max)
    cond_steps = int(saved.cond_steps)
    rollout_steps = int(cfg.rollout_steps)
    ema_decay = float(cfg.ema_decay)
    batch_size = min(int(cfg.batch_size), int(features.shape[0]))
    length = int(features.shape[1])
    real_consistency = position_velocity_consistency(raw, fps)
    real_rot6d = rot6d_gram_error(raw)
    real_motion = object_motion_before_touch(raw, skip_tail=skip_tail)
    renderer = WarpSampleRenderer(cfg.data.root, cfg.robot.urdf)
    _start_swanlab(cfg, config)
    print(
        f"fine-tuning {cfg.swanlab.name} on {features.shape[0]} clips "
        f"of length {features.shape[1]} ({device})",
        flush=True,
    )
    print(
        f"advantage beta {beta}, weight cap {weight_max}, behavior-cloning weight {bc_weight}",
        flush=True,
    )
    print(
        f"denoising targets are {rollout_steps}-step samples from the EMA model",
        flush=True,
    )

    step = start
    running = {"weighted": 0.0, "bc": 0.0, "grad": 0.0, "reward": 0.0, "wmax": 0.0}
    seen = 0
    keep_last = int(cfg.checkpoint.keep_last)
    try:
        while step < int(cfg.steps):
            step += 1
            index = torch.randint(0, features.shape[0], (batch_size,), device=device)
            prefix = features[index, :cond_steps] if cond_steps else None
            geom = shapes[index]
            sampled = ema.sample(
                (batch_size, length, FEATURE_DIM),
                steps=rollout_steps,
                prefix=prefix,
                geom=geom,
            )
            reward = combined_reward(
                clip_penalties(sampled * std + mean, fps, skip_tail),
                scales,
            )
            weights = advantage_weights(reward, beta, weight_max)
            x_t, noise, t_frac = module.corrupt(sampled, cond_steps)
            pred = module.model(x_t, t_frac, geom)
            denoising = _noise_error(pred, noise, cond_steps)
            weighted = (weights.detach() * denoising).mean()
            with torch.no_grad():
                reference = teacher.model(x_t, t_frac, geom)
            behavior = _noise_error(pred, reference, cond_steps).mean()
            loss = weighted + bc_weight * behavior
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            else:
                grad_norm = _global_grad_norm(module)
            optimizer.step()
            _update_ema(ema, module, ema_decay)

            running["weighted"] += float(weighted.detach())
            running["bc"] += float(behavior.detach())
            running["grad"] += float(grad_norm)
            running["reward"] += float(reward.detach().mean())
            running["wmax"] += float(weights.detach().max())
            seen += 1
            if step % int(cfg.log_every) == 0 or step == int(cfg.steps):
                linear_norm, norm_norm = _weight_norms(module)
                swanlab.log(
                    {
                        "train/weighted_loss": running["weighted"] / seen,
                        "train/bc": running["bc"] / seen,
                        "train/grad_norm": running["grad"] / seen,
                        "train/reward": running["reward"] / seen,
                        "train/weight_max": running["wmax"] / seen,
                        "train/weight_norm_linear": linear_norm,
                        "train/weight_norm_norm": norm_norm,
                    },
                    step=step,
                )
                print(
                    f"step {step}  weighted {running['weighted'] / seen:.5f}  "
                    f"bc {running['bc'] / seen:.5f}  "
                    f"reward {running['reward'] / seen:.3f}  "
                    f"wmax {running['wmax'] / seen:.2f}",
                    flush=True,
                )
                running = {key: 0.0 for key in running}
                seen = 0
            if step % int(cfg.sample.every) == 0 or step == int(cfg.steps):
                _log_samples(
                    ema,
                    std,
                    mean,
                    fps,
                    step,
                    int(features.shape[1]),
                    int(cfg.sample.batch),
                    int(cfg.sample.steps),
                    int(cfg.sample.count),
                    int(cfg.sample.playback_hz)
                    if "playback_hz" in cfg.sample and cfg.sample.playback_hz
                    else fps,
                    sample_dir / f"step_{step:07d}",
                    renderer,
                    object_names,
                    shapes,
                    real_consistency,
                    real_rot6d,
                    real_motion,
                    skip_tail,
                    _sample_condition(
                        features,
                        rotation_anchors,
                        int(cfg.sample.batch),
                        cond_steps,
                        device,
                    ),
                )
            if step % int(cfg.checkpoint.every) == 0 and step != int(cfg.steps):
                _save(
                    checkpoint_dir,
                    step,
                    module,
                    optimizer,
                    ema,
                    mean,
                    std,
                    scales,
                    fps,
                    config,
                    named=True,
                    keep_last=keep_last,
                )
            if step >= int(cfg.steps):
                break
    finally:
        _save(
            checkpoint_dir,
            step,
            module,
            optimizer,
            ema,
            mean,
            std,
            scales,
            fps,
            config,
            named=True,
            keep_last=keep_last,
        )
        print(f"view GIFs in {sample_dir}", flush=True)
        swanlab.finish()
