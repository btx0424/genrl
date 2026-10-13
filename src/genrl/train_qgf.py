"""Train a QGF critic on top of a frozen flow-matching checkpoint.

The flow is loaded from ``flow`` and not updated. Q is fit to dataset clips and
to clips sampled from that flow, both labeled with the object-still reward.
Guidance at sample time adds the critic gradient to the frozen velocity.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import swanlab
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from genrl.data.segments import (
    FEATURE_DIM,
    clip_penalties,
    combined_reward,
    object_motion_before_touch,
    penalty_scales,
    position_velocity_consistency,
    rot6d_gram_error,
)
from genrl.model.flow_matching import FlowMatching
from genrl.model.qgf import IQL, qgf_sample
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


def _scale_reward(
    reward: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (reward - mean) / std


def _load_flow(path: str | None, device: torch.device) -> tuple[FlowMatching, dict, DictConfig]:
    """Load a finished flow-matching run and freeze it."""
    if path is None or str(path).lower() in ("", "null", "none"):
        raise ValueError(
            "QGF does not train the flow. Pass a flow-matching checkpoint, "
            "for example flow=outputs/flow_matching/<run>/checkpoints/last.pt."
        )
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Flow checkpoint not found: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema" not in loaded or "config" not in loaded:
        raise ValueError(
            f"{checkpoint_path} is not a flow-matching checkpoint. "
            "Train one with scripts/train_flow_matching.py and pass its last.pt."
        )
    saved = OmegaConf.create(loaded["config"])
    flow = FlowMatching(
        build_backbone(saved),
        t_eps=float(saved.t_eps),
        guidance=_guidance(saved),
    )
    flow.load_state_dict(loaded["ema"])
    flow.to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    print(
        f"frozen flow from {checkpoint_path} (step {int(loaded['step'])})",
        flush=True,
    )
    return flow, loaded, saved


def _check_flow_data(saved: DictConfig, cfg: DictConfig, raw: torch.Tensor, fps: int) -> None:
    """The frozen flow only matches clips built the same way as its training run."""
    mismatches = []
    saved_data = saved.data
    for key in ("sample_hz", "relative_rotation", "duration"):
        trained = saved_data[key]
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
        detail = "; ".join(mismatches)
        raise ValueError(f"QGF data does not match the flow checkpoint. {detail}.")


def _save(
    directory: Path,
    step: int,
    flow: FlowMatching,
    critic: IQL,
    critic_optimizer: torch.optim.Optimizer,
    mean: torch.Tensor,
    std: torch.Tensor,
    reward_mean: torch.Tensor,
    reward_std: torch.Tensor,
    scales: dict,
    fps: int,
    config: dict,
    *,
    named: bool,
    keep_last: int,
) -> None:
    payload = {
        "step": step,
        "flow": flow.state_dict(),
        "critic": critic.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "reward_mean": reward_mean.detach().cpu(),
        "reward_std": reward_std.detach().cpu(),
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
    flow, flow_checkpoint, flow_config = _load_flow(cfg.flow, device)
    raw, fps, rotation_anchors, skip_tail, object_names = load_raw_segments(cfg)
    _check_flow_data(flow_config, cfg, raw, fps)
    config = _public_config(cfg)
    config["fps"] = fps
    config["feature_dim"] = FEATURE_DIM
    config["segment_length"] = int(raw.shape[1])
    config["flow"] = str(cfg.flow)

    mean = flow_checkpoint["mean"]
    std = flow_checkpoint["std"]
    features = (raw - mean) / std
    dataset_penalties = clip_penalties(raw, fps, skip_tail)
    resume = cfg.checkpoint.resume
    if resume:
        loaded = torch.load(resume, map_location="cpu", weights_only=False)
        if "critic" not in loaded or "penalty_scales" not in loaded:
            raise ValueError(
                f"{resume} has no critic trained on the current penalties. "
                "Resume from a QGF checkpoint, and pass the flow as flow=..."
            )
        scales = {name: torch.tensor(value) for name, value in loaded["penalty_scales"].items()}
        reward_mean = loaded["reward_mean"]
        reward_std = loaded["reward_std"]
        print(f"resuming critic from {resume}", flush=True)
    else:
        loaded = None
        scales = penalty_scales(dataset_penalties)
        reward = combined_reward(dataset_penalties, scales)
        reward_mean = reward.mean()
        reward_std = reward.std().clamp_min(1e-4)
    reward = _scale_reward(combined_reward(dataset_penalties, scales), reward_mean, reward_std)

    iql_cfg = cfg.iql
    critic = IQL(
        FEATURE_DIM,
        hidden=int(iql_cfg.hidden),
        ensemble=int(iql_cfg.ensemble),
        expectile=float(iql_cfg.expectile),
        discount=float(iql_cfg.discount),
    ).to(device)
    critic_count = sum(
        parameter.numel() for parameter in critic.parameters() if parameter.requires_grad
    )
    print(f"critic {critic_count / 1e6:.2f}M parameters", flush=True)
    mean = mean.to(device)
    std = std.to(device)
    reward_mean = reward_mean.to(device)
    reward_std = reward_std.to(device)
    scales = {name: value.to(device) for name, value in scales.items()}
    critic_optimizer = torch.optim.AdamW(
        _adamw_param_groups(critic, float(iql_cfg.weight_decay)),
        lr=float(iql_cfg.lr),
    )
    start = 0
    if loaded is not None:
        critic.load_state_dict(loaded["critic"])
        critic_optimizer.load_state_dict(loaded["critic_optimizer"])
        start = int(loaded["step"])
        print(f"resumed at step {start}", flush=True)
    target_decay = float(iql_cfg.target_decay)
    guidance = float(cfg.guidance)
    cond_steps = int(flow_config.cond_steps)

    shapes = object_shape_table(Path(cfg.data.root), object_names)
    loader = DataLoader(
        TensorDataset(features, shapes, reward),
        batch_size=min(int(cfg.batch_size), features.shape[0]),
        shuffle=True,
        drop_last=features.shape[0] >= int(cfg.batch_size),
    )
    real_consistency = position_velocity_consistency(raw, fps)
    real_rot6d = rot6d_gram_error(raw)
    real_motion = object_motion_before_touch(raw, skip_tail=skip_tail)
    renderer = WarpSampleRenderer(cfg.data.root, cfg.robot.urdf)
    _start_swanlab(cfg, config)
    print(
        f"training {cfg.swanlab.name} on {features.shape[0]} clips "
        f"of length {features.shape[1]} ({device})",
        flush=True,
    )
    print(
        "real object speed before touch "
        f"{real_motion['speed_before_touch']:.4f} m/s, "
        f"at end {real_motion['speed_at_end']:.4f} m/s",
        flush=True,
    )
    print(f"QGF guidance weight {guidance}", flush=True)
    generated_steps = int(iql_cfg.generated_steps)
    generated_weight = float(iql_cfg.generated_weight)
    if generated_weight > 0 and generated_steps > 0:
        print(
            f"Q is also fit to flow samples ({generated_steps} Euler steps, "
            f"weight {generated_weight})",
            flush=True,
        )

    step = start
    running = {
        "q": 0.0,
        "q_gen": 0.0,
        "v": 0.0,
        "grad": 0.0,
        "reward": 0.0,
        "gen_reward": 0.0,
    }
    seen = 0
    keep_last = int(cfg.checkpoint.keep_last)
    try:
        while step < int(cfg.steps):
            for batch, batch_geom, batch_reward in loader:
                step += 1
                batch = batch.to(device)
                batch_geom = batch_geom.to(device)
                batch_reward = batch_reward.to(device)
                q_loss, v_loss = critic.losses(batch, batch, batch_reward)
                q_gen = torch.zeros((), device=device)
                gen_reward_value = torch.zeros((), device=device)
                if generated_weight > 0 and generated_steps > 0:
                    prefix = batch[:, :cond_steps] if cond_steps else None
                    generated = flow.sample(
                        tuple(batch.shape),
                        steps=generated_steps,
                        prefix=prefix,
                        geom=batch_geom,
                    )
                    physical_reward = combined_reward(
                        clip_penalties(generated * std + mean, fps, skip_tail),
                        scales,
                    )
                    gen_reward_value = physical_reward.mean()
                    q_gen = critic.reward_loss(
                        generated,
                        generated,
                        _scale_reward(physical_reward, reward_mean, reward_std),
                    )
                critic_optimizer.zero_grad(set_to_none=True)
                (q_loss + v_loss + generated_weight * q_gen).backward()
                trainable = [parameter for parameter in critic.parameters() if parameter.requires_grad]
                if cfg.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                else:
                    grad_norm = _global_grad_norm(critic)
                critic_optimizer.step()
                _update_ema(critic.q_target, critic.q, target_decay)
                _update_ema(critic.v_target, critic.v, target_decay)

                running["q"] += float(q_loss.detach())
                running["q_gen"] += float(q_gen.detach())
                running["v"] += float(v_loss.detach())
                running["grad"] += float(grad_norm)
                running["reward"] += float(batch_reward.detach().mean())
                running["gen_reward"] += float(gen_reward_value.detach())
                seen += 1
                if step % int(cfg.log_every) == 0 or step == int(cfg.steps):
                    linear_norm, norm_norm = _weight_norms(critic)
                    with torch.no_grad():
                        q_mean = float(critic.q_min(batch, batch).mean())
                    swanlab.log(
                        {
                            "train/q_loss": running["q"] / seen,
                            "train/q_loss_generated": running["q_gen"] / seen,
                            "train/v_loss": running["v"] / seen,
                            "train/generated_reward": running["gen_reward"] / seen,
                            "train/grad_norm": running["grad"] / seen,
                            "train/reward": running["reward"] / seen,
                            "train/q_mean": q_mean,
                            "train/weight_norm_linear": linear_norm,
                            "train/weight_norm_norm": norm_norm,
                        },
                        step=step,
                    )
                    print(
                        f"step {step}  q {running['q'] / seen:.5f}  "
                        f"q_gen {running['q_gen'] / seen:.5f}  "
                        f"v {running['v'] / seen:.5f}  Q {q_mean:.3f}",
                        flush=True,
                    )
                    running = {key: 0.0 for key in running}
                    seen = 0
                if step % int(cfg.sample.every) == 0 or step == int(cfg.steps):
                    prefix, anchor, index = _sample_condition(
                        features,
                        rotation_anchors,
                        int(cfg.sample.batch),
                        cond_steps,
                        device,
                    )
                    length = int(features.shape[1])
                    sample_steps = int(cfg.sample.steps)

                    geom = shapes[index].to(device)

                    def generate(
                        prefix=prefix,
                        length=length,
                        sample_steps=sample_steps,
                        geom=geom,
                    ):
                        return qgf_sample(
                            flow,
                            critic.q_min,
                            (
                                prefix.shape[0] if prefix is not None else int(cfg.sample.batch),
                                length,
                                FEATURE_DIM,
                            ),
                            steps=sample_steps,
                            guidance=guidance,
                            prefix=prefix,
                            geom=geom,
                        )

                    _log_samples(
                        flow,
                        std,
                        mean,
                        fps,
                        step,
                        length,
                        int(cfg.sample.batch),
                        sample_steps,
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
                        (prefix, anchor, index),
                        generate=generate,
                    )
                if step % int(cfg.checkpoint.every) == 0 and step != int(cfg.steps):
                    _save(
                        checkpoint_dir,
                        step,
                        flow,
                        critic,
                        critic_optimizer,
                        mean,
                        std,
                        reward_mean,
                        reward_std,
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
            flow,
            critic,
            critic_optimizer,
            mean,
            std,
            reward_mean,
            reward_std,
            scales,
            fps,
            config,
            named=True,
            keep_last=keep_last,
        )
        print(f"view GIFs in {sample_dir}", flush=True)
        swanlab.finish()
