"""Training loop, checkpoints, and sample GIFs for the segment generators."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import swanlab
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from genrl.data.segments import (
    FEATURE_DIM,
    OBJECT_ROT,
    PELVIS_ROT,
    clips_to_features,
    compose_rotation_anchor,
    make_rotations_relative,
    object_motion_before_touch,
    position_velocity_consistency,
    project_rot6d,
    rot6d_gram_error,
)
from genrl.data.weave import WeavePreInteractionDataset
from genrl.model.diffusion import GaussianDiffusion
from genrl.model.dit import MotionDiT
from genrl.model.flow_matching import FlowMatching
from genrl.model.modules import TemporalUNet
from genrl.viz.render import ClipRenderer, expand_time, write_sample_gifs


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_segments(cfg: DictConfig) -> tuple[torch.Tensor, int, torch.Tensor | None, int]:
    """Unnormalized segments, their rate, rotation anchors, and onset frames to skip.

    The last item is 1 when the clip still ends on the interaction onset, and 0
    when stride subsampling dropped that frame. The object-still reward uses it
    so a touched frame is not penalized for moving.
    """
    data = cfg.data
    dataset = WeavePreInteractionDataset(
        list(data.objects),
        root=data.root,
        split=data.split,
        duration_s=data.duration,
        speed_threshold=data.speed_threshold,
    )
    features = torch.from_numpy(clips_to_features(dataset))
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {features.shape[-1]}.")
    sample_hz = data.sample_hz if "sample_hz" in data else None
    native_hz = int(dataset.fps)
    native_length = int(features.shape[1])
    if sample_hz is None:
        stored, fps, anchors = _with_relative_rotations(data, features, native_hz)
        return stored, fps, anchors, 1
    sample_hz = int(sample_hz)
    if native_hz % sample_hz != 0:
        raise ValueError(f"Dataset rate {native_hz} Hz is not divisible by sample_hz {sample_hz}.")
    stride = native_hz // sample_hz
    features = features[:, ::stride].contiguous()
    # The onset is the last native frame. It stays in the clip only on a stride
    # that lands on it. Otherwise every stored frame is still before the touch.
    skip_tail = 1 if (native_length - 1) % stride == 0 else 0
    print(
        f"training at {sample_hz} Hz ({features.shape[1]} frames, stride {stride})",
        flush=True,
    )
    stored, fps, anchors = _with_relative_rotations(data, features, sample_hz)
    return stored, fps, anchors, skip_tail


def _with_relative_rotations(
    data: DictConfig, features: torch.Tensor, fps: int
) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    if "relative_rotation" in data and data.relative_rotation:
        features, anchors = make_rotations_relative(features)
        print("rotations are relative to the pinned first frame", flush=True)
        return features, fps, anchors
    return features, fps, None


def _normalize(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-dimension z-score. 6D rotations stay in their raw columns.

    A rotation column is not a Euclidean coordinate. Scaling it makes a small
    sampling error into an orientation twitch after the columns are
    orthonormalized for drawing.
    """
    mean = features.mean(dim=(0, 1), keepdim=True)
    std = features.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    for rotation in (PELVIS_ROT, OBJECT_ROT):
        mean[..., rotation] = 0
        std[..., rotation] = 1
    return (features - mean) / std, mean, std


def build_backbone(cfg: DictConfig) -> nn.Module:
    """The temporal network, without the flow-matching or diffusion wrapper."""
    architecture = cfg.model
    backbone = str(architecture.name)
    if backbone == "dit":
        network = MotionDiT(
            FEATURE_DIM,
            hidden=int(architecture.hidden),
            depth=int(architecture.depth),
            heads=int(architecture.heads),
            mlp_ratio=float(architecture.mlp_ratio),
        )
    elif backbone == "unet":
        network = TemporalUNet(
            FEATURE_DIM,
            model_dim=int(architecture.model_dim),
            kernel_size=int(architecture.kernel_size),
            upsample=str(architecture.upsample),
        )
    else:
        raise ValueError(f"model.name must be 'unet' or 'dit', got {backbone!r}.")
    count = sum(parameter.numel() for parameter in network.parameters())
    print(f"backbone {backbone}  {count / 1e6:.2f}M parameters", flush=True)
    return network


def build_model(cfg: DictConfig) -> nn.Module:
    network = build_backbone(cfg)
    if cfg.name == "diffusion":
        return GaussianDiffusion(network, timesteps=cfg.timesteps)
    if cfg.name == "flow_matching":
        return FlowMatching(network, t_eps=float(cfg.t_eps))
    raise ValueError(f"Unknown model {cfg.name!r}.")


def _update_ema(ema: nn.Module, module: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema.parameters(), module.parameters(), strict=True):
            ema_param.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), module.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def _public_config(cfg: DictConfig) -> dict:
    container = OmegaConf.to_container(cfg, resolve=True)
    swanlab_cfg = container.get("swanlab")
    if isinstance(swanlab_cfg, dict):
        swanlab_cfg.pop("key", None)
    return container


def _device(cfg: DictConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Checkpointer:
    """Write ``last.pt`` every call and a pruned history of ``step_*.pt`` files.

    Each file contains the weights, optimizer, normalization, step, and config
    needed to resume.
    """

    def __init__(self, directory: Path, every: int, keep_last: int):
        if every < 1:
            raise ValueError(f"checkpoint.every must be positive, got {every}.")
        if keep_last < 1:
            raise ValueError(f"checkpoint.keep_last must be positive, got {keep_last}.")
        self.directory = directory
        self.every = every
        self.keep_last = keep_last
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        fps: int,
        config: dict,
        *,
        named: bool,
    ) -> None:
        payload = {
            "step": step,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "fps": fps,
            "config": config,
        }
        last = self.directory / "last.pt"
        temporary = self.directory / "last.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(last)
        if named:
            path = self.directory / f"step_{step:07d}.pt"
            shutil.copy2(last, path)
            history = sorted(self.directory.glob("step_*.pt"))
            for old in history[: -self.keep_last]:
                old.unlink()
        print(f"saved {last}", flush=True)

    def load(self, path: Path, module: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint


def _global_grad_norm(module: nn.Module) -> torch.Tensor:
    squared = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squared:
        return torch.zeros((), device=next(module.parameters()).device)
    return torch.sqrt(torch.stack(squared).sum())


_LINEAR_TYPES = (
    nn.Linear,
    nn.MultiheadAttention,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)
_NORM_TYPES = (
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _adamw_param_groups(module: nn.Module, weight_decay: float) -> list[dict]:
    """Decay conv and linear weights. Leave normalization affine parameters alone."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for submodule in module.modules():
        if isinstance(submodule, _LINEAR_TYPES):
            bucket = decay
        elif isinstance(submodule, _NORM_TYPES):
            bucket = no_decay
        else:
            continue
        for parameter in submodule.parameters(recurse=False):
            if parameter.requires_grad and id(parameter) not in seen:
                bucket.append(parameter)
                seen.add(id(parameter))
    for parameter in module.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _parameter_norm(parameters) -> float:
    squared = [parameter.detach().square().sum() for parameter in parameters]
    if not squared:
        return 0.0
    return float(torch.sqrt(torch.stack(squared).sum()))


def _weight_norms(module: nn.Module) -> tuple[float, float]:
    """L2 of conv/linear parameters, and L2 of normalization parameters.

    Convolution is grouped with linear maps. Normalization is GroupNorm,
    BatchNorm, LayerNorm, and InstanceNorm, including affine scale and bias.
    """
    linear = []
    norm = []
    for submodule in module.modules():
        if isinstance(submodule, _LINEAR_TYPES):
            linear.extend(submodule.parameters(recurse=False))
        elif isinstance(submodule, _NORM_TYPES):
            norm.extend(submodule.parameters(recurse=False))
    return _parameter_norm(linear), _parameter_norm(norm)


def _sample_condition(
    features: torch.Tensor,
    anchors: torch.Tensor | None,
    batch: int,
    cond_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if cond_steps == 0:
        return None, None
    index = torch.randint(0, features.shape[0], (batch,))
    prefix = features[index, :cond_steps].to(device)
    anchor = None if anchors is None else anchors[index].to(device)
    return prefix, anchor


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def _start_swanlab(cfg: DictConfig, config: dict) -> None:
    key_path = Path(cfg.swanlab.key)
    if not key_path.is_file():
        raise FileNotFoundError(f"SwanLab API key file not found: {key_path}")
    api_key = key_path.read_text().strip()
    if not api_key:
        raise ValueError(f"SwanLab API key file is empty: {key_path}")
    swanlab.login(api_key=api_key)
    swanlab.init(project=cfg.swanlab.project, name=cfg.swanlab.name, config=config)


def train(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = _device(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    raw, fps, rotation_anchors, skip_tail = load_raw_segments(cfg)
    config = _public_config(cfg)
    config["fps"] = fps
    config["feature_dim"] = FEATURE_DIM
    config["segment_length"] = int(raw.shape[1])

    resume = cfg.checkpoint.resume
    if resume:
        loaded = torch.load(resume, map_location="cpu", weights_only=False)
        mean = loaded["mean"]
        std = loaded["std"]
        print(f"resuming normalization and weights from {resume}", flush=True)
    else:
        loaded = None
        _, mean, std = _normalize(raw)
    features = (raw - mean) / std

    module = build_model(cfg).to(device)
    mean = mean.to(device)
    std = std.to(device)
    optimizer = torch.optim.AdamW(
        _adamw_param_groups(module, float(cfg.weight_decay)),
        lr=cfg.lr,
    )
    checkpointer = Checkpointer(checkpoint_dir, int(cfg.checkpoint.every), int(cfg.checkpoint.keep_last))
    ema = copy.deepcopy(module).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    start = 0
    if loaded is not None:
        loaded = checkpointer.load(Path(resume), module, optimizer, device)
        if "ema" in loaded:
            ema.load_state_dict(loaded["ema"])
        else:
            ema.load_state_dict(module.state_dict())
        start = int(loaded["step"])
        print(f"resumed at step {start}", flush=True)
    ema_decay = float(cfg.ema_decay)

    loader = DataLoader(
        TensorDataset(features),
        batch_size=min(int(cfg.batch_size), features.shape[0]),
        shuffle=True,
        drop_last=features.shape[0] >= int(cfg.batch_size),
    )
    real_consistency = position_velocity_consistency(raw, fps)
    real_rot6d = rot6d_gram_error(raw)
    real_motion = object_motion_before_touch(raw, skip_tail=skip_tail)
    cond_steps = int(cfg.cond_steps) if "cond_steps" in cfg else 0
    renderer = ClipRenderer(cfg.robot.urdf)
    _start_swanlab(cfg, config)
    print(
        f"training {cfg.swanlab.name} on {features.shape[0]} clips "
        f"of length {features.shape[1]} ({device})",
        flush=True,
    )
    print(f"checkpoints: {checkpoint_dir}", flush=True)
    print(f"samples: {sample_dir}", flush=True)
    print(
        "real position-velocity gap: "
        + ", ".join(f"{name} {value:.4f}" for name, value in real_consistency.items()),
        flush=True,
    )
    print(
        "real 6D rotation error: "
        + ", ".join(f"{name} {value:.4f}" for name, value in real_rot6d.items()),
        flush=True,
    )
    if cond_steps:
        print(f"inpainting the first {cond_steps} frame(s)", flush=True)
    print(
        "real object speed before touch "
        f"{real_motion['speed_before_touch']:.4f} m/s, "
        f"at end {real_motion['speed_at_end']:.4f} m/s",
        flush=True,
    )

    step = start
    running = 0.0
    running_grad = 0.0
    seen = 0
    try:
        while step < int(cfg.steps):
            for (batch,) in loader:
                step += 1
                batch = batch.to(device)
                if cond_steps:
                    loss = module.loss(batch, cond_steps=cond_steps)
                else:
                    loss = module.loss(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        module.parameters(), cfg.grad_clip
                    )
                else:
                    grad_norm = _global_grad_norm(module)
                optimizer.step()
                _update_ema(ema, module, ema_decay)
                running += float(loss.detach())
                running_grad += float(grad_norm)
                seen += 1
                if step % int(cfg.log_every) == 0 or step == int(cfg.steps):
                    avg = running / seen
                    avg_grad = running_grad / seen
                    linear_norm, norm_norm = _weight_norms(module)
                    swanlab.log(
                        {
                            "train/loss": avg,
                            "train/grad_norm": avg_grad,
                            "train/weight_norm_linear": linear_norm,
                            "train/weight_norm_norm": norm_norm,
                        },
                        step=step,
                    )
                    print(
                        f"step {step}  loss {avg:.5f}  grad {avg_grad:.4f}  "
                        f"linear {linear_norm:.4f}  norm {norm_norm:.4f}",
                        flush=True,
                    )
                    running = 0.0
                    running_grad = 0.0
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
                        int(cfg.sample.playback_hz) if "playback_hz" in cfg.sample and cfg.sample.playback_hz else fps,
                        sample_dir / f"step_{step:07d}",
                        renderer,
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
                    checkpointer.save(
                        step, module, optimizer, ema, mean, std, fps, config, named=True
                    )
                if step >= int(cfg.steps):
                    break
    finally:
        checkpointer.save(step, module, optimizer, ema, mean, std, fps, config, named=True)
        print(f"view GIFs in {sample_dir}", flush=True)
        swanlab.finish()


def _log_samples(
    module: nn.Module,
    std: torch.Tensor,
    mean: torch.Tensor,
    fps: int,
    step: int,
    length: int,
    batch: int,
    sample_steps: int,
    count: int,
    playback_hz: int,
    directory: Path,
    renderer: ClipRenderer,
    real_consistency: dict[str, float],
    real_rot6d: dict[str, float],
    real_motion: dict[str, float],
    skip_tail: int,
    condition: tuple[torch.Tensor | None, torch.Tensor | None],
    generate=None,
) -> None:
    """Log one batch of samples. ``generate`` returns normalized clips when set.

    The default draws from ``module.sample``. A guided sampler can replace that
    without changing the metrics or the GIFs.
    """
    prefix, anchor = condition
    training = module.training
    module.eval()
    with torch.no_grad():
        if generate is None:
            if prefix is None:
                generated = module.sample((batch, length, FEATURE_DIM), steps=sample_steps)
            else:
                generated = module.sample(
                    (batch, length, FEATURE_DIM),
                    steps=sample_steps,
                    prefix=prefix,
                )
        else:
            generated = generate()
        generated = generated * std + mean
    module.train(training)
    generated_motion = object_motion_before_touch(generated, skip_tail=skip_tail)
    skip = 0 if prefix is None else int(prefix.shape[1])
    generated_consistency = position_velocity_consistency(generated, fps)
    generated_rot6d = rot6d_gram_error(generated, skip_frames=skip)
    shown = project_rot6d(generated[:count])
    if anchor is not None:
        shown = compose_rotation_anchor(shown, anchor[:count])
    clips = expand_time(shown.detach().cpu().numpy(), fps, playback_hz)
    paths = write_sample_gifs(clips, directory, playback_hz, renderer=renderer)
    captions = [f"step {step} sample {index}" for index in range(len(paths))]
    swanlab.log(
        {
            "sample/abs_mean": float(generated.abs().mean()),
            "sample/std": float(generated.std()),
            **_prefixed(generated_consistency, "consistency"),
            **_prefixed(real_consistency, "consistency_real"),
            **_prefixed(generated_rot6d, "rot6d"),
            **_prefixed(real_rot6d, "rot6d_real"),
            **_prefixed(generated_motion, "object"),
            **_prefixed(real_motion, "object_real"),
        },
        step=step,
    )
    swanlab.log_video(
        key="samples",
        data=[str(path) for path in paths],
        caption=captions,
        step=step,
    )
    print(
        "consistency: "
        + ", ".join(f"{name} {value:.4f}" for name, value in generated_consistency.items()),
        flush=True,
    )
    print(
        "6D rotation error: "
        + ", ".join(f"{name} {value:.4f}" for name, value in generated_rot6d.items()),
        flush=True,
    )
    print(
        "object speed before touch "
        f"{generated_motion['speed_before_touch']:.4f} m/s "
        f"(real {real_motion['speed_before_touch']:.4f})",
        flush=True,
    )
    print(f"wrote {len(paths)} GIFs to {directory}", flush=True)
