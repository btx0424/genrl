"""Fine-tune a pretrained diffusion model with advantage-weighted denoising."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from genrl.train_awd import train


@hydra.main(version_base=None, config_path="../configs", config_name="awd")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
