"""Train a shape-conditioned diffusion model on fixed-length pre-interaction segments."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from genrl.train_common import train


@hydra.main(version_base=None, config_path="../configs", config_name="diffusion")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
