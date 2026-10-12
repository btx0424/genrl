"""Train flow matching with a QGF critic on pre-interaction segments."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from genrl.train_qgf import train


@hydra.main(version_base=None, config_path="../configs", config_name="qgf")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
