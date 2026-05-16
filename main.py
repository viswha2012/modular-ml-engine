import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.ml_engine.config import MainConfig
from src.ml_engine.train import run_training_pipeline
from src.ml_engine.utils import get_logger, timer

logger = get_logger("__main__")


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
@timer
def main(cfg: DictConfig):
    try:
        raw_cfg = OmegaConf.to_container(cfg, resolve=True)
        config = MainConfig(**raw_cfg)
        run_training_pipeline(config)
        logger.info("Main Pipeline run successfully.")
    except Exception as e:
        logger.critical(f"System Failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
