import sys
from pathlib import Path

from src.ml_engine.config import MainConfig
from src.ml_engine.train import run_training_pipeline
from src.ml_engine.utils import get_logger, timer

logger = get_logger("__main__")


@timer
def main():
    try:
        config_path = Path("configs/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file absent at {config_path}")

        config = MainConfig.load_from_yaml(config_path)
        run_training_pipeline(config)
        logger.info("Main Pipeline run successfully.")
    except Exception as e:
        logger.critical(f"System Failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
