import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.ml_engine.config import MainConfig
from src.ml_engine.utils import get_logger, log_call, timer

logger = get_logger("model_wrapper")


class BaseMLModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class ModelWrapper(BaseMLModel):
    def __init__(self, config: MainConfig) -> None:
        self.config = config
        self.weights = None
        self.is_trained = False
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=config.model.random_seed
        )
        logger.info(f"Model {config.model.name} initialised-")
        logger.info(
            f"lr: {config.model.learning_rate}, batch_size: {config.model.batch_size}"
        )

    @log_call(logger)
    @timer
    def fit(self, X_train, y_train):
        # self.weights = np.random.rand(X_train.shape)
        self.model.fit(X_train, y_train)
        logger.info(
            f"Training complete - {len(X_train)} samples for {self.config.model.epochs} epochs"
        )
        self.is_trained = True

    @timer
    def predict(self, X_test):
        if not self.is_trained:
            logger.error(
                f"{self.config.model.name} has not been trained. Call train() first"
            )

        preds = self.model.predict(X_test)
        logger.info(f"Generated {len(X_test)}  preidctions.")
        return preds

    @log_call(logger)
    async def save(self):
        filename = f"{self.config.experiment.name}_{self.config.model.name}.joblib"
        save_path = self.config.output.model_path.parent / filename

        save_path.parent.mkdir(parents=True, exist_ok=True)

        def _write_weights():
            joblib.dump(self.model, save_path)

        await asyncio.to_thread(_write_weights)
        logger.info(f"{self.config.model.name} saved to {save_path}")

    @log_call(logger)
    @classmethod
    async def load(cls, config: MainConfig, load_path: Path):
        if not load_path.exists():
            logger.error(f"Model doesn't exist at {load_path}")

        instance = cls(config)
        instance.model = joblib.load(load_path)
        logger.info(f"Model has been loaded from {load_path}")
        return instance
