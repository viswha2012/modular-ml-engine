from typing import Tuple, cast

import pandas as pd
from pandas.errors import EmptyDataError
from pydantic import BaseModel, Field, field_validator
from sklearn.model_selection import train_test_split

from src.ml_engine.config import MainConfig
from src.ml_engine.utils import get_logger, log_call, retry, timer

logger = get_logger("data_processor")


class DataSchema(BaseModel):
    feature_1: float
    feature_2: float
    label: float


class DataProcessor:
    def __init__(self, config: MainConfig) -> None:
        self.config = config
        self.df = None

    @timer
    @retry()
    @log_call(logger)
    def load_and_validate(self):
        path = self.config.data.raw_path
        if not path.exists():
            logger.error("File not found.")
            raise FileNotFoundError(f"Data file not found at: {path.absolute()}")
        self.df = pd.read_csv(path)
        logger.info(f"Loaded data from {path} - shape: {self.df.shape}")
        try:
            [DataSchema(**row) for row in self.df.to_dict("records")]
            logger.info(f"Data validation passed for {len(self.df)} rows")
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise

    @log_call(logger)
    def get_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.df is None:
            logger.error("DataFrame Empty")
            raise EmptyDataError

        if self.config.data.target_column not in self.df.columns:
            logger.error("Target column not found in DataFrame")
            raise KeyError("Target column not found in DataFrame")

        X = self.df.drop(columns=[self.config.data.target_column])
        y = cast(pd.Series, self.df[self.config.data.target_column])

        return (X, y)

    @log_call(logger)
    def prepare_split(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.data.test_size,
            random_state=self.config.model.random_seed,
        )
        logger.info(
            f"Split Complete. Train Size: {len(X_train)}, Test_Size: {len(X_test)}"
        )

        return X_train, X_test, y_train, y_test

    @log_call(logger)
    def run_full_pipeline(self):
        self.load_and_validate()
        X, y = self.get_features_target()
        return self.prepare_split(X, y)
