import asyncio

from src.ml_engine.config import MainConfig
from src.ml_engine.data_ import DataProcessor
from src.ml_engine.evaluate import Evaluator
from src.ml_engine.model import ModelWrapper
from src.ml_engine.utils import get_logger, log_call, timer

logger = get_logger("trainer")


@timer
@log_call(logger)
async def run_training_pipeline(config: MainConfig):
    processor = DataProcessor(config)
    X_train, X_test, y_train, y_test = processor.run_full_pipeline()

    model = ModelWrapper(config)
    model.fit(X_train, y_train)

    evaluator = Evaluator(config)
    metrics = await evaluator.run_evaluation(model, X_test, y_test)

    await asyncio.gather(evaluator.save_metrics(metrics), model.save())

    logger.info(
        f"Pipeline finished succesfully\nAccuracy: {metrics.get('accuracy'):.4f}"
    )
