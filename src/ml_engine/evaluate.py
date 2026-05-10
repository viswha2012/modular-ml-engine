import json
from typing import Any, Dict, cast

from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report

from src.ml_engine.config import MainConfig
from src.ml_engine.utils import get_logger, log_call, timer

logger = get_logger("evaluator")


class EvaluationMetrics(BaseModel):
    model_name: str
    experiment_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class Evaluator:
    def __init__(self, config: MainConfig):
        self.config = config

    @log_call(logger)
    @timer
    def run_evaluation(self, model, X_test, y_test):
        y_preds = model.predict(X_test)

        acc = accuracy_score(y_test, y_preds)
        report = cast(
            Dict[str, Any], classification_report(y_test, y_preds, output_dict=True)
        )

        results = {
            "model_name": self.config.model.name,
            "experiment_name": self.config.experiment.name,
            "accuracy": acc,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
        }

        metrics = EvaluationMetrics(**results)
        logger.info(f"Evaluation finished for {self.config.model.name}")
        return metrics.model_dump()

    @log_call(logger)
    def save_metrics(self, metrics: dict):
        filename = (
            f"{self.config.experiment.name}_{self.config.model.name}_metrics.json"
        )
        save_path = self.config.output.metrics_path.parent / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {save_path}")
