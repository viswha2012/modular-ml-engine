from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator


class ModelConfig(BaseModel):
    name: str
    learning_rate: PositiveFloat
    batch_size: PositiveInt
    epochs: PositiveInt = 10
    random_seed: PositiveInt = 42


class DataConfig(BaseModel):
    raw_path: Path
    processed_path: Path
    test_size: PositiveFloat = Field(gt=0, lt=1, default=0.2)
    target_column: str


class OutputConfig(BaseModel):
    model_path: Path
    metrics_path: Path
    log_path: Path

    @field_validator("model_path", "metrics_path", mode="after")
    @classmethod
    def make_absolute(cls, v: Path) -> Path:
        if not v.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            return project_root / v
        return v


class ExperimentConfig(BaseModel):
    name: str
    track: bool = True


class MainConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
    experiment: ExperimentConfig

    @classmethod
    def load_from_yaml(cls, yaml_path: Path):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
