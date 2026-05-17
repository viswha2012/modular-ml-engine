# Production-Grade ML Engineering Pipeline

A modular, type-safe Machine Learning framework designed for reproducibility and production deployment. This project demonstrates advanced Python engineering practices applied to a standard ML workflow.

## Key Engineering Features
- **Strict Data Contracts:** Uses **Pydantic** for runtime data validation and configuration management.
- **Modular Architecture:** Decoupled modules for Data Processing, Model Wrapping, and Evaluation.
- **Production Observability:** Implemented custom decorators for automated execution timing and granular logging.
- **Environment Agnostic:** Dynamic absolute path resolution for reliable execution across different local/server environments.
- **Abstract Base Classes (ABC):** Enforcement of model interfaces to ensure structural integrity.

## Release History

### v1.2.0 — Concurrent Execution Layer
- **Async Engine Integration:** Introduced asynchronous execution patterns into writing operations.

### v1.1.0 — Hydra Config Engine
- **Modular configs:** Enabled modular, hierarchical config swapping (`data`, `model`).
- **Pydantic Scheme Bridge:** Flattens `DictConfig` structures with **Pydantic** validation.

## Project Structure
```text
project_root/
├── configs/          # YAML configurations(hydra)
├── src/ml_engine/    # Core logic (Config, Data, Model, Train, Eval, Utils)
├── logs/             # Automated execution logs
├── outputs/          # Versioned model artifacts and metrics
└── main.py           # System entry point
