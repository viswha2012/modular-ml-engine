# Production-Grade ML Engineering Pipeline

A modular, type-safe Machine Learning framework designed for reproducibility and production deployment. This project demonstrates advanced Python engineering practices applied to a standard ML workflow.

## Key Engineering Features
- **Strict Data Contracts:** Uses **Pydantic** for runtime data validation and configuration management.
- **Modular Architecture:** Decoupled modules for Data Processing, Model Wrapping, and Evaluation.
- **Production Observability:** Implemented custom decorators for automated execution timing and granular logging.
- **Environment Agnostic:** Dynamic absolute path resolution for reliable execution across different local/server environments.
- **Abstract Base Classes (ABC):** Enforcement of model interfaces to ensure structural integrity.

## Project Structure
```text
project_root/
├── configs/          # YAML configurations
├── src/ml_engine/    # Core logic (Config, Data, Model, Train, Eval, Utils)
├── logs/             # Automated execution logs
├── outputs/          # Versioned model artifacts and metrics
└── main.py           # System entry point
