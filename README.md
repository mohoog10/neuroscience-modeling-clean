# Neuroscience Modeling Framework

Unified pipelines for classic ML (scikit-learn) and neural (Keras/TensorFlow) models, with a clean architecture (interface → manager → registry → models) and Optuna-powered hyperparameter tuning.

> **License:** MIT  
> **Language:** Python 3.10+

## What It Does

- **Model Registry Pattern**: Registers and runs models through a shared manager: KMeans, Logistic Regression, Random Forest, and Keras classifiers (Conv1D / GRU architectures defined in JSON config).
- **Single-Model Runs**: Execute individual model training/validation/testing via `main.py`.
- **Multi-Model Optuna Tuning**: Automated hyperparameter search across multiple models via `optuna_main.py`.
- **Synthetic Data Generation**: Includes CSV generation and preprocessing helpers under `process_csv_files/`.
- **Reporting**: Saves classification reports, confusion matrices, and Optuna trial tables to `results/` when enabled.
- **Unit Tests**: Test suite under `tests/` covering registry, preprocessing, manager pipeline, and model flows.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Core ML | scikit-learn (KMeans, LogisticRegression, RandomForest) |
| Deep Learning | TensorFlow / Keras (Conv1D, GRU, Dense networks) |
| Hyperparameter Tuning | Optuna |
| Keras-sklearn Bridge | scikeras |
| Data Processing | pandas, NumPy, SciPy |
| Visualization | matplotlib, seaborn |

## Project Structure

```
neuroscience-modeling/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── process_csv_files/          # Synthetic data + preprocessing helpers
│   ├── simulated.py            # Generate synthetic datasets
│   ├── downsampling_semi-controlled-data.py
│   └── interpolation_semi-controlled-data.py
├── src/
│   ├── interface/              # CLI and GUI adapters
│   │   ├── interface.py        # Abstract interface base
│   │   ├── interface_cli.py    # Command-line interface
│   │   ├── interface_tkinter.py
│   │   └── interface_kmeans.py
│   ├── manager/
│   │   └── manager.py          # Orchestrates interface + registry + models
│   ├── registry/
│   │   └── model_registry.py   # Model registration and lookup
│   ├── models/
│   │   ├── model.py            # Abstract Model base class
│   │   ├── kmeansmodel.py      # KMeans clustering
│   │   ├── logistic_model.py   # Logistic Regression
│   │   ├── rfc_model.py        # Random Forest Classifier
│   │   ├── keras_model.py      # Keras neural networks with Optuna
│   │   └── hyperparam_search.py
│   └── processing/
│       └── datapreprocessor.py # Data loading, splitting, scaling
├── tests/
│   └── test_core.py            # Unit test suite
├── main.py                     # Single-model pipeline entry point
├── optuna_main.py              # Multi-model Optuna tuning entry point
├── default_config.json         # Single-CSV smoke-test config
├── test_own_data.json          # Multi-model Optuna example config
├── requirements.txt            # Python dependencies
├── requirements-cpu.txt        # CPU-only dependencies (for CI)
├── Dockerfile
├── docker-compose.yml
└── LICENSE                     # MIT
```

Additional documentation: `QUICKSTART.md`, `ARCHITECTURE.md`, `PROJECT_STRUCTURE.md`, `PROJECT_SUMMARY.md`, `GITHUB_SETUP.md`.

## Setup

### Prerequisites

- Python 3.10 or higher
- pip (package installer)
- Optional: Docker 20.10+ for containerized runs

### Installation (pip)

```bash
# Clone the repository
git clone <repository-url>
cd neuroscience-modeling

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only environments (no GPU):

```bash
pip install -r requirements-cpu.txt
```

### Installation (Docker)

```bash
docker compose build
docker compose run --rm neuroscience-modeling
```

The compose file mounts `./outputs` and `./data` into the container.

## How to Run

### 1. Generate Synthetic Dataset

```bash
python process_csv_files/simulated.py
```

This creates sample CSV files under `simulated_dataset/`.

### 2. Train a Single Model

```bash
python main.py --model KMeans --mode train --config default_config.json
```

Available models: `KMeans`, `LogisticRegression`, `RandomForest`, `KerasClassifier`

### 3. Run Optuna Hyperparameter Tuning

```bash
python optuna_main.py --config test_own_data.json
```

This tunes all models defined in the config file with Optuna search.

### macOS Note

On macOS, you may need to set this environment variable to avoid Accelerate framework issues:

```bash
NPY_DISABLE_MACOS_ACCELERATE_CHECK=1 python main.py ...
```

## Configuration

Model configurations are JSON-based:

- **`default_config.json`**: Single-CSV config for quick smoke tests (KMeans, Logistic, RandomForest, Keras).
- **`test_own_data.json`**: Directory-based data loading with RandomForest + Keras Conv1D + Keras GRU blocks and `optuna_trials`.

Model registration names: `kmeans`, `logistic`, `random_forest`, `keras_classifier`.

### Example Model Config Block

```json
{
  "name": "KerasConv1D",
  "type": "keras_classifier",
  "build_mode": "grid",
  "epochs": 30,
  "batch_size": 16,
  "optuna_trials": 10,
  "layers": [
    {"type": "Conv1D", "filters": 64, "kernel_size": 5, "activation": "relu", "padding": "same"},
    {"type": "MaxPooling1D", "pool_size": 2},
    {"type": "Flatten"},
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "Dropout", "rate": 0.4}
  ]
}
```

## Tests

Run the test suite:

```bash
# Using pytest
pytest tests/ -v

# Or using unittest
python -m unittest discover tests
```

### Test Coverage

Tests in `tests/test_core.py` cover:

| Test | Description |
|------|-------------|
| `test_register_and_get` | Model registry registration and lookup |
| `test_split_dataframe` | DataPreprocessor split functionality |
| `test_load_all_csvs` | Multi-file CSV loading |
| `test_logistic_pipeline` | Manager + Logistic Regression train/predict |
| `test_train_predict` | KMeans train/validate flow |

## Output Artifacts

When enabled in config, the framework generates:

- `results/reports/<Model>/classification_report.txt`
- `results/reports/<Model>/confusion_matrix.png`
- `results/<Model>_trials_results.csv` (Optuna trial table)
- `results/best_results.csv` (best trial summaries)

## Results

<!-- 
Results placeholder: Add your model performance metrics here after running experiments.
Example format:
| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| RandomForest | 0.XX | 0.XX | XX trials |
| KerasConv1D | 0.XX | 0.XX | XX epochs |
-->

_Results will be populated after running experiments with your data._

## Extending the Framework

1. Subclass `Model` in `src/models/`.
2. Register in `main.py` or `optuna_main.py` via `registry.register_model("name", YourClass)`.
3. Add a JSON config block with `type`, hyperparameters, and optional `optuna_trials` / `layers`.
4. Add a unit test under `tests/`.

## Known Limitations

- TensorFlow may log warnings when loading optimizer states with different variable counts; models still function correctly.
- Keras Transformer-style architectures require appropriate kernels/GPU support.
- Heavy models may be slow on CPU-only environments.

## License

MIT License - see [LICENSE](LICENSE) for details.
