# Neuroscience Modeling Framework

Unified pipelines for classic ML (sklearn) and neural (Keras) models, built with a clean architecture and Optuna-powered hyperparameter tuning.

## 🏗️ Repository Layout

```
neuroscience-modeling/
├── process_csv_files/        # synthetic data + preprocessing helpers
├── results/                  # Optuna reports & artifacts
├── src/
│   ├── interface/            # CLI adapters
│   ├── manager/              # orchestrates interface + registry + models
│   ├── registry/             # model registration
│   └── models/               # sklearn & Keras implementations
├── main.py                   # single-model pipeline entry point
├── optuna_main.py            # multi-model Optuna tuning entry point
├── tests/                    # python -m unittest discover tests
└── requirements.txt
```

## ✨ Key Features

- **Clean Architecture** separation (interface → manager → registry → models).
- **Model zoo**: KMeans, LogisticRegression, RandomForest, Keras Conv1D, GRU (Transformer-ready).
- **Optuna integration**: auto-tunes every configured model and saves best-trial weights + reports.
- **Synthetic dataset workflow**: generate padded CSVs for experimentation via `process_csv_files/simulated.py`.
- **Rich reporting**: confusion matrices & classification reports per model in `results/reports/<Model>/`.
- **Unit tests**: registry, preprocessing, manager, and KMeans flows covered in `tests/test_core.py`.

## 📋 Requirements

- Python 3.10+ (virtualenv recommended)
- pip
- (Optional) Docker 20.10+/Compose 1.29+ for containerized demos

## 🚀 Quickstart (Local)

```bash
git clone https://github.com/mohoog10/neuroscience-modeling.git
cd neuroscience-modeling
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# 1. Generate synthetic dataset
python process_csv_files/simulated.py

# 2. Train a single model (uses default_config.json)
NPY_DISABLE_MACOS_ACCELERATE_CHECK=1 python main.py \
  --model KMeans --mode train --config default_config.json

# 3. Tune all models defined in test_own_data.json
NPY_DISABLE_MACOS_ACCELERATE_CHECK=1 python optuna_main.py \
  --config test_own_data.json

# 4. Run unit tests
python -m unittest discover tests
```

## 🔧 Config-Driven Pipelines

- `default_config.json` → points to a single CSV (`simulated_dataset/sample_0.csv`) for quick KMeans/logistic smoke tests.
- `test_own_data.json` → loads the entire `simulated_dataset/` directory, splits into train/val/test, and tunes:
  - RandomForest (Optuna grid)
  - KerasConv1D (short epochs/trials for CPU)
  - KerasGRU (short epochs/trials for CPU)
  - (Transformer entry available but disabled by default on CPU-only machines)

Each model block supports `optuna_trials`, `epochs`, `batch_size`, and a `layers` list for Keras architectures.

## 🧪 Model Summary

| Name             | Stack      | Notes                                        |
|------------------|------------|----------------------------------------------|
| `KMeans`         | sklearn    | Unsupervised clustering baseline             |
| `LogisticRegression` | sklearn | Fast supervised baseline                     |
| `RandomForest`   | sklearn    | Tuned via Optuna, saves confusion matrices   |
| `KerasConv1D`    | Keras CNN  | Lightweight 1D convolutional classifier      |
| `KerasGRU`       | Keras RNN  | Mask-aware GRU stack for padded sequences    |

## 📊 Output Artifacts

- `results/reports/<Model>/classification_report.txt`
- `results/reports/<Model>/confusion_matrix.png`
- `results/<Model>_trials_results.csv` (full Optuna trial table)
- `results/best_results.csv` (best trial summaries)

Example Optuna run (`NPY_DISABLE_MACOS_ACCELERATE_CHECK=1 python optuna_main.py --config test_own_data.json`) prints progress like:

```
RandomForest  ... finished in 100.6s (reports saved)
KerasConv1D   ... finished in 12.8s (reports saved)
KerasGRU      ... finished in 55.7s (reports saved)
All 3 models finished in 239.1 seconds.
```

## 🧰 Extending

1. Add a subclass of `Model` under `src/models/`.
2. Register it in `main.py` / `optuna_main.py` via `registry.register_model("name", Class)`.
3. Add a config block (or new JSON file) referencing the model + any Optuna grids/layers.
4. Optional: update `tests/` with a lightweight smoke test for the new component.

## 🧪 Tests

Run all unit tests:

```bash
python -m unittest discover tests
```

`tests/test_core.py` currently covers:
- Model registry registration/lookup
- `DataPreprocessor.split` and multi-file loader
- Logistic manager pipeline (train + predict)
- KMeans train/validate flow

## 📝 Notes

- TensorFlow warns when loading optimizer states with different variable counts; models still run.
- When using Keras Transformer layers, a CUDA-enabled GPU (or CPU kernels with BroadcastTo support) is required. On CPU-only Macs, keep the transformer config disabled.
- `NPY_DISABLE_MACOS_ACCELERATE_CHECK=1` prevents macOS Accelerate sanity-check crashes.

Happy modeling! 🎓🧠
