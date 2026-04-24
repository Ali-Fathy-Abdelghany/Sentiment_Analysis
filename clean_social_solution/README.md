# CleanSocial Sentiment Project

A complete sentiment analysis pipeline implementation aligned with the 4 course tasks:
- Task 1: data collection audit + minimum records check
- Task 2: configurable preprocessing pipeline
- Task 3: labeling, representation, lexical/ML modeling, benchmarking
- Task 4: optimization, error analysis, deployment (FastAPI + Streamlit)

This project now uses a **src-first structure** where all runnable code lives under `src/clean_social`.

## 1) Quick Start

```powershell
cd c:\Users\Fares\Documents\university\CleanSocial\clean_social_solution
python -m pip install -r requirements.txt
python -m pip install -e .
python -m textblob.download_corpora
```

Run full pipeline in one command:

```powershell
python -m clean_social.cli.run_all
```

Run full pipeline with explicit local GloVe file:

```powershell
python -m clean_social.cli.run_all --glove-path c:/Users/Fares/Documents/university/CleanSocial/glove.6B.100d.txt
```

Run task coverage check:

```powershell
python -m clean_social.cli.check_requirements
```

Start API + Streamlit together:

```powershell
python -m clean_social.cli.start_apps
```

Start API + Streamlit in background (command exits, services keep running):

```powershell
python -m clean_social.cli.start_apps --detach
```

Run API:

```powershell
python -m uvicorn clean_social.apps.api:app --host 127.0.0.1 --port 8000
```

Run Streamlit:

```powershell
python -m streamlit run src/clean_social/apps/streamlit_app.py --server.port 8501
```

---

## 2) Project Architecture

### High-level flow
1. Raw reviews are audited and preprocessed.
2. 400 records are sampled for annotation (balanced classes + negation quotas).
3. Annotators are aggregated with majority voting and Cohen's Kappa.
4. Text is represented with TF-IDF and GloVe-like vectors.
5. Models are benchmarked (18 evaluated variants).
6. Selected ML candidates are optimized and compared pre/post tuning.
7. Errors are analyzed for failure patterns.
8. Best serving artifact is exposed through FastAPI and consumed by Streamlit.

### Why src-first?
- Keeps all business logic, CLI entrypoints, and app entrypoints under one package.
- Makes imports consistent (`clean_social.*`) and easier to maintain.
- Avoids having logic spread across root/scripts/deployment folders.

---

## 3) Folder-by-Folder Guide

### `src/clean_social/`
Core Python package.

#### `src/clean_social/cli/`
Command-line entrypoints for each stage.

- `main.py`
  - Task 2 named main entrypoint module.
  - Delegates directly to preprocessing CLI.
- `preprocess.py`
  - Configurable preprocessing pipeline (`argparse` flags).
  - All flags default to False.
- `audit_data.py`
  - Data audit summary for Task 1 requirements.
- `generate_preprocessed_sets.py`
  - Generates 3 preprocessing schemes (`s1`, `s2`, `s3`).
- `prepare_annotations.py`
  - Samples 400 records with balanced sampling and negation-aware quotas.
- `fill_annotations.py`
  - Optional helper to auto-fill annotator sheets.
- `aggregate_annotations.py`
  - Majority vote + Cohen's Kappa report.
- `build_features.py`
  - Builds TF-IDF and GloVe representations.
- `train_models.py`
  - Benchmarks lexical + ML models and saves all 12 ML artifacts.
- `optimize_models.py`
  - Hyperparameter optimization on selected candidates and pre/post comparison.
- `error_analysis.py`
  - Extracts and summarizes misclassification patterns.
- `train_serving_model.py`
  - Trains and saves deployable model artifact.
- `run_all.py`
  - One-command orchestrator for full pipeline.
- `check_requirements.py`
  - Verifies generated outputs and core task-alignment checks.

#### `src/clean_social/apps/`
User-facing application entrypoints.

- `api.py`
  - FastAPI app with `/health` and `/predict`.
  - Loads `artifacts/models/deployment_model.joblib`.
- `streamlit_app.py`
  - Streamlit UI that calls API `/predict`.

#### `src/clean_social/preprocessing/`
Text cleaning and metadata tagging modules.

- `cleaners.py`
  - URL/emoji/punctuation removal, lemmatization, spell correction, stopword handling.
- `pipeline.py`
  - `PreprocessingConfig` + ordered application of cleaning steps.
- `tagging.py`
  - Regex-based category tagging.

#### `src/clean_social/labeling/`
Annotation and label aggregation logic.

- `rules.py`
  - Label normalization, rating-to-label mapping, negation detection.
- `sampling.py`
  - Balanced sampling + negation quota logic.
- `aggregation.py`
  - Majority voting and Kappa computations.

#### `src/clean_social/features/`
Feature representation builders.

- `representations.py`
  - TF-IDF and GloVe matrix creation (with hash fallback).

#### `src/clean_social/models/`
Model-specific algorithms.

- `lexical.py`
  - VADER classifier and AFINN-from-scratch with improved negation handling.

#### `src/clean_social/evaluation/`
Metrics and evaluation helpers.

- `metrics.py`
  - Accuracy, precision/recall/F1, confusion matrix, ROC-AUC (OVR).

#### `src/clean_social/utils/`
Shared utility functions.

- `io_utils.py`
  - CSV loading/saving and required column checks.
- `paths.py`
  - Project root resolver used across CLI and app modules.

### `data/`
Data lifecycle files.

- `data/raw/`
  - Source files (`ali-express_reviews.csv`, `AFINN-en-165.txt`).
- `data/processed/`
  - Preprocessed schemes (`s1_minimal.csv`, `s2_standard.csv`, `s3_extended.csv`).
- `data/annotations/`
  - Annotation sheets and final labeled set (`labels_400.csv`).

### `artifacts/`
Generated machine-readable outputs.

- `artifacts/features/`
  - TF-IDF/GloVe matrices, text subsets, metadata.
- `artifacts/models/benchmark/`
  - 12 saved ML benchmark models (Task 3 matrix).
- `artifacts/models/optimized/`
  - tuned models from optimization.
- `artifacts/models/deployment_model.joblib`
  - serving model for API.

### `reports/`
Human-readable and tabular reports.

- `data_audit.*`
- `annotation_agreement.json`
- `model_metrics.csv`, `model_metrics_detailed.json`
- `optimization_results.csv`, `optimization_results_detailed.json`
- `error_analysis.csv`, `error_analysis.md`
- `serving_model_metrics.json`
- `task_coverage_check.md`, `task_coverage_check.json`
- `plots/roc_auc_before_optimization.png`
- `plots/roc_auc_after_optimization.png`

### Root files
- `pyproject.toml` package config
- `requirements.txt` dependencies
- `README.md` this guide

---

## 4) Command-by-Command Usage

### Task 1
```powershell
python -m clean_social.cli.audit_data
```

### Task 2
Named `main.py` module behavior:
```powershell
python -m clean_social.cli.main --input data/raw/ali-express_reviews.csv --output data/processed/custom.csv --extract-tags
```

Equivalent direct module:
```powershell
python -m clean_social.cli.preprocess --input data/raw/ali-express_reviews.csv --output data/processed/custom.csv --extract-tags
```

Generate 3 schemes for downstream testing:
```powershell
python -m clean_social.cli.generate_preprocessed_sets
```

### Task 3
```powershell
python -m clean_social.cli.prepare_annotations --sample-size 400 --min-negation-ratio 0.10 --sampling-label-source vader
python -m clean_social.cli.fill_annotations --overwrite
python -m clean_social.cli.aggregate_annotations
python -m clean_social.cli.build_features
python -m clean_social.cli.train_models
```

### Task 4
```powershell
python -m clean_social.cli.optimize_models
python -m clean_social.cli.error_analysis
python -m clean_social.cli.train_serving_model
```

### Full automation
```powershell
python -m clean_social.cli.run_all
```

Run only a range:
```powershell
python -m clean_social.cli.run_all --from-step build_features --to-step train_serving_model
```

### Task-coverage check
```powershell
python -m clean_social.cli.check_requirements
```

---

## 5) API and UI

### FastAPI
- Module: `clean_social.apps.api`
- Endpoints:
  - `GET /health`
  - `POST /predict`

Example request:
```json
{ "text": "I love this product" }
```

Example response:
```json
{ "sentiment": "Positive", "confidence": 0.94 }
```

### Streamlit
- Module file: `src/clean_social/apps/streamlit_app.py`
- Default API URL in UI: `http://127.0.0.1:8000`

### Combined launcher
- Module: `clean_social.cli.start_apps`
- Starts both services and prints URLs.
- If a port is already in use, it assumes that service is already running and does not relaunch it.

---

## 6) Notes on Model Counts

- Task 3 benchmark evaluates 18 variants (3 schemes x 6 variants per scheme).
- 12 ML models are saved as artifacts (SVM + Linear Regression across TF-IDF/GloVe x 3 schemes).
- Lexical models (VADER/AFINN) are rule-based and evaluated but not persisted as fitted model files.

---

## 7) Troubleshooting

- If imports fail, reinstall editable package:
```powershell
python -m pip install -e .
```

- If TextBlob complains about corpora:
```powershell
python -m textblob.download_corpora
```

- If API says model not found, regenerate serving artifact:
```powershell
python -m clean_social.cli.train_serving_model
```

- If Streamlit cannot predict, ensure API is running on `127.0.0.1:8000`.

- Backward-compatibility (old commands still supported):
```powershell
python -m uvicorn deployment.api.main:app --host 127.0.0.1 --port 8000
python -m streamlit run deployment/streamlit_app.py --server.port 8501
```
