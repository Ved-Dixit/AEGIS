# AEGIS Predictive Fraud Intelligence System

AEGIS is a Python-based fraud intelligence platform built from the project synopsis in `/Users/veddixit/Desktop/AEGIS_Project_Synopsis.pdf`. It detects fraud **before** fund disbursement by combining tabular fraud classification, graph analytics, duplicate detection, anomaly scoring, and explainable AI. The backend is implemented with FastAPI and the operator console is implemented with Streamlit.

## What this project includes

- A synthetic supply-chain training dataset that injects the fraud patterns described in the synopsis: cross-lender duplicate financing, buyer-supplier collusion, and rapid network expansion.
- A strong ensemble model with four specialists:
  - `risk classifier`: supervised fraud probability using `XGBoost` when available, otherwise a balanced tree ensemble fallback.
  - `anomaly detector`: `IsolationForest` for rare behavioral signatures.
  - `duplicate detector`: trade identity similarity search using cosine distance.
  - `graph engine`: entity-network centrality and pair-density scoring with `NetworkX`.
- Explainability via SHAP-backed local explanations, with a deterministic fallback explanation path if SHAP cannot run.
- A FastAPI backend with train, sources, and prediction endpoints.
- A Streamlit console for manual scoring and model inspection.
- SQLite-backed persistence for prediction history, dataset imports, and training runs.
- Public dataset connectors and adapters so Kaggle/GitHub datasets can be normalized into the AEGIS schema.
- A hybrid training path that combines real DataCo supply-chain records with injected fraud patterns.

## Open-source dataset strategy

The synopsis explicitly asked for Kaggle and GitHub data sources. Those are wired into the dataset catalog at [`/Users/veddixit/ageix/aegis/data/sources.py`](/Users/veddixit/ageix/aegis/data/sources.py), including:

- IEEE-CIS Fraud Detection on Kaggle
- Credit Card Fraud Detection (MLG-ULB) on Kaggle
- PaySim on Kaggle
- Elliptic++ on GitHub
- DataCo Supply Chain Analytics on GitHub

Because public, labeled supply-chain fraud data is limited and Kaggle access often requires credentials, the default local training path uses the synthetic generator in [`/Users/veddixit/ageix/aegis/data/synthetic.py`](/Users/veddixit/ageix/aegis/data/synthetic.py). This keeps the project runnable immediately while preserving clear upgrade paths to larger public benchmarks.

## Project structure

- [`/Users/veddixit/ageix/aegis/data/synthetic.py`](/Users/veddixit/ageix/aegis/data/synthetic.py): generates a supply-chain dataset with explicit fraud modes.
- [`/Users/veddixit/ageix/aegis/data/adapters.py`](/Users/veddixit/ageix/aegis/data/adapters.py): maps Kaggle/GitHub datasets into the unified AEGIS schema.
- [`/Users/veddixit/ageix/aegis/data/connectors.py`](/Users/veddixit/ageix/aegis/data/connectors.py): download helpers and manual-fetch commands for public sources.
- [`/Users/veddixit/ageix/aegis/data/hybrid.py`](/Users/veddixit/ageix/aegis/data/hybrid.py): builds a labeled hybrid dataset from real DataCo rows plus injected fraud patterns.
- [`/Users/veddixit/ageix/aegis/features.py`](/Users/veddixit/ageix/aegis/features.py): schema normalization, historical feature engineering, and graph feature construction.
- [`/Users/veddixit/ageix/aegis/models.py`](/Users/veddixit/ageix/aegis/models.py): duplicate detector, anomaly detector, ensemble model, and explainability logic.
- [`/Users/veddixit/ageix/aegis/persistence.py`](/Users/veddixit/ageix/aegis/persistence.py): SQLite history store for runs, imports, and predictions.
- [`/Users/veddixit/ageix/aegis/service.py`](/Users/veddixit/ageix/aegis/service.py): train/save/load helpers shared by every surface.
- [`/Users/veddixit/ageix/backend/main.py`](/Users/veddixit/ageix/backend/main.py): FastAPI application.
- [`/Users/veddixit/ageix/streamlit_app.py`](/Users/veddixit/ageix/streamlit_app.py): Streamlit operator console.
- [`/Users/veddixit/ageix/scripts/fetch_public_data.py`](/Users/veddixit/ageix/scripts/fetch_public_data.py): fetches direct GitHub data or prints Kaggle manual commands.
- [`/Users/veddixit/ageix/scripts/prepare_external_dataset.py`](/Users/veddixit/ageix/scripts/prepare_external_dataset.py): normalizes external raw datasets into AEGIS format.
- [`/Users/veddixit/ageix/scripts/bootstrap_data.py`](/Users/veddixit/ageix/scripts/bootstrap_data.py): create demo data and export the open-source dataset catalog.
- [`/Users/veddixit/ageix/scripts/train_aegis.py`](/Users/veddixit/ageix/scripts/train_aegis.py): training CLI.

## Quick start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Bootstrap demo data:

```bash
python3 scripts/bootstrap_data.py --rows 8000
```

Download or inspect a public source:

```bash
python3 scripts/fetch_public_data.py --source dataco
```

Load Kaggle datasets interactively from the terminal:

```bash
python3 scripts/load_kaggle_datasets.py
```

The script will:
- prompt for Kaggle username and API key if `~/.kaggle/kaggle.json` does not exist
- ask which Kaggle datasets to download
- download and unzip them
- optionally normalize supported files into AEGIS CSVs automatically

By default, the interactive loader pulls `creditcardfraud` and `paysim`. `ieee_cis` is optional and only works after you accept the competition rules on Kaggle.

Prepare an external dataset into AEGIS schema:

```bash
python3 scripts/prepare_external_dataset.py --source dataco --input artifacts/data/raw/dataco/DataCoSupplyChainDataset.csv
```

Build the stronger hybrid training dataset:

```bash
python3 scripts/build_hybrid_dataset.py --dataco-path artifacts/data/dataco_aegis.csv --rows 60000 --output artifacts/data/dataco_hybrid_aegis.csv
```

Build the blended open-datasets training set used by the current saved model:

```bash
python3 scripts/build_multisource_dataset.py --rows 32000 --output artifacts/data/multisource_open_aegis_32000.csv
```

Train on the hybrid dataset:

```bash
python3 scripts/train_aegis.py --data-path artifacts/data/dataco_hybrid_aegis.csv --source-name dataco_hybrid --rows 60000
```

Train on the blended Kaggle + GitHub dataset mix:

```bash
python3 scripts/train_aegis.py --data-path artifacts/data/multisource_open_aegis_32000.csv --source-name multisource_open_blend_32000 --rows 32000
```

Train the model:

```bash
python3 scripts/train_aegis.py --rows 8000
```

Run the backend:

```bash
uvicorn backend.main:app --reload
```

Run the Streamlit console:

```bash
streamlit run streamlit_app.py
```

Run tests:

```bash
pytest
```

## API overview

- `GET /health`: backend heartbeat.
- `GET /sources`: returns the Kaggle/GitHub dataset catalog.
- `GET /sources/public`: returns download/preparation presets for supported public datasets.
- `POST /train`: trains a new model bundle from the synthetic dataset.
- `POST /predict/transaction`: scores one transaction.
- `POST /predict/batch`: scores multiple transactions.
- `POST /datasets/fetch`: downloads a public GitHub source or returns Kaggle/GitHub manual commands.
- `POST /datasets/prepare`: converts a raw public dataset into AEGIS schema.
- `GET /history/training`, `GET /history/predictions`, `GET /history/datasets`: inspect backend activity history.

## Notes on model quality

The backend is intentionally straightforward, per your request. The AI layer is where most of the quality effort went:

- The feature builder uses buyer, supplier, product, invoice, and lender history instead of only raw transaction fields.
- The graph layer models fraud propagation and dense collusive relationships.
- The duplicate detector gives pre-disbursal cross-lender visibility for invoice reuse.
- The ensemble exposes every specialist score separately so a lender can understand **why** a transaction was flagged.

Most modules include short comments and docstrings so the reasoning stays easy to follow without drowning the code in noise.
