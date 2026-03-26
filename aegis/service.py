"""Training and scoring helpers shared by CLI scripts, API, and Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from aegis.config import DATA_DIR, DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, MODEL_DIR, RAW_DATA_DIR, TrainingConfig
from aegis.data.adapters import prepare_dataset
from aegis.data.connectors import describe_connectors, fetch_public_source
from aegis.data.sources import describe_sources
from aegis.data.synthetic import generate_synthetic_supply_chain_dataset
from aegis.features import normalize_transactions
from aegis.models import AegisFraudEngine
from aegis.persistence import AegisRepository


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_or_generate_dataset(rows: int, data_path: str | Path | None = None) -> pd.DataFrame:
    ensure_directories()
    candidate = Path(data_path or DEFAULT_DATA_PATH)
    if candidate.exists():
        return pd.read_csv(candidate, parse_dates=["invoice_date", "due_date"])

    frame = generate_synthetic_supply_chain_dataset(rows=rows)
    frame.to_csv(candidate, index=False)
    return frame


def _compute_metrics(target: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    threshold = float(np.quantile(probabilities, 0.9))
    predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
        "average_precision": round(float(average_precision_score(target, probabilities)), 4),
        "precision_at_top_10pct": round(float(precision_score(target, predictions, zero_division=0)), 4),
        "recall_at_top_10pct": round(float(recall_score(target, predictions, zero_division=0)), 4),
        "fraud_rate": round(float(target.mean()), 4),
    }


def train_model(
    rows: int = 8_000,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_frame: pd.DataFrame | None = None,
    source_name: str = "synthetic",
    repository: AegisRepository | None = None,
) -> dict[str, Any]:
    """
    Train and persist the AEGIS fraud engine.

    The function validates quality on a holdout split first, then refits on the
    full dataset so the saved artifact contains all available history.
    """

    ensure_directories()
    frame = normalize_transactions(data_frame if data_frame is not None else load_or_generate_dataset(rows))
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["is_fraud"],
    )

    validation_engine = AegisFraudEngine(random_state=42).fit(train_frame)
    validation_results = validation_engine.predict(test_frame)
    validation_scores = np.array([row["final_score"] for row in validation_results])
    metrics = _compute_metrics(test_frame["is_fraud"].astype(int), validation_scores)

    final_engine = AegisFraudEngine(random_state=42).fit(frame)
    final_engine.metrics_ = metrics
    final_engine.training_metadata_ = {
        "rows": int(len(frame)),
        "fraud_rows": int(frame["is_fraud"].sum()),
        "dataset_sources": describe_sources(),
    }
    saved_path = final_engine.save(model_path)

    summary = {
        "model_path": str(saved_path),
        "rows": int(len(frame)),
        "fraud_rows": int(frame["is_fraud"].sum()),
        "metrics": metrics,
    }
    (repository or AegisRepository()).record_training_run(summary, source_name=source_name)
    return summary


class AegisService:
    """Lazy loader used by the API and Streamlit app."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        repository: AegisRepository | None = None,
    ):
        self.model_path = Path(model_path)
        self.repository = repository or AegisRepository()
        self._engine: AegisFraudEngine | None = None

    def ensure_model(self, rows: int = 6_000) -> dict[str, Any]:
        if self.model_path.exists():
            return {
                "model_path": str(self.model_path),
                "status": "ready",
            }
        return train_model(rows=rows, model_path=self.model_path, repository=self.repository)

    def load(self) -> AegisFraudEngine:
        if self._engine is None:
            self._engine = AegisFraudEngine.load(self.model_path)
        return self._engine

    def predict_record(self, record: dict[str, Any]) -> dict[str, Any]:
        engine = self.load()
        result = engine.predict(pd.DataFrame([record]))[0]
        self.repository.record_prediction(record, result)
        return result

    def predict_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        engine = self.load()
        results = engine.predict(pd.DataFrame(records))
        for request_record, response_record in zip(records, results):
            self.repository.record_prediction(request_record, response_record)
        return results

    def fetch_public_source(self, source_name: str, execute_kaggle: bool = False) -> dict[str, Any]:
        return fetch_public_source(source_name=source_name, output_dir=RAW_DATA_DIR, execute_kaggle=execute_kaggle)

    def prepare_external_dataset(
        self,
        source_name: str,
        input_paths: list[str | Path],
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        summary = prepare_dataset(source_name=source_name, input_paths=input_paths, output_path=output_path)
        self.repository.record_dataset_import(
            source_name=summary["source_name"],
            input_paths=summary["input_paths"],
            output_path=summary["output_path"],
            row_count=summary["row_count"],
            label_summary=summary["label_summary"],
            notes=summary["notes"],
        )
        return summary

    def recent_predictions(self, limit: int = 25) -> list[dict[str, Any]]:
        return self.repository.list_prediction_events(limit=limit)

    def recent_training_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.repository.list_training_runs(limit=limit)

    def recent_dataset_imports(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.repository.list_dataset_imports(limit=limit)

    @staticmethod
    def public_source_catalog() -> list[dict[str, Any]]:
        return describe_connectors()
