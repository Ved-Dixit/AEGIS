"""FastAPI backend for the AEGIS fraud intelligence platform."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI

from aegis.data.sources import describe_sources
from aegis.schemas import (
    BatchPredictionRequest,
    DatasetFetchRequest,
    DatasetPrepareRequest,
    TrainRequest,
    TransactionInput,
    model_to_dict,
)
from aegis.service import AegisService, train_model

app = FastAPI(
    title="AEGIS Fraud Intelligence API",
    description="Predictive fraud scoring for supply-chain finance transactions.",
    version="1.0.0",
)

service = AegisService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sources")
def sources() -> list[dict[str, str]]:
    return describe_sources()


@app.get("/sources/public")
def public_sources() -> list[dict[str, object]]:
    return service.public_source_catalog()


@app.post("/train")
def train_endpoint(request: TrainRequest) -> dict[str, object]:
    data_frame = None
    if request.data_path:
        data_frame = pd.read_csv(request.data_path, parse_dates=["invoice_date", "due_date"])
    summary = train_model(rows=request.rows, data_frame=data_frame, source_name=request.source_name)
    service._engine = None
    return summary


@app.post("/predict/transaction")
def predict_transaction(payload: TransactionInput) -> dict[str, object]:
    service.ensure_model()
    return service.predict_record(model_to_dict(payload))


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictionRequest) -> list[dict[str, object]]:
    service.ensure_model()
    records = [model_to_dict(transaction) for transaction in payload.transactions]
    return service.predict_batch(records)


@app.post("/datasets/fetch")
def fetch_dataset(payload: DatasetFetchRequest) -> dict[str, object]:
    return service.fetch_public_source(
        source_name=payload.source_name,
        execute_kaggle=payload.execute_kaggle,
    )


@app.post("/datasets/prepare")
def prepare_dataset(payload: DatasetPrepareRequest) -> dict[str, object]:
    return service.prepare_external_dataset(
        source_name=payload.source_name,
        input_paths=payload.input_paths,
        output_path=payload.output_path,
    )


@app.get("/history/training")
def history_training(limit: int = 10) -> list[dict[str, object]]:
    return service.recent_training_runs(limit=limit)


@app.get("/history/predictions")
def history_predictions(limit: int = 25) -> list[dict[str, object]]:
    return service.recent_predictions(limit=limit)


@app.get("/history/datasets")
def history_datasets(limit: int = 10) -> list[dict[str, object]]:
    return service.recent_dataset_imports(limit=limit)
