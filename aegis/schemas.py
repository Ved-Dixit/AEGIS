"""Shared request/response models used by the API and Streamlit app."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field


def _default_due_date() -> date:
    return date.today() + timedelta(days=30)


class TrainRequest(BaseModel):
    """Minimal training options exposed by the API."""

    rows: int = Field(default=8_000, ge=1_000, le=100_000)
    data_path: str | None = None
    source_name: str = "synthetic"


class TransactionInput(BaseModel):
    """Single supply-chain transaction passed into the scoring engine."""

    transaction_id: str = "txn_manual_001"
    invoice_id: str = "inv_manual_001"
    buyer_id: str = "BUY_001"
    supplier_id: str = "SUP_001"
    lender_id: str = "LEND_001"
    product_id: str = "PROD_001"
    quantity: float = Field(default=120.0, gt=0)
    unit_price: float = Field(default=54.0, gt=0)
    invoice_amount: float = Field(default=6_480.0, gt=0)
    loan_amount: float = Field(default=5_300.0, gt=0)
    invoice_date: date = Field(default_factory=date.today)
    due_date: date = Field(default_factory=_default_due_date)
    payment_term_days: int = Field(default=30, ge=1, le=365)
    shipment_distance_km: float = Field(default=180.0, ge=0)
    buyer_risk_rating: float = Field(default=0.18, ge=0, le=1)
    supplier_risk_rating: float = Field(default=0.14, ge=0, le=1)
    historic_late_payments: int = Field(default=1, ge=0)
    prior_financing_count: int = Field(default=3, ge=0)
    channel: str = "portal"
    currency: str = "USD"


class BatchPredictionRequest(BaseModel):
    """Convenience wrapper for scoring several transactions in one call."""

    transactions: list[TransactionInput]


class DatasetPrepareRequest(BaseModel):
    """Prepare a raw public dataset into the unified AEGIS schema."""

    source_name: str
    input_paths: list[str]
    output_path: str | None = None


class DatasetFetchRequest(BaseModel):
    """Download or describe the files for a supported public source."""

    source_name: str
    execute_kaggle: bool = False


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Handle both Pydantic v1 and v2 without forcing a specific version."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
