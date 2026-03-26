from pathlib import Path

import pandas as pd

from aegis.data.adapters import adapt_paysim_dataset
from aegis.data.hybrid import build_hybrid_supply_chain_dataset
from aegis.persistence import AegisRepository


def test_paysim_adapter_maps_into_aegis_schema() -> None:
    raw = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 1250.0,
                "nameOrig": "C123",
                "nameDest": "M456",
                "oldbalanceOrg": 2000.0,
                "newbalanceOrig": 700.0,
                "oldbalanceDest": 300.0,
                "newbalanceDest": 1550.0,
                "isFraud": 1,
            },
            {
                "step": 2,
                "type": "CASH_OUT",
                "amount": 210.0,
                "nameOrig": "C124",
                "nameDest": "M457",
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 290.0,
                "oldbalanceDest": 100.0,
                "newbalanceDest": 310.0,
                "isFraud": 0,
            },
        ]
    )

    adapted = adapt_paysim_dataset(raw)
    assert set(
        [
            "transaction_id",
            "invoice_id",
            "buyer_id",
            "supplier_id",
            "loan_amount",
            "is_fraud",
            "fraud_type",
        ]
    ).issubset(adapted.columns)
    assert adapted["is_fraud"].tolist() == [1, 0]
    assert adapted["buyer_id"].tolist() == ["C123", "C124"]


def test_repository_records_history(tmp_path: Path) -> None:
    repository = AegisRepository(tmp_path / "history.sqlite3")
    repository.record_training_run(
        {
            "model_path": str(tmp_path / "bundle.joblib"),
            "rows": 100,
            "fraud_rows": 12,
            "metrics": {"roc_auc": 0.9},
        },
        source_name="synthetic",
    )
    repository.record_prediction(
        {"transaction_id": "txn_1", "invoice_id": "inv_1"},
        {"transaction_id": "txn_1", "invoice_id": "inv_1", "final_score": 0.72, "risk_band": "high"},
    )
    repository.record_dataset_import(
        source_name="dataco",
        input_paths=[str(tmp_path / "raw.csv")],
        output_path=str(tmp_path / "prepared.csv"),
        row_count=50,
        label_summary={"0": 50},
        notes="prepared",
    )

    assert repository.list_training_runs(limit=5)[0]["source_name"] == "synthetic"
    assert repository.list_prediction_events(limit=5)[0]["risk_band"] == "high"
    assert repository.list_dataset_imports(limit=5)[0]["source_name"] == "dataco"


def test_hybrid_builder_creates_labeled_supply_chain_data(tmp_path: Path) -> None:
    dataco_like = pd.DataFrame(
        [
            {
                "transaction_id": f"txn_{index}",
                "invoice_id": f"inv_{index}",
                "buyer_id": f"BUY_{index % 5}",
                "supplier_id": f"SUP_{index % 7}",
                "lender_id": "PAYMENT_DEBIT",
                "product_id": f"PROD_{index % 3}",
                "quantity": 2,
                "unit_price": 10.0 + index,
                "invoice_amount": 20.0 + index,
                "loan_amount": 16.0 + index,
                "invoice_date": "2024-01-01",
                "due_date": "2024-01-08",
                "payment_term_days": 7,
                "shipment_distance_km": 0.0,
                "buyer_risk_rating": 0.1,
                "supplier_risk_rating": 0.1,
                "historic_late_payments": 0,
                "prior_financing_count": 1,
                "channel": "standard",
                "currency": "USD",
                "is_fraud": 0,
                "fraud_type": "legit",
            }
            for index in range(200)
        ]
    )
    dataco_path = tmp_path / "dataco_like.csv"
    dataco_like.to_csv(dataco_path, index=False)

    hybrid = build_hybrid_supply_chain_dataset(dataco_path, rows=300, fraud_ratio=0.2, seed=9)
    assert len(hybrid) == 300
    assert int(hybrid["is_fraud"].sum()) > 0
    assert {"duplicate_financing", "buyer_supplier_collusion", "rapid_network_expansion"}.issubset(
        set(hybrid["fraud_type"].unique())
    )
