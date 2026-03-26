from pathlib import Path

import pandas as pd

from aegis.data.blend import BlendSourceSpec, build_multisource_dataset


def _write_source(path: Path, prefix: str, fraud_rows: int, legit_rows: int) -> None:
    rows = []
    for index in range(legit_rows):
        rows.append(
            {
                "transaction_id": f"{prefix}_legit_{index}",
                "invoice_id": f"{prefix}_inv_l_{index}",
                "buyer_id": f"{prefix}_BUY_{index % 3}",
                "supplier_id": f"{prefix}_SUP_{index % 4}",
                "lender_id": "LENDER_A",
                "product_id": "PROD_A",
                "quantity": 1,
                "unit_price": 10.0,
                "invoice_amount": 10.0,
                "loan_amount": 8.0,
                "invoice_date": "2025-01-01",
                "due_date": "2025-01-10",
                "payment_term_days": 9,
                "shipment_distance_km": 12.0,
                "buyer_risk_rating": 0.1,
                "supplier_risk_rating": 0.1,
                "historic_late_payments": 0,
                "prior_financing_count": 1,
                "channel": "portal",
                "currency": "USD",
                "is_fraud": 0,
                "fraud_type": "legit",
            }
        )
    for index in range(fraud_rows):
        rows.append(
            {
                "transaction_id": f"{prefix}_fraud_{index}",
                "invoice_id": f"{prefix}_inv_f_{index}",
                "buyer_id": f"{prefix}_BUY_{index % 2}",
                "supplier_id": f"{prefix}_SUP_{index % 2}",
                "lender_id": "LENDER_B",
                "product_id": "PROD_B",
                "quantity": 1,
                "unit_price": 200.0,
                "invoice_amount": 200.0,
                "loan_amount": 198.0,
                "invoice_date": "2025-02-01",
                "due_date": "2025-02-05",
                "payment_term_days": 4,
                "shipment_distance_km": 1.0,
                "buyer_risk_rating": 0.8,
                "supplier_risk_rating": 0.85,
                "historic_late_payments": 7,
                "prior_financing_count": 8,
                "channel": "manual",
                "currency": "USD",
                "is_fraud": 1,
                "fraud_type": "duplicate_financing",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_multisource_dataset_balances_sources(tmp_path: Path) -> None:
    source_a = tmp_path / "source_a.csv"
    source_b = tmp_path / "source_b.csv"
    _write_source(source_a, prefix="A", fraud_rows=2, legit_rows=20)
    _write_source(source_b, prefix="B", fraud_rows=1, legit_rows=18)

    blended = build_multisource_dataset(
        [
            BlendSourceSpec(name="source_a", path=source_a, rows=24, fraud_ratio=0.25, chunksize=8),
            BlendSourceSpec(name="source_b", path=source_b, rows=16, fraud_ratio=0.25, chunksize=8),
        ],
        seed=5,
    )

    assert len(blended) == 40
    assert set(blended["source_name"].unique()) == {"source_a", "source_b"}
    assert int(blended["is_fraud"].sum()) >= 10
    assert {"duplicate_financing", "legit"} == set(blended["fraud_type"].unique())
