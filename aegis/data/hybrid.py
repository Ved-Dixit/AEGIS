"""Hybrid dataset builders that combine real operations with injected fraud."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from aegis.features import normalize_transactions


def build_hybrid_supply_chain_dataset(
    dataco_prepared_path: str | Path,
    rows: int = 60_000,
    fraud_ratio: float = 0.14,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a training set from real DataCo operations plus injected fraud cases.

    Real supply-chain records provide the legitimate structure. Fraud rows are
    injected on top so the classifier learns the exact risk patterns from the
    AEGIS synopsis without depending only on fully synthetic legitimate data.
    """

    rng = np.random.default_rng(seed)
    base_frame = pd.read_csv(dataco_prepared_path, parse_dates=["invoice_date", "due_date"])
    base_frame = normalize_transactions(base_frame)
    base_frame = base_frame.copy()
    base_frame["is_fraud"] = 0
    base_frame["fraud_type"] = "legit"

    legit_target = max(int(round(rows * (1.0 - fraud_ratio))), 1)
    legit_sample = base_frame.sample(
        n=min(legit_target, len(base_frame)),
        replace=len(base_frame) < legit_target,
        random_state=seed,
    ).reset_index(drop=True)

    fraud_target = max(rows - len(legit_sample), 1)
    fraud_chunks: list[pd.DataFrame] = []

    duplicate_count = max(int(fraud_target * 0.45), 1)
    collusion_count = max(int(fraud_target * 0.3), 1)
    expansion_count = max(fraud_target - duplicate_count - collusion_count, 1)

    duplicate_seed = legit_sample.sample(
        n=min(duplicate_count, len(legit_sample)),
        replace=len(legit_sample) < duplicate_count,
        random_state=seed + 11,
    ).reset_index(drop=True)
    duplicate_rows = duplicate_seed.copy()
    duplicate_rows["transaction_id"] = [f"hyb_dup_{index:06d}" for index in range(len(duplicate_rows))]
    duplicate_rows["lender_id"] = duplicate_rows["lender_id"].astype(str) + "_ALT"
    duplicate_rows["channel"] = "manual"
    duplicate_rows["loan_amount"] = (duplicate_rows["loan_amount"] * rng.uniform(0.96, 1.04, len(duplicate_rows))).round(2)
    duplicate_rows["invoice_date"] = pd.to_datetime(duplicate_rows["invoice_date"]) + pd.to_timedelta(
        rng.integers(0, 3, len(duplicate_rows)),
        unit="D",
    )
    duplicate_rows["due_date"] = pd.to_datetime(duplicate_rows["invoice_date"]) + pd.to_timedelta(
        duplicate_rows["payment_term_days"],
        unit="D",
    )
    duplicate_rows["buyer_risk_rating"] = np.clip(duplicate_rows["buyer_risk_rating"] + rng.uniform(0.08, 0.22, len(duplicate_rows)), 0, 1)
    duplicate_rows["supplier_risk_rating"] = np.clip(duplicate_rows["supplier_risk_rating"] + rng.uniform(0.07, 0.2, len(duplicate_rows)), 0, 1)
    duplicate_rows["historic_late_payments"] = duplicate_rows["historic_late_payments"] + rng.integers(2, 6, len(duplicate_rows))
    duplicate_rows["prior_financing_count"] = duplicate_rows["prior_financing_count"] + rng.integers(1, 4, len(duplicate_rows))
    duplicate_rows["is_fraud"] = 1
    duplicate_rows["fraud_type"] = "duplicate_financing"
    fraud_chunks.append(duplicate_rows)

    collusion_seed = legit_sample.sample(
        n=min(collusion_count, len(legit_sample)),
        replace=len(legit_sample) < collusion_count,
        random_state=seed + 19,
    ).reset_index(drop=True)
    collusion_rows = collusion_seed.copy()
    collusion_rows["transaction_id"] = [f"hyb_col_{index:06d}" for index in range(len(collusion_rows))]
    collusion_rows["invoice_id"] = [
        f"collusive_{buyer}_{supplier}_{index:05d}"
        for index, (buyer, supplier) in enumerate(zip(collusion_rows["buyer_id"], collusion_rows["supplier_id"]))
    ]
    collusion_rows["invoice_amount"] = (collusion_rows["invoice_amount"] * rng.uniform(1.22, 1.7, len(collusion_rows))).round(2)
    collusion_rows["loan_amount"] = (collusion_rows["invoice_amount"] * rng.uniform(0.9, 0.99, len(collusion_rows))).round(2)
    collusion_rows["shipment_distance_km"] = (collusion_rows["shipment_distance_km"] * rng.uniform(0.15, 0.55, len(collusion_rows))).round(2)
    collusion_rows["payment_term_days"] = rng.choice([7, 10, 14, 15], len(collusion_rows))
    collusion_rows["due_date"] = pd.to_datetime(collusion_rows["invoice_date"]) + pd.to_timedelta(
        collusion_rows["payment_term_days"],
        unit="D",
    )
    collusion_rows["channel"] = "manual"
    collusion_rows["buyer_risk_rating"] = np.clip(collusion_rows["buyer_risk_rating"] + rng.uniform(0.18, 0.32, len(collusion_rows)), 0, 1)
    collusion_rows["supplier_risk_rating"] = np.clip(collusion_rows["supplier_risk_rating"] + rng.uniform(0.18, 0.34, len(collusion_rows)), 0, 1)
    collusion_rows["historic_late_payments"] = collusion_rows["historic_late_payments"] + rng.integers(4, 10, len(collusion_rows))
    collusion_rows["prior_financing_count"] = collusion_rows["prior_financing_count"] + rng.integers(5, 14, len(collusion_rows))
    collusion_rows["is_fraud"] = 1
    collusion_rows["fraud_type"] = "buyer_supplier_collusion"
    fraud_chunks.append(collusion_rows)

    expansion_seed = legit_sample.sample(
        n=min(expansion_count, len(legit_sample)),
        replace=len(legit_sample) < expansion_count,
        random_state=seed + 29,
    ).reset_index(drop=True)
    expansion_rows = expansion_seed.copy()
    expansion_rows["transaction_id"] = [f"hyb_exp_{index:06d}" for index in range(len(expansion_rows))]
    expansion_rows["supplier_id"] = "RAPID_SUP_" + pd.Series(rng.integers(1, 45, len(expansion_rows))).astype(str).str.zfill(3)
    expansion_rows["invoice_id"] = [f"rapid_growth_{index:06d}" for index in range(len(expansion_rows))]
    expansion_rows["payment_term_days"] = rng.choice([5, 7, 9, 12], len(expansion_rows))
    expansion_rows["invoice_date"] = pd.to_datetime(expansion_rows["invoice_date"]) + pd.to_timedelta(
        rng.integers(0, 14, len(expansion_rows)),
        unit="D",
    )
    expansion_rows["due_date"] = pd.to_datetime(expansion_rows["invoice_date"]) + pd.to_timedelta(
        expansion_rows["payment_term_days"],
        unit="D",
    )
    expansion_rows["loan_amount"] = (expansion_rows["invoice_amount"] * rng.uniform(0.92, 1.0, len(expansion_rows))).round(2)
    expansion_rows["buyer_risk_rating"] = np.clip(expansion_rows["buyer_risk_rating"] + rng.uniform(0.1, 0.24, len(expansion_rows)), 0, 1)
    expansion_rows["supplier_risk_rating"] = np.clip(expansion_rows["supplier_risk_rating"] + rng.uniform(0.2, 0.38, len(expansion_rows)), 0, 1)
    expansion_rows["historic_late_payments"] = expansion_rows["historic_late_payments"] + rng.integers(2, 7, len(expansion_rows))
    expansion_rows["channel"] = rng.choice(["manual", "portal"], len(expansion_rows), p=[0.75, 0.25])
    expansion_rows["is_fraud"] = 1
    expansion_rows["fraud_type"] = "rapid_network_expansion"
    fraud_chunks.append(expansion_rows)

    fraud_frame = pd.concat(fraud_chunks, ignore_index=True)
    hybrid = pd.concat([legit_sample, fraud_frame], ignore_index=True)
    hybrid = hybrid.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return normalize_transactions(hybrid.iloc[:rows].copy())
