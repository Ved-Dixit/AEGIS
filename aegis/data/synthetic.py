"""Synthetic supply-chain data generator aligned with the AEGIS synopsis."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def _entity_list(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index:03d}" for index in range(1, count + 1)]


def generate_synthetic_supply_chain_dataset(
    rows: int = 8_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a demo dataset that matches the fraud modes described in the PDF.

    Public supply-chain fraud datasets are scarce, so this generator intentionally
    creates realistic buyer-supplier-lender behaviour and then injects several
    fraud patterns: duplicate financing, collusion, anomalous spikes, and rapid
    network expansion. The downstream models can therefore learn the exact shape
    of the AEGIS problem instead of a generic credit-card task.
    """

    rng = np.random.default_rng(seed)

    buyers = _entity_list("BUY", 70)
    suppliers = _entity_list("SUP", 110)
    lenders = _entity_list("LEND", 18)
    products = _entity_list("PROD", 45)
    channels = np.array(["portal", "edi", "manual"])
    currencies = np.array(["USD", "EUR", "INR"])
    regions = np.array(["north", "south", "west", "east"])

    product_price = {
        product: float(rng.uniform(20, 180))
        for product in products
    }
    supplier_region = {
        supplier: str(rng.choice(regions))
        for supplier in suppliers
    }
    buyer_region = {
        buyer: str(rng.choice(regions))
        for buyer in buyers
    }

    preferred_suppliers = {
        buyer: list(rng.choice(suppliers, size=rng.integers(5, 9), replace=False))
        for buyer in buyers
    }

    start_date = pd.Timestamp("2024-01-01")
    base_count = max(rows - max(rows // 7, 250), 1)

    records: list[dict[str, object]] = []
    for index in range(base_count):
        buyer_id = str(rng.choice(buyers))
        buyer_pool = preferred_suppliers[buyer_id]
        supplier_id = str(rng.choice(buyer_pool if rng.random() < 0.82 else suppliers))
        product_id = str(rng.choice(products))
        lender_id = str(rng.choice(lenders))

        base_qty = max(rng.lognormal(mean=4.2, sigma=0.35), 12.0)
        price = product_price[product_id] * rng.normal(1.0, 0.06)
        invoice_amount = float(base_qty * max(price, 5.0))
        loan_ratio = float(rng.uniform(0.68, 0.9))
        loan_amount = invoice_amount * loan_ratio

        invoice_date = start_date + timedelta(days=int(rng.integers(0, 420)))
        payment_term_days = int(rng.choice([15, 30, 45, 60], p=[0.12, 0.46, 0.25, 0.17]))
        due_date = invoice_date + timedelta(days=payment_term_days)
        channel = str(rng.choice(channels, p=[0.52, 0.36, 0.12]))
        currency = str(rng.choice(currencies, p=[0.45, 0.12, 0.43]))
        distance = float(abs(rng.normal(240, 95)))
        buyer_risk = float(np.clip(rng.beta(2.0, 12.0), 0.01, 0.95))
        supplier_risk = float(np.clip(rng.beta(2.2, 12.0), 0.01, 0.95))
        late_payments = int(rng.poisson(1.1))
        prior_financing = int(rng.poisson(4.5))

        if buyer_region[buyer_id] == supplier_region[supplier_id]:
            distance *= 0.55

        records.append(
            {
                "transaction_id": f"txn_{index:06d}",
                "invoice_id": f"inv_{index:06d}",
                "buyer_id": buyer_id,
                "supplier_id": supplier_id,
                "lender_id": lender_id,
                "product_id": product_id,
                "quantity": float(round(base_qty, 2)),
                "unit_price": float(round(max(price, 5.0), 2)),
                "invoice_amount": float(round(invoice_amount, 2)),
                "loan_amount": float(round(loan_amount, 2)),
                "invoice_date": invoice_date.normalize(),
                "due_date": due_date.normalize(),
                "payment_term_days": payment_term_days,
                "shipment_distance_km": float(round(distance, 2)),
                "buyer_risk_rating": round(buyer_risk, 4),
                "supplier_risk_rating": round(supplier_risk, 4),
                "historic_late_payments": late_payments,
                "prior_financing_count": prior_financing,
                "channel": channel,
                "currency": currency,
                "is_fraud": 0,
                "fraud_type": "legit",
            }
        )

    base_frame = pd.DataFrame(records)

    fraud_records: list[dict[str, object]] = []

    duplicate_seeds = base_frame.sample(
        n=min(max(rows // 16, 120), len(base_frame)),
        random_state=seed,
    )
    for offset, (_, row) in enumerate(duplicate_seeds.iterrows(), start=1):
        duplicate = row.to_dict()
        alternate_lenders = [lender for lender in lenders if lender != row["lender_id"]]
        duplicate["transaction_id"] = f"fraud_dup_{offset:05d}"
        duplicate["lender_id"] = str(rng.choice(alternate_lenders))
        duplicate["channel"] = "manual"
        duplicate["loan_amount"] = float(round(float(row["loan_amount"]) * rng.uniform(0.96, 1.05), 2))
        duplicate["invoice_date"] = pd.Timestamp(row["invoice_date"]) + timedelta(days=int(rng.integers(0, 3)))
        duplicate["due_date"] = pd.Timestamp(duplicate["invoice_date"]) + timedelta(days=int(row["payment_term_days"]))
        duplicate["buyer_risk_rating"] = round(min(float(row["buyer_risk_rating"]) + rng.uniform(0.12, 0.3), 0.98), 4)
        duplicate["supplier_risk_rating"] = round(min(float(row["supplier_risk_rating"]) + rng.uniform(0.1, 0.24), 0.98), 4)
        duplicate["historic_late_payments"] = int(row["historic_late_payments"]) + int(rng.integers(2, 6))
        duplicate["prior_financing_count"] = int(row["prior_financing_count"]) + int(rng.integers(1, 4))
        duplicate["is_fraud"] = 1
        duplicate["fraud_type"] = "duplicate_financing"
        fraud_records.append(duplicate)

    collusive_pairs = [
        (str(rng.choice(buyers)), str(rng.choice(suppliers)))
        for _ in range(max(rows // 50, 70))
    ]
    for offset, (buyer_id, supplier_id) in enumerate(collusive_pairs, start=1):
        product_id = str(rng.choice(products))
        quantity = float(round(max(rng.lognormal(mean=4.5, sigma=0.22), 30.0), 2))
        unit_price = float(round(product_price[product_id] * rng.uniform(1.25, 1.7), 2))
        invoice_amount = float(round(quantity * unit_price, 2))
        loan_amount = float(round(invoice_amount * rng.uniform(0.88, 0.98), 2))
        invoice_date = start_date + timedelta(days=int(rng.integers(150, 420)))
        payment_term_days = int(rng.choice([15, 21, 30], p=[0.5, 0.35, 0.15]))
        fraud_records.append(
            {
                "transaction_id": f"fraud_collusion_{offset:05d}",
                "invoice_id": f"collusion_inv_{buyer_id}_{supplier_id}_{offset:04d}",
                "buyer_id": buyer_id,
                "supplier_id": supplier_id,
                "lender_id": str(rng.choice(lenders)),
                "product_id": product_id,
                "quantity": quantity,
                "unit_price": unit_price,
                "invoice_amount": invoice_amount,
                "loan_amount": loan_amount,
                "invoice_date": invoice_date.normalize(),
                "due_date": (invoice_date + timedelta(days=payment_term_days)).normalize(),
                "payment_term_days": payment_term_days,
                "shipment_distance_km": float(round(abs(rng.normal(55, 18)), 2)),
                "buyer_risk_rating": round(float(np.clip(rng.beta(4.5, 6.0), 0.2, 0.98)), 4),
                "supplier_risk_rating": round(float(np.clip(rng.beta(5.0, 5.5), 0.18, 0.98)), 4),
                "historic_late_payments": int(rng.integers(4, 11)),
                "prior_financing_count": int(rng.integers(8, 20)),
                "channel": "manual",
                "currency": str(rng.choice(currencies)),
                "is_fraud": 1,
                "fraud_type": "buyer_supplier_collusion",
            }
        )

    burst_suppliers = list(rng.choice(suppliers, size=max(rows // 80, 40), replace=False))
    for offset, supplier_id in enumerate(burst_suppliers, start=1):
        for hop in range(int(rng.integers(2, 5))):
            buyer_id = str(rng.choice(buyers))
            product_id = str(rng.choice(products))
            quantity = float(round(max(rng.lognormal(mean=4.1, sigma=0.28), 18.0), 2))
            unit_price = float(round(product_price[product_id] * rng.uniform(0.95, 1.18), 2))
            invoice_amount = float(round(quantity * unit_price, 2))
            invoice_date = start_date + timedelta(days=int(rng.integers(250, 420)))
            payment_term_days = int(rng.choice([7, 10, 14], p=[0.45, 0.35, 0.2]))
            fraud_records.append(
                {
                    "transaction_id": f"fraud_burst_{offset:04d}_{hop:02d}",
                    "invoice_id": f"burst_inv_{supplier_id}_{offset:04d}_{hop:02d}",
                    "buyer_id": buyer_id,
                    "supplier_id": supplier_id,
                    "lender_id": str(rng.choice(lenders)),
                    "product_id": product_id,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "invoice_amount": invoice_amount,
                    "loan_amount": float(round(invoice_amount * rng.uniform(0.9, 0.99), 2)),
                    "invoice_date": invoice_date.normalize(),
                    "due_date": (invoice_date + timedelta(days=payment_term_days)).normalize(),
                    "payment_term_days": payment_term_days,
                    "shipment_distance_km": float(round(abs(rng.normal(110, 32)), 2)),
                    "buyer_risk_rating": round(float(np.clip(rng.beta(3.2, 7.5), 0.12, 0.9)), 4),
                    "supplier_risk_rating": round(float(np.clip(rng.beta(5.2, 4.4), 0.25, 0.99)), 4),
                    "historic_late_payments": int(rng.integers(3, 9)),
                    "prior_financing_count": int(rng.integers(1, 6)),
                    "channel": str(rng.choice(["manual", "portal"], p=[0.7, 0.3])),
                    "currency": str(rng.choice(currencies)),
                    "is_fraud": 1,
                    "fraud_type": "rapid_network_expansion",
                }
            )

    fraud_frame = pd.DataFrame(fraud_records)
    frame = pd.concat([base_frame, fraud_frame], ignore_index=True)
    if len(frame) < rows:
        missing = rows - len(frame)
        # Top up with extra legitimate trades so the caller always gets the requested size.
        extra_frame = base_frame.sample(n=missing, replace=True, random_state=seed).copy().reset_index(drop=True)
        extra_frame["transaction_id"] = [f"txn_extra_{index:05d}" for index in range(1, missing + 1)]
        extra_frame["invoice_id"] = [f"inv_extra_{index:05d}" for index in range(1, missing + 1)]
        extra_offsets = rng.integers(0, 21, size=missing)
        extra_frame["invoice_date"] = pd.to_datetime(extra_frame["invoice_date"]) + pd.to_timedelta(extra_offsets, unit="D")
        extra_frame["due_date"] = pd.to_datetime(extra_frame["invoice_date"]) + pd.to_timedelta(
            extra_frame["payment_term_days"],
            unit="D",
        )
        frame = pd.concat([frame, extra_frame], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return frame.iloc[:rows].copy()
