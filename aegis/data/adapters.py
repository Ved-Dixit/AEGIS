"""Dataset adapters that map public fraud datasets into the AEGIS schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis.features import normalize_transactions

AEGIS_COLUMNS = [
    "transaction_id",
    "invoice_id",
    "buyer_id",
    "supplier_id",
    "lender_id",
    "product_id",
    "quantity",
    "unit_price",
    "invoice_amount",
    "loan_amount",
    "invoice_date",
    "due_date",
    "payment_term_days",
    "shipment_distance_km",
    "buyer_risk_rating",
    "supplier_risk_rating",
    "historic_late_payments",
    "prior_financing_count",
    "channel",
    "currency",
    "is_fraud",
    "fraud_type",
]


def _scaled_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    low = float(numeric.quantile(0.05))
    high = float(max(numeric.quantile(0.95), low + 1e-6))
    return ((numeric - low) / (high - low)).clip(0.0, 1.0)


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    finalized = normalize_transactions(frame)
    for column in AEGIS_COLUMNS:
        if column not in finalized:
            finalized[column] = np.nan
    return finalized[AEGIS_COLUMNS].copy()


def adapt_credit_card_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the Kaggle MLG-ULB credit card benchmark into AEGIS fields."""

    working = frame.copy()
    amount = pd.to_numeric(working.get("Amount"), errors="coerce").fillna(0.0).clip(lower=1.0)
    seconds = pd.to_numeric(working.get("Time"), errors="coerce").fillna(0)
    base_date = pd.Timestamp("2024-01-01")
    invoice_date = base_date + pd.to_timedelta(seconds.astype(int), unit="s")

    v1 = pd.to_numeric(working.get("V1"), errors="coerce").fillna(0.0)
    v2 = pd.to_numeric(working.get("V2"), errors="coerce").fillna(0.0)
    v14 = pd.to_numeric(working.get("V14"), errors="coerce").fillna(0.0)
    v17 = pd.to_numeric(working.get("V17"), errors="coerce").fillna(0.0)

    adapted = pd.DataFrame(
        {
            "transaction_id": [f"cc_txn_{index:07d}" for index in range(len(working))],
            "invoice_id": [f"cc_inv_{index:07d}" for index in range(len(working))],
            "buyer_id": "CARD_" + pd.qcut(v1.rank(method="first"), q=32, labels=False, duplicates="drop").astype(str),
            "supplier_id": "MERCHANT_" + pd.qcut(v2.rank(method="first"), q=32, labels=False, duplicates="drop").astype(str),
            "lender_id": "CARD_NETWORK_001",
            "product_id": "CC_TXN",
            "quantity": 1.0,
            "unit_price": amount,
            "invoice_amount": amount,
            "loan_amount": (amount * 0.94).round(2),
            "invoice_date": invoice_date,
            "due_date": invoice_date + pd.Timedelta(days=30),
            "payment_term_days": 30,
            "shipment_distance_km": 0.0,
            "buyer_risk_rating": _scaled_score(v14.abs() + v17.abs()),
            "supplier_risk_rating": _scaled_score(v2.abs() + v1.abs()),
            "historic_late_payments": (seconds // 18_000).astype(int) % 6,
            "prior_financing_count": (amount.rank(method="dense") % 12).astype(int),
            "channel": "card_network",
            "currency": "USD",
            "is_fraud": pd.to_numeric(working.get("Class"), errors="coerce").fillna(0).astype(int),
            "fraud_type": np.where(
                pd.to_numeric(working.get("Class"), errors="coerce").fillna(0).astype(int) == 1,
                "card_payment_fraud",
                "legit",
            ),
        }
    )
    return _finalize(adapted)


def adapt_paysim_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Map PaySim mobile-money events into the AEGIS schema."""

    working = frame.copy()
    amount = pd.to_numeric(working.get("amount"), errors="coerce").fillna(0.0).clip(lower=1.0)
    step = pd.to_numeric(working.get("step"), errors="coerce").fillna(0).astype(int)
    base_date = pd.Timestamp("2024-01-01")
    invoice_date = base_date + pd.to_timedelta(step, unit="h")
    old_balance_org = pd.to_numeric(working.get("oldbalanceOrg"), errors="coerce").fillna(0.0)
    new_balance_org = pd.to_numeric(working.get("newbalanceOrig"), errors="coerce").fillna(0.0)
    old_balance_dest = pd.to_numeric(working.get("oldbalanceDest"), errors="coerce").fillna(0.0)
    new_balance_dest = pd.to_numeric(working.get("newbalanceDest"), errors="coerce").fillna(0.0)

    adapted = pd.DataFrame(
        {
            "transaction_id": [f"ps_txn_{index:07d}" for index in range(len(working))],
            "invoice_id": [f"ps_inv_{index:07d}" for index in range(len(working))],
            "buyer_id": working.get("nameOrig", pd.Series(["ORIG_UNKNOWN"] * len(working))).astype(str),
            "supplier_id": working.get("nameDest", pd.Series(["DEST_UNKNOWN"] * len(working))).astype(str),
            "lender_id": "PAYSIM_" + working.get("type", pd.Series(["UNKNOWN"] * len(working))).astype(str),
            "product_id": working.get("type", pd.Series(["UNKNOWN"] * len(working))).astype(str),
            "quantity": 1.0,
            "unit_price": amount,
            "invoice_amount": amount,
            "loan_amount": (amount * 0.9).round(2),
            "invoice_date": invoice_date,
            "due_date": invoice_date + pd.Timedelta(days=1),
            "payment_term_days": 1,
            "shipment_distance_km": 0.0,
            "buyer_risk_rating": _scaled_score((old_balance_org - new_balance_org).abs() + amount),
            "supplier_risk_rating": _scaled_score((new_balance_dest - old_balance_dest).abs() + amount),
            "historic_late_payments": (step % 8).astype(int),
            "prior_financing_count": pd.qcut(amount.rank(method="first"), q=10, labels=False, duplicates="drop").astype(int),
            "channel": "mobile_money",
            "currency": "USD",
            "is_fraud": pd.to_numeric(working.get("isFraud"), errors="coerce").fillna(0).astype(int),
            "fraud_type": np.where(
                pd.to_numeric(working.get("isFraud"), errors="coerce").fillna(0).astype(int) == 1,
                "mobile_money_fraud",
                "legit",
            ),
        }
    )
    return _finalize(adapted)


def adapt_ieee_dataset(transactions: pd.DataFrame, identities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map IEEE-CIS transactions into the AEGIS schema."""

    working = transactions.copy()
    if identities is not None and "TransactionID" in identities:
        identity_subset = identities.drop_duplicates("TransactionID")
        working = working.merge(identity_subset, on="TransactionID", how="left", suffixes=("", "_identity"))

    amount = pd.to_numeric(working.get("TransactionAmt"), errors="coerce").fillna(0.0).clip(lower=1.0)
    txn_seconds = pd.to_numeric(working.get("TransactionDT"), errors="coerce").fillna(0).astype(int)
    base_date = pd.Timestamp("2024-01-01")
    invoice_date = base_date + pd.to_timedelta(txn_seconds, unit="s")
    dist1 = pd.to_numeric(working.get("dist1"), errors="coerce").fillna(0.0)
    dist2 = pd.to_numeric(working.get("dist2"), errors="coerce").fillna(0.0)

    buyer_id = (
        "CARD_"
        + working.get("card1", pd.Series(["0000"] * len(working))).astype(str)
        + "_"
        + working.get("card2", pd.Series(["00"] * len(working))).fillna("00").astype(str)
    )
    supplier_id = (
        "ADDR_"
        + working.get("addr1", pd.Series(["000"] * len(working))).fillna("000").astype(str)
        + "_"
        + working.get("ProductCD", pd.Series(["UNK"] * len(working))).fillna("UNK").astype(str)
    )
    lender_id = "CARD6_" + working.get("card6", pd.Series(["unknown"] * len(working))).fillna("unknown").astype(str)
    product_id = working.get("ProductCD", pd.Series(["UNKNOWN"] * len(working))).fillna("UNKNOWN").astype(str)

    adapted = pd.DataFrame(
        {
            "transaction_id": "ieee_" + working.get("TransactionID", pd.Series(range(len(working)))).astype(str),
            "invoice_id": "ieee_inv_" + working.get("TransactionID", pd.Series(range(len(working)))).astype(str),
            "buyer_id": buyer_id,
            "supplier_id": supplier_id,
            "lender_id": lender_id,
            "product_id": product_id,
            "quantity": 1.0,
            "unit_price": amount,
            "invoice_amount": amount,
            "loan_amount": (amount * 0.92).round(2),
            "invoice_date": invoice_date,
            "due_date": invoice_date + pd.Timedelta(days=21),
            "payment_term_days": 21,
            "shipment_distance_km": (dist1.fillna(0.0) + dist2.fillna(0.0)).clip(lower=0.0),
            "buyer_risk_rating": _scaled_score(
                pd.to_numeric(working.get("D1"), errors="coerce").fillna(0.0).abs()
                + pd.to_numeric(working.get("D4"), errors="coerce").fillna(0.0).abs()
            ),
            "supplier_risk_rating": _scaled_score(dist1.abs() + dist2.abs()),
            "historic_late_payments": pd.to_numeric(working.get("D15"), errors="coerce").fillna(0).clip(lower=0).astype(int),
            "prior_financing_count": pd.to_numeric(working.get("card3"), errors="coerce").fillna(0).clip(lower=0).astype(int) % 15,
            "channel": working.get("DeviceType", pd.Series(["web"] * len(working))).fillna("web").astype(str).str.lower(),
            "currency": "USD",
            "is_fraud": pd.to_numeric(working.get("isFraud"), errors="coerce").fillna(0).astype(int),
            "fraud_type": np.where(
                pd.to_numeric(working.get("isFraud"), errors="coerce").fillna(0).astype(int) == 1,
                "transaction_identity_fraud",
                "legit",
            ),
        }
    )
    return _finalize(adapted)


def adapt_dataco_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Map DataCo supply-chain operations data into AEGIS fields.

    DataCo is not fraud-labeled, so the adapter keeps `is_fraud` at zero and
    exposes operational-risk proxies via the buyer/supplier risk fields. This
    makes it useful for graph enrichment, anomaly calibration, and UI demos.
    """

    working = frame.copy()
    invoice_date = pd.to_datetime(working.get("order date (DateOrders)"), errors="coerce")
    shipping_date = pd.to_datetime(working.get("shipping date (DateOrders)"), errors="coerce")
    quantity = pd.to_numeric(working.get("Order Item Quantity"), errors="coerce").fillna(1.0).clip(lower=1.0)
    price = pd.to_numeric(working.get("Order Item Product Price"), errors="coerce").fillna(
        pd.to_numeric(working.get("Product Price"), errors="coerce").fillna(1.0)
    )
    total = pd.to_numeric(working.get("Order Item Total"), errors="coerce").fillna(
        pd.to_numeric(working.get("Sales"), errors="coerce").fillna(quantity * price)
    )
    late_risk = pd.to_numeric(working.get("Late_delivery_risk"), errors="coerce").fillna(0.0)
    profit = pd.to_numeric(working.get("Order Profit Per Order"), errors="coerce").fillna(0.0)

    adapted = pd.DataFrame(
        {
            "transaction_id": "dataco_" + working.get("Order Item Id", pd.Series(range(len(working)))).astype(str),
            "invoice_id": "dataco_inv_" + working.get("Order Id", pd.Series(range(len(working)))).astype(str),
            "buyer_id": "CUSTOMER_" + working.get("Customer Id", working.get("Order Customer Id", pd.Series(["unknown"] * len(working)))).astype(str),
            "supplier_id": (
                "SUPPLY_"
                + working.get("Department Id", pd.Series(["0"] * len(working))).astype(str)
                + "_"
                + working.get("Category Id", pd.Series(["0"] * len(working))).astype(str)
            ),
            "lender_id": "PAYMENT_" + working.get("Type", pd.Series(["unknown"] * len(working))).astype(str),
            "product_id": "PRODUCT_" + working.get("Product Card Id", pd.Series(["0"] * len(working))).astype(str),
            "quantity": quantity,
            "unit_price": price,
            "invoice_amount": total.abs(),
            "loan_amount": (total.abs() * 0.82).round(2),
            "invoice_date": invoice_date,
            "due_date": shipping_date,
            "payment_term_days": pd.to_numeric(working.get("Days for shipment (scheduled)"), errors="coerce").fillna(7).clip(lower=1),
            "shipment_distance_km": 0.0,
            "buyer_risk_rating": _scaled_score(late_risk + profit.lt(0).astype(float)),
            "supplier_risk_rating": _scaled_score(pd.to_numeric(working.get("Benefit per order"), errors="coerce").fillna(0.0).abs()),
            "historic_late_payments": late_risk.astype(int),
            "prior_financing_count": pd.qcut(total.abs().rank(method="first"), q=10, labels=False, duplicates="drop").astype(int),
            "channel": working.get("Shipping Mode", pd.Series(["standard"] * len(working))).fillna("standard").astype(str).str.lower(),
            "currency": "USD",
            "is_fraud": 0,
            "fraud_type": "supply_chain_activity",
        }
    )
    return _finalize(adapted)


def adapt_elliptic_dataset(features: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """
    Map Elliptic++ transactions into AEGIS fields.

    The raw dataset is graph-first, so several business values are proxy values
    derived from feature magnitude and timestep. That still makes it useful for
    graph and anomaly experimentation inside the AEGIS schema.
    """

    features_frame = features.copy()
    classes_frame = classes.copy()

    if "txId" not in features_frame.columns:
        features_frame = features_frame.copy()
        features_frame.columns = ["txId", "time_step"] + [f"feature_{index:03d}" for index in range(1, len(features_frame.columns) - 1)]
    if "txId" not in classes_frame.columns:
        classes_frame = classes_frame.copy()
        classes_frame.columns = ["txId", "class"]

    merged = features_frame.merge(classes_frame, on="txId", how="left")
    time_step_series = merged["time_step"] if "time_step" in merged.columns else merged.get("Time step")
    if time_step_series is None:
        time_step_series = pd.Series(np.zeros(len(merged)))
    time_step = pd.to_numeric(time_step_series, errors="coerce").fillna(0).astype(int)
    base_date = pd.Timestamp("2024-01-01")
    invoice_date = base_date + pd.to_timedelta(time_step, unit="D")

    numeric_features = merged.select_dtypes(include=[np.number]).drop(columns=["time_step", "Time step"], errors="ignore")
    amount_proxy = numeric_features.abs().sum(axis=1).replace(0, np.nan).fillna(1.0)
    illicit = merged.get("class", pd.Series(["unknown"] * len(merged))).astype(str).str.lower().isin({"1", "illicit"})

    first_feature = numeric_features.iloc[:, 0] if not numeric_features.empty else pd.Series(np.zeros(len(merged)))
    second_feature = numeric_features.iloc[:, 1] if numeric_features.shape[1] > 1 else first_feature

    adapted = pd.DataFrame(
        {
            "transaction_id": "elliptic_" + merged["txId"].astype(str),
            "invoice_id": "elliptic_inv_" + merged["txId"].astype(str),
            "buyer_id": "WALLET_" + pd.qcut(first_feature.rank(method="first"), q=32, labels=False, duplicates="drop").astype(str),
            "supplier_id": "COUNTERPARTY_" + pd.qcut(second_feature.rank(method="first"), q=32, labels=False, duplicates="drop").astype(str),
            "lender_id": "BLOCKCHAIN_" + (time_step % 12).astype(str),
            "product_id": "BTC_TRANSFER",
            "quantity": 1.0,
            "unit_price": amount_proxy,
            "invoice_amount": amount_proxy,
            "loan_amount": amount_proxy,
            "invoice_date": invoice_date,
            "due_date": invoice_date + pd.Timedelta(days=1),
            "payment_term_days": 1,
            "shipment_distance_km": 0.0,
            "buyer_risk_rating": _scaled_score(amount_proxy),
            "supplier_risk_rating": _scaled_score(second_feature.abs()),
            "historic_late_payments": (time_step % 5).astype(int),
            "prior_financing_count": pd.qcut(amount_proxy.rank(method="first"), q=10, labels=False, duplicates="drop").astype(int),
            "channel": "blockchain",
            "currency": "BTC",
            "is_fraud": illicit.astype(int),
            "fraud_type": np.where(illicit, "crypto_illicit_flow", "legit"),
        }
    )
    return _finalize(adapted)


def prepare_dataset(
    source_name: str,
    input_paths: list[str | Path],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load source files, adapt them to AEGIS, and optionally persist a CSV."""

    paths = [Path(path) for path in input_paths]
    source_key = source_name.strip().lower()

    if source_key == "creditcardfraud":
        adapted = adapt_credit_card_dataset(pd.read_csv(paths[0]))
        notes = "Mapped the Kaggle MLG-ULB benchmark into AEGIS transaction fields."
    elif source_key == "paysim":
        adapted = adapt_paysim_dataset(pd.read_csv(paths[0]))
        notes = "Mapped PaySim mobile-money records into the AEGIS schema."
    elif source_key == "ieee_cis":
        transactions = pd.read_csv(paths[0])
        identities = pd.read_csv(paths[1]) if len(paths) > 1 else None
        adapted = adapt_ieee_dataset(transactions, identities)
        notes = "Mapped IEEE-CIS transaction data into supply-chain-style AEGIS records."
    elif source_key == "dataco":
        adapted = adapt_dataco_dataset(pd.read_csv(paths[0], encoding="latin1"))
        notes = "Mapped DataCo operations data for graph/anomaly enrichment; labels remain non-fraud."
    elif source_key == "ellipticplusplus":
        features = pd.read_csv(paths[0])
        classes = pd.read_csv(paths[1])
        adapted = adapt_elliptic_dataset(features, classes)
        notes = "Mapped Elliptic++ graph transactions into AEGIS fields using proxy business values."
    else:
        raise ValueError(f"Unsupported source '{source_name}'.")

    destination = Path(output_path) if output_path else paths[0].with_name(f"{source_key}_aegis_prepared.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_csv(destination, index=False)

    label_summary = {
        str(key): int(value)
        for key, value in adapted["is_fraud"].value_counts().sort_index().to_dict().items()
    }
    return {
        "source_name": source_key,
        "input_paths": [str(path) for path in paths],
        "output_path": str(destination),
        "row_count": int(len(adapted)),
        "fraud_rows": int(adapted["is_fraud"].sum()),
        "label_summary": label_summary,
        "notes": notes,
    }
