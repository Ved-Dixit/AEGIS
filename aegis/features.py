"""Feature engineering and graph analytics for AEGIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    values = numerator / denominator
    return values.replace([np.inf, -np.inf], np.nan).fillna(default)


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types and fill predictable defaults.

    This keeps API, Streamlit, CSV, and synthetic records on the same schema.
    """

    normalized = frame.copy()
    defaults: dict[str, Any] = {
        "transaction_id": "txn_unknown",
        "invoice_id": "inv_unknown",
        "buyer_id": "BUY_UNKNOWN",
        "supplier_id": "SUP_UNKNOWN",
        "lender_id": "LEND_UNKNOWN",
        "product_id": "PROD_UNKNOWN",
        "quantity": 1.0,
        "unit_price": 1.0,
        "invoice_amount": np.nan,
        "loan_amount": np.nan,
        "payment_term_days": 30,
        "shipment_distance_km": 0.0,
        "buyer_risk_rating": 0.15,
        "supplier_risk_rating": 0.15,
        "historic_late_payments": 0,
        "prior_financing_count": 0,
        "channel": "portal",
        "currency": "USD",
        "is_fraud": 0,
        "fraud_type": "unknown",
    }
    for column, default in defaults.items():
        if column not in normalized:
            normalized[column] = default
        normalized[column] = normalized[column].fillna(default)

    if "invoice_date" not in normalized:
        normalized["invoice_date"] = pd.NaT
    if "due_date" not in normalized:
        normalized["due_date"] = pd.NaT

    normalized["invoice_date"] = pd.to_datetime(normalized["invoice_date"], errors="coerce")
    normalized["due_date"] = pd.to_datetime(normalized["due_date"], errors="coerce")
    normalized["invoice_date"] = normalized["invoice_date"].fillna(pd.Timestamp("2025-01-01"))
    normalized["due_date"] = normalized["due_date"].fillna(
        normalized["invoice_date"] + pd.to_timedelta(normalized["payment_term_days"], unit="D")
    )

    numeric_columns = [
        "quantity",
        "unit_price",
        "invoice_amount",
        "loan_amount",
        "payment_term_days",
        "shipment_distance_km",
        "buyer_risk_rating",
        "supplier_risk_rating",
        "historic_late_payments",
        "prior_financing_count",
        "is_fraud",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized["invoice_amount"] = normalized["invoice_amount"].fillna(
        normalized["quantity"] * normalized["unit_price"]
    )
    normalized["loan_amount"] = normalized["loan_amount"].fillna(normalized["invoice_amount"] * 0.82)
    normalized["payment_term_days"] = normalized["payment_term_days"].fillna(30).clip(lower=1)

    string_columns = [
        "transaction_id",
        "invoice_id",
        "buyer_id",
        "supplier_id",
        "lender_id",
        "product_id",
        "channel",
        "currency",
        "fraud_type",
    ]
    for column in string_columns:
        normalized[column] = normalized[column].astype(str)

    return normalized


def _pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["buyer_id"].astype(str) + "::" + frame["supplier_id"].astype(str)


@dataclass(slots=True)
class GraphFeatureBuilder:
    """NetworkX-powered graph intelligence layer used by the ensemble."""

    degree_centrality_: dict[str, float] = field(default_factory=dict)
    pagerank_: dict[str, float] = field(default_factory=dict)
    clustering_: dict[str, float] = field(default_factory=dict)
    pair_stats_: pd.DataFrame | None = None
    scale_: dict[str, float] = field(default_factory=dict)

    graph_columns: tuple[str, ...] = (
        "buyer_degree",
        "supplier_degree",
        "lender_degree",
        "buyer_pagerank",
        "supplier_pagerank",
        "supplier_clustering",
        "pair_txn_count",
        "pair_total_amount",
        "pair_unique_lenders",
        "pair_unique_products",
        "graph_rule_score",
    )

    def fit(self, frame: pd.DataFrame) -> "GraphFeatureBuilder":
        graph = nx.Graph()

        # The graph mirrors the PDF architecture: buyers, suppliers, and lenders
        # become nodes, while invoices become weighted relationships.
        for row in frame.itertuples(index=False):
            self._add_weighted_edge(graph, f"buyer::{row.buyer_id}", f"supplier::{row.supplier_id}", float(row.invoice_amount))
            self._add_weighted_edge(graph, f"supplier::{row.supplier_id}", f"lender::{row.lender_id}", float(row.loan_amount))
            self._add_weighted_edge(graph, f"buyer::{row.buyer_id}", f"lender::{row.lender_id}", float(row.loan_amount) * 0.2)

        self.degree_centrality_ = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
        self.pagerank_ = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
        self.clustering_ = nx.clustering(graph, weight="weight") if graph.number_of_nodes() else {}

        pair_frame = (
            frame.assign(buyer_supplier_pair=_pair_key(frame))
            .groupby("buyer_supplier_pair", as_index=True)
            .agg(
                pair_txn_count=("transaction_id", "count"),
                pair_total_amount=("invoice_amount", "sum"),
                pair_unique_lenders=("lender_id", "nunique"),
                pair_unique_products=("product_id", "nunique"),
            )
        )
        self.pair_stats_ = pair_frame
        self.scale_ = {
            "pair_txn_count": float(max(pair_frame["pair_txn_count"].quantile(0.95), 1.0)),
            "pair_total_amount": float(max(pair_frame["pair_total_amount"].quantile(0.95), 1.0)),
            "pair_unique_lenders": float(max(pair_frame["pair_unique_lenders"].quantile(0.95), 1.0)),
            "buyer_degree": float(max(self.degree_centrality_.values(), default=1.0)),
            "supplier_pagerank": float(max(self.pagerank_.values(), default=1.0)),
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pair_stats_ is None:
            raise RuntimeError("GraphFeatureBuilder must be fitted before transform().")

        pair_key = _pair_key(frame)
        features = pd.DataFrame(index=frame.index)
        features["buyer_degree"] = frame["buyer_id"].map(
            lambda value: self.degree_centrality_.get(f"buyer::{value}", 0.0)
        )
        features["supplier_degree"] = frame["supplier_id"].map(
            lambda value: self.degree_centrality_.get(f"supplier::{value}", 0.0)
        )
        features["lender_degree"] = frame["lender_id"].map(
            lambda value: self.degree_centrality_.get(f"lender::{value}", 0.0)
        )
        features["buyer_pagerank"] = frame["buyer_id"].map(
            lambda value: self.pagerank_.get(f"buyer::{value}", 0.0)
        )
        features["supplier_pagerank"] = frame["supplier_id"].map(
            lambda value: self.pagerank_.get(f"supplier::{value}", 0.0)
        )
        features["supplier_clustering"] = frame["supplier_id"].map(
            lambda value: self.clustering_.get(f"supplier::{value}", 0.0)
        )
        for column in ("pair_txn_count", "pair_total_amount", "pair_unique_lenders", "pair_unique_products"):
            features[column] = pair_key.map(self.pair_stats_[column]).fillna(0.0)

        # This heuristic score gives the ensemble a fast graph-native opinion
        # before the main classifier probability is combined in the decision layer.
        features["graph_rule_score"] = np.clip(
            0.28 * features["pair_txn_count"] / self.scale_["pair_txn_count"]
            + 0.24 * features["pair_total_amount"] / self.scale_["pair_total_amount"]
            + 0.22 * features["pair_unique_lenders"] / self.scale_["pair_unique_lenders"]
            + 0.16 * features["buyer_degree"] / self.scale_["buyer_degree"]
            + 0.10 * features["supplier_pagerank"] / self.scale_["supplier_pagerank"],
            0.0,
            1.0,
        )
        return features[list(self.graph_columns)]

    @staticmethod
    def _add_weighted_edge(graph: nx.Graph, source: str, target: str, weight: float) -> None:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
            graph[source][target]["count"] += 1
            return
        graph.add_edge(source, target, weight=weight, count=1)


@dataclass(slots=True)
class FeatureBuilder:
    """Stateful feature builder so training and inference use identical logic."""

    global_amount_mean_: float = 1.0
    global_unit_price_mean_: float = 1.0
    reference_date_: pd.Timestamp | None = None
    buyer_amount_mean_: dict[str, float] = field(default_factory=dict)
    buyer_count_: dict[str, int] = field(default_factory=dict)
    supplier_amount_mean_: dict[str, float] = field(default_factory=dict)
    supplier_price_mean_: dict[str, float] = field(default_factory=dict)
    supplier_count_: dict[str, int] = field(default_factory=dict)
    product_price_mean_: dict[str, float] = field(default_factory=dict)
    pair_amount_mean_: dict[str, float] = field(default_factory=dict)
    pair_count_: dict[str, int] = field(default_factory=dict)
    pair_lender_count_: dict[str, int] = field(default_factory=dict)
    invoice_count_: dict[str, int] = field(default_factory=dict)
    invoice_lender_count_: dict[str, int] = field(default_factory=dict)
    graph_builder_: GraphFeatureBuilder | None = None

    categorical_columns: tuple[str, ...] = (
        "buyer_id",
        "supplier_id",
        "lender_id",
        "product_id",
        "channel",
        "currency",
        "buyer_supplier_pair",
    )
    numeric_columns: tuple[str, ...] = (
        "quantity",
        "unit_price",
        "invoice_amount",
        "loan_amount",
        "payment_term_days",
        "shipment_distance_km",
        "buyer_risk_rating",
        "supplier_risk_rating",
        "historic_late_payments",
        "prior_financing_count",
        "loan_to_invoice_ratio",
        "buyer_amount_ratio",
        "supplier_amount_ratio",
        "product_price_ratio",
        "pair_amount_ratio",
        "buyer_transaction_count",
        "supplier_transaction_count",
        "pair_transaction_count",
        "pair_lender_count",
        "known_invoice_count",
        "invoice_lender_count",
        "duplicate_invoice_hint",
        "days_to_due",
        "days_from_reference",
        "invoice_month",
        "invoice_weekday",
        "historic_pressure",
        "manual_channel_flag",
    )

    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        normalized = normalize_transactions(frame)
        pair_key = _pair_key(normalized)

        self.global_amount_mean_ = float(normalized["invoice_amount"].mean())
        self.global_unit_price_mean_ = float(normalized["unit_price"].mean())
        self.reference_date_ = normalized["invoice_date"].max()

        self.buyer_amount_mean_ = normalized.groupby("buyer_id")["invoice_amount"].mean().to_dict()
        self.buyer_count_ = normalized.groupby("buyer_id")["transaction_id"].count().to_dict()
        self.supplier_amount_mean_ = normalized.groupby("supplier_id")["invoice_amount"].mean().to_dict()
        self.supplier_price_mean_ = normalized.groupby("supplier_id")["unit_price"].mean().to_dict()
        self.supplier_count_ = normalized.groupby("supplier_id")["transaction_id"].count().to_dict()
        self.product_price_mean_ = normalized.groupby("product_id")["unit_price"].mean().to_dict()
        self.pair_amount_mean_ = normalized.assign(pair_key=pair_key).groupby("pair_key")["invoice_amount"].mean().to_dict()
        self.pair_count_ = normalized.assign(pair_key=pair_key).groupby("pair_key")["transaction_id"].count().to_dict()
        self.pair_lender_count_ = normalized.assign(pair_key=pair_key).groupby("pair_key")["lender_id"].nunique().to_dict()
        self.invoice_count_ = normalized.groupby("invoice_id")["transaction_id"].count().to_dict()
        self.invoice_lender_count_ = normalized.groupby("invoice_id")["lender_id"].nunique().to_dict()

        self.graph_builder_ = GraphFeatureBuilder().fit(normalized)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.graph_builder_ is None or self.reference_date_ is None:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")

        normalized = normalize_transactions(frame)
        pair_key = _pair_key(normalized)

        features = pd.DataFrame(index=normalized.index)
        features["buyer_id"] = normalized["buyer_id"]
        features["supplier_id"] = normalized["supplier_id"]
        features["lender_id"] = normalized["lender_id"]
        features["product_id"] = normalized["product_id"]
        features["channel"] = normalized["channel"]
        features["currency"] = normalized["currency"]
        features["buyer_supplier_pair"] = pair_key

        buyer_amount_mean = normalized["buyer_id"].map(self.buyer_amount_mean_).fillna(self.global_amount_mean_)
        supplier_amount_mean = normalized["supplier_id"].map(self.supplier_amount_mean_).fillna(self.global_amount_mean_)
        product_price_mean = normalized["product_id"].map(self.product_price_mean_).fillna(self.global_unit_price_mean_)
        supplier_price_mean = normalized["supplier_id"].map(self.supplier_price_mean_).fillna(self.global_unit_price_mean_)
        pair_amount_mean = pair_key.map(self.pair_amount_mean_).fillna(self.global_amount_mean_)

        features["quantity"] = normalized["quantity"].clip(lower=0.0)
        features["unit_price"] = normalized["unit_price"].clip(lower=0.0)
        features["invoice_amount"] = normalized["invoice_amount"].clip(lower=0.0)
        features["loan_amount"] = normalized["loan_amount"].clip(lower=0.0)
        features["payment_term_days"] = normalized["payment_term_days"].clip(lower=1.0)
        features["shipment_distance_km"] = normalized["shipment_distance_km"].clip(lower=0.0)
        features["buyer_risk_rating"] = normalized["buyer_risk_rating"].clip(lower=0.0, upper=1.0)
        features["supplier_risk_rating"] = normalized["supplier_risk_rating"].clip(lower=0.0, upper=1.0)
        features["historic_late_payments"] = normalized["historic_late_payments"].clip(lower=0.0)
        features["prior_financing_count"] = normalized["prior_financing_count"].clip(lower=0.0)
        features["loan_to_invoice_ratio"] = _safe_ratio(normalized["loan_amount"], normalized["invoice_amount"], default=0.0)
        features["buyer_amount_ratio"] = _safe_ratio(normalized["invoice_amount"], buyer_amount_mean, default=1.0)
        features["supplier_amount_ratio"] = _safe_ratio(normalized["invoice_amount"], supplier_amount_mean, default=1.0)
        features["product_price_ratio"] = _safe_ratio(normalized["unit_price"], product_price_mean, default=1.0)
        features["pair_amount_ratio"] = _safe_ratio(normalized["invoice_amount"], pair_amount_mean, default=1.0)
        features["buyer_transaction_count"] = normalized["buyer_id"].map(self.buyer_count_).fillna(0.0)
        features["supplier_transaction_count"] = normalized["supplier_id"].map(self.supplier_count_).fillna(0.0)
        features["pair_transaction_count"] = pair_key.map(self.pair_count_).fillna(0.0)
        features["pair_lender_count"] = pair_key.map(self.pair_lender_count_).fillna(0.0)
        features["known_invoice_count"] = normalized["invoice_id"].map(self.invoice_count_).fillna(0.0)
        features["invoice_lender_count"] = normalized["invoice_id"].map(self.invoice_lender_count_).fillna(0.0)
        features["duplicate_invoice_hint"] = (features["invoice_lender_count"] > 1).astype(float)
        features["days_to_due"] = (
            (normalized["due_date"] - normalized["invoice_date"]).dt.days.clip(lower=0).fillna(0)
        )
        features["days_from_reference"] = (
            (self.reference_date_ - normalized["invoice_date"]).dt.days.abs().fillna(0)
        )
        features["invoice_month"] = normalized["invoice_date"].dt.month.fillna(1)
        features["invoice_weekday"] = normalized["invoice_date"].dt.weekday.fillna(0)
        features["historic_pressure"] = _safe_ratio(
            normalized["historic_late_payments"],
            normalized["prior_financing_count"] + 1,
            default=0.0,
        )
        features["manual_channel_flag"] = (normalized["channel"].str.lower() == "manual").astype(float)

        # This ratio is a useful operational signal for invoice inflation.
        features["product_price_ratio"] = 0.65 * features["product_price_ratio"] + 0.35 * _safe_ratio(
            normalized["unit_price"],
            supplier_price_mean,
            default=1.0,
        )

        graph_features = self.graph_builder_.transform(normalized)
        model_frame = pd.concat([features, graph_features], axis=1)

        ordered_columns = list(self.categorical_columns) + list(self.numeric_columns) + list(self.graph_builder_.graph_columns)
        return model_frame[ordered_columns]
