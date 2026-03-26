"""Modeling layer for AEGIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from aegis.features import FeatureBuilder, normalize_transactions

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - covers missing wheels and broken local runtime libs.
    XGBClassifier = None


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - for older scikit-learn versions.
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _humanize_feature_name(name: str) -> str:
    return name.replace("cat__", "").replace("num__", "").replace("_", " ")


@dataclass(slots=True)
class DuplicateDetector:
    """Trade identity similarity engine inspired by the synopsis."""

    vectorizer_: DictVectorizer | None = None
    neighbor_model_: NearestNeighbors | None = None
    reference_: pd.DataFrame | None = None
    invoice_lender_map_: dict[str, set[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "DuplicateDetector":
        normalized = normalize_transactions(frame)
        records = self._to_records(normalized)
        self.vectorizer_ = DictVectorizer(sparse=True)
        matrix = self.vectorizer_.fit_transform(records)
        neighbor_count = max(1, min(5, len(normalized)))
        self.neighbor_model_ = NearestNeighbors(metric="cosine", n_neighbors=neighbor_count)
        self.neighbor_model_.fit(matrix)
        self.reference_ = normalized.reset_index(drop=True)
        self.invoice_lender_map_ = {
            invoice_id: set(group["lender_id"].astype(str))
            for invoice_id, group in normalized.groupby("invoice_id")
        }
        return self

    def score(self, frame: pd.DataFrame, top_k: int = 3) -> list[dict[str, Any]]:
        if self.vectorizer_ is None or self.neighbor_model_ is None or self.reference_ is None:
            raise RuntimeError("DuplicateDetector must be fitted before score().")

        normalized = normalize_transactions(frame)
        matrix = self.vectorizer_.transform(self._to_records(normalized))
        distances, indices = self.neighbor_model_.kneighbors(matrix, n_neighbors=min(top_k, len(self.reference_)))

        results: list[dict[str, Any]] = []
        for row_index, row in normalized.reset_index(drop=True).iterrows():
            similarities = (1.0 - distances[row_index]).tolist()
            neighbor_rows = self.reference_.iloc[indices[row_index]]

            exact_invoice_seen = row["invoice_id"] in self.invoice_lender_map_
            cross_lender_match = exact_invoice_seen and row["lender_id"] not in self.invoice_lender_map_[row["invoice_id"]]
            base_score = max(similarities) if similarities else 0.0
            duplicate_score = float(np.clip(max(base_score, 0.98 if cross_lender_match else 0.0), 0.0, 1.0))

            matches: list[dict[str, Any]] = []
            for similarity, (_, neighbor) in zip(similarities, neighbor_rows.iterrows()):
                matches.append(
                    {
                        "transaction_id": str(neighbor["transaction_id"]),
                        "invoice_id": str(neighbor["invoice_id"]),
                        "lender_id": str(neighbor["lender_id"]),
                        "similarity": round(float(similarity), 4),
                        "fraud_type": str(neighbor.get("fraud_type", "unknown")),
                    }
                )

            explanation_parts = []
            if cross_lender_match:
                explanation_parts.append("invoice id already exists with a different lender")
            if similarities and similarities[0] > 0.9:
                explanation_parts.append("trade identity vector is nearly identical to a historical record")

            results.append(
                {
                    "duplicate_score": duplicate_score,
                    "duplicate_matches": matches,
                    "duplicate_reason": "; ".join(explanation_parts) or "no strong duplicate evidence detected",
                }
            )
        return results

    @staticmethod
    def _to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        identity_rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            identity_rows.append(
                {
                    "buyer_id": str(row["buyer_id"]),
                    "supplier_id": str(row["supplier_id"]),
                    "lender_id": str(row["lender_id"]),
                    "product_id": str(row["product_id"]),
                    "invoice_id": str(row["invoice_id"]),
                    "channel": str(row["channel"]),
                    "currency": str(row["currency"]),
                    "rounded_quantity": round(float(row["quantity"]), 1),
                    "rounded_unit_price": round(float(row["unit_price"]), 1),
                    "rounded_invoice_amount": round(float(row["invoice_amount"]), -1),
                    "rounded_loan_amount": round(float(row["loan_amount"]), -1),
                    "payment_term_days": int(row["payment_term_days"]),
                }
            )
        return identity_rows


@dataclass(slots=True)
class AnomalyDetector:
    """Isolation Forest wrapper with stable 0..1 scoring."""

    random_state: int = 42
    model_: IsolationForest | None = None
    low_: float = 0.0
    high_: float = 1.0

    def fit(self, numeric_frame: pd.DataFrame, target: pd.Series) -> "AnomalyDetector":
        clean_numeric = numeric_frame.fillna(0.0)
        train_frame = clean_numeric[target == 0] if int((target == 0).sum()) >= 20 else clean_numeric

        contamination = float(np.clip(target.mean(), 0.02, 0.2))
        self.model_ = IsolationForest(
            n_estimators=250,
            contamination=contamination,
            random_state=self.random_state,
        )
        self.model_.fit(train_frame)

        raw_scores = -self.model_.score_samples(train_frame)
        self.low_ = float(np.quantile(raw_scores, 0.05))
        self.high_ = float(max(np.quantile(raw_scores, 0.95), self.low_ + 1e-6))
        return self

    def score(self, numeric_frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("AnomalyDetector must be fitted before score().")
        raw_scores = -self.model_.score_samples(numeric_frame.fillna(0.0))
        scaled = (raw_scores - self.low_) / (self.high_ - self.low_)
        return np.clip(scaled, 0.0, 1.0)


@dataclass(slots=True)
class AegisFraudEngine:
    """
    End-to-end fraud engine combining supervised, anomaly, duplicate, graph, and XAI layers.
    """

    random_state: int = 42
    feature_builder_: FeatureBuilder = field(default_factory=FeatureBuilder)
    duplicate_detector_: DuplicateDetector = field(default_factory=DuplicateDetector)
    anomaly_detector_: AnomalyDetector | None = None
    risk_pipeline_: Pipeline | None = None
    background_frame_: pd.DataFrame | None = None
    transformed_feature_names_: list[str] = field(default_factory=list)
    metrics_: dict[str, float] = field(default_factory=dict)
    training_metadata_: dict[str, Any] = field(default_factory=dict)
    _explainer: Any = field(default=None, repr=False)

    def fit(self, frame: pd.DataFrame) -> "AegisFraudEngine":
        normalized = normalize_transactions(frame)
        feature_frame = self.feature_builder_.fit(normalized).transform(normalized)
        target = normalized["is_fraud"].astype(int)

        categorical_columns = list(self.feature_builder_.categorical_columns)
        numeric_columns = [column for column in feature_frame.columns if column not in categorical_columns]

        positive_count = max(int(target.sum()), 1)
        negative_count = max(len(target) - positive_count, 1)
        scale_pos_weight = max(negative_count / positive_count, 1.0)

        if XGBClassifier is not None:
            estimator: Any = XGBClassifier(
                n_estimators=320,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.85,
                min_child_weight=2.0,
                reg_lambda=1.5,
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                n_jobs=2,
            )
        else:  # pragma: no cover - fallback only matters in a reduced environment.
            estimator = RandomForestClassifier(
                n_estimators=140,
                random_state=self.random_state,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
            )

        self.risk_pipeline_ = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            ("cat", _make_one_hot_encoder(), categorical_columns),
                            ("num", "passthrough", numeric_columns),
                        ]
                    ),
                ),
                ("model", estimator),
            ]
        )
        self.risk_pipeline_.fit(feature_frame, target)
        self.transformed_feature_names_ = list(
            self.risk_pipeline_.named_steps["preprocessor"].get_feature_names_out()
        )

        self.anomaly_detector_ = AnomalyDetector(random_state=self.random_state).fit(
            feature_frame[numeric_columns],
            target,
        )
        self.duplicate_detector_.fit(normalized)
        self.background_frame_ = feature_frame.sample(
            n=min(len(feature_frame), 240),
            random_state=self.random_state,
        ).copy()
        self._explainer = None
        return self

    def predict(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if self.risk_pipeline_ is None or self.anomaly_detector_ is None:
            raise RuntimeError("AegisFraudEngine must be fitted before predict().")

        normalized = normalize_transactions(frame)
        feature_frame = self.feature_builder_.transform(normalized)
        categorical_columns = list(self.feature_builder_.categorical_columns)
        numeric_columns = [column for column in feature_frame.columns if column not in categorical_columns]

        classifier_probability = self.risk_pipeline_.predict_proba(feature_frame)[:, 1]
        anomaly_probability = self.anomaly_detector_.score(feature_frame[numeric_columns])
        duplicate_results = self.duplicate_detector_.score(normalized)
        duplicate_probability = np.array([item["duplicate_score"] for item in duplicate_results])
        graph_probability = feature_frame["graph_rule_score"].to_numpy(dtype=float)

        # Weighted late fusion keeps every specialist model visible in the final score.
        final_probability = np.clip(
            0.58 * classifier_probability
            + 0.17 * anomaly_probability
            + 0.15 * duplicate_probability
            + 0.10 * graph_probability,
            0.0,
            1.0,
        )

        shap_messages = self._explain(feature_frame)

        results: list[dict[str, Any]] = []
        for index, row in normalized.reset_index(drop=True).iterrows():
            explanations = list(shap_messages[index])
            explanations.append(f"duplicate detector: {duplicate_results[index]['duplicate_reason']}")
            if graph_probability[index] > 0.6:
                explanations.append("graph layer: buyer/supplier relationship is unusually dense or lender-diverse")
            if anomaly_probability[index] > 0.6:
                explanations.append("behavioral layer: transaction pattern looks statistically rare")

            results.append(
                {
                    "transaction_id": str(row["transaction_id"]),
                    "invoice_id": str(row["invoice_id"]),
                    "classifier_score": round(float(classifier_probability[index]), 4),
                    "anomaly_score": round(float(anomaly_probability[index]), 4),
                    "duplicate_score": round(float(duplicate_probability[index]), 4),
                    "graph_score": round(float(graph_probability[index]), 4),
                    "final_score": round(float(final_probability[index]), 4),
                    "risk_band": self._risk_band(float(final_probability[index])),
                    "explanations": explanations[:6],
                    "duplicate_matches": duplicate_results[index]["duplicate_matches"],
                }
            )
        return results

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "AegisFraudEngine":
        engine = joblib.load(Path(path))
        engine._explainer = None
        return engine

    def _explain(self, feature_frame: pd.DataFrame) -> list[list[str]]:
        if self.risk_pipeline_ is None:
            return [["model explanation unavailable"] for _ in range(len(feature_frame))]

        try:
            transformed = self.risk_pipeline_.named_steps["preprocessor"].transform(feature_frame)
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.risk_pipeline_.named_steps["model"])
            shap_values = self._explainer.shap_values(transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            messages: list[list[str]] = []
            for row_values in np.asarray(shap_values):
                ranked = sorted(
                    zip(self.transformed_feature_names_, row_values),
                    key=lambda pair: abs(pair[1]),
                    reverse=True,
                )[:4]
                messages.append(
                    [
                        f"model: {_humanize_feature_name(name)} pushed risk {'up' if value >= 0 else 'down'}"
                        for name, value in ranked
                    ]
                )
            return messages
        except Exception:
            fallback_messages: list[list[str]] = []
            for _, row in feature_frame.iterrows():
                ranked = sorted(
                    [
                        ("loan to invoice ratio", abs(float(row.get("loan_to_invoice_ratio", 0.0)) - 0.82)),
                        ("pair transaction count", float(row.get("pair_transaction_count", 0.0))),
                        ("duplicate invoice hint", float(row.get("duplicate_invoice_hint", 0.0))),
                        ("graph rule score", float(row.get("graph_rule_score", 0.0))),
                    ],
                    key=lambda pair: abs(pair[1]),
                    reverse=True,
                )[:4]
                fallback_messages.append(
                    [f"model fallback: {name} is materially unusual for this transaction" for name, _ in ranked]
                )
            return fallback_messages

    @staticmethod
    def _risk_band(score: float) -> str:
        if score >= 0.85:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"
