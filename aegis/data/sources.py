"""Open-source dataset catalog used to ground the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Metadata for a public dataset or repository relevant to AEGIS."""

    name: str
    provider: str
    category: str
    url: str
    purpose: str
    access_notes: str


OPEN_SOURCE_DATASETS: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="IEEE-CIS Fraud Detection",
        provider="Kaggle",
        category="tabular fraud benchmark",
        url="https://www.kaggle.com/c/ieee-fraud-detection",
        purpose="High-cardinality transaction fraud benchmark for supervised classification.",
        access_notes="Requires Kaggle credentials; useful when you want a larger production-style benchmark.",
    ),
    DatasetSource(
        name="Credit Card Fraud Detection (MLG-ULB)",
        provider="Kaggle",
        category="tabular fraud benchmark",
        url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        purpose="Classic benchmark for severe class imbalance and rare-event scoring.",
        access_notes="Requires Kaggle credentials; easy to map into the AEGIS tabular pipeline.",
    ),
    DatasetSource(
        name="PaySim Mobile Money Fraud",
        provider="Kaggle",
        category="behavioral anomaly benchmark",
        url="https://www.kaggle.com/datasets/ealaxi/paysim1",
        purpose="Useful for sequential patterns, bursty transactions, and anomaly signatures.",
        access_notes="Requires Kaggle credentials; strong source for behavior-focused experimentation.",
    ),
    DatasetSource(
        name="Elliptic++",
        provider="GitHub",
        category="graph fraud benchmark",
        url="https://github.com/git-disl/EllipticPlusPlus",
        purpose="Graph-centric illicit transaction detection with rich entity relationships.",
        access_notes="Public GitHub repository; ideal for graph feature research and future GNN upgrades.",
    ),
    DatasetSource(
        name="DataCo Supply Chain Analytics",
        provider="GitHub",
        category="supply-chain operations benchmark",
        url="https://github.com/McGill-MMA-EnterpriseAnalytics/DataCo_Supply_Chain",
        purpose="Supply-chain context that helps bridge fraud scoring with buyer/supplier behavior.",
        access_notes="Public GitHub project referencing a Kaggle-backed supply-chain dataset.",
    ),
)


def describe_sources() -> list[dict[str, str]]:
    """Return a JSON-friendly version of the public dataset catalog."""

    return [asdict(source) for source in OPEN_SOURCE_DATASETS]

