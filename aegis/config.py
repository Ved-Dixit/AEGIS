"""Central configuration for the AEGIS project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ARTIFACT_DIR / "data"
MODEL_DIR = ARTIFACT_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"

DEFAULT_MODEL_PATH = MODEL_DIR / "aegis_bundle.joblib"
DEFAULT_DATA_PATH = DATA_DIR / "synthetic_supply_chain.csv"
DEFAULT_DB_PATH = ARTIFACT_DIR / "aegis_history.sqlite3"
RANDOM_SEED = 42


@dataclass(slots=True)
class TrainingConfig:
    """Runtime options for training and saving the fraud engine."""

    rows: int = 8_000
    test_size: float = 0.2
    random_state: int = RANDOM_SEED
    model_path: Path = DEFAULT_MODEL_PATH
    data_path: Path = DEFAULT_DATA_PATH
