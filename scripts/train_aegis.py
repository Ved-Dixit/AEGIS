"""CLI entrypoint for training the AEGIS fraud engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from aegis.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH
from aegis.service import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AEGIS fraud engine.")
    parser.add_argument("--rows", type=int, default=8_000, help="Synthetic rows to create when no CSV exists.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="Optional training CSV.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Where to save the model bundle.")
    parser.add_argument("--source-name", type=str, default="synthetic", help="Training source label stored in history.")
    args = parser.parse_args()

    data_frame = None
    if args.data_path and Path(args.data_path).exists():
        data_frame = pd.read_csv(args.data_path, parse_dates=["invoice_date", "due_date"])

    summary = train_model(
        rows=args.rows,
        model_path=args.model_path,
        data_frame=data_frame,
        source_name=args.source_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
