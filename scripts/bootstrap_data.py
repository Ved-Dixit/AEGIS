"""Generate a local demo dataset and store the public source catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from aegis.config import DATA_DIR, DEFAULT_DATA_PATH
from aegis.data.sources import describe_sources
from aegis.data.synthetic import generate_synthetic_supply_chain_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap demo data for AEGIS.")
    parser.add_argument("--rows", type=int, default=8_000, help="Number of synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="CSV destination for the generated dataset.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_supply_chain_dataset(rows=args.rows, seed=args.seed)
    frame.to_csv(args.output, index=False)

    catalog_path = DATA_DIR / "open_source_catalog.json"
    catalog_path.write_text(json.dumps(describe_sources(), indent=2), encoding="utf-8")

    print(f"wrote dataset: {args.output}")
    print(f"wrote catalog: {catalog_path}")
    print(frame[['is_fraud', 'fraud_type']].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
