"""Build a hybrid AEGIS training set from real DataCo rows plus injected fraud."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from aegis.data.hybrid import build_hybrid_supply_chain_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hybrid supply-chain fraud dataset.")
    parser.add_argument(
        "--dataco-path",
        required=True,
        help="Prepared DataCo AEGIS CSV path.",
    )
    parser.add_argument("--rows", type=int, default=60000, help="Target output row count.")
    parser.add_argument("--fraud-ratio", type=float, default=0.14, help="Fraction of injected fraud rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    frame = build_hybrid_supply_chain_dataset(
        dataco_prepared_path=args.dataco_path,
        rows=args.rows,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(output_path)
    print(frame[["is_fraud", "fraud_type"]].value_counts().to_string())


if __name__ == "__main__":
    main()
