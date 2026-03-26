"""Fetch or describe public Kaggle/GitHub sources used by AEGIS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from aegis.service import AegisService


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or describe a public source used by AEGIS.")
    parser.add_argument("--source", required=True, help="Source key: creditcardfraud, paysim, ieee_cis, dataco, ellipticplusplus")
    parser.add_argument(
        "--execute-kaggle",
        action="store_true",
        help="Actually run the Kaggle CLI command if the source uses Kaggle.",
    )
    args = parser.parse_args()

    service = AegisService()
    summary = service.fetch_public_source(args.source, execute_kaggle=args.execute_kaggle)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
