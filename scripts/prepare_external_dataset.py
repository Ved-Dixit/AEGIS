"""Prepare a public dataset into the normalized AEGIS schema."""

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
    parser = argparse.ArgumentParser(description="Prepare a public dataset into the AEGIS schema.")
    parser.add_argument("--source", required=True, help="Source key: creditcardfraud, paysim, ieee_cis, dataco, ellipticplusplus")
    parser.add_argument("--input", action="append", required=True, help="Input file path. Use twice for multi-file sources.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    service = AegisService()
    summary = service.prepare_external_dataset(
        source_name=args.source,
        input_paths=args.input,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
