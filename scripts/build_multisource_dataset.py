"""Build a blended AEGIS training set from open-source prepared datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from aegis.data.blend import BlendSourceSpec, build_multisource_dataset


def _allocate_rows(total_rows: int) -> list[BlendSourceSpec]:
    """
    Scale a fixed source plan to the requested total size.

    The mix is intentionally biased toward sources with stronger fraud signal
    while still keeping every public dataset represented.
    """

    source_templates = [
        {
            "name": "dataco_hybrid",
            "path": ROOT_DIR / "artifacts" / "data" / "dataco_hybrid_aegis_20000.csv",
            "weight": 1.0,
            "fraud_ratio": 0.22,
        },
        {
            "name": "elliptic",
            "path": ROOT_DIR / "artifacts" / "data" / "elliptic_aegis.csv",
            "weight": 1.2,
            "fraud_ratio": 0.28,
        },
        {
            "name": "creditcardfraud",
            "path": ROOT_DIR / "artifacts" / "data" / "creditcardfraud_aegis.csv",
            "weight": 0.7,
            "fraud_ratio": 0.16,
        },
        {
            "name": "paysim",
            "path": ROOT_DIR / "artifacts" / "data" / "paysim_aegis.csv",
            "weight": 1.1,
            "fraud_ratio": 0.18,
        },
    ]

    total_weight = sum(item["weight"] for item in source_templates)
    allocated = []
    running_rows = 0
    for index, template in enumerate(source_templates):
        if index == len(source_templates) - 1:
            source_rows = total_rows - running_rows
        else:
            source_rows = int(round(total_rows * template["weight"] / total_weight))
            running_rows += source_rows
        allocated.append(
            BlendSourceSpec(
                name=template["name"],
                path=template["path"],
                rows=max(source_rows, 0),
                fraud_ratio=float(template["fraud_ratio"]),
            )
        )
    return allocated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a blended AEGIS dataset from multiple prepared sources.")
    parser.add_argument("--rows", type=int, default=40_000, help="Total number of output rows.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "artifacts" / "data" / "multisource_open_aegis_40000.csv"),
        help="Where to save the blended dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducible sampling.")
    args = parser.parse_args()

    specs = _allocate_rows(args.rows)
    blended = build_multisource_dataset(specs, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(output_path, index=False)

    source_breakdown = (
        blended.groupby(["source_name", "is_fraud"], as_index=False)
        .size()
        .sort_values(["source_name", "is_fraud"])
        .to_dict(orient="records")
    )
    summary = {
        "output_path": str(output_path),
        "rows": int(len(blended)),
        "fraud_rows": int(blended["is_fraud"].sum()),
        "source_breakdown": source_breakdown,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
