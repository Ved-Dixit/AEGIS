"""Interactive terminal loader for Kaggle datasets used by AEGIS."""

from __future__ import annotations

import csv
import getpass
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from aegis.service import AegisService

KAGGLE_SOURCES = {
    "1": "creditcardfraud",
    "2": "paysim",
    "3": "ieee_cis",
}
DEFAULT_SOURCES = ("creditcardfraud", "paysim")


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle"):
        return
    print("`kaggle` CLI not found. Installing Python package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)


def ensure_kaggle_credentials() -> Path:
    target = Path.home() / ".kaggle" / "kaggle.json"
    if target.exists():
        print(f"Using existing credentials at {target}")
        return target

    print("Kaggle credentials not found.")
    print("You can get the API key from https://www.kaggle.com/settings -> API -> Create New Token")
    username = input("Kaggle username: ").strip()
    key = getpass.getpass("Kaggle API key: ").strip()

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    target.chmod(0o600)
    print(f"Wrote credentials to {target}")
    return target


def select_sources() -> list[str]:
    print("Select Kaggle datasets to load:")
    print("  1. creditcardfraud")
    print("  2. paysim")
    print("  3. ieee_cis (optional competition dataset; requires Kaggle rule acceptance)")
    raw = input("Enter numbers separated by commas [default: 1,2]: ").strip()
    if not raw:
        return list(DEFAULT_SOURCES)

    selected: list[str] = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token in KAGGLE_SOURCES:
            selected.append(KAGGLE_SOURCES[token])
    return selected or list(DEFAULT_SOURCES)


def find_csv_by_header(root: Path, required_columns: set[str]) -> Path | None:
    for path in sorted(root.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                header = next(csv.reader([handle.readline().strip()]))
        except (StopIteration, OSError):
            continue
        if required_columns.issubset(set(header)):
            return path
    return None


def resolve_source_inputs(source_name: str, download_dir: Path) -> list[str]:
    if source_name == "creditcardfraud":
        exact = next((path for path in download_dir.rglob("*.csv") if path.name.lower() == "creditcard.csv"), None)
        if exact:
            return [str(exact)]
        detected = find_csv_by_header(download_dir, {"Time", "Amount", "Class"})
        return [str(detected)] if detected else []

    if source_name == "paysim":
        detected = find_csv_by_header(
            download_dir,
            {"step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig", "isFraud"},
        )
        return [str(detected)] if detected else []

    if source_name == "ieee_cis":
        transaction = next((path for path in download_dir.rglob("train_transaction.csv")), None)
        identity = next((path for path in download_dir.rglob("train_identity.csv")), None)
        if transaction and identity:
            return [str(transaction), str(identity)]
        if transaction:
            return [str(transaction)]
        detected = find_csv_by_header(download_dir, {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"})
        return [str(detected)] if detected else []

    return []


def main() -> None:
    ensure_kaggle_cli()
    ensure_kaggle_credentials()
    service = AegisService()

    sources = select_sources()
    prepare_now = input("Prepare downloaded datasets into AEGIS schema too? [Y/n]: ").strip().lower() not in {"n", "no"}

    for source_name in sources:
        print(f"\n=== Loading {source_name} ===")
        summary = service.fetch_public_source(source_name, execute_kaggle=True)
        print(json.dumps(summary, indent=2))

        if summary.get("status") != "downloaded":
            continue

        if not prepare_now:
            continue

        input_paths = resolve_source_inputs(source_name, Path(summary["output_dir"]))
        if not input_paths:
            print(f"Could not auto-detect raw CSV files for {source_name}.")
            continue

        prepared_output = ROOT_DIR / "artifacts" / "data" / f"{source_name}_aegis.csv"
        prepared = service.prepare_external_dataset(
            source_name=source_name,
            input_paths=input_paths,
            output_path=prepared_output,
        )
        print(json.dumps(prepared, indent=2))


if __name__ == "__main__":
    main()
