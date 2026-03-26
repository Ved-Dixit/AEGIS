"""Download helpers for public Kaggle and GitHub sources used by AEGIS."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aegis.config import RAW_DATA_DIR


@dataclass(frozen=True, slots=True)
class SourceConnector:
    key: str
    provider: str
    source_url: str
    access_mode: str
    description: str
    cli_args: tuple[str, ...] = ()
    direct_files: tuple[str, ...] = ()


PUBLIC_SOURCE_CONNECTORS: dict[str, SourceConnector] = {
    "creditcardfraud": SourceConnector(
        key="creditcardfraud",
        provider="Kaggle",
        source_url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        access_mode="kaggle_cli",
        description="Classic severe class-imbalance fraud dataset.",
        cli_args=("datasets", "download", "-d", "mlg-ulb/creditcardfraud", "--unzip"),
    ),
    "paysim": SourceConnector(
        key="paysim",
        provider="Kaggle",
        source_url="https://www.kaggle.com/datasets/ealaxi/paysim1",
        access_mode="kaggle_cli",
        description="Mobile money fraud simulation dataset.",
        cli_args=("datasets", "download", "-d", "ealaxi/paysim1", "--unzip"),
    ),
    "ieee_cis": SourceConnector(
        key="ieee_cis",
        provider="Kaggle",
        source_url="https://www.kaggle.com/c/ieee-fraud-detection",
        access_mode="kaggle_cli",
        description="IEEE-CIS fraud competition dataset.",
        cli_args=("competitions", "download", "-c", "ieee-fraud-detection"),
    ),
    "dataco": SourceConnector(
        key="dataco",
        provider="GitHub",
        source_url="https://github.com/McGill-MMA-EnterpriseAnalytics/DataCo_Supply_Chain",
        access_mode="direct_download",
        description="Supply-chain operations data useful for AEGIS graph enrichment.",
        direct_files=(
            "https://raw.githubusercontent.com/McGill-MMA-EnterpriseAnalytics/DataCo_Supply_Chain/main/data/raw/DataCoSupplyChainDataset.csv",
            "https://raw.githubusercontent.com/McGill-MMA-EnterpriseAnalytics/DataCo_Supply_Chain/main/data/raw/Q1_2015.csv",
            "https://raw.githubusercontent.com/McGill-MMA-EnterpriseAnalytics/DataCo_Supply_Chain/main/data/raw/future_data.csv",
        ),
    ),
    "ellipticplusplus": SourceConnector(
        key="ellipticplusplus",
        provider="GitHub",
        source_url="https://github.com/git-disl/EllipticPlusPlus",
        access_mode="direct_download",
        description="Graph-centric illicit transaction benchmark served via GitHub media URLs.",
        direct_files=(
            "https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/main/Transactions%20Dataset/txs_classes.csv",
            "https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/main/Transactions%20Dataset/txs_features.csv",
            "https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/main/Transactions%20Dataset/txs_edgelist.csv",
        ),
    ),
}


def describe_connectors() -> list[dict[str, Any]]:
    return [asdict(connector) for connector in PUBLIC_SOURCE_CONNECTORS.values()]


def fetch_public_source(
    source_name: str,
    output_dir: str | Path = RAW_DATA_DIR,
    execute_kaggle: bool = False,
) -> dict[str, Any]:
    """
    Download source files when possible, or return the exact manual commands.

    Kaggle sources require credentials. GitHub sources are downloaded directly
    when raw file URLs are available. Elliptic++ uses Git LFS, so the helper
    returns the manual clone command instead of pretending to fetch the data.
    """

    source_key = source_name.strip().lower()
    if source_key not in PUBLIC_SOURCE_CONNECTORS:
        raise ValueError(f"Unsupported public source '{source_name}'.")

    connector = PUBLIC_SOURCE_CONNECTORS[source_key]
    target_dir = Path(output_dir) / source_key
    target_dir.mkdir(parents=True, exist_ok=True)

    if connector.access_mode == "direct_download":
        downloaded_files = []
        for url in connector.direct_files:
            destination = target_dir / Path(url).name
            urllib.request.urlretrieve(url, destination)
            downloaded_files.append(str(destination))
        return {
            "source_name": source_key,
            "status": "downloaded",
            "output_dir": str(target_dir),
            "files": downloaded_files,
            "notes": "Downloaded direct GitHub assets.",
        }

    if connector.access_mode == "kaggle_cli":
        kaggle_executable = _resolve_kaggle_executable()
        kaggle_launcher = [kaggle_executable] if kaggle_executable else [str(Path(sys.executable).with_name("kaggle"))]
        command = [*kaggle_launcher, *connector.cli_args, "-p", str(target_dir)]
        if execute_kaggle:
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                _extract_zip_files(target_dir)
                return {
                    "source_name": source_key,
                    "status": "downloaded",
                    "output_dir": str(target_dir),
                    "files": [str(path) for path in sorted(target_dir.rglob("*")) if path.is_file()],
                    "notes": "Kaggle download completed successfully.",
                }
            except FileNotFoundError:
                pass
            except subprocess.CalledProcessError as exc:
                return {
                    "source_name": source_key,
                    "status": "manual_required",
                    "output_dir": str(target_dir),
                    "commands": [" ".join(command)],
                    "notes": _format_kaggle_failure_note(exc),
                }
        return {
            "source_name": source_key,
            "status": "manual_required",
            "output_dir": str(target_dir),
            "commands": [" ".join(command)],
            "notes": "Install the Kaggle CLI, configure credentials, and rerun with execute_kaggle=True.",
        }

    return {
        "source_name": source_key,
        "status": "manual_required",
        "output_dir": str(target_dir),
        "commands": ["kaggle or git-lfs manual download required"],
        "notes": "This source still requires manual setup.",
    }


def save_connector_manifest(output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(describe_connectors(), indent=2), encoding="utf-8")
    return destination


def _extract_zip_files(target_dir: Path) -> None:
    """Unpack Kaggle archives so dataset preparation can work immediately."""

    for zip_path in target_dir.rglob("*.zip"):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(zip_path.with_suffix(""))


def _resolve_kaggle_executable() -> str | None:
    """Find the Kaggle console script even when user PATH is incomplete."""

    if shutil.which("kaggle"):
        return shutil.which("kaggle")
    sibling = Path(sys.executable).with_name("kaggle")
    if sibling.exists():
        return str(sibling)
    return None


def _format_kaggle_failure_note(error: subprocess.CalledProcessError) -> str:
    """Turn Kaggle CLI failures into a user-facing next step."""

    raw_message = (error.stderr or error.stdout or "").strip()
    lowered = raw_message.lower()
    if "403" in lowered or "forbidden" in lowered:
        return (
            "Kaggle denied the download. For competition datasets such as IEEE-CIS, "
            "open the competition page in your browser, accept the rules, then rerun the command."
        )
    return raw_message or "Kaggle CLI failed; review stderr and credentials."
