"""Blend multiple prepared AEGIS datasets into one training corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aegis.features import normalize_transactions


@dataclass(frozen=True, slots=True)
class BlendSourceSpec:
    """
    Sampling plan for one prepared AEGIS dataset.

    `rows` is the number of rows to pull from the source. `fraud_ratio`
    controls how many of those rows should come from the fraud class so the
    final blend is more informative than the raw class imbalance alone.
    """

    name: str
    path: str | Path
    rows: int
    fraud_ratio: float = 0.15
    chunksize: int = 100_000


def build_multisource_dataset(
    source_specs: list[BlendSourceSpec],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build one shuffled training frame from multiple prepared AEGIS CSVs.

    The helper reads large CSVs in chunks, keeps only a compact label-aware
    sample from each source, and adds `source_name` for auditability.
    """

    if not source_specs:
        raise ValueError("At least one blend source is required.")

    sampled_parts = [
        _sample_source(spec, seed=seed + index * 97)
        for index, spec in enumerate(source_specs)
        if spec.rows > 0
    ]
    sampled_parts = [part for part in sampled_parts if not part.empty]
    if not sampled_parts:
        raise ValueError("Blend sampling produced no rows.")

    blended = pd.concat(sampled_parts, ignore_index=True)
    blended = blended.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return normalize_transactions(blended)


def _sample_source(spec: BlendSourceSpec, seed: int) -> pd.DataFrame:
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")

    fraud_target = int(round(spec.rows * float(spec.fraud_ratio)))
    fraud_target = min(max(fraud_target, 0), spec.rows)
    legit_target = max(spec.rows - fraud_target, 0)

    legit_pool: pd.DataFrame | None = None
    fraud_pool: pd.DataFrame | None = None
    expected_columns: list[str] | None = None
    legit_rng = np.random.default_rng(seed + 11)
    fraud_rng = np.random.default_rng(seed + 23)

    for chunk in pd.read_csv(path, chunksize=spec.chunksize, low_memory=False):
        if "is_fraud" not in chunk.columns:
            raise ValueError(f"Prepared dataset is missing 'is_fraud': {path}")
        if expected_columns is None:
            expected_columns = list(chunk.columns)

        labels = pd.to_numeric(chunk["is_fraud"], errors="coerce").fillna(0).astype(int)
        legit_chunk = chunk.loc[labels == 0].copy()
        fraud_chunk = chunk.loc[labels == 1].copy()

        legit_pool = _update_priority_pool(legit_pool, legit_chunk, legit_target, legit_rng)
        fraud_pool = _update_priority_pool(fraud_pool, fraud_chunk, fraud_target, fraud_rng)

    if expected_columns is None:
        raise ValueError(f"Prepared dataset is empty: {path}")

    legit_sample = _finalize_priority_pool(
        pool=legit_pool,
        target=legit_target,
        columns=expected_columns,
        seed=seed + 31,
        label="legitimate",
        source_name=spec.name,
    )
    fraud_sample = _finalize_priority_pool(
        pool=fraud_pool,
        target=fraud_target,
        columns=expected_columns,
        seed=seed + 43,
        label="fraud",
        source_name=spec.name,
    )

    sampled = pd.concat([legit_sample, fraud_sample], ignore_index=True)
    if sampled.empty:
        raise ValueError(f"Sampling plan for '{spec.name}' produced no rows.")

    sampled["source_name"] = spec.name
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _update_priority_pool(
    pool: pd.DataFrame | None,
    chunk: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
) -> pd.DataFrame | None:
    if target <= 0 or chunk.empty:
        return pool

    scored = chunk.copy()
    scored["_priority"] = rng.random(len(scored))
    combined = scored if pool is None else pd.concat([pool, scored], ignore_index=True)

    # Trim aggressively so the sampler stays memory-safe on large CSVs.
    if len(combined) > max(target * 2, target + len(scored)):
        combined = combined.nsmallest(target, "_priority")
    return combined


def _finalize_priority_pool(
    pool: pd.DataFrame | None,
    target: int,
    columns: list[str],
    seed: int,
    label: str,
    source_name: str,
) -> pd.DataFrame:
    if target <= 0:
        return pd.DataFrame(columns=columns)
    if pool is None or pool.empty:
        raise ValueError(f"Source '{source_name}' has no {label} rows to sample.")

    selected = pool.nsmallest(target, "_priority").drop(columns="_priority").reset_index(drop=True)
    if len(selected) >= target:
        return selected.iloc[:target].copy()

    # When a source has fewer fraud rows than requested, duplicate the best
    # sample rows so the blend still hits the planned class mix.
    top_up = selected.sample(n=target - len(selected), replace=True, random_state=seed)
    return pd.concat([selected, top_up], ignore_index=True)
