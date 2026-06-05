"""Shared helpers for the buyer segmentation project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1].parent


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def candidate_data_paths(filename: str) -> list[Path]:
    root = project_root()
    workspace = workspace_root()
    return [
        root / "data" / "raw" / filename,
        workspace / "Parcl Co Limited" / filename,
    ]


def resolve_data_path(filename: str) -> Path:
    for candidate in candidate_data_paths(filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in any known data location.")


def standardize_column_names(columns: Iterable[str]) -> list[str]:
    standardized = []
    for column in columns:
        name = column.strip().lower()
        name = name.replace("%", "pct")
        name = name.replace("-", "_")
        name = name.replace(" ", "_")
        standardized.append(name)
    return standardized


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = standardize_column_names(normalized.columns)
    object_columns = normalized.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        normalized[column] = normalized[column].astype("string").str.strip()
    return normalized


def save_figure(fig, output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    return output


def save_dataframe(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    df.to_csv(output, index=index)
    return output


def set_random_seed(seed: int = 42) -> None:
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
