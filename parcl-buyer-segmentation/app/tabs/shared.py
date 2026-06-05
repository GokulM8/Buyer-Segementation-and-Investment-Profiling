"""Shared helpers for Streamlit dashboard tabs."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def processed_dir() -> Path:
    return project_root() / 'data' / 'processed'


def load_cluster_frame() -> pd.DataFrame | None:
    proc = processed_dir()
    combined_fp = proc / 'clients_clusters_all.csv'
    k_fp = proc / 'clients_clusters_kmeans.csv'
    if combined_fp.exists():
        return pd.read_csv(combined_fp, index_col=0)
    if k_fp.exists():
        return pd.read_csv(k_fp, index_col=0)
    return None


def label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith('_label') or c.endswith('_cluster')]


def default_label_choice(label_cols: list[str]) -> str:
    return 'kmeans_label' if 'kmeans_label' in label_cols else label_cols[0]
