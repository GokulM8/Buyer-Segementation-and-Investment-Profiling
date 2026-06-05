"""Clustering models and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .utils import ensure_dir


def train_kmeans(
    data: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: int = 10,
) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    model.fit(data)
    return model


def train_hierarchical(
    data: np.ndarray,
    n_clusters: int = 4,
    linkage: str = "ward",
) -> AgglomerativeClustering:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(data)
    return model


def evaluate_clusters(data: np.ndarray, labels: Iterable[int]) -> dict[str, float]:
    label_array = np.asarray(list(labels))
    unique_labels = np.unique(label_array)

    metrics: dict[str, float] = {"clusters": float(len(unique_labels))}
    if len(unique_labels) > 1 and len(label_array) > len(unique_labels):
        metrics["silhouette_score"] = float(silhouette_score(data, label_array))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(data, label_array))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(data, label_array))
    return metrics


def assign_cluster_labels(model, data: np.ndarray) -> np.ndarray:
    if hasattr(model, "labels_"):
        return np.asarray(model.labels_)
    return np.asarray(model.predict(data))


def save_model(model, output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    joblib.dump(model, output)
    return output


def load_model(model_path: str | Path):
    return joblib.load(model_path)
