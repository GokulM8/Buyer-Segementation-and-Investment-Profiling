"""Data loading and preprocessing helpers for buyer segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import normalize_dataframe, resolve_data_path, save_dataframe


def load_raw_data(
    clients_filename: str = "clients.csv",
    properties_filename: str = "properties.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(resolve_data_path(clients_filename))
    properties = pd.read_csv(resolve_data_path(properties_filename))
    return clients, properties


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_dataframe(df)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in df.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                    ]
                ),
                list(categorical_features),
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_feature_matrix(
    df: pd.DataFrame,
    drop_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, ColumnTransformer, list[str]]:
    working = clean_dataframe(df)
    if drop_columns:
        working = working.drop(columns=[column for column in drop_columns if column in working.columns], errors="ignore")

    numeric_features, categorical_features = split_feature_types(working)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    matrix = preprocessor.fit_transform(working)

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        feature_names = [f"feature_{index}" for index in range(matrix.shape[1])]

    transformed = pd.DataFrame(matrix, columns=feature_names)
    return transformed, preprocessor, feature_names


def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    return save_dataframe(df, output_path, index=False)
