"""Domain-specific preprocessing and feature engineering helpers.

Functions:
- engineer_property_features(properties_df)
- engineer_client_features(clients_df)
- aggregate_client_stats(properties_df)
- build_client_feature_table(clients_df, properties_df)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from dateutil import parser


def _robust_parse_date_series(series: pd.Series) -> pd.Series:
    def _parse(s):
        if pd.isna(s) or str(s).strip() == "":
            return pd.NaT
        try:
            return parser.parse(str(s), dayfirst=False)
        except Exception:
            try:
                return parser.parse(str(s), dayfirst=True)
            except Exception:
                return pd.NaT

    return series.apply(_parse)


def engineer_property_features(properties: pd.DataFrame) -> pd.DataFrame:
    p = properties.copy()
    if "transaction_date" in p.columns:
        p["transaction_date"] = pd.to_datetime(p["transaction_date"], errors="coerce")

    # clean sale price to numeric
    if "sale_price" in p.columns:
        p["sale_price_num"] = (
            p["sale_price"].astype(str).str.replace("[$,]", "", regex=True).replace("", np.nan).astype(float)
        )
    else:
        p["sale_price_num"] = np.nan

    # price per sqft
    if "floor_area_sqft" in p.columns:
        p["price_per_sqft"] = p["sale_price_num"] / p["floor_area_sqft"]
    else:
        p["price_per_sqft"] = np.nan

    # listing status boolean
    if "listing_status" in p.columns:
        p["is_sold"] = p["listing_status"].astype(str).str.lower() == "sold"
    else:
        p["is_sold"] = False

    return p


def engineer_client_features(clients: pd.DataFrame, ref_date: str = "2024-01-01") -> pd.DataFrame:
    c = clients.copy()
    if "date_of_birth" in c.columns:
        c["date_of_birth"] = _robust_parse_date_series(c["date_of_birth"])
        ref = pd.to_datetime(ref_date)
        c["age"] = (ref - c["date_of_birth"]).dt.days // 365
    else:
        c["age"] = np.nan

    # normalize boolean-like fields
    if "loan_applied" in c.columns:
        c["loan_applied_flag"] = c["loan_applied"].astype(str).str.lower().isin(["yes", "y", "true", "1"]) 
    else:
        c["loan_applied_flag"] = False

    return c


def aggregate_client_stats(properties: pd.DataFrame, client_id_col: str = "client_ref") -> pd.DataFrame:
    p = properties.copy()
    # consider sold transactions only for purchase stats
    sold = p[p.get("is_sold", True) == True].copy()

    agg = sold.groupby(client_id_col).agg(
        purchases_count=("listing_id", "count"),
        avg_sale_price=("sale_price_num", "mean"),
        median_sale_price=("sale_price_num", "median"),
        avg_price_per_sqft=("price_per_sqft", "mean"),
        last_purchase_date=("transaction_date", "max"),
    )

    # overall listing counts (including available)
    total_listings = p.groupby(client_id_col).agg(total_listings=("listing_id", "count"))
    agg = agg.join(total_listings, how="outer")
    agg = agg.reset_index().rename(columns={client_id_col: "client_id"})
    return agg


def build_client_feature_table(clients: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    clients_e = engineer_client_features(clients)
    properties_e = engineer_property_features(properties)
    agg = aggregate_client_stats(properties_e, client_id_col="client_ref")

    # clients' client_id column name may be 'client_id'
    merged = clients_e.merge(agg, left_on="client_id", right_on="client_id", how="left")

    # fill numeric NaNs
    num_cols = merged.select_dtypes(include=["number"]).columns
    merged[num_cols] = merged[num_cols].fillna(0)

    return merged


if __name__ == "__main__":
    # quick local test when executed directly
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    clients = pd.read_csv(raw / "clients.csv")
    properties = pd.read_csv(raw / "properties.csv")
    out = build_client_feature_table(clients, properties)
    out_path = root / "data" / "processed" / "clients_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("Wrote", out_path)
