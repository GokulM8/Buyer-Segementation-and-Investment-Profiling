"""Generate EDA figures and save them to outputs/figures.

Run:
    python3 scripts/generate_eda_figures.py

This script reads CSVs from `data/raw`, performs light cleaning, and writes PNGs.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load
clients_fp = RAW / "clients.csv"
properties_fp = RAW / "properties.csv"
if not clients_fp.exists() or not properties_fp.exists():
    raise FileNotFoundError("data/raw/clients.csv or properties.csv not found")

clients = pd.read_csv(clients_fp)
properties = pd.read_csv(properties_fp)

# robust date parser

def _robust_parse_date(s):
    if pd.isna(s) or str(s).strip() == "":
        return pd.NaT
    try:
        return parser.parse(str(s), dayfirst=False)
    except Exception:
        try:
            return parser.parse(str(s), dayfirst=True)
        except Exception:
            return pd.NaT

clients["date_of_birth"] = clients["date_of_birth"].apply(_robust_parse_date)
properties["transaction_date"] = pd.to_datetime(properties["transaction_date"], errors="coerce")

# compute age
ref_date = pd.to_datetime("2024-01-01")
clients["age"] = (ref_date - clients["date_of_birth"]).dt.days // 365

# clean price
properties["sale_price_num"] = (
    properties["sale_price"].astype(str).str.replace("[$,]", "", regex=True).replace("", np.nan).astype(float)
)
properties["price_per_sqft"] = properties["sale_price_num"] / properties["floor_area_sqft"]
properties["is_sold"] = properties["listing_status"].str.lower() == "sold"

sns.set(style="whitegrid")

# helper to save
def save_fig(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print("Saved", out)

# price distribution
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(properties["sale_price_num"].dropna(), bins=60, kde=True, ax=ax)
ax.set_title("Sale price distribution")
save_fig(fig, "price_distribution.png")
plt.close(fig)

# area distribution
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(properties["floor_area_sqft"].dropna(), bins=60, kde=True, ax=ax)
ax.set_title("Floor area (sqft) distribution")
save_fig(fig, "area_distribution.png")
plt.close(fig)

# price per sqft distribution
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(properties["price_per_sqft"].dropna(), bins=60, kde=True, ax=ax)
ax.set_title("Price per sqft distribution")
save_fig(fig, "price_per_sqft_distribution.png")
plt.close(fig)

# categorical counts
fig, axes = plt.subplots(1,2, figsize=(12,4))
sns.countplot(x="listing_status", data=properties, ax=axes[0])
axes[0].set_title("Listing status")
sns.countplot(x="unit_category", data=properties, ax=axes[1])
axes[1].set_title("Unit category")
save_fig(fig, "categorical_counts.png")
plt.close(fig)

# client distributions
fig, axes = plt.subplots(2,2, figsize=(12,8))
sns.countplot(x="client_type", data=clients, ax=axes[0,0])
axes[0,0].set_title("Client type")
sns.countplot(x="acquisition_purpose", data=clients, ax=axes[0,1])
axes[0,1].set_title("Acquisition purpose")
sns.countplot(x="referral_channel", data=clients, ax=axes[1,0])
axes[1,0].set_title("Referral channel")
sns.histplot(clients["satisfaction_score"].dropna(), bins=5, ax=axes[1,1])
axes[1,1].set_title("Satisfaction score")
save_fig(fig, "client_distributions.png")
plt.close(fig)

# price vs area scatter (sampled)
sample = properties.sample(frac=0.2, random_state=1) if len(properties) > 2000 else properties
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x="floor_area_sqft", y="sale_price_num", hue="unit_category", data=sample, alpha=0.7, ax=ax)
ax.set_title("Sale price vs area")
save_fig(fig, "price_vs_area.png")
plt.close(fig)

# missingness heatmap (sample cols)
try:
    subset = properties.join(clients.set_index('client_id'), on='client_ref').iloc[:200, :40]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(subset.isna(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missingness (sample of rows/cols)')
    save_fig(fig, 'missingness_sample.png')
    plt.close(fig)
except Exception:
    pass

print('All figures saved to', OUT_DIR)
