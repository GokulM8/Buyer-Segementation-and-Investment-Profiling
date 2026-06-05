from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

proj_root = Path(__file__).resolve().parents[1]
PROC = proj_root / 'data' / 'processed'
OUT = proj_root / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

combined_fp = PROC / 'clients_clusters_all.csv'
k_fp = PROC / 'clients_clusters_kmeans.csv'
if combined_fp.exists():
    df = pd.read_csv(combined_fp, index_col=0)
elif k_fp.exists():
    df = pd.read_csv(k_fp, index_col=0)
else:
    raise FileNotFoundError('No cluster label CSV found; run clustering notebook or scripts first')

print('Loaded', df.shape, 'rows')

metrics = {}
numeric = df.select_dtypes(include=[np.number]).copy()
label_cols = [c for c in df.columns if c.endswith('_label') or c.endswith('_cluster')]

feature_cols = [c for c in numeric.columns if c not in label_cols]
X = numeric[feature_cols].fillna(numeric[feature_cols].median()).to_numpy() if len(feature_cols)>0 else None

for lbl in label_cols:
    labels = df[lbl].to_numpy()
    if X is None or len(set(labels))<=1:
        metrics[lbl] = {'silhouette': None, 'calinski_harabasz': None}
        continue
    try:
        sil = silhouette_score(X, labels) if len(set(labels))>1 else None
    except Exception:
        sil = None
    try:
        ch = calinski_harabasz_score(X, labels) if len(set(labels))>1 else None
    except Exception:
        ch = None
    metrics[lbl] = {'silhouette': sil, 'calinski_harabasz': ch}

print('Evaluation metrics:')
for k,v in metrics.items():
    print(k, v)

profiles = {}
for lbl in label_cols:
    grp = df.groupby(lbl)
    sizes = grp.size().rename('size')
    means = grp[feature_cols].mean()
    pct = (sizes / sizes.sum()).rename('pct')
    profile = pd.concat([sizes, pct, means], axis=1).reset_index().rename(columns={lbl: 'label'})
    profiles[lbl] = profile
    profile_fp = PROC / f'profile_{lbl}.csv'
    profile.to_csv(profile_fp, index=False)
    print('Saved profile', profile_fp)

# Visuals for kmeans_label if present
if 'kmeans_label' in profiles:
    p = profiles['kmeans_label'].copy()
    plt.figure(figsize=(6,4))
    sns.barplot(data=p, x='label', y='size', palette='tab10')
    plt.title('Segment sizes (KMeans)')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(OUT / 'segment_sizes_kmeans.png', bbox_inches='tight')
    plt.close()

    means = p[feature_cols].set_index(p['label']).T
    s = StandardScaler()
    means_std = pd.DataFrame(s.fit_transform(means.T).T, index=means.index, columns=means.columns)
    plt.figure(figsize=(10,6))
    sns.heatmap(means_std, cmap='vlag', center=0)
    plt.title('Standardized feature means by cluster (KMeans)')
    plt.tight_layout()
    plt.savefig(OUT / 'cluster_profile_heatmap_kmeans.png', bbox_inches='tight')
    plt.close()

print('Saved figures to', OUT)
print('Done')
