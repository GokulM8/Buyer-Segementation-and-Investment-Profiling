from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
import matplotlib.pyplot as plt
import joblib

proj_root = Path(__file__).resolve().parents[1]
RAW = proj_root / 'data' / 'raw'
PROC = proj_root / 'data' / 'processed'
OUT = proj_root / 'outputs' / 'figures'
MODELS = proj_root / 'outputs' / 'models'
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

clients_fp = PROC / 'clients_features.csv'
if not clients_fp.exists():
    raise FileNotFoundError(f'Processed clients file not found: {clients_fp}')

print('Loading', clients_fp)
df = pd.read_csv(clients_fp, index_col=0)
if 'client_ref' in df.columns:
    ids = df['client_ref'].astype(str)
else:
    ids = df.index.astype(str)

num = df.select_dtypes(include=[np.number]).copy()
num = num.loc[:, num.isna().mean() < 0.4]
num = num.fillna(num.median())

scaler = StandardScaler()
X = scaler.fit_transform(num)
joblib.dump(scaler, MODELS / 'scaler.joblib')
print('Prepared feature matrix', X.shape)

# K range
ks = list(range(2,9))
sil_scores = []
ch_scores = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels))>1 else np.nan
    ch = calinski_harabasz_score(X, labels) if len(set(labels))>1 else np.nan
    sil_scores.append(sil)
    ch_scores.append(ch)

plt.figure(figsize=(8,4))
plt.plot(ks, sil_scores, marker='o', label='Silhouette')
plt.plot(ks, ch_scores, marker='o', label='Calinski-Harabasz')
plt.xlabel('k')
plt.ylabel('score')
plt.legend()
plt.title('Clustering quality vs k (KMeans)')
plt.grid(True)
plt.savefig(OUT / 'k_vs_scores.png', bbox_inches='tight')
plt.close()

best_idx = int(np.nanargmax(sil_scores))
best_k = ks[best_idx]
print('Best k (by silhouette):', best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X)
df_out = df.copy()
df_out['kmeans_label'] = k_labels

df_out.to_csv(PROC / 'clients_clusters_kmeans.csv')
joblib.dump(kmeans, MODELS / f'kmeans_k{best_k}.joblib')
print('Saved KMeans labels and model')

agg = AgglomerativeClustering(n_clusters=best_k)
a_labels = agg.fit_predict(X)
print('ARI between KMeans and Agglomerative:', adjusted_rand_score(k_labels, a_labels))
df_out['agg_label'] = a_labels
df_out.to_csv(PROC / 'clients_clusters_agg.csv')
print('Saved Agglomerative labels')

# DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(X)
print('DBSCAN produced', len(set(db_labels)), 'labels (including noise)')
df_out['dbscan_label'] = db_labels
df_out.to_csv(PROC / 'clients_clusters_dbscan.csv')
print('Saved DBSCAN labels')

# PCA viz
pca = PCA(n_components=2)
Z = pca.fit_transform(X)
plt.figure(figsize=(7,6))
plt.scatter(Z[:,0], Z[:,1], c=k_labels, cmap='tab10', s=20, alpha=0.8)
plt.title(f'KMeans (k={best_k}) in PCA space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.savefig(OUT / 'kmeans_pca_scatter.png', bbox_inches='tight')
plt.close()

# save final combined labels

df_out.to_csv(PROC / 'clients_clusters_all.csv')
print('Saved combined cluster labels to', (PROC / 'clients_clusters_all.csv'))

print('Done. Plots saved to', OUT)
