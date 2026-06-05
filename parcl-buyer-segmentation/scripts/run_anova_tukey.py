from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

from statsmodels.stats.multicomp import pairwise_tukeyhsd

proj_root = Path(__file__).resolve().parents[1]
PROC = proj_root / 'data' / 'processed'

combined_fp = PROC / 'clients_clusters_all.csv'
k_fp = PROC / 'clients_clusters_kmeans.csv'
if combined_fp.exists():
    df = pd.read_csv(combined_fp, index_col=0)
elif k_fp.exists():
    df = pd.read_csv(k_fp, index_col=0)
else:
    raise FileNotFoundError('No cluster labels found; run clustering first')

if 'kmeans_label' not in df.columns:
    raise ValueError('kmeans_label column not found in cluster file')

numeric = df.select_dtypes(include=[np.number]).copy()
label_cols = [c for c in df.columns if c.endswith('_label') or c.endswith('_cluster')]
feature_cols = [c for c in numeric.columns if c not in label_cols]
if len(feature_cols) == 0:
    raise ValueError('No numeric feature columns found to test')

ranges = numeric[feature_cols].max() - numeric[feature_cols].min()
top_features = list(ranges.sort_values(ascending=False).head(5).index)

anova_results = []
for feat in top_features:
    groups = [group[feat].dropna().values for _, group in df.groupby('kmeans_label')]
    try:
        F, p = stats.f_oneway(*groups)
    except Exception:
        F, p = np.nan, np.nan
    anova_results.append({'feature': feat, 'F': F, 'p': p})

anova_df = pd.DataFrame(anova_results).sort_values('p')
anova_df.to_csv(PROC / 'anova_results_top_features.csv', index=False)
print('Saved ANOVA results to', PROC / 'anova_results_top_features.csv')

# Tukey HSD
for feat in top_features:
    try:
        res = pairwise_tukeyhsd(df[feat].dropna(), df.loc[df[feat].notna(), 'kmeans_label'])
        txt = res.summary().as_text()
        with open(PROC / f'tukey_{feat}.txt', 'w') as f:
            f.write(txt)
        print('Saved Tukey summary for', feat)
    except Exception as e:
        print('Tukey failed for', feat, e)

print('Done')
