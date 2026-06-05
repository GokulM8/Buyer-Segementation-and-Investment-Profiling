from pathlib import Path
import pandas as pd

proj_root = Path(__file__).resolve().parents[1]
PROC = proj_root / 'data' / 'processed'
OUT = proj_root / 'outputs' / 'figures'
PAPER = proj_root / 'paper'
PAPER.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

profile_fp = PROC / 'profile_kmeans_label.csv'
anova_fp = PROC / 'anova_results_top_features.csv'

if not profile_fp.exists():
    raise FileNotFoundError(profile_fp)

p = pd.read_csv(profile_fp)
if anova_fp.exists():
    anova = pd.read_csv(anova_fp)
else:
    anova = None

# basic cluster summary
clusters = p[['label','size','pct']].copy()
clusters['pct'] = clusters['pct'] * 100
clusters = clusters.sort_values('label')

# top features
feat_cols = [c for c in p.columns if c not in ['label','size','pct']]
# compute ranges
ranges = (p[feat_cols].max() - p[feat_cols].min()).sort_values(ascending=False)
top_feats = list(ranges.head(5).index)

# read Tukey significant CSVs
tukey_sigs = {}
for feat in top_feats:
    sig_fp = PROC / f'tukey_significant_{feat}.csv'
    if sig_fp.exists():
        try:
            tdf = pd.read_csv(sig_fp)
            tukey_sigs[feat] = tdf
        except Exception:
            tukey_sigs[feat] = None
    else:
        tukey_sigs[feat] = None

# build summary text
lines = []
lines.append('# One-page Cluster Summary')
lines.append('')
lines.append('## Overview')
lines.append('- Clustering method: KMeans (best k determined by silhouette)')
lines.append(f"- Total clusters: {len(clusters)}")
lines.append('')
lines.append('## Cluster sizes')
for _, r in clusters.iterrows():
    lines.append(f"- Cluster {int(r['label'])}: {int(r['size'])} clients ({r['pct']:.1f}% of dataset)")
lines.append('')
lines.append('## Top differentiating features')
for i,f in enumerate(top_feats, start=1):
    lines.append(f"{i}. {f}")
lines.append('')
if anova is not None:
    lines.append('## ANOVA (top features)')
    for _, row in anova.sort_values('p').iterrows():
        lines.append(f"- {row['feature']}: F={row['F']:.3f}, p={row['p']:.3e}" )
    lines.append('')

lines.append('## Tukey HSD significant pairs (p<0.05)')
any_sig = False
for feat, tdf in tukey_sigs.items():
    if tdf is not None and not tdf.empty:
        any_sig = True
        lines.append(f"- {feat}:")
        # show group1, group2, p-adj
        for _, r in tdf.iterrows():
            lines.append(f"  - {int(r['group1'])} vs {int(r['group2'])}: p={r['p-adj']:.3g}")
if not any_sig:
    lines.append('- No significant pairwise differences found for the top features')

lines.append('')
lines.append('## Interpretation & Recommendations')
lines.append('- Cluster 0: small, high-value/high-frequency buyers — prioritize premium listings and retention campaigns.')
lines.append('- Cluster 1: mid-priced segment — recommend upsell and targeted listing recommendations.')
lines.append('- Cluster 2: largest mainstream segment — focus on broad acquisition and affordability messaging.')
lines.append('')
lines.append('## Artifacts')
lines.append('- Profiles: `data/processed/profile_kmeans_label.csv`')
lines.append('- ANOVA results: `data/processed/anova_results_top_features.csv`')
lines.append('- Tukey summaries: `data/processed/tukey_significant_<feature>.csv`')
lines.append('- Visuals: see `outputs/figures` (e.g., `kmeans_pca_scatter.png`, `segment_sizes_kmeans.png`, `cluster_profile_heatmap_kmeans.png`)')

md = '\n'.join(lines)

# save
paper_fp = PAPER / 'one_page_summary.md'
text_fp = OUT / 'cluster_one_page_summary.txt'
paper_fp.write_text(md)
text_fp.write_text(md)
print('Saved one-page summary to', paper_fp, 'and', text_fp)
