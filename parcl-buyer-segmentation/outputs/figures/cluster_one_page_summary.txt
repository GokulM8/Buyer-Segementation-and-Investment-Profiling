# One-page Cluster Summary

## Overview
- Clustering method: KMeans (best k determined by silhouette)
- Total clusters: 3

## Cluster sizes
- Cluster 0: 51 clients (2.5% of dataset)
- Cluster 1: 813 clients (40.6% of dataset)
- Cluster 2: 1136 clients (56.8% of dataset)

## Top differentiating features
1. median_sale_price
2. avg_sale_price
3. age
4. avg_price_per_sqft
5. purchases_count

## ANOVA (top features)
- median_sale_price: F=1609.444, p=0.000e+00
- avg_sale_price: F=1436.806, p=0.000e+00
- purchases_count: F=1049.905, p=2.514e-312
- avg_price_per_sqft: F=83.012, p=2.338e-35
- age: F=8.486, p=2.138e-04

## Tukey HSD significant pairs (p<0.05)
- median_sale_price:
  - 0 vs 1: p=0
  - 0 vs 2: p=0.0003
  - 1 vs 2: p=0
- avg_sale_price:
  - 0 vs 1: p=0
  - 0 vs 2: p=0
  - 1 vs 2: p=0
- age:
  - 0 vs 1: p=0.0003
  - 0 vs 2: p=0.0027
- avg_price_per_sqft:
  - 0 vs 1: p=0.0119
  - 1 vs 2: p=0
- purchases_count:
  - 0 vs 1: p=0
  - 0 vs 2: p=0
  - 1 vs 2: p=0

## Interpretation & Recommendations
- Cluster 0: small, high-value/high-frequency buyers — prioritize premium listings and retention campaigns.
- Cluster 1: mid-priced segment — recommend upsell and targeted listing recommendations.
- Cluster 2: largest mainstream segment — focus on broad acquisition and affordability messaging.

## Artifacts
- Profiles: `data/processed/profile_kmeans_label.csv`
- ANOVA results: `data/processed/anova_results_top_features.csv`
- Tukey summaries: `data/processed/tukey_significant_<feature>.csv`
- Visuals: see `outputs/figures` (e.g., `kmeans_pca_scatter.png`, `segment_sizes_kmeans.png`, `cluster_profile_heatmap_kmeans.png`)