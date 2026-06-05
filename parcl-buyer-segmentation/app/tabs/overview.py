"""Overview tab for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from app.tabs.shared import default_label_choice, label_columns, load_cluster_frame


def render() -> None:
    st.subheader("Cluster Distribution")
    st.write(
        "Use this tab to display the overall segment mix, cluster counts, and high-level profile summaries."
    )

    proj_root = Path(__file__).resolve().parents[2]
    PROC = proj_root / 'data' / 'processed'

    df = load_cluster_frame()
    profile_fp = PROC / 'profile_kmeans_label.csv'

    if df is None:
        st.info("Connect a processed dataset and cluster labels to visualize the distribution here.")
        return

    label_cols = label_columns(df)
    if not label_cols:
        st.info('No cluster label columns found')
        return

    label_choice = st.session_state.get('dashboard_label_choice', default_label_choice(label_cols))
    if label_choice not in label_cols:
        label_choice = default_label_choice(label_cols)

    cluster_filter = st.session_state.get('dashboard_cluster_filter', 'All')
    counts = df[label_choice].value_counts().sort_index()
    st.markdown('### Segment mix')
    c1, c2, c3 = st.columns(3)
    c1.metric('Clients', f'{len(df):,}')
    c2.metric('Clusters', f'{len(counts):,}')
    c3.metric('Largest cluster', f'{counts.max():,}')
    st.bar_chart(counts)

    if cluster_filter != 'All' and label_choice in df.columns:
        st.markdown(f'### Selected cluster: {cluster_filter}')
        st.write(df[df[label_choice] == cluster_filter].head(20))

    if profile_fp.exists():
        p = pd.read_csv(profile_fp)
        feat_cols = [c for c in p.columns if c not in ['label', 'size', 'pct']]
        if feat_cols:
            if cluster_filter != 'All' and 'label' in p.columns and str(cluster_filter).isdigit():
                row = p[p['label'] == int(cluster_filter)]
            elif cluster_filter != 'All' and 'label' in p.columns:
                row = p[p['label'] == cluster_filter]
            else:
                row = p.sort_values('size', ascending=False).head(1)
            if not row.empty:
                row = row.iloc[0]
                top_feats = sorted(feat_cols, key=lambda c: abs(row[c] - p[c].mean()), reverse=True)[:3]
                st.markdown('### Narrative snapshot')
                st.info(
                    f"Top signals for this view: {', '.join(top_feats)}. "
                    f"The selected segment is characterized by size={int(row['size'])} and share={row['pct']*100:.1f}% of the dataset."
                )

    # show quick profile table if available
    if profile_fp.exists():
        st.markdown('### Cluster profile summary')
        # display label, size, pct and top 4 features
        feat_cols = [c for c in p.columns if c not in ['label','size','pct']][:4]
        cols = ['label','size','pct'] + feat_cols
        st.dataframe(p[cols].sort_values('label'))
    else:
        st.info('Profile CSV not found; run the interpretation script to generate `profile_kmeans_label.csv`.')
