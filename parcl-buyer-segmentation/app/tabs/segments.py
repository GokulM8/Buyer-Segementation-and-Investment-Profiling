"""Segments tab for the Streamlit dashboard.

Features added:
- KPI cards for a selected cluster
- Cluster profile heatmap across numeric features
- Editable persona labels and recommended actions (saved to `data/processed/cluster_personas.csv`)
"""

from __future__ import annotations

from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _load_clusters(proc: Path):
    combined_fp = proc / 'clients_clusters_all.csv'
    k_fp = proc / 'clients_clusters_kmeans.csv'
    if combined_fp.exists():
        return pd.read_csv(combined_fp, index_col=0)
    if k_fp.exists():
        return pd.read_csv(k_fp, index_col=0)
    return None


def render() -> None:
    import traceback

    try:
        st.subheader("Segments")
        st.write("Explore cluster labels, segment sizes, KPI cards, profile heatmaps, and persona recommendations.")

        proj_root = Path(__file__).resolve().parents[2]
        PROC = proj_root / 'data' / 'processed'
        OUT = proj_root / 'outputs' / 'figures'
        PROC.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)

        df = _load_clusters(PROC)
        if df is None:
            st.error('No cluster label CSV found; run clustering scripts/notebooks first')
            return

        # identify label columns
        label_cols = [c for c in df.columns if c.endswith('_label') or c.endswith('_cluster')]
        if not label_cols:
            st.error('No cluster label columns found in the dataset')
            return

        label_choice = st.selectbox('Choose label column', label_cols, index=label_cols.index('kmeans_label') if 'kmeans_label' in label_cols else 0)

        # ensure numeric features and feature columns
        numeric = df.select_dtypes(include=[np.number]).copy()
        feat_cols = [c for c in numeric.columns if c not in label_cols]

        counts = df[label_choice].value_counts().sort_index()
        st.markdown('### Cluster sizes')
        st.bar_chart(counts)

        # read or compute profile
        profile_fp = PROC / f'profile_{label_choice}.csv'
        if profile_fp.exists():
            profile = pd.read_csv(profile_fp)
        else:
            if feat_cols:
                profile = df.groupby(label_choice)[feat_cols].mean().reset_index()
                # add size and pct
                sizes = df[label_choice].value_counts().sort_index()
                profile_index = label_choice
                profile['size'] = profile[profile_index].map(sizes.to_dict())
                profile['pct'] = profile['size'] / profile['size'].sum()
            else:
                profile = pd.DataFrame()

        # determine profile index column (profile files sometimes use 'label')
        if not profile.empty:
            if label_choice in profile.columns:
                profile_index = label_choice
            elif 'label' in profile.columns:
                profile_index = 'label'
            else:
                profile_index = profile.columns[0]

        # Heatmap: show normalized feature means across clusters
        if not profile.empty and feat_cols:
            heat_df = profile.set_index(profile_index)[feat_cols]
            # normalize features (min-max) for visualization
            norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())
            fig, ax = plt.subplots(figsize=(max(6, len(feat_cols) * 0.8), max(3, len(norm) * 0.8)))
            sns.heatmap(norm, cmap='viridis', annot=False, cbar=True, ax=ax)
            ax.set_ylabel('Cluster')
            ax.set_xlabel('Feature')
            ax.set_title('Cluster profile heatmap (normalized)')
            st.pyplot(fig)
            # save heatmap
            out_fp = OUT / f'cluster_profile_heatmap_{label_choice}.png'
            fig.savefig(out_fp, bbox_inches='tight')

        # Select a cluster to inspect
        sel = st.selectbox('Inspect cluster', sorted(counts.index.tolist()))

        # KPI cards: show a few key metrics if present in profile or df
        kpi_cols = [c for c in ['size', 'median_sale_price', 'avg_sale_price', 'avg_price_per_sqft', 'purchases_count', 'age'] if c in profile.columns or c in df.columns]
        # prepare KPI values from profile if possible
        kpi_vals = {}
        if not profile.empty and 'size' in profile.columns:
            for c in kpi_cols:
                if c in profile.columns:
                    val = profile.loc[profile[profile_index] == sel, c]
                    if not val.empty:
                        kpi_vals[c] = val.values[0]
        # fallback to computing from df
        for c in kpi_cols:
            if c not in kpi_vals and c in df.columns:
                series = df.loc[df[label_choice] == sel, c].dropna()
                if not series.empty:
                    if c == 'size':
                        kpi_vals[c] = int(series.shape[0])
                    else:
                        kpi_vals[c] = float(series.mean())

        # display KPI metrics
        if kpi_vals:
            cols = st.columns(min(4, len(kpi_vals)))
            for i, (k, v) in enumerate(kpi_vals.items()):
                label = k.replace('_', ' ').title()
                # present percentage for pct
                if k == 'pct':
                    cols[i].metric(label, f"{v*100:.1f}%")
                elif k in ('size',):
                    cols[i].metric(label, f"{int(v)}")
                elif isinstance(v, float):
                    # format currency-like fields
                    if 'price' in k or 'value' in k:
                        cols[i].metric(label, f"${v:,.0f}")
                    else:
                        cols[i].metric(label, f"{v:.2f}")

        # show persona labels and recommendations; allow editing and saving
        personas_fp = PROC / 'cluster_personas.csv'
        # load existing personas if present
        if personas_fp.exists():
            try:
                personas_df = pd.read_csv(personas_fp)
            except Exception:
                personas_df = pd.DataFrame()
        else:
            personas_df = pd.DataFrame()

        st.markdown('### Personas & Recommendations')

        # if no personas exist, create sensible defaults based on sizes
        if personas_df.empty:
            size_order = counts.sort_values()
            defaults = {}
            # smallest -> high-value frequent, largest -> mainstream
            labels_sorted = list(size_order.index)
            for idx, lbl in enumerate(labels_sorted):
                if idx == 0:
                    defaults[lbl] = ('High-value Frequent', 'Prioritize premium listings and retention campaigns')
                elif idx == len(labels_sorted) - 1:
                    defaults[lbl] = ('Mainstream Value Seekers', 'Focus on acquisition and affordability messaging')
                else:
                    defaults[lbl] = ('Mid-price Segment', 'Recommend upsell and targeted listing recommendations')
            # build dataframe
            personas_df = pd.DataFrame([{'label': int(k), 'persona': v[0], 'recommendation': v[1]} for k, v in defaults.items()])

        # present editable fields per cluster
        persona_cols = {}
        rec_cols = {}
        for _, row in personas_df.sort_values('label').iterrows():
            lbl = int(row['label'])
            with st.expander(f"Cluster {lbl} persona", expanded=(lbl == sel)):
                persona_text = st.text_input(f'Persona (cluster {lbl})', value=str(row.get('persona', '')), key=f'persona_{lbl}')
                rec_text = st.text_area(f'Recommendation (cluster {lbl})', value=str(row.get('recommendation', '')), key=f'rec_{lbl}')
                persona_cols[lbl] = persona_text
                rec_cols[lbl] = rec_text

        if st.button('Save personas and recommendations'):
            out_rows = []
            for lbl in persona_cols:
                out_rows.append({'label': int(lbl), 'persona': persona_cols[lbl], 'recommendation': rec_cols[lbl]})
            out_df = pd.DataFrame(out_rows)
            out_df.to_csv(personas_fp, index=False)
            st.success(f'Saved personas to {personas_fp}')

        # show persona for selected cluster
        if sel in persona_cols:
            st.markdown('**Selected cluster persona**')
            st.write(persona_cols[sel])
            st.markdown('**Recommendation**')
            st.write(rec_cols[sel])

        # preview rows
        n_preview = st.slider('Rows to preview', min_value=5, max_value=200, value=10, key='preview_slider')
        st.markdown('### Sample rows')
        st.dataframe(df[df[label_choice] == sel].head(n_preview))

        # allow download of profiles and personas
        if not profile.empty:
            prof_csv = profile.to_csv(index=False).encode('utf-8')
            st.download_button('Download cluster profile CSV', data=prof_csv, file_name=f'profile_{label_choice}.csv')

        if not personas_df.empty:
            pers_csv = personas_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download personas CSV', data=pers_csv, file_name='cluster_personas.csv')
    except Exception as e:
        st.error('An error occurred while rendering the Segments tab — see traceback below')
        st.text(str(e))
        st.text(traceback.format_exc())
