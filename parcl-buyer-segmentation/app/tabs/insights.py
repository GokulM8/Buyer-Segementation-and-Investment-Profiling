"""Insights tab for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import pandas as pd
import glob

from app.tabs.shared import default_label_choice, label_columns, load_cluster_frame



def render() -> None:
    st.subheader("Segment Insights")
    st.write(
        "Use this tab to summarize the defining traits of each segment and surface actionable recommendations."
    )
    proj_root = Path(__file__).resolve().parents[2]
    PROC = proj_root / 'data' / 'processed'

    df = load_cluster_frame()
    label_choice = None
    if df is not None:
        label_cols = label_columns(df)
        if label_cols:
            label_choice = st.session_state.get('dashboard_label_choice', default_label_choice(label_cols))
            if label_choice not in label_cols:
                label_choice = default_label_choice(label_cols)

    anova_fp = PROC / 'anova_results_top_features.csv'
    if anova_fp.exists():
        anova = pd.read_csv(anova_fp)
        st.markdown('### ANOVA results (top features)')
        st.dataframe(anova)
    else:
        st.info('ANOVA results not found (run analysis scripts)')

    # list Tukey significant CSVs
    tukey_files = sorted(PROC.glob('tukey_significant_*.csv'))
    if tukey_files:
        st.markdown('### Tukey HSD — significant pairwise differences')
        for f in tukey_files:
            st.markdown(f'**{f.name}**')
            try:
                tdf = pd.read_csv(f)
                st.dataframe(tdf)
            except Exception:
                st.write('Could not read', f.name)
    else:
        st.info('No Tukey summary CSVs found')

    # personas and recommendations
    personas_fp = PROC / 'cluster_personas.csv'
    if personas_fp.exists():
        p = pd.read_csv(personas_fp)
        st.markdown('### Personas & Recommendations')
        st.dataframe(p)
        if label_choice is not None and st.session_state.get('dashboard_cluster_filter', 'All') != 'All':
            selected_cluster = st.session_state['dashboard_cluster_filter']
            st.markdown(f'### Selected cluster focus: {selected_cluster}')
            match = p[p['label'] == int(selected_cluster)] if str(selected_cluster).isdigit() else p[p['label'] == selected_cluster]
            if not match.empty:
                st.write(match.iloc[0].to_dict())
    else:
        st.info('No saved personas found. Create them in the Segments tab.')

    if df is not None and label_choice is not None and 'size' in locals():
        st.markdown('### Narrative summary')
        cluster_filter = st.session_state.get('dashboard_cluster_filter', 'All')
        if cluster_filter != 'All' and label_choice in df.columns:
            subset = df[df[label_choice] == cluster_filter]
            if not subset.empty:
                avg_price = subset['avg_sale_price'].median() if 'avg_sale_price' in subset.columns else None
                purchase_count = subset['purchases_count'].median() if 'purchases_count' in subset.columns else None
                parts = [f'Cluster {cluster_filter} is the active focus.']
                if avg_price is not None:
                    parts.append(f'Its median sale price is about ${avg_price:,.0f}.')
                if purchase_count is not None:
                    parts.append(f'Median purchases count is {purchase_count:.1f}.')
                st.success(' '.join(parts))
        elif 'cluster_personas.csv' and personas_fp.exists():
            st.info('Use the sidebar cluster filter to switch the narrative to a specific segment.')
