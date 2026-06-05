"""Geographic tab for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from app.tabs.shared import default_label_choice, label_columns, load_cluster_frame


def render() -> None:
    st.subheader("Geographic Analysis")
    st.write(
        "Use this tab to show where each segment concentrates by market, region, or property location."
    )

    proj_root = Path(__file__).resolve().parents[2]
    PROC = proj_root / 'data' / 'processed'

    df = load_cluster_frame()
    if df is None:
        st.info('Run clustering first to generate geographic visualizations')
        return

    label_cols = label_columns(df)
    if not label_cols:
        st.info('No cluster labels found')
        return
    label_choice = st.session_state.get('dashboard_label_choice', default_label_choice(label_cols))
    if label_choice not in label_cols:
        label_choice = default_label_choice(label_cols)

    cluster_filter = st.session_state.get('dashboard_cluster_filter', 'All')
    if cluster_filter != 'All':
        df = df[df[label_choice] == cluster_filter].copy()
        if df.empty:
            st.info('No rows match the selected cluster filter')
            return

    # if lat/lon present, show on map
    if {'latitude','longitude'}.issubset(set(df.columns)):
        st.write('Map: cluster locations')
        # show selected cluster or all
        sel = st.selectbox('Cluster to show', options=['All'] + sorted(df[label_choice].unique().tolist()))
        plot_df = df if sel == 'All' else df[df[label_choice] == sel]
        map_df = plot_df[['latitude', 'longitude']].dropna().copy()
        if not map_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.map(map_df, use_container_width=True)
            with c2:
                st.markdown('### Cluster geography')
                st.metric('Points on map', f'{len(map_df):,}')
                st.metric('Selected view', str(sel))
                if 'region' in plot_df.columns:
                    top_region = plot_df['region'].mode().iloc[0]
                    st.write(f'Most common region: {top_region}')
        else:
            st.info('No valid latitude/longitude rows found for the selected cluster')
    else:
        # fallback: show region or country counts
        if 'region' in df.columns:
            st.write('Cluster counts by region')
            ct = df.groupby(['region', label_choice]).size().unstack(fill_value=0)
            st.dataframe(ct)
            st.bar_chart(ct.sum(axis=1))
        elif 'country' in df.columns:
            st.write('Cluster counts by country')
            ct = df.groupby(['country', label_choice]).size().unstack(fill_value=0)
            st.dataframe(ct)
            st.bar_chart(ct.sum(axis=1))
        else:
            st.info('No geographic columns (latitude/longitude/region/country) found in processed data')
