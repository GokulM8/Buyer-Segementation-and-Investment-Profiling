"""Behavior tab for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from app.tabs.shared import default_label_choice, label_columns, load_cluster_frame


def render() -> None:
    st.subheader("Investor Behavior")
    st.write(
        "Use this tab to compare trading frequency, holding patterns, deal size, and other behavioral signals."
    )

    proj_root = Path(__file__).resolve().parents[2]
    PROC = proj_root / 'data' / 'processed'

    df = load_cluster_frame()
    if df is None:
        st.info('Run clustering first to generate behavior visualizations')
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

    st.markdown('### Behavioral KPIs by cluster')
    # purchases_count distribution
    if 'purchases_count' in df.columns:
        st.write('Purchases count distribution by cluster')
        fig, ax = plt.subplots()
        sns.boxplot(x=label_choice, y='purchases_count', data=df, ax=ax)
        st.pyplot(fig)

    # recency (days since last purchase)
    if 'last_purchase_date' in df.columns:
        try:
            df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
            df['recency_days'] = (pd.Timestamp.today() - df['last_purchase_date']).dt.days
            st.write('Recency (days since last purchase) by cluster')
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=label_choice, y='recency_days', data=df, ax=ax2)
            st.pyplot(fig2)
        except Exception:
            st.info('Could not compute recency from last_purchase_date')

    # average sale price distribution
    if 'avg_sale_price' in df.columns:
        st.write('Average sale price by cluster')
        st.bar_chart(df.groupby(label_choice)['avg_sale_price'].median())
