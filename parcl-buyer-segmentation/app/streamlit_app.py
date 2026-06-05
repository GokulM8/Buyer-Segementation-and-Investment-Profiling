"""Streamlit entry point for the buyer segmentation dashboard."""

from __future__ import annotations

from pathlib import Path
import sys
import streamlit as st

# Ensure project root is on sys.path so `import app` works when launched
# from the workspace root or other locations.
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from app.tabs.behavior import render as render_behavior
from app.tabs.geographic import render as render_geographic
from app.tabs.insights import render as render_insights
from app.tabs.overview import render as render_overview
from app.tabs.export import render as render_export
from app.tabs.segments import render as render_segments
from app.tabs.shared import default_label_choice, label_columns, load_cluster_frame


st.set_page_config(page_title="Parcl Buyer Segmentation", page_icon="🏠", layout="wide")

st.markdown(
        """
        <style>
        :root {
            --card-bg: rgba(255,255,255,0.03);
            --muted: rgba(255,255,255,0.6);
            --accent: #2b7cff;
        }
        .block-container { padding-top: 1.5rem; padding-left: 2rem; padding-right: 2rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        .stMetric { background: var(--card-bg); padding: 0.75rem; border-radius: 0.75rem; }
        section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); padding-top: 1rem; }
        .card { background: var(--card-bg); padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.75rem; }
        .bundle-btn { background: var(--accent); color: white !important; padding: 0.6rem 1.0rem; border-radius: 0.5rem; }
        .file-list { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color: var(--muted); }
        .narrative { background: linear-gradient(90deg, rgba(43,124,255,0.06), rgba(43,124,255,0.02)); padding: 0.75rem; border-radius: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
)

df = load_cluster_frame()
label_cols = label_columns(df) if df is not None else []
if label_cols:
    st.session_state.setdefault('dashboard_label_choice', default_label_choice(label_cols))
    st.session_state.setdefault('dashboard_cluster_filter', 'All')
    with st.sidebar:
        st.markdown('### Dashboard Controls')
        st.session_state['dashboard_label_choice'] = st.selectbox(
            'Cluster label',
            label_cols,
            index=label_cols.index(st.session_state['dashboard_label_choice']) if st.session_state['dashboard_label_choice'] in label_cols else 0,
        )
        unique_clusters = []
        if df is not None and st.session_state['dashboard_label_choice'] in df.columns:
            unique_clusters = sorted(df[st.session_state['dashboard_label_choice']].dropna().unique().tolist())
        cluster_options = ['All'] + unique_clusters
        st.session_state['dashboard_cluster_filter'] = st.selectbox(
            'Cluster filter',
            cluster_options,
            index=cluster_options.index(st.session_state.get('dashboard_cluster_filter', 'All')) if st.session_state.get('dashboard_cluster_filter', 'All') in cluster_options else 0,
        )
        st.caption('These controls are shared across tabs.')
else:
    st.sidebar.info('Run clustering first to enable shared dashboard controls.')

st.title("Parcl Buyer Segmentation")
st.caption("Interactive dashboard for buyer segments, behavior, geography, and cluster insights.")

overview_tab, behavior_tab, geographic_tab, insights_tab, segments_tab, export_tab = st.tabs(
    ["Overview", "Behavior", "Geographic", "Insights", "Segments", "Export"]
)

with overview_tab:
    render_overview()

with behavior_tab:
    render_behavior()

with geographic_tab:
    render_geographic()

with insights_tab:
    render_insights()

with segments_tab:
    render_segments()

with export_tab:
    render_export()

