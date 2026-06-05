"""Export tab for the Streamlit dashboard (polished report center)."""

from __future__ import annotations

from pathlib import Path
import io
import zipfile
import streamlit as st
import pandas as pd

from app.tabs.shared import label_columns, load_cluster_frame, default_label_choice


def render() -> None:
    st.subheader('Export & Reporting Bundle')
    st.write('A single downloadable reporting bundle with the key artifacts, plus individual file links for inspection.')

    proj_root = Path(__file__).resolve().parents[2]
    PROC = proj_root / 'data' / 'processed'
    OUT = proj_root / 'outputs' / 'figures'

    df = load_cluster_frame()
    label_cols = label_columns(df) if df is not None else []

    # Top metrics
    c1, c2, c3 = st.columns([2, 2, 2])
    c1.metric('Clients', f"{len(df) if df is not None else 0:,}")
    c2.metric('Processed files', f"{len(list(PROC.glob('*.csv'))):,}")
    c3.metric('Figures', f"{len(list(OUT.glob('*.png'))):,}")

    st.markdown('')

    # Bundle creation and download (centered)
    bundle_col1, bundle_col2, _ = st.columns([3, 3, 1])
    with bundle_col1:
        st.markdown('#### Reporting bundle')
        st.write('Includes processed cluster CSVs, profile tables, ANOVA/Tukey outputs, personas, and key figures.')
    with bundle_col2:
        bundle_buffer = io.BytesIO()
        with zipfile.ZipFile(bundle_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            file_map = {
                'clients_clusters_all.csv': PROC / 'clients_clusters_all.csv',
                'clients_clusters_kmeans.csv': PROC / 'clients_clusters_kmeans.csv',
                'profile_kmeans_label.csv': PROC / 'profile_kmeans_label.csv',
                'anova_results_top_features.csv': PROC / 'anova_results_top_features.csv',
                'cluster_personas.csv': PROC / 'cluster_personas.csv',
                'one_page_summary.md': proj_root / 'paper' / 'one_page_summary.md',
            }
            for name, fp in file_map.items():
                if fp.exists():
                    try:
                        zf.writestr(name, fp.read_bytes())
                    except Exception:
                        pass
            for fp in sorted(OUT.glob('*.png')):
                try:
                    zf.writestr(f'figures/{fp.name}', fp.read_bytes())
                except Exception:
                    pass
        bundle_buffer.seek(0)
        st.download_button('Download reporting bundle (.zip)', data=bundle_buffer.getvalue(), file_name='parcl_reporting_bundle.zip', key='bundle_dl', use_container_width=True)

    st.markdown('---')
    st.markdown('### Included files')
    file_map_list = [
        ('Cluster labels (all)', PROC / 'clients_clusters_all.csv'),
        ('KMeans clusters', PROC / 'clients_clusters_kmeans.csv'),
        ('Cluster profile', PROC / 'profile_kmeans_label.csv'),
        ('ANOVA results', PROC / 'anova_results_top_features.csv'),
        ('Personas', PROC / 'cluster_personas.csv'),
        ('One-page summary', proj_root / 'paper' / 'one_page_summary.md'),
    ]
    for label, fp in file_map_list:
        if fp.exists():
            cols = st.columns([6, 2])
            cols[0].markdown(f'**{label}** — {fp.name}')
            try:
                cols[1].download_button('Download', data=fp.read_bytes(), file_name=fp.name, key=f'dl_{fp.name}')
            except Exception:
                cols[1].button('Open')
        else:
            st.markdown(f'• {label}: *not found*')

    st.markdown('---')
    st.markdown('### Figures (preview)')
    figure_files = sorted(OUT.glob('*.png'))
    if figure_files:
        preview_cols = st.columns(min(4, len(figure_files[:8])))
        for i, fp in enumerate(figure_files[:8]):
            try:
                preview_cols[i % len(preview_cols)].image(str(fp), use_column_width=True, caption=fp.name)
            except Exception:
                preview_cols[i % len(preview_cols)].write(fp.name)
    else:
        st.info('No figures found in outputs/figures yet.')
