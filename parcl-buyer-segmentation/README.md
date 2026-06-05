# Parcl Buyer Segmentation

This project scaffolds a buyer segmentation and investment profiling workflow for the Parcl data set.

## Structure

- `data/raw/` for source files copied from the original data drop
- `data/processed/` for cleaned and encoded outputs
- `notebooks/` for EDA, preprocessing, clustering, and interpretation
- `src/` for reusable preprocessing and clustering utilities
- `app/` for the Streamlit dashboard
- `outputs/` for figures and saved models
- `paper/` for the written research deliverable

## Data handling

The reusable loaders first look in `data/raw/`. If the CSVs are still in the sibling `Parcl Co Limited/` folder, the helper functions can read them there without changing the original files.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebooks in order:

- `notebooks/01_eda.ipynb`
- `notebooks/02_preprocessing.ipynb`
- `notebooks/03_clustering.ipynb`
- `notebooks/04_interpretation.ipynb`

3. Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Notes

- Keep the original CSV files unchanged.
- Save generated charts to `outputs/figures/`.
- Save trained models to `outputs/models/`.
