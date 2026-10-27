# Hastings FNOL Case Study

This repository contains analysis and modeling work for the Hastings First Notice of Loss (FNOL) data science case study.

## Project layout

- `brief/` — original case-study prompt material.
- `data/` — datasets by lifecycle stage:
  - `data/raw/` immutable input data.
  - `data/interim/` temporary transformed outputs.
  - `data/processed/` model-ready datasets.
- `environment/` — reproducible environment specification.
- `notebooks/` — exploratory and reporting notebooks.
- `src/` — reusable Python modules for data, features, models, and utilities.
- `reports/figures/` — exported visualizations and report assets.
- `tests/` — lightweight automated tests.

## Quick start

1. Create and activate environment from `environment/environment.yml`.
2. Register the kernel for notebook use.
3. Keep notebook experimentation in `notebooks/` and move reusable logic into `src/`.

## Recommended workflow

1. Load data with functions from `src/data/`.
2. Build features in `src/features/`.
3. Train/evaluate baselines in `src/models/`.
4. Export charts to `reports/figures/` and summarize findings in a notebook/report.

## Notes

- Your source Excel currently sits in `data/`.
- Best practice is to place it under `data/raw/` once your notebook paths are updated.