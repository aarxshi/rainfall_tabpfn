# Rainfall Forecast Post-Processing with TabPFN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project exploring **forecast post-processing** for daily rainfall prediction in India, using TabPFN to correct systematic bias in climatological baseline forecasts.

## Overview

This project uses **TabPFN**, a transformer-based, prior-fitted tabular foundation model, to improve the accuracy of daily rainfall forecasts. Rather than predicting rainfall directly, TabPFN is trained to predict the *error* of a baseline climatological forecast (the historical "daily normal" for a given location and date). Adding this predicted error back to the baseline produces an adjusted forecast that is, on average, closer to the true observed rainfall.

## Motivation

Operational rainfall forecasts that reach farmers, irrigation departments, and disaster-preparedness agencies are typically built on numerical weather prediction (NWP) models or long-term climatological averages. While useful, these baselines carry systematic biases — they can consistently over- or under-estimate rainfall for a given location, season, or year, and often miss short-term, localized deviations from the historical pattern.

Forecast post-processing addresses this without replacing the baseline: instead of solving the harder problem of predicting rainfall from scratch, a model only needs to learn how wrong the baseline tends to be, and under what conditions. This is the same logic used in adjacent domains — for example, residual-correction techniques are used to improve solar and wind energy forecasts by correcting NWP biases. Applying it to rainfall has direct relevance for agricultural planning and water resource management.

## Dataset

- **Source:** [NDAP India – Rainfall Normal Dataset](https://ndap.niti.gov.in/dataset/7319), published on India's National Data and Analytics Platform (NITI Aayog, Government of India).
- **Size:** 961,752 rows x 30 columns, covering daily, weekly, cumulative, and monthly rainfall records at the district level across India.
- **Granularity:** One row per district per calendar day.

**Columns used:**

| Column | Description |
|---|---|
| `Calendar Day` | Date of observation |
| `State` / `District` | Location identifiers |
| `Daily actual` | Measured rainfall (mm) |
| `Daily normal` | Historical climatological average rainfall (mm) — used as the baseline forecast |
| `Percentage of daily departure` | Departure of actual from normal |

Roughly 10–11% of rows are missing `Daily actual` or `Daily normal` and were dropped prior to modelling.

## Methodology

1. **Baseline forecast:** `Daily normal` is treated as the initial, uncorrected forecast.
2. **Target definition:** `forecast_error = Daily actual - Daily normal` becomes the regression target, not rainfall itself.
3. **Feature engineering:** `Month` and `Year` are extracted from the date; `State` and `District` are label-encoded into numeric features.
4. **Model:** TabPFN is trained to predict `forecast_error` from the baseline value, temporal features, and location identifiers. As a prior-fitted model, it requires no manual hyperparameter tuning.
5. **Sampling:** TabPFN has practical limits on training-set size, so the model is trained on a representative sample of 1,000 rows and evaluated on a held-out sample of 500 rows (80/20 split, fixed seed).
6. **Forecast adjustment:** `Adjusted Forecast = Daily normal + Predicted Forecast Error`.
7. **Evaluation:** Mean Absolute Error (MAE) between the adjusted forecast and true observed rainfall.

## Results

The adjusted forecast achieved an **MAE of 3.71 mm** against true observed daily rainfall on the held-out 500-row test sample.

Plots are available in `results/plots/`:

- `Daily actual rainfall.png` — distribution of daily actual rainfall
- `Daily actual vs normal.png` — baseline ("Daily normal") vs. actual rainfall
- `Daily forecast errors.png` — distribution of daily forecast errors (actual − normal)
- `Forecast error by month.png` — seasonal pattern in forecast error
- `Model improvement.png` — baseline forecast vs. TabPFN-adjusted forecast, compared against true rainfall
- `Actual rainfall per month.png` — monthly rainfall totals
- `Average monthly rain.png` — average rainfall by month
- `Average rainfall across states.png` — average daily rainfall by state, illustrating geographic variation
- `Time series - actual rainfall.png` — actual rainfall over time

## Project Structure
```
rainfall-tabpfn/

├── rainfall_forecast_adjustment.ipynb   # Main notebook

├── README.md                            # This file

├── requirements.txt                     # Python dependencies

└── results/

└── plots/                           # Generated figures
```
## Getting Started

**Prerequisites:**
- Python >= 3.8
- Jupyter Notebook or JupyterLab

**Installation and execution:**

```bash
git clone https://github.com/aarxshi/rainfall_tabpfn.git
cd rainfall_tabpfn
pip install -r requirements.txt
```

> Note: TabPFN installation may require specific dependencies or a working PyTorch installation.

Place the input dataset at `data/data.csv`, then launch Jupyter and run `rainfall_forecast_adjustment.ipynb` top to bottom. All sampling steps use a fixed `random_state=42` for reproducibility.

## Limitations

- TabPFN's training-set constraints mean the model sees only 1,000 of roughly 850,000 usable rows, which likely limits generalization across India's diverse climatic regions.
- The baseline (unadjusted) MAE is not currently computed on the same test rows for a direct side-by-side comparison — a natural next step.
- No explicit geospatial features (e.g. latitude/longitude) are used; location is captured only via label-encoded categorical identifiers.

## Future Work

- Compute baseline MAE on the same test split for a direct before/after comparison.
- Incorporate explicit geospatial features (latitude/longitude).
- Add lagged rainfall values or external weather variables.
- Compare TabPFN against other regression models (e.g. gradient boosting, LSTMs).
- Extend the framework to weekly or monthly forecast horizons.
- Apply the same post-processing approach to other forecasting domains, such as solar or wind energy.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Issues and pull requests are welcome.
