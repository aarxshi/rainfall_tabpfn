# 🌧️Rainfall Forecast Post-Processing with TabPFN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
**GSoC Application Support Material:** This project was developed in preparation for Google Summer of Code, exploring machine learning for climate applications using open data.

## Overview

This project explores the use of **TabPFN**, a Transformer-based model optimized for tabular data, to **improve the accuracy of daily rainfall forecasts**. By learning patterns in historical forecast errors derived from open environmental data, TabPFN acts as a post-processing step to adjust initial simulations, leading to more accurate predictions. The core methodology involves machine learning, time-series analysis, and working with open climate datasets.

## Motivation & Relevance

Traditional rainfall forecasts often rely on numerical weather prediction (NWP) models or historical climatology (like 'daily normal' values). While valuable, these methods can exhibit systematic biases or fail to capture localized, short-term deviations accurately.

Machine learning offers a powerful approach to **post-process** these initial forecasts. By training a model on historical data, we can learn to predict the *expected error* of the baseline forecast under different conditions (e.g., time of year, location). Adding this predicted error back to the baseline forecast yields an adjusted, often more accurate, prediction.

**Why this matters for GSoC & OpenClimateFix:**

1.  **Improving Environmental Forecasts:** Accurate rainfall prediction is vital for agriculture, water resource management, and disaster preparedness.
2.  **ML for Climate Science:** Demonstrates a practical application of modern ML (Transformers for tabular data) on a real-world environmental problem.
3.  **Methodological Applicability:** The **post-processing technique** used here is highly relevant to other forecasting domains. For instance, similar methods are used by organizations like **OpenClimateFix** to improve solar energy forecasts by correcting biases in NWP model outputs.
4.  **Open Data & Reproducibility:** Leverages open government data and emphasizes a clear, reproducible workflow.

## Key Features

* **TabPFN Integration:** Utilizes the prior-diffusion-based TabPFN model for fast and effective tabular regression.
* **Forecast Error Correction:** Focuses on modeling the *error* of a baseline forecast (`Daily normal`) rather than predicting rainfall directly.
* **Open Dataset:** Uses publicly available rainfall data from India's NDAP.
* **Time-Series & Categorical Features:** Incorporates temporal features (month, year) and geographical identifiers (state, district) into the model.
* **Evaluation Framework:** Compares the Mean Absolute Error (MAE) of the adjusted forecast against the baseline.

## Dataset

* **Source:** [NDAP India – Rainfall Normal Dataset](https://ndap.niti.gov.in/) (Note: Provide the most direct link if possible, otherwise the main portal is fine).
* **Description:** Contains historical daily rainfall measurements and climatological 'normal' values aggregated at the district level across India.
* **Key Columns Used:**
    * `Calendar Day`: Date of observation.
    * `State`: State name.
    * `District`: District name.
    * `Daily actual`: Measured rainfall for the day (mm).
    * `Daily normal`: Historical average/climatological rainfall for that day and location (mm). Used here as the baseline forecast.
    * `Percentage of daily departure`: Calculated departure from normal.

## Methodology

The core workflow implemented in the accompanying notebook (`rainfall_forecast_tabpfn.ipynb`) is:

1.  **Load & Preprocess Data:** Read the dataset using Pandas, handle missing values, and convert date columns.
2.  **Baseline Forecast Simulation:** Use the `Daily normal` column as the initial, uncorrected forecast.
3.  **Calculate Forecast Error:** Compute the error of the baseline: `forecast_error = Daily actual - Daily normal`. This becomes the target variable for TabPFN.
4.  **Feature Engineering:** Extract temporal features (`Month`, `Year`) from the date. Encode categorical features (`State`, `District`) numerically.
5.  **Train TabPFN Regressor:** Train TabPFN to predict the `forecast_error` using features like the baseline forecast (`Daily normal`), temporal features, and location identifiers.
    * *Note:* Due to TabPFN's computational requirements on large datasets, training is performed on a representative sample.
6.  **Predict Forecast Error:** Use the trained TabPFN model to predict the forecast error on unseen test data.
7.  **Adjust Forecast:** Calculate the final, adjusted forecast: `Adjusted Forecast = Baseline Forecast (Daily normal) + Predicted Forecast Error`.
8.  **Evaluate:** Compare the MAE between the `Adjusted Forecast` and the `Daily actual` rainfall against the MAE between the `Baseline Forecast` and the `Daily actual` rainfall. Visualize prediction quality.

## Project Structure

```
rainfall-tabpfn/
├── rainfall_forecast_tabpfn.ipynb     # Main notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── rainfall_data.csv              # Input dataset
└── results/
    └── plots                          # Model outputs
```
## Getting Started

**Prerequisites:**

* Python (>= 3.8 recommended)
* Jupyter Notebook or JupyterLab

**Installation & Execution:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aarxshi/rainfall-tabpfn.git]
    cd rainfall-tabpfn
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: TabPFN installation might require specific dependencies or PyTorch)*
3.  **Prepare Data:** Ensure the input dataset (`data/data.csv`) is present.
4.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    # or
    # jupyter lab
    ```
5.  **Run the Notebook:** Open and execute the cells in `rainfall_forecast_tabpfn.ipynb`.

## GSoC Context & Learning Outcomes

This project served as a practical exercise in preparation for GSoC, focusing on techniques relevant to organizations like **OpenClimateFix**.

* **Alignment:** Demonstrates experience in applying ML to improve environmental time-series forecasts using open data – a core theme in climate-focused open-source work.
* **Methodology:** The ML post-processing approach is directly analogous to methods used for improving solar PV or wind energy forecasts based on NWP outputs.
* **Learning:** Provided hands-on experience with:
    * Using the novel TabPFN model for regression on real-world tabular data.
    * Preprocessing and feature engineering for environmental time-series data.
    * Evaluating forecast model performance (MAE, visualizations).
    * Managing dependencies and ensuring reproducibility (`requirements.txt`).
    * Working with geospatial variations within a tabular ML framework.

## Challenges & Learnings

* **Data Sparsity/Noise:** Handling missing values and potential inconsistencies in real-world datasets requires careful preprocessing and robust modeling choices.
* **Computational Constraints:** TabPFN, while powerful, has limitations on dataset size for training. Sampling strategies were necessary, highlighting trade-offs between performance and computational resources.
* **Generalization:** Ensuring the model generalizes across diverse geographical regions (different states/districts) requires careful feature engineering and validation. Learned the importance of spatial considerations even in non-explicitly geospatial models.

## Future Work

* **Incorporate Geospatial Features:** Explicitly add latitude/longitude or other spatial features.
* **Advanced Feature Engineering:** Include lagged rainfall values, weather variables from other sources (if available), or satellite data.
* **Explore Other Models:** Compare TabPFN performance against other ML models (e.g., Gradient Boosting, LSTMs).
* **Different Time Horizons:** Adapt the framework for weekly or monthly rainfall forecasts.
* **Transfer Learning:** Investigate if models trained on one region could be fine-tuned for another.
* **Application to Other Domains:** Apply the post-processing framework to solar PV or temperature forecast data.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue or submit a pull request.

## Contact

Interested in discussing ML for climate, open data, or GSoC? Feel free to connect via GitHub.
