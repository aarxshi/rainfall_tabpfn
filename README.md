# Rainfall Forecast Adjustment using TabPFN

This project demonstrates how rainfall forecast accuracy can be improved using TabPFN — a transformer-based meta-learning model for tabular data. It aligns with GSoC themes involving machine learning, open environmental data, and climate informatics.

---

## Objective

To post-process traditional rainfall simulations using TabPFN and improve forecast accuracy by learning from historical forecast errors.

---

## Why This Matters

Most rainfall forecasts rely on physical models or historical climatology, which may not capture localized deviations well. By using machine learning to learn from past forecast errors, we can make corrections that improve real-world relevance.

This is especially relevant for GSoC in areas such as:

- Earth observation and climate forecasting
- Environmental informatics and sustainability
- Open data science
- Meta-learning and few-shot learning models

---

## Project Structure

```
rainfall-tabpfn/
├── rainfall_forecast_tabpfn.ipynb     # Main notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── rainfall_data.csv              # Input dataset
└── results/
    └── predictions.csv, plots/        # Model outputs
```

---

## Dataset

- **Source**: NDAP India – [Rainfall Normal Dataset](https://ndap.niti.gov.in/dataset/7319)
- Contains daily rainfall values (actual and normal) by district and state
- Columns used: `Calendar Day`, `State`, `District`, `Daily actual`, `Daily normal`

---

## How It Works

1. Load and clean the rainfall dataset
2. Simulate a basic forecast using `Daily normal`
3. Train a `TabPFNRegressor` on:
   - `simulated_forecast`, `Month`, `Year`, `State`, `District`
4. Predict the forecast error
5. Adjust the forecast to reduce error

---

## Evaluation

- Metric: **Mean Absolute Error (MAE)**
- Comparison: Naive forecast vs. TabPFN-adjusted forecast
- Visualization: Scatter plot of true vs. predicted forecast error

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rainfall-tabpfn.git
   cd rainfall-tabpfn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook
   ```
4. Open and run `rainfall_forecast_tabpfn.ipynb`

---

## GSoC Relevance

This project showcases:

- Use of **TabPFN**, a state-of-the-art few-shot learner for tabular data
- Application of **machine learning in climate informatics**
- Integration of **open government datasets** (NDAP India)
- Reproducible ML pipelines in **Jupyter Notebooks**

### Potential Extensions:
- Add **geospatial or satellite-based features**
- Expand to **weekly/monthly forecasts**
- Incorporate **citizen science rainfall reports**
- Use **other climate indicators** as targets

---

## License

This project is open source and uses the **MIT License**.

---

## Contact

Feel free to fork, raise issues, or contribute! This work is part of my exploration in applying ML for climate data, and is potentially aligned with a GSoC proposal.