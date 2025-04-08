# 🌧️ Rainfall Forecast Adjustment using TabPFN

This project demonstrates how rainfall forecast accuracy can be improved using **TabPFN** — a transformer-based meta-learning model for tabular data. It involves machine learning, open environmental data, and climate informatics.

---

## Objective

To post-process traditional rainfall simulations using TabPFN and improve forecast accuracy by learning from historical forecast errors.

---

## Why This Matters

Most rainfall forecasts rely on physical models or historical climatology, which may not capture localized deviations well. By using machine learning to learn from past forecast errors, we can make corrections that improve real-world relevance.

This approach has applications in:

- Earth observation and climate forecasting  
- Environmental informatics and sustainability  
- Open data science  
- Meta-learning and few-shot learning research  

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
- Key columns used:
  - `Calendar Day`
  - `State`
  - `District`
  - `Daily actual`
  - `Daily normal`

---

## How It Works

1. Load and clean the rainfall dataset  
2. Simulate a basic forecast using `Daily normal`  
3. Train a `TabPFNRegressor` on:
   - `simulated_forecast`, `Month`, `Year`, `State`, `District`  
4. Predict the forecast error  
5. Adjust the forecast to reduce the error  

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

This prototype was developed as part of my GSoC preparation with **OpenClimateFix**. It demonstrates:

- Hands-on use of **TabPFN** for real-world time-series regression  
- Practical ML integration with **open government datasets**  
- Commitment to **reproducible, documented, and open-source** climate tools  
- Exploration of **ML post-processing** for improving weather forecasts  

## Challenges Faced

- Managing sparse or noisy inputs across districts  
- Avoiding overfitting on small windows of error data  
- Ensuring spatial generalization across Indian regions  

---

## Future Work

- Add **satellite and geospatial features**  
- Expand to **weekly/monthly rainfall** trends  
- Use **ensemble forecasts** for more robust inputs  
- Apply the same framework to **solar PV or temperature forecasts**

---

## License

This project is released under the **MIT License**. Contributions welcome!

---

## Contact

I'm exploring how open-source ML tools can solve real-world climate challenges.  
Feel free to open issues, submit PRs, or connect via GitHub.
