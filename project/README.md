# 📈 Stock Predictor

End-to-end workspace for training, evaluating, and visualizing stock-price prediction models on NSE equities (HDFCBANK, ICICIBANK, INFY, RELIANCE, SBIN, TCS, etc.).  
Both **Linear Regression** and **Random Forest** regressors convert OHLCV signals into next-close forecasts, while companion scripts/notebooks surface up/down classification metrics, confusion matrices, and dashboards.

---

## Quick Start

```bash
cd stock_predictor
python -m venv .venv && .venv\Scripts\activate        # or use your preferred env
pip install -r requirements.txt                       # create one via pip freeze if missing
```

> **Note**  
> Core dependencies: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `streamlit`, `yfinance`.

---

## 🗂 Key Components

| File / Folder | Purpose |
| --- | --- |
| `data/` | Cleaned NSE CSVs (`*_NS.csv`) used by all training flows. |
| `run_training.py` | Batch Linear Regression trainer + metrics exporter (`accuracy_report.csv`) + latest predictions per stock. |
| `train_random_forest_model.py` | Batch Random Forest trainer, writes `accuracy_report_RF.csv` and `model_comparison.csv`. |
| `evaluate_lr_updown.py` / `evaluate_rf_updown.py` | Turn regression outputs into **Up/Down** classification metrics (Accuracy, Precision/Recall/F1) + confusion matrices. |
| `model_training.ipynb` | Interactive notebook: data exploration, LR training, plotting, buy/sell signals, plus Step 8/9 tables showing both regression and classification performance for LR & RF. |
| `app_dashboard.py` | Streamlit dashboard for live inspection of predictions, indicators, and signals. |
| `PROJECT_ANALYSIS_REPORT.md` / `MODEL_PERFORMANCE_ANALYSIS.md` | Narrative reports summarizing datasets, modeling choices, KPIs, and insights. |
| `models/` | Persisted `.joblib` artifacts (both LR and RF variants). |

---

## 📊 Training & Evaluation Flows

### 1. Linear Regression (baseline)

```bash
python run_training.py
```

- Loads every `*_NS.csv`, trains LR model, saves to `models/<STOCK>_NS_model.joblib`.
- Metrics saved in `accuracy_report.csv`.
- Prints latest-day prediction summary (BUY/SELL suggestion based on delta).

### 2. Random Forest (ensemble)

```bash
python train_random_forest_model.py
```

- Mirrors LR pipeline using tuned `RandomForestRegressor`.
- Outputs `accuracy_report_RF.csv` and comparison table vs LR (`model_comparison.csv`).

### 3. Up/Down Classification Metrics

```bash
python evaluate_lr_updown.py
python evaluate_rf_updown.py
```

- Reconstructs the 80/20 split, compares predicted next-close vs previous close to define **Up/Down** labels.
- Prints per-stock Accuracy / Precision (weighted) / Recall (weighted) / F1 (weighted) + confusion matrices.
- CSV summaries: `lr_updown_classification_report.csv`, `rf_updown_classification_report.csv`.

---

## 🧪 Notebook Workflow (`model_training.ipynb`)

Open in Jupyter/VSCode and run sequentially:

1. **Setup & data discovery** – lists stocks, loads sample frames.
2. **Training loop** – fits LR models, logs MAE/RMSE/R², saves artifacts.
3. **Visualization** – actual vs predicted plots, buy/sell markers, recent predictions.
4. **Dashboards/Backtests** – optional cells for cumulative profit & signal inspection.
5. **Step 8 / Step 9** – display regression performance matrices (LR & RF) and Up/Down classification tables/confusion matrices inline.

Snapshot the run with `jupyter nbconvert --execute --inplace model_training.ipynb` (kernel `python3`).

---

## 🖥️ Dashboard

```bash
$Env:PYTHONUTF8='1'            # avoids Windows emoji encoding issues
streamlit run app_dashboard.py
```

- Visit `http://localhost:8501` (Streamlit prints the local URL).
- Select stock, view current predictions, indicators, and buy/sell recommendations.
- Warnings shown in terminal:
  - `InconsistentVersionWarning`: re-save models with current scikit-learn if needed.
  - `use_container_width` deprecation: update Streamlit calls to `width='stretch'` when convenient.

---

## 📁 Generated Artifacts

- `accuracy_report.csv` / `accuracy_report_RF.csv` – regression KPIs.
- `model_comparison.csv` – LR vs RF Test_R²/MAE per stock.
- `lr_updown_classification_report.csv` / `rf_updown_classification_report.csv`.
- Streamlit cache files (auto-created).

---

## ⚠️ Known Considerations

- **Model compatibility**: Models saved with older scikit-learn emit `InconsistentVersionWarning`. Retrain under the current version to silence.
- **UTF-8 on Windows**: Set `PYTHONUTF8=1` or remove emoji prints in scripts to avoid codec errors.
- **Streamlit API changes**: Replace `use_container_width` with `width='stretch'` before 2025‑12‑31.

---

## Development Guidelines

1. Keep data in `data/` clean and organized.
2. Update relevant reports (`PROJECT_ANALYSIS_REPORT.md`) and README to reflect major changes.
3. Test new features thoroughly before deployment.

Happy modeling! ✨
