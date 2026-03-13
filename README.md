# 📈 StockSense AI — Time Series Forecasting Dashboard

> End-to-end stock market prediction system using ARIMA, LSTM & Prophet with an interactive Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

---

## 🚀 Features

- **Real-time Data** — Fetches live stock data via `yfinance`
- **3 Forecasting Models** — ARIMA, LSTM (Deep Learning), Facebook Prophet
- **Technical Indicators** — RSI, MACD, Bollinger Bands, Moving Averages
- **Anomaly Detection** — Z-score based spike detection
- **Interactive Charts** — Fully interactive Plotly visualizations
- **Model Comparison** — Side-by-side RMSE / MAE / MAPE metrics
- **Professional UI** — Dark-themed Streamlit dashboard

---

## 📁 Project Structure

```
stock_prediction/
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb       # Exploratory Data Analysis
│   ├── 02_ARIMA_Model.ipynb                 # ARIMA Forecasting
│   ├── 03_LSTM_Model.ipynb                  # LSTM Deep Learning
│   └── 04_Prophet_Model.ipynb               # Facebook Prophet
├── app/
│   └── streamlit_app.py                     # Main Streamlit Application
├── models/
│   └── saved_models/                        # Persisted model files
├── data/                                    # Cached CSV data
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository
bash
git clone https://github.com/yourname/stocksense-ai.git
cd stocksense-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
bash
pip install -r requirements.txt


### 4. Run Streamlit App
```bash
streamlit run app.py
```

---

## 📓 Jupyter Notebooks

Run notebooks in order for full pipeline:

| Notebook | Purpose |
|---|---|
| `01_EDA_and_Preprocessing.ipynb` | Data loading, visualization, stationarity tests |
| `02_ARIMA_Model.ipynb` | Classical time series forecasting |
| `03_LSTM_Model.ipynb` | Deep learning sequence modeling |
| `04_Prophet_Model.ipynb` | Facebook Prophet with seasonality |

```bash
jupyter notebook notebooks/
```

---

## 🏗️ Models Overview

### ARIMA
- ADF Test for stationarity
- Auto-selection of p, d, q parameters
- Confidence intervals on forecasts

### LSTM
- Sliding window sequences (60-day lookback)
- 2-layer LSTM with Dropout regularization
- MinMaxScaler normalization

### Prophet
- Handles holidays and seasonality automatically
- Uncertainty intervals
- Trend changepoint detection

---

## 📊 Metrics Used

| Metric | Description |
|---|---|
| RMSE | Root Mean Square Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |

---

## 🌐 Deployment

 http://localhost:8501

---

## 📌 Supported Tickers

Any valid Yahoo Finance ticker: `AAPL`, `GOOGL`, `MSFT`, `TSLA`, `BTC-USD`, `ETH-USD`, `NIFTY50.NS`, `RELIANCE.NS`, etc.

---

## 🛠️ Tech Stack

| Tool | Version |
|---|---|
| Python | 3.10+ |
| Streamlit | 1.31 |
| TensorFlow/Keras | 2.15 |
| Prophet | 1.1.5 |
| Plotly | 5.18 |
| yfinance | 0.2.36 |

---

## 📜 License
MIT License — free for personal and commercial use.
## Author
by Sai vignesh
