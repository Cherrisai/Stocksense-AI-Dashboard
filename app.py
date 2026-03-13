"""
StockSense AI by Saivignesh — Professional Stock Market Prediction Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI by Saivignesh",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #050810;
    color: #e2e8f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #060912 100%);
    border-right: 1px solid #1e2a45;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1526, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 20px rgba(0,180,255,0.05);
  }
  [data-testid="stMetricLabel"]  { color: #64748b !important; font-size: 0.78rem !important; }
  [data-testid="stMetricValue"]  { color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; }
  [data-testid="stMetricDelta"]  { font-size: 0.78rem !important; }

  /* Header strip */
  .hero-header {
    background: linear-gradient(135deg, #0d1526 0%, #0a1628 50%, #050810 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .hero-subtitle { color: #64748b; font-size: 0.95rem; margin-top: 0.3rem; }

  /* Section cards */
  .section-card {
    background: linear-gradient(135deg, #0d1526, #0a1220);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
  }
  .section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #00d4ff;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* Ticker badge */
  .ticker-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0a3a5c, #0d2a45);
    border: 1px solid #00d4ff44;
    border-radius: 8px;
    padding: 0.2rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #00d4ff;
    font-weight: 600;
  }

  /* Anomaly highlight */
  .anomaly-box {
    background: linear-gradient(135deg, #2d0a0a, #1a0505);
    border: 1px solid #dc262644;
    border-radius: 10px;
    padding: 1rem;
    color: #fca5a5;
    font-size: 0.88rem;
  }
  .good-box {
    background: linear-gradient(135deg, #0a2d1a, #051a0a);
    border: 1px solid #16a34a44;
    border-radius: 10px;
    padding: 1rem;
    color: #86efac;
    font-size: 0.88rem;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #0a0f1e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0a3a5c, #0d2a45) !important;
    color: #00d4ff !important;
  }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #0a3a5c, #0d2a45);
    border: 1px solid #00d4ff44;
    color: #00d4ff;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    width: 100%;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    border-color: #00d4ff;
    box-shadow: 0 0 20px rgba(0,212,255,0.2);
  }

  /* Selectbox / slider */
  .stSelectbox > div > div, .stMultiSelect > div > div {
    background: #0d1526 !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
  }

  div[data-testid="stSlider"] .rc-slider-track { background: #00d4ff; }
  div[data-testid="stSlider"] .rc-slider-handle {
    border-color: #00d4ff;
    background: #00d4ff;
  }

  /* Divider */
  hr { border-color: #1e3a5f; margin: 1rem 0; }

  /* Table */
  .dataframe { font-size: 0.82rem !important; }

  /* Spinner */
  .stSpinner > div { border-top-color: #00d4ff !important; }

  /* Footer */
  footer { visibility: hidden; }
  .footer-custom {
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #1e3a5f;
    margin-top: 2rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "NFLX", "BTC-USD", "ETH-USD", "RELIANCE.NS", "TCS.NS",
    "INFY.NS", "SPY", "QQQ",
]

@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Technical indicators
    df["MA_20"]  = df["Close"].rolling(20).mean()
    df["MA_50"]  = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(20).std() * np.sqrt(252) * 100
    return df.dropna()


def get_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


def rsi_signal(rsi: float) -> tuple:
    if rsi < 30:
        return "🟢 Oversold — Potential Buy", "#22c55e"
    elif rsi > 70:
        return "🔴 Overbought — Potential Sell", "#ef4444"
    else:
        return "🟡 Neutral", "#eab308"


def detect_anomalies(series: pd.Series, z_thresh: float = 2.5) -> pd.Series:
    mean = series.rolling(30).mean()
    std  = series.rolling(30).std()
    return np.abs((series - mean) / (std + 1e-9)) > z_thresh


def fmt(val, prefix="$", decimals=2):
    if pd.isna(val):
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1.5rem;'>
      <div style='font-size:2.5rem'></div>
      <div style='font-size:1.3rem; font-weight:700; background:linear-gradient(135deg,#00d4ff,#a855f7);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        StockSense AI
      </div>
      <div style='color:#334155; font-size:0.78rem; margin-top:2px;'>
        ML-Powered Forecasting
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Configuration")

    ticker_input = st.text_input("🔍 Ticker Symbol", value="AAPL", max_chars=15)
    ticker = ticker_input.strip().upper()

    st.markdown("**Popular Tickers**")
    cols_t = st.columns(3)
    for i, t in enumerate(POPULAR_TICKERS[:9]):
        if cols_t[i % 3].button(t, key=f"btn_{t}"):
            ticker = t

    st.markdown("---")

    period_map = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y", "2 Years": "2y", "5 Years": "5y",
    }
    period_label = st.selectbox(" Data Period", list(period_map.keys()), index=3)
    period = period_map[period_label]

    forecast_days = st.slider(" Forecast Horizon (days)", 7, 90, 30, step=7)

    model_choice = st.selectbox(
        " Forecast Model",
        ["ARIMA (Classical)", "LSTM (Deep Learning)", "Prophet (Meta AI)", "All Models"],
    )

    st.markdown("---")
    st.markdown("### Chart Options")
    show_ma   = st.checkbox("Moving Averages", value=True)
    show_bb   = st.checkbox("Bollinger Bands", value=True)
    show_vol  = st.checkbox("Volume", value=True)
    show_rsi  = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=False)

    st.markdown("---")
    run_btn = st.button("Analyze & Forecast", type="primary")

    st.markdown("""
    <div style='color:#334155; font-size:0.72rem; margin-top:1rem; line-height:1.6;'>
      ⚠️ Educational only. Not financial advice.<br>
      Data: Yahoo Finance · Models: ARIMA, LSTM, Prophet
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='hero-header'>
  <div class='hero-title'>StockSense AI Dashboard</div>
  <div class='hero-subtitle'>
    Machine Learning · Technical Analysis · Multi-Model Forecasting
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span class='ticker-badge'>{ticker}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner(f"Loading {ticker} data…"):
    df = load_data(ticker, period)

if df.empty:
    st.error(f"Could not load data for **{ticker}**. Please check the ticker symbol.")
    st.stop()

info = get_info(ticker)

# ── KPI Row ──────────────────────────────────────────────────────────────
latest       = df["Close"].iloc[-1]
prev         = df["Close"].iloc[-2]
day_chg      = latest - prev
day_chg_pct  = (day_chg / prev) * 100
week_high    = df["High"].tail(5).max()
week_low     = df["Low"].tail(5).min()
rsi_val      = df["RSI"].iloc[-1]
volatility   = df["Volatility"].iloc[-1]
avg_volume   = df["Volume"].tail(20).mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(" Price",      fmt(latest),          f"{day_chg:+.2f} ({day_chg_pct:+.2f}%)")
col2.metric(" 5D High",    fmt(week_high))
col3.metric("5D Low",     fmt(week_low))
col4.metric("RSI (14)",    f"{rsi_val:.1f}")
col5.metric(" Volatility", f"{volatility:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Company Info ─────────────────────────────────────────────────────────
if info:
    name    = info.get("longName", ticker)
    sector  = info.get("sector", "—")
    market_cap = info.get("marketCap", None)
    pe_ratio   = info.get("trailingPE", None)
    div_yield  = info.get("dividendYield", None)

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.markdown(f"<div class='section-card'><div class='section-title'> Company</div><b>{name}</b><br><span style='color:#64748b;font-size:0.82rem'>{sector}</span></div>", unsafe_allow_html=True)
    ic2.metric("Market Cap",  f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
    ic3.metric("P/E Ratio",   f"{pe_ratio:.1f}" if pe_ratio else "N/A")
    ic4.metric("Div. Yield",  f"{div_yield*100:.2f}%" if div_yield else "N/A")


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Chart & Indicators",
    "AI Forecast",
    " Anomaly Detection",
    " Model Comparison",
    " Data Explorer",
])


# ══════════════════════════════════════════════
#  TAB 1 — CHART & INDICATORS
# ══════════════════════════════════════════════
with tab1:
    rows = 1
    if show_vol:  rows += 1
    if show_rsi:  rows += 1
    if show_macd: rows += 1

    row_heights = [0.55] + [0.15] * (rows - 1)
    specs       = [[{"type": "candlestick"}]] + [[{"type": "scatter"}]] * (rows - 1)

    fig = make_subplots(
        rows=rows, cols=1,
        row_heights=row_heights,
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=specs,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    if show_ma:
        for ma, color, w in [("MA_20","#00d4ff",1.2), ("MA_50","#ff6b35",1.2), ("MA_200","#a855f7",1.5)]:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma,
                line=dict(color=color, width=w)), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="#eab308", dash="dot", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="#eab308", dash="dot", width=1),
            fill="tonexty", fillcolor="rgba(234,179,8,0.04)"), row=1, col=1)

    current_row = 2
    if show_vol:
        colors = ["#22c55e" if r >= 0 else "#ef4444" for r in df["Daily_Return"]]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.6), row=current_row, col=1)
        current_row += 1

    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#00d4ff", width=1.2)), row=current_row, col=1)
        fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=0.8), row=current_row, col=1)
        fig.add_hline(y=30, line=dict(color="#22c55e", dash="dash", width=0.8), row=current_row, col=1)
        current_row += 1

    if show_macd:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#00d4ff", width=1.2)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#ff6b35", width=1)), row=current_row, col=1)
        macd_hist = df["MACD"] - df["MACD_Signal"]
        fig.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram",
            marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in macd_hist],
            opacity=0.5), row=current_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#050810",
        plot_bgcolor="#050810",
        height=700,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(bgcolor="#0d1526", bordercolor="#1e3a5f",
                    borderwidth=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#0d1f35", row=i, col=1)
        fig.update_yaxes(gridcolor="#0d1f35", row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # RSI Signal
    sig_text, sig_color = rsi_signal(rsi_val)
    st.markdown(f"""
    <div class='section-card' style='border-color:{sig_color}33;'>
      <div class='section-title'> RSI Signal</div>
      <span style='color:{sig_color}; font-weight:600; font-size:1rem;'>{sig_text}</span>
      <span style='color:#64748b; margin-left:1rem; font-size:0.85rem;'>RSI = {rsi_val:.1f}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — AI FORECAST
# ══════════════════════════════════════════════
with tab2:
    close_series = df["Close"].dropna()

    def run_arima(series, steps):
        order = (2, 1, 2)
        try:
            m = ARIMA(series, order=order).fit()
            fc   = m.get_forecast(steps=steps)
            mean = fc.predicted_mean
            ci   = fc.conf_int()
            future_idx = pd.bdate_range(start=series.index[-1], periods=steps + 1)[1:]
            return pd.Series(mean.values, index=future_idx), ci.values, None
        except Exception as e:
            return None, None, str(e)

    def run_lstm_simple(series, steps, lookback=60):
        """Simplified LSTM using only numpy — avoids Keras import issues in prod."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            vals   = series.values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(vals)
            X, y = [], []
            for i in range(lookback, len(scaled)):
                X.append(scaled[i-lookback:i, 0])
                y.append(scaled[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            split = int(len(X) * 0.85)
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(32),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X[:split], y[:split], epochs=20, batch_size=32, verbose=0)
            # Forecast
            last_seq = scaled[-lookback:].reshape(1, lookback, 1)
            preds = []
            for _ in range(steps):
                p = model.predict(last_seq, verbose=0)[0, 0]
                preds.append(p)
                last_seq = np.roll(last_seq, -1, axis=1)
                last_seq[0, -1, 0] = p
            fc_vals = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            future_idx = pd.bdate_range(start=series.index[-1], periods=steps + 1)[1:]
            return pd.Series(fc_vals, index=future_idx), None
        except Exception as e:
            return None, str(e)

    def run_prophet_simple(series, steps):
        try:
            from prophet import Prophet
            pdf = series.reset_index()
            pdf.columns = ["ds", "y"]
            pdf["ds"] = pd.to_datetime(pdf["ds"])
            m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                        yearly_seasonality=True, interval_width=0.95)
            m.fit(pdf)
            future  = m.make_future_dataframe(periods=steps)
            fc      = m.predict(future)
            fc_future = fc[fc["ds"] > series.index[-1]]
            idx = pd.DatetimeIndex(fc_future["ds"].values)
            return (pd.Series(fc_future["yhat"].values, index=idx),
                    fc_future[["yhat_lower","yhat_upper"]].values, None)
        except Exception as e:
            return None, None, str(e)

    models_to_run = {
        "ARIMA (Classical)": "arima",
        "LSTM (Deep Learning)": "lstm",
        "Prophet (Meta AI)": "prophet",
        "All Models": "all",
    }[model_choice]

    run_arima_flag   = models_to_run in ("arima",   "all")
    run_lstm_flag    = models_to_run in ("lstm",    "all")
    run_prophet_flag = models_to_run in ("prophet", "all")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=close_series.index[-120:], y=close_series[-120:],
        name="Historical", line=dict(color="#94a3b8", width=1.5),
    ))

    results_summary = {}

    if run_arima_flag:
        with st.spinner("Running ARIMA model…"):
            fc_arima, ci_arima, err = run_arima(close_series, forecast_days)
        if fc_arima is not None:
            fig_fc.add_trace(go.Scatter(x=fc_arima.index, y=fc_arima,
                name="ARIMA", line=dict(color="#ff6b35", dash="dash", width=2)))
            if ci_arima is not None:
                fig_fc.add_trace(go.Scatter(
                    x=list(fc_arima.index) + list(reversed(fc_arima.index)),
                    y=list(ci_arima[:,1]) + list(reversed(ci_arima[:,0])),
                    fill="toself", fillcolor="rgba(255,107,53,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), name="ARIMA CI", showlegend=False))
            # Backtest metrics
            split = int(len(close_series) * 0.85)
            train_s, test_s = close_series[:split], close_series[split:]
            try:
                bt_m = ARIMA(train_s, order=(2,1,2)).fit()
                bt_pred = bt_m.forecast(len(test_s))
                rmse = np.sqrt(mean_squared_error(test_s, bt_pred))
                mae  = mean_absolute_error(test_s, bt_pred)
                mape = np.mean(np.abs((test_s.values - bt_pred.values) / test_s.values)) * 100
                results_summary["ARIMA"] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
            except Exception:
                pass
        else:
            st.warning(f"ARIMA error: {err}")

    if run_lstm_flag:
        with st.spinner("Training LSTM model (may take ~1 min)…"):
            fc_lstm, err = run_lstm_simple(close_series, forecast_days)
        if fc_lstm is not None:
            fig_fc.add_trace(go.Scatter(x=fc_lstm.index, y=fc_lstm,
                name="LSTM", line=dict(color="#a855f7", dash="dash", width=2)))
        else:
            st.warning(f"LSTM error: {err}")

    if run_prophet_flag:
        with st.spinner("Running Prophet model…"):
            fc_prophet, ci_prophet, err = run_prophet_simple(close_series, forecast_days)
        if fc_prophet is not None:
            fig_fc.add_trace(go.Scatter(x=fc_prophet.index, y=fc_prophet,
                name="Prophet", line=dict(color="#22c55e", dash="dash", width=2)))
            if ci_prophet is not None:
                fig_fc.add_trace(go.Scatter(
                    x=list(fc_prophet.index) + list(reversed(fc_prophet.index)),
                    y=list(ci_prophet[:,1]) + list(reversed(ci_prophet[:,0])),
                    fill="toself", fillcolor="rgba(34,197,94,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), name="Prophet CI", showlegend=False))
        else:
            st.warning(f"Prophet error: {err}")

    fig_fc.update_layout(
        title=f"{ticker} — {forecast_days}-Day AI Forecast",
        template="plotly_dark",
        paper_bgcolor="#050810",
        plot_bgcolor="#050810",
        height=480,
        legend=dict(bgcolor="#0d1526", bordercolor="#1e3a5f", borderwidth=1),
        xaxis=dict(gridcolor="#0d1f35"),
        yaxis=dict(gridcolor="#0d1f35", title="Price (USD)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Metrics cards
    if results_summary:
        st.markdown("<div class='section-title'> Backtest Metrics</div>", unsafe_allow_html=True)
        mc = st.columns(len(results_summary))
        for i, (mdl_name, metrics) in enumerate(results_summary.items()):
            with mc[i]:
                st.markdown(f"""
                <div class='section-card' style='text-align:center;'>
                  <div style='color:#00d4ff; font-weight:700; font-size:1rem; margin-bottom:0.8rem;'>
                    {mdl_name}
                  </div>
                  <div style='font-family:JetBrains Mono,monospace; font-size:0.88rem; line-height:2;'>
                    RMSE &nbsp; <b style='color:#e2e8f0'>{metrics['RMSE']:.2f}</b><br>
                    MAE &nbsp;&nbsp; <b style='color:#e2e8f0'>{metrics['MAE']:.2f}</b><br>
                    MAPE &nbsp; <b style='color:#e2e8f0'>{metrics['MAPE']:.1f}%</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab3:
    anomalies    = detect_anomalies(df["Close"])
    anom_dates   = df.index[anomalies]
    anom_prices  = df["Close"][anomalies]
    anom_count   = anomalies.sum()

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
        line=dict(color="#64748b", width=1), opacity=0.8))
    fig_anom.add_trace(go.Scatter(x=df.index, y=df["Close"].where(anomalies),
        name="⚠️ Anomaly", mode="markers",
        marker=dict(color="#ef4444", size=9, symbol="x",
                    line=dict(width=1.5, color="#fca5a5"))))
    fig_anom.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(30).mean(),
        name="30D Mean", line=dict(color="#a855f7", dash="dot", width=1.2)))
    fig_anom.update_layout(
        title=f"{ticker} — Anomaly Detection (Z-Score > 2.5)",
        template="plotly_dark", paper_bgcolor="#050810", plot_bgcolor="#050810",
        height=420, xaxis=dict(gridcolor="#0d1f35"), yaxis=dict(gridcolor="#0d1f35"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    if anom_count > 0:
        st.markdown(f"""
        <div class='anomaly-box'>
          ⚠️ <b>{anom_count} anomalous price events detected</b> in the selected period.<br>
          These are statistically unusual movements (Z-Score &gt; 2.5) that may indicate
          news events, earnings releases, or market disruptions.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        anom_df = pd.DataFrame({
            "Date":  anom_dates.strftime("%Y-%m-%d"),
            "Price": anom_prices.values.round(2),
            "Daily Return": df["Daily_Return"][anomalies].mul(100).round(2).astype(str) + "%",
        })
        st.dataframe(anom_df.set_index("Date"), use_container_width=True)
    else:
        st.markdown("""
        <div class='good-box'>
          <b>No anomalies detected</b> in the selected period. Price movement appears statistically normal.
        </div>
        """, unsafe_allow_html=True)

    # Daily returns distribution
    st.markdown("###  Daily Returns Distribution")
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(x=df["Daily_Return"].dropna() * 100,
        nbinsx=60, name="Returns",
        marker=dict(color="#00d4ff", opacity=0.7, line=dict(color="#050810", width=0.3))))
    fig_ret.update_layout(
        template="plotly_dark", paper_bgcolor="#050810", plot_bgcolor="#050810",
        height=300, xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_ret, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab4:
    st.markdown("###  Model Performance Comparison")
    st.info("ℹRun forecasts in the **AI Forecast** tab first to populate backtest metrics. This tab shows comparative charts.")

    # Correlation heatmap
    st.markdown("### 🔗 Feature Correlation")
    corr_cols = ["Close", "Volume", "RSI", "MACD", "Volatility", "MA_20", "MA_50"]
    corr = df[corr_cols].corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0, "#1e0a3a"], [0.5, "#050810"], [1, "#003d5c"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
    ))
    fig_corr.update_layout(
        template="plotly_dark", paper_bgcolor="#050810",
        height=380, margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Rolling volatility vs price
    st.markdown("###  Price vs Annualised Volatility")
    fig_vp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_vp.add_trace(go.Scatter(x=df.index, y=df["Close"],
        name="Price", line=dict(color="#00d4ff", width=1.5)), secondary_y=False)
    fig_vp.add_trace(go.Scatter(x=df.index, y=df["Volatility"],
        name="Volatility %", line=dict(color="#ff6b35", width=1.2)), secondary_y=True)
    fig_vp.update_layout(
        template="plotly_dark", paper_bgcolor="#050810", plot_bgcolor="#050810",
        height=350, margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(gridcolor="#0d1f35"), legend=dict(bgcolor="#0d1526"),
    )
    st.plotly_chart(fig_vp, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════
with tab5:
    st.markdown("###  Raw Data Table")

    display_cols = st.multiselect(
        "Select columns to display",
        options=df.columns.tolist(),
        default=["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MA_20"],
    )
    if display_cols:
        display_df = df[display_cols].tail(100).round(4)
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True, height=400)

    dl1, dl2 = st.columns(2)
    with dl1:
        csv = df.to_csv().encode("utf-8")
        st.download_button("⬇Download Full Data (CSV)", csv,
            file_name=f"{ticker}_data.csv", mime="text/csv")
    with dl2:
        st.markdown(f"""
        <div class='section-card'>
          <div class='section-title'> Dataset Info</div>
          Rows: <b>{len(df)}</b><br>
          From: <b>{df.index[0].date()}</b><br>
          To: <b>{df.index[-1].date()}</b><br>
          Columns: <b>{len(df.columns)}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###  Descriptive Statistics")
    st.dataframe(df[["Close","Volume","RSI","MACD","Volatility"]].describe().round(3),
                 use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer-custom'>
  StockSense AI by Sai vignesh &nbsp;|&nbsp; Built with Streamlit · Plotly · ARIMA · LSTM · Prophet &nbsp;|&nbsp;
  Data: Yahoo Finance &nbsp;|&nbsp; ⚠️ For educational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
