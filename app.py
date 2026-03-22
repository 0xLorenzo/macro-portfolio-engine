# =====================================
# Macro Portfolio Strategy App V2
# Production-Ready (Streamlit + FRED)
# =====================================
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

st.set_page_config(page_title="Macro Portfolio Engine V2", layout="wide")

# =============================
# CONFIG
# =============================
FRED_API_KEY = st.secrets["FRED_API_KEY"] # <-- 替换

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "BTC-USD"]

BASE_WEIGHTS = {
    "SPY": 0.25,
    "QQQ": 0.20,
    "GLD": 0.10,
    "TLT": 0.15,
    "BTC-USD": 0.10,
    "CASH": 0.20
}

# =============================
# DATA
# =============================
@st.cache_data
def load_prices():
    data = yf.download(ASSETS, period="2y")["Close"]
    return data.dropna()

prices = load_prices()
returns = prices.pct_change().dropna()

# =============================
# FRED MACRO
# =============================
def get_fred(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    r = requests.get(url, params=params).json()
    df = pd.DataFrame(r["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df["value"].dropna().astype(float)

@st.cache_data
def load_macro():
    rates = get_fred("FEDFUNDS")
    cpi = get_fred("CPIAUCSL")
    return rates, cpi

try:
    rates, cpi = load_macro()
    macro_ok = True
except:
    macro_ok = False

# =============================
# SIGNALS
# =============================
def momentum(price):
    ma200 = price.rolling(200).mean()
    return 1.2 if price.iloc[-1] > ma200.iloc[-1] else 0.7


def volatility(r):
    return r.std() * np.sqrt(252)


def liquidity_signal():
    if not macro_ok:
        return 1.0

    latest_rate = rates.iloc[-1]
    latest_cpi = cpi.iloc[-1]

    # 简化逻辑
    if latest_rate < 3:
        return 1.2
    elif latest_rate > 5:
        return 0.8
    else:
        return 1.0

# =============================
# WEIGHT ENGINE
# =============================
def compute_weights():
    L = liquidity_signal()
    weights = {}

    for asset in ASSETS:
        base = BASE_WEIGHTS[asset]
        T = momentum(prices[asset])
        vol = volatility(returns[asset])
        R = 1 / (vol + 1e-6)

        weights[asset] = base * L * T * R

    total_risk = sum(weights.values())
    weights["CASH"] = max(0.1, 1 - total_risk)

    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    return weights

weights = compute_weights()

# =============================
# BACKTEST
# =============================
def backtest():
    w = pd.Series(weights).drop("CASH")
    aligned_returns = returns[w.index]
    portfolio = (aligned_returns * w).sum(axis=1)
    return portfolio

portfolio_returns = backtest()

# =============================
# METRICS
# =============================
def sharpe(r):
    return (r.mean() / r.std()) * np.sqrt(252)


def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

sr = sharpe(portfolio_returns)
dd = max_drawdown(portfolio_returns)

# =============================
# RISK CONTROL
# =============================
if dd < -0.20:
    for k in weights:
        if k != "CASH":
            weights[k] *= 0.7
    weights["CASH"] += 0.3

# =============================
# UI
# =============================
st.title("📊 Macro Portfolio Engine V2")

st.subheader("📌 Allocation")
st.write(pd.Series(weights).sort_values(ascending=False))

st.subheader("📈 Metrics")
st.write(f"Sharpe Ratio: {sr:.2f}")
st.write(f"Max Drawdown: {dd:.2%}")

st.subheader("📉 Portfolio Curve")
st.line_chart((1 + portfolio_returns).cumprod())

st.subheader("📊 Asset Prices")
st.line_chart(prices)

# =============================
# EXPLANATION
# =============================
st.subheader("🧠 Strategy Logic")
for asset, w in weights.items():
    if asset == "CASH":
        st.write(f"Cash: {w:.1%} (Risk buffer)")
    else:
        trend = "UP" if momentum(prices[asset]) > 1 else "DOWN"
        st.write(f"{asset}: {w:.1%} | Trend: {trend}")

# =============================
# FOOTER
# =============================
st.markdown("""
### 🚀 Next Upgrade Ideas
- VIX 风控
- AI新闻情绪
- 自动调仓执行
- 多周期回测
""")
