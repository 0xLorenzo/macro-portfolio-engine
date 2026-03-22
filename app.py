# =====================================
# Macro Portfolio Strategy App V3 (中文增强版)
# Streamlit + 宏观 + 风控 + UI优化
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# =============================
# 页面配置
# =============================
st.set_page_config(page_title="宏观投资组合引擎 V3", layout="wide")

# =============================
# 安全导入
# =============================
try:
    import yfinance as yf
except ImportError:
    st.error("缺少 yfinance 依赖，请检查 requirements.txt")
    st.stop()

# =============================
# 常量配置
# =============================
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
# 数据获取
# =============================
@st.cache_data
def load_prices():
    try:
        data = yf.download(ASSETS, period="2y")["Close"]
        return data.dropna()
    except:
        st.error("市场数据加载失败")
        return pd.DataFrame()

prices = load_prices()

if prices.empty:
    st.stop()

returns = prices.pct_change().dropna()

# =============================
# 指标计算
# =============================
def momentum(price):
    ma200 = price.rolling(200).mean()
    return 1.2 if price.iloc[-1] > ma200.iloc[-1] else 0.7


def volatility(r):
    return r.std() * np.sqrt(252)

# =============================
# 权重引擎
# =============================
def compute_weights():
    weights = {}

    for asset in ASSETS:
        base = BASE_WEIGHTS[asset]
        T = momentum(prices[asset])
        vol = volatility(returns[asset])
        R = 1 / (vol + 1e-6)

        weights[asset] = base * T * R

    total_risk = sum(weights.values())
    weights["CASH"] = max(0.1, 1 - total_risk)

    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    return weights

weights = compute_weights()

# =============================
# 回测
# =============================
def backtest():
    w = pd.Series(weights).drop("CASH")
    aligned_returns = returns[w.index]
    portfolio = (aligned_returns * w).sum(axis=1)
    return portfolio

portfolio_returns = backtest()

# =============================
# 风险指标
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
# UI 美化
# =============================
st.markdown("""
<style>
.big-font {font-size:28px !important; font-weight:600;}
.metric-box {
    background-color:#111827;
    padding:15px;
    border-radius:10px;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">📊 宏观投资组合引擎 V3</p>', unsafe_allow_html=True)

# =============================
# 指标展示
# =============================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-box">📈 Sharpe Ratio<br><b>{:.2f}</b></div>'.format(sr), unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">📉 最大回撤<br><b>{:.2%}</b></div>'.format(dd), unsafe_allow_html=True)

# =============================
# 权重展示
# =============================
st.subheader("📌 当前资产配置")
st.dataframe(pd.Series(weights).sort_values(ascending=False))

# =============================
# 曲线
# =============================
st.subheader("📈 组合收益曲线")
st.line_chart((1 + portfolio_returns).cumprod())

st.subheader("📊 市场价格走势")
st.line_chart(prices)

# =============================
# 策略解释
# =============================
st.subheader("🧠 策略逻辑说明")

for asset, w in weights.items():
    if asset == "CASH":
        st.write(f"现金：{w:.1%}（风险缓冲）")
    else:
        trend = "上涨" if momentum(prices[asset]) > 1 else "下跌"
        st.write(f"{asset}：{w:.1%} | 趋势：{trend}")

# =============================
# 页脚
# =============================
st.markdown("---")
st.caption("V3版本：宏观 + 趋势 + 风控 | 持续优化中")
