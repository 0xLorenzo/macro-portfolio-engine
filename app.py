# =====================================
# Macro Portfolio Strategy App V4
# 宏观状态机 + AI决策引擎（简化版）
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(page_title="宏观投资引擎 V4", layout="wide")

# =============================
# 安全导入
# =============================
try:
    import yfinance as yf
except ImportError:
    st.error("缺少 yfinance 依赖")
    st.stop()

# =============================
# 参数
# =============================
ASSETS = ["SPY", "QQQ", "GLD", "TLT", "BTC-USD", "^VIX"]

BASE_WEIGHTS = {
    "SPY": 0.25,
    "QQQ": 0.20,
    "GLD": 0.10,
    "TLT": 0.15,
    "BTC-USD": 0.10,
    "CASH": 0.20
}

# =============================
# 数据
# =============================
@st.cache_data
def load_data():
    data = yf.download(ASSETS, period="2y")["Close"]
    return data.dropna()

prices = load_data()

if prices.empty:
    st.error("数据加载失败")
    st.stop()

returns = prices.pct_change().dropna()

# =============================
# 宏观状态机（核心）
# =============================
def macro_regime():
    spy_trend = prices["SPY"].iloc[-1] > prices["SPY"].rolling(200).mean().iloc[-1]
    tlt_trend = prices["TLT"].iloc[-1] > prices["TLT"].rolling(200).mean().iloc[-1]

    if spy_trend and not tlt_trend:
        return "复苏（Risk-On）"
    elif spy_trend and tlt_trend:
        return "过热（通胀）"
    elif not spy_trend and tlt_trend:
        return "衰退（避险）"
    else:
        return "滞胀（Stagflation）"

regime = macro_regime()

# =============================
# AI决策引擎（规则版模拟）
# =============================
def ai_decision(regime):
    if "复苏" in regime:
        return {"SPY":1.3, "QQQ":1.3, "GLD":0.8, "TLT":0.7, "BTC-USD":1.2}
    elif "过热" in regime:
        return {"SPY":0.9, "QQQ":0.9, "GLD":1.3, "TLT":0.8, "BTC-USD":1.0}
    elif "衰退" in regime:
        return {"SPY":0.6, "QQQ":0.6, "GLD":1.2, "TLT":1.4, "BTC-USD":0.7}
    else:
        return {"SPY":0.7, "QQQ":0.7, "GLD":1.4, "TLT":1.0, "BTC-USD":0.8}

ai_weights = ai_decision(regime)

# =============================
# VIX风险控制（Taleb）
# =============================
def vix_control(weights):
    vix = prices["^VIX"].iloc[-1]
    if vix > 30:
        for k in weights:
            if k != "CASH":
                weights[k] *= 0.6
        weights["CASH"] += 0.4
    return weights, vix

# =============================
# 权重计算
# =============================
def compute_weights():
    weights = {}

    for asset in BASE_WEIGHTS:
        if asset == "CASH":
            continue
        base = BASE_WEIGHTS[asset]
        adj = ai_weights.get(asset,1)
        vol = returns[asset].std() * np.sqrt(252)
        risk = 1 / (vol + 1e-6)

        weights[asset] = base * adj * risk

    total_risk = sum(weights.values())
    weights["CASH"] = max(0.1, 1 - total_risk)

    weights, vix = vix_control(weights)

    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    return weights, vix

weights, vix = compute_weights()

# =============================
# 回测
# =============================
def backtest():
    w = pd.Series(weights).drop("CASH")
    aligned = returns[w.index]
    return (aligned * w).sum(axis=1)

portfolio_returns = backtest()

# =============================
# 指标
# =============================
def sharpe(r):
    return (r.mean()/r.std())*np.sqrt(252)


def max_dd(r):
    cum = (1+r).cumprod()
    peak = cum.cummax()
    return ((cum-peak)/peak).min()

sr = sharpe(portfolio_returns)
dd = max_dd(portfolio_returns)

# =============================
# UI
# =============================
st.title("🧠 宏观投资引擎 V4")

col1,col2,col3 = st.columns(3)
col1.metric("Sharpe", f"{sr:.2f}")
col2.metric("最大回撤", f"{dd:.2%}")
col3.metric("VIX", f"{vix:.1f}")

st.subheader("🌍 当前宏观状态")
st.success(regime)

st.subheader("📊 资产配置")
st.dataframe(pd.Series(weights).sort_values(ascending=False))

st.subheader("📈 收益曲线")
st.line_chart((1+portfolio_returns).cumprod())

st.subheader("🧠 AI决策解释")
for k,v in ai_weights.items():
    st.write(f"{k}: 调整系数 {v}")

st.caption("V4：宏观状态机 + AI决策 + VIX风控")
