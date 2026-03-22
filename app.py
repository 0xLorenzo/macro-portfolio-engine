# =====================================
# Macro Portfolio Strategy App V4.1
# 中文增强 + 可解释性 + UI优化
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="宏观投资引擎 V4.1", layout="wide")

# =============================
# 安全导入
# =============================
try:
    import yfinance as yf
except ImportError:
    st.error("缺少 yfinance 依赖，请检查 requirements.txt")
    st.stop()

# =============================
# 样式
# =============================
st.markdown("""
<style>
.block {padding:14px;border-radius:10px;background:#111; color:#eee}
.small {color:#aaa;font-size:12px}
.title {font-size:26px;font-weight:600}
.dot {font-size:18px;margin-right:6px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">● 宏观投资引擎 V4.1</div>', unsafe_allow_html=True)

# =============================
# 数据
# =============================
ASSETS = ["SPY","QQQ","GLD","TLT","BTC-USD","^VIX"]

@st.cache_data
def load():
    data = yf.download(ASSETS, period="2y")["Close"]
    return data.dropna()

prices = load()
if prices.empty:
    st.error("数据加载失败")
    st.stop()

returns = prices.pct_change().dropna()

# =============================
# 宏观状态机
# =============================
def macro_regime():
    spy_ma = prices["SPY"].rolling(200).mean().iloc[-1]
    tlt_ma = prices["TLT"].rolling(200).mean().iloc[-1]

    spy_up = prices["SPY"].iloc[-1] > spy_ma
    tlt_up = prices["TLT"].iloc[-1] > tlt_ma

    if spy_up and not tlt_up:
        return "复苏"
    elif spy_up and tlt_up:
        return "过热"
    elif not spy_up and tlt_up:
        return "衰退"
    else:
        return "滞胀"

regime = macro_regime()

# =============================
# AI决策
# =============================
def ai_weights(regime):
    if regime == "复苏":
        return {"SPY":1.3,"QQQ":1.3,"GLD":0.8,"TLT":0.7,"BTC-USD":1.2}
    if regime == "过热":
        return {"SPY":0.9,"QQQ":0.9,"GLD":1.3,"TLT":0.8,"BTC-USD":1.0}
    if regime == "衰退":
        return {"SPY":0.6,"QQQ":0.6,"GLD":1.2,"TLT":1.4,"BTC-USD":0.7}
    return {"SPY":0.7,"QQQ":0.7,"GLD":1.4,"TLT":1.0,"BTC-USD":0.8}

ai_adj = ai_weights(regime)

# =============================
# 权重计算
# =============================
def compute_weights():
    base = {"SPY":0.25,"QQQ":0.2,"GLD":0.1,"TLT":0.15,"BTC-USD":0.1}
    w = {}
    for a in base:
        vol = returns[a].std()*np.sqrt(252)
        w[a] = base[a]*ai_adj[a]*(1/(vol+1e-6))
    total = sum(w.values())
    for k in w:
        w[k]/=total
    return w

weights = compute_weights()

# =============================
# 回测
# =============================
w_series = pd.Series(weights)
port = (returns[w_series.index]*w_series).sum(axis=1)

# =============================
# 指标
# =============================
sharpe = (port.mean()/port.std())*np.sqrt(252)

cum = (1+port).cumprod()
peak = cum.cummax()
dd_series = (cum-peak)/peak
max_dd = dd_series.min()

start_dd = dd_series.idxmin()

vix = prices["^VIX"].iloc[-1]

# =============================
# UI展示
# =============================
col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='block'>● Sharpe Ratio<br>{sharpe:.2f}</div>",unsafe_allow_html=True)
    with st.expander("说明"):
        st.write("Sharpe = (平均收益 - 无风险利率) / 波动率 × sqrt(252)")
        st.write("用于衡量单位风险收益")

with col2:
    st.markdown(f"<div class='block'>● 最大回撤<br>{max_dd:.2%}</div>",unsafe_allow_html=True)
    with st.expander("说明"):
        st.write(f"最大回撤起点时间：{start_dd}")
        st.write("计算方式：历史净值峰值到最低点跌幅")

with col3:
    st.markdown(f"<div class='block'>● VIX<br>{vix:.2f}</div>",unsafe_allow_html=True)
    with st.expander("说明"):
        st.write("VIX来源：CBOE波动率指数（yfinance ^VIX）")
        st.write("代表市场恐慌程度：>30 高风险")

# =============================
# 宏观状态解释
# =============================
st.subheader("● 当前宏观状态")
st.write(regime)
with st.expander("判断依据"):
    st.write("SPY 与 TLT 是否高于200日均线组合判断")

# =============================
# 资产配置
# =============================
st.subheader("● 当前资产配置")
alloc = {k:round(v*100,2) for k,v in weights.items()}
st.dataframe(pd.Series(alloc))

# =============================
# 曲线
# =============================
st.subheader("● 收益曲线")
st.line_chart(cum)

# =============================
# AI解释
# =============================
st.subheader("● AI决策解释")
with st.expander("展开查看"):
    st.write("基于宏观状态机调整权重系数")
    st.write(f"当前状态：{regime}")
    st.write("参数说明：")
    st.write(ai_adj)
    st.write("计算逻辑：基础权重 × AI系数 × 波动率倒数")
    st.write("历史回溯：可通过回测曲线观察不同周期表现")

st.caption("V4.1：高可解释性版本")
