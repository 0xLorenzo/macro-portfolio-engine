# =====================================
# Macro Portfolio Strategy App V5.3
# GPT多维宏观判断 + 利率曲线 + 通胀 + 回测验证 + 风控
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import openai

st.set_page_config(page_title="宏观投资引擎 V5.3", layout="wide")

# =============================
# OpenAI Key
# =============================
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.warning("未配置 OPENAI_API_KEY，将使用默认概率")

# =============================
# 数据下载
# =============================
ASSETS = ["SPY","QQQ","GLD","TLT","BTC-USD","^VIX","DX-Y.NYB","^TNX","^IRX"]
@st.cache_data
def load():
    data = yf.download(ASSETS, period="2y")["Close"]
    return data.dropna()

prices = load()
returns = prices.pct_change().dropna()

# =============================
# 宏观特征提取 + 利率曲线 + 通胀 proxy
# =============================
def macro_features():
    spy_trend = "上涨" if prices["SPY"].iloc[-1] > prices["SPY"].rolling(200).mean().iloc[-1] else "下跌"
    tlt_trend = "上涨" if prices["TLT"].iloc[-1] > prices["TLT"].rolling(200).mean().iloc[-1] else "下跌"
    vix = prices["^VIX"].iloc[-1]
    dxy = prices.get("DX-Y.NYB", pd.Series([0])).iloc[-1]

    # 利率曲线：短期利率(^IRX) vs 长期利率(^TNX)
    short_rate = prices.get("^IRX", pd.Series([0])).iloc[-1]
    long_rate = prices.get("^TNX", pd.Series([0])).iloc[-1]
    curve_slope = long_rate - short_rate

    # 通胀 proxy：使用TLT价格反向代理长期通胀预期
    inflation_proxy = round(float(prices["TLT"].pct_change(252).iloc[-1]*100),2)

    return {
        "SPY趋势": spy_trend,
        "TLT趋势": tlt_trend,
        "VIX": round(float(vix),2),
        "美元指数DXY": round(float(dxy),2),
        "短期利率": round(float(short_rate),2),
        "长期利率": round(float(long_rate),2),
        "利率曲线斜率": round(float(curve_slope),2),
        "通胀proxy": inflation_proxy
    }

features = macro_features()

# =============================
# GPT宏观判断 + 风控
# =============================
def gpt_macro(features):
    try:
        prompt = f"""
        当前宏观数据如下：
        - SPY趋势：{features['SPY趋势']}
        - TLT趋势：{features['TLT趋势']}
        - VIX：{features['VIX']}
        - 美元指数DXY：{features['美元指数DXY']}
        - 短期利率：{features['短期利率']}
        - 长期利率：{features['长期利率']}
        - 利率曲线斜率：{features['利率曲线斜率']}
        - 通胀proxy：{features['通胀proxy']}

        请判断四种宏观状态概率，并解释原因，给出潜在最大风险。输出JSON：
        {{"probabilities":{{...}},"reasoning":"...","risk":"..."}}
        """
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        import json
        return json.loads(res.choices[0].message.content)
    except:
        return {
            "probabilities": {"复苏":0.25,"过热":0.25,"衰退":0.25,"滞胀":0.25},
            "reasoning": "使用默认概率", 
            "risk": "无法判断"
        }

ai_output = gpt_macro(features)
probs = ai_output["probabilities"]
reasoning = ai_output["reasoning"]
risk_note = ai_output["risk"]

# =============================
# 权重矩阵 + 概率加权 + 风控
# =============================
def regime_weights():
    return {
        "复苏": {"SPY":1.3,"QQQ":1.3,"GLD":0.8,"TLT":0.7,"BTC-USD":1.2},
        "过热": {"SPY":0.9,"QQQ":0.9,"GLD":1.3,"TLT":0.8,"BTC-USD":1.0},
        "衰退": {"SPY":0.6,"QQQ":0.6,"GLD":1.2,"TLT":1.4,"BTC-USD":0.7},
        "滞胀": {"SPY":0.7,"QQQ":0.7,"GLD":1.4,"TLT":1.0,"BTC-USD":0.8}
    }
regime_map = regime_weights()

combined = {}
for asset in ["SPY","QQQ","GLD","TLT","BTC-USD"]:
    combined[asset] = sum(probs[r]*regime_map[r][asset] for r in probs)

# 权重计算 + 风控
def compute_weights():
    base = {"SPY":0.25,"QQQ":0.2,"GLD":0.1,"TLT":0.15,"BTC-USD":0.1}
    w = {}
    for a in base:
        vol = returns[a].std()*np.sqrt(252)
        w[a] = base[a]*combined[a]*(1/(vol+1e-6))
    # 风控：单资产最大0.4
    for k in w:
        if w[k] > 0.4: w[k] = 0.4
    total = sum(w.values())
    for k in w: w[k]/=total
    return w
weights = compute_weights()

# =============================
# 回测验证
# =============================
w_series = pd.Series(weights)
port = (returns[w_series.index]*w_series).sum(axis=1)

sharpe = (port.mean()/port.std())*np.sqrt(252)
cum = (1+port).cumprod()
peak = cum.cummax()
dd_series = (cum-peak)/peak
max_dd = dd_series.min()
dd_date = dd_series.idxmin()

# =============================
# UI展示
# =============================
st.title("● 宏观投资引擎 V5.3（多维AI+利率+通胀+回测+风控）")

col1,col2,col3 = st.columns(3)
with col1: st.markdown(f"● Sharpe {sharpe:.2f}")
with col2: st.markdown(f"● 最大回撤 {max_dd:.2%} (发生时间: {dd_date})")
with col3: st.markdown(f"● VIX {features['VIX']:.2f}")

st.subheader("● 宏观输入数据")
st.json(features)
st.subheader("● 宏观概率（AI）")
st.dataframe(pd.Series({k:round(v*100,2) for k,v in probs.items()}))
st.subheader("● 资产配置")
st.dataframe(pd.Series({k:round(v*100,2) for k,v in weights.items()}))
st.subheader("● 收益曲线")
st.line_chart(cum)

st.subheader("● AI决策解释与风险提示")
with st.expander("展开查看"):
    st.write("AI判断理由:")
    st.write(reasoning)
    st.write("潜在风险:")
    st.write(risk_note)
    st.write("权重计算逻辑：概率加权 × 波动率倒数，单资产最大权重0.4")

st.caption("V5.3：多维宏观输入 + 利率曲线 + 通胀 + GPT判断 + 回测验证 + 风控")
