# =====================================

# 宏观投资引擎 V5.3 GitHub 模板

# 完整可运行 Streamlit 项目

# =====================================

# 文件结构示例:

# macro_portfolio_engine/

# ├── app.py

# ├── requirements.txt

# ├── runtime.txt

# └── .streamlit/secrets.toml

# =====================================

# app.py

# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# OpenAI安全调用

try:
import openai
openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
GPT_AVAILABLE = openai.api_key is not None
except ModuleNotFoundError:
GPT_AVAILABLE = False

st.set_page_config(page_title="宏观投资引擎 V5.3", layout="wide")

ASSETS = ["SPY","QQQ","GLD","TLT","BTC-USD","^VIX","DX-Y.NYB","^TNX","^IRX"]

@st.cache_data
def load_prices():
try:
data = yf.download(ASSETS, period="3y")["Close"]
return data.dropna()
except Exception as e:
st.error(f"数据下载失败: {e}")
return pd.DataFrame()

prices = load_prices()
returns = prices.pct_change().dropna()

# 宏观特征

def macro_features():
spy_trend = ("数据不足" if len(prices.get("SPY", [])) < 200 else
"上涨" if prices["SPY"].iloc[-1] > prices["SPY"].rolling(200).mean().iloc[-1] else "下跌")
tlt_trend = ("数据不足" if len(prices.get("TLT", [])) < 200 else
"上涨" if prices["TLT"].iloc[-1] > prices["TLT"].rolling(200).mean().iloc[-1] else "下跌")
vix = prices.get("^VIX", pd.Series([0])).iloc[-1]
dxy = prices.get("DX-Y.NYB", pd.Series([0])).iloc[-1]
short_rate = prices.get("^IRX", pd.Series([0])).iloc[-1]
long_rate = prices.get("^TNX", pd.Series([0])).iloc[-1]
curve_slope = long_rate - short_rate
inflation_proxy = (round(float(prices.get("TLT", pd.Series([0])).pct_change(252).iloc[-1]*100),2)
if len(prices.get("TLT", []))>=252 else 0)
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

# GPT宏观概率判断

def gpt_macro(features):
if not GPT_AVAILABLE:
return {"probabilities":{"复苏":0.25,"过热":0.25,"衰退":0.25,"滞胀":0.25},
"reasoning":"未配置 OpenAI Key，使用默认概率", "risk":"无法判断"}
try:
prompt = f"""
当前宏观数据如下：

* SPY趋势：{features['SPY趋势']}
* TLT趋势：{features['TLT趋势']}
* VIX：{features['VIX']}
* 美元指数DXY：{features['美元指数DXY']}
* 短期利率：{features['短期利率']}
* 长期利率：{features['长期利率']}
* 利率曲线斜率：{features['利率曲线斜率']}
* 通胀proxy：{features['通胀proxy']}
  请判断四种宏观状态概率，并解释原因，输出JSON。
  """
  res = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
  import json
  return json.loads(res.choices[0].message.content)
  except:
  return {"probabilities":{"复苏":0.25,"过热":0.25,"衰退":0.25,"滞胀":0.25},
  "reasoning":"GPT调用失败，使用默认概率", "risk":"无法判断"}

ai_output = gpt_macro(features)
probs = ai_output["probabilities"]
reasoning = ai_output["reasoning"]
risk_note = ai_output["risk"]

# 权重矩阵 + 概率加权

def regime_weights():
return {"复苏":{"SPY":1.3,"QQQ":1.3,"GLD":0.8,"TLT":0.7,"BTC-USD":1.2},
"过热":{"SPY":0.9,"QQQ":0.9,"GLD":1.3,"TLT":0.8,"BTC-USD":1.0},
"衰退":{"SPY":0.6,"QQQ":0.6,"GLD":1.2,"TLT":1.4,"BTC-USD":0.7},
"滞胀":{"SPY":0.7,"QQQ":0.7,"GLD":1.4,"TLT":1.0,"BTC-USD":0.8}}

regime_map = regime_weights()
combined = {asset: sum(probs[r]*regime_map[r][asset] for r in probs) for asset in ["SPY","QQQ","GLD","TLT","BTC-USD"]}

# 权重计算 + 风控

base = {"SPY":0.25,"QQQ":0.2,"GLD":0.1,"TLT":0.15,"BTC-USD":0.1}
weights = {}
for a in base:
vol = returns[a].std()*np.sqrt(252) if a in returns else 0.1
weights[a] = base[a]*combined[a]*(1/(vol+1e-6))
for k in weights:
if weights[k]>0.4: weights[k]=0.4
total = sum(weights.values())
for k in weights: weights[k]/=total

# 回测

w_series = pd.Series(weights)
port = (returns[w_series.index]*w_series).sum(axis=1) if not returns.empty else pd.Series([0])
sharpe = (port.mean()/port.std()*np.sqrt(252)) if port.std()>0 else 0
cum = (1+port).cumprod() if not port.empty else pd.Series([1])
peak = cum.cummax()
dd_series = (cum-peak)/peak
max_dd = dd_series.min() if not dd_series.empty else 0

# UI

st.title("● 宏观投资引擎 V5.3")
col1,col2,col3 = st.columns(3)
with col1: st.markdown(f"● Sharpe {sharpe:.2f}")
with col2: st.markdown(f"● 最大回撤 {max_dd:.2%}")
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
st.caption("V5.3 安全版：多维宏观输入 + GPT判断（可选） + 风控 + 回测 + 数据安全")

# =====================================

# requirements.txt

# =====================================

# streamlit

# pandas

# numpy

# yfinance

# openai

# matplotlib

# plotly

# requests

# =====================================

# runtime.txt

# =====================================

# python-3.11.8

# =====================================

# .streamlit/secrets.toml (可选)

# =====================================

#[general]
#OPENAI_API_KEY = "你的OpenAI Key"
