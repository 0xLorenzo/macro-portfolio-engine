import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="宏观投资引擎 测试版", layout="wide")

st.title("● 宏观投资引擎 测试版")

# 模拟价格数据
assets = ["SPY","QQQ","GLD","TLT","BTC-USD"]
dates = pd.date_range(end=pd.Timestamp.today(), periods=252)
prices = pd.DataFrame({a: np.cumprod(1 + np.random.normal(0,0.01,252)) for a in assets}, index=dates)
returns = prices.pct_change().dropna()

# 模拟宏观特征
features = {
    "SPY趋势":"上涨",
    "TLT趋势":"下跌",
    "VIX":20.5,
    "美元指数DXY":102.3,
    "短期利率":4.5,
    "长期利率":4.8,
    "利率曲线斜率":0.3,
    "通胀proxy":2.5
}

st.subheader("● 宏观输入数据")
st.json(features)

# 模拟权重
weights = {"SPY":0.25,"QQQ":0.2,"GLD":0.15,"TLT":0.25,"BTC-USD":0.15}
st.subheader("● 资产配置")
st.dataframe(pd.Series({k:round(v*100,2) for k,v in weights.items()}))

# 计算组合收益和夏普
w_series = pd.Series(weights)
port = (returns[w_series.index]*w_series).sum(axis=1)
sharpe = port.mean()/port.std()*np.sqrt(252)
cum = (1+port).cumprod()
peak = cum.cummax()
dd_series = (cum-peak)/peak
max_dd = dd_series.min()

col1,col2,col3 = st.columns(3)
with col1: st.markdown(f"● Sharpe {sharpe:.2f}")
with col2: st.markdown(f"● 最大回撤 {max_dd:.2%}")
with col3: st.markdown("● VIX 20.5")

st.subheader("● 收益曲线")
st.line_chart(cum)

st.subheader("● AI决策解释与风险提示")
with st.expander("展开查看"):
    st.write("未启用 GPT，使用默认概率和模拟逻辑")
