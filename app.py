"""
IDLC Securities DSE Sector Rotation Dashboard
Main landing page. Navigation to the three analysis sections is in the sidebar.
"""

import streamlit as st

st.set_page_config(
    page_title="IDLC DSE Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DSE Sector Rotation Dashboard")
st.markdown("**Prepared for IDLC Securities | Senior Data Scientist Assignment**")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This dashboard is built as part of a data science assignment for IDLC Securities Limited.
    The goal is to analyze the Dhaka Stock Exchange (DSE) dataset and provide findings
    that can help generate better investment returns for clients.

    The analysis uses historical daily price and volume data from **2012 to 2026**,
    covering **18 DSE sectors** and over **480 listed stocks**. Every section of this
    dashboard ends with a concrete investment recommendation so that the findings
    are directly usable by portfolio managers and the research team at IDLC.

    **Please use the sidebar on the left to navigate between the three sections.**

    **Task 1: EDA and Market Intelligence**

    This section explores the DSE dataset in depth. It covers sector returns,
    risk metrics, technical indicators, calendar effects specific to Bangladesh
    such as the Eid holiday pattern and the June budget effect, and market
    regime detection using a Hidden Markov Model.

    **Task 2: Sector Rotation Strategy**

    This section builds a complete sector rotation system. It shows where
    money has historically moved between sectors, identifies momentum and
    relative strength patterns, trains machine learning models to predict
    next month's leading sectors, and backtests the strategy on data
    the model has never seen (2019 to 2026).

    **Bonus: AI Signal Engine and Investment Chatbot**

    This section demonstrates what a production-ready system could look like
    for IDLC. It includes a live signal engine, portfolio risk management
    tools (VaR, CVaR, Kelly Criterion), a margin financing risk guide
    for the lending desk, and an AI chatbot that answers investment questions
    based on the actual analysis data.
    """)

with col2:
    st.markdown("### Key Numbers")
    st.metric("Stocks Covered", "480+")
    st.metric("Sectors Tracked", "18")
    st.metric("Historical Period", "2012 to 2026")
    st.metric("Backtest Period", "2019 to 2026")
    st.metric("Risk Free Rate", "7.5% per year")

st.divider()

st.markdown("### How the Analysis Was Done")

st.markdown("""
| Step | Details |
|------|---------|
| Data Source | Mendeley DSE dataset with adjusted daily OHLCV prices for all listed stocks |
| Sector Mapping | 18 DSE sectors mapped manually using the dsebd.org sector classification |
| Returns | Daily percentage returns, resampled to monthly for rotation signals |
| Rotation Signal | 35 percent Momentum plus 30 percent Relative Strength plus 35 percent XGBoost ML probability |
| Backtest Setup | Monthly rebalancing with walk-forward validation and 0.65 percent round-trip brokerage cost |
| Risk Management | Value at Risk (VaR), Conditional VaR, Kelly Criterion, BSEC-aligned margin risk ratings |

All machine learning models are trained on data up to the end of 2018.
The period from 2019 to 2026 is kept fully out of sample for honest performance testing.
""")

st.caption("Built by Mustain Billah | Python, Streamlit, XGBoost, Groq Llama 3.1")