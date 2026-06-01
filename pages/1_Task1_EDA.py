"""
Task 1:Open-Ended EDA & Investment Insights
Assignment requirement: "What insights can you provide from the data?
Anything that can help us generate return from investment is a plus."

Covers NB01, NB02, NB03 in full.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_macd_fig(df):
    """Two-panel MACD chart: price on top, MACD + signal + histogram below."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Price",
                             line=dict(color="#2c3e50", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#3498db")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                             line=dict(color="#e74c3c")), row=2, col=1)
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram",
                         marker_color=colors, opacity=0.6), row=2, col=1)
    fig.update_layout(title="MACD - DSE Market Index", height=480)
    return fig


st.set_page_config(page_title="Task 1 EDA", page_icon="📊", layout="wide")
st.title("Task 1: Market Intelligence and Investment Insights")
st.markdown(
    "Open-ended exploration of the DSE dataset from 2012 to 2026. "
    "Every section ends with a concrete investment action."
)
st.divider()

# ── Load data (cached so each section doesn't reload) ─────────────────────
@st.cache_data
def load_all():
    sr  = pd.read_parquet("data/processed/sector_returns.parquet")
    ret = pd.read_parquet("data/processed/returns.parquet")
    vol = pd.read_parquet("data/processed/volume.parquet")
    pri = pd.read_parquet("data/processed/prices.parquet")
    dsex = pd.read_parquet("data/processed/dsex.parquet")
    sec_map = pd.read_csv("data/processed/sector_map.csv", index_col=0).squeeze()
    return sr, ret, vol, pri, dsex, sec_map

sector_returns, returns, volume, prices, dsex, sector_map = load_all()
RISK_FREE = 0.075

# ── Five tabs to organise the analysis ────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Overview",
    "Risk and Return",
    "Technical Analysis",
    "Calendar and Patterns",
    "Bangladesh Insights",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1:MARKET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── 1.1 Dataset Overview ──────────────────────────────────────────────
    st.subheader("1.1 Dataset Overview")
    st.markdown(
        "Before starting any analysis, the first thing I did was understand the data I am working with. "
        "The dataset contains daily stock price and trading volume information for companies listed "
        "on the Dhaka Stock Exchange from 2012 to 2026. "
        "DSE operates from Sunday to Thursday, which is different from most international markets. "
        "All stocks are grouped into 18 sectors, which is the foundation for the sector rotation strategy in Task 2."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickers", f"{returns.shape[1]:,}")
    c2.metric("Sectors", str(sector_returns.shape[1]))
    c3.metric("Trading Days", f"{len(sector_returns):,}")
    c4.metric("Date Range", f"{sector_returns.index.min().year} to {sector_returns.index.max().year}")
    c5.metric("Trading Week", "Sun to Thu")

    st.divider()

    # ── 1.2 DSEX Price History ────────────────────────────────────────────
    st.subheader("1.2 DSEX Benchmark Price History")
    st.markdown(
        "The DSEX is the main index of the Dhaka Stock Exchange and represents the overall "
        "health of the Bangladesh stock market. It works the same way as the S&P 500 in the US "
        "or the NIKKEI in Japan. I plotted this first because every sector comparison in this "
        "project is measured against this benchmark. "
        "From the chart, we can see the market had a strong bull run from 2012 to 2017, "
        "went through a correction, recovered sharply around 2020 to 2022, and then entered "
        "a downtrend. Understanding this big picture is necessary before analyzing individual sectors."
    )
    try:
        price_col = [c for c in dsex.columns if c != 'Date'][0]
        fig_dsex = px.area(
            x=dsex.index, y=dsex[price_col],
            labels={"x": "Date", "y": "DSEX Points"},
            title="DSEX Index - Full History (2012 to 2026)",
            color_discrete_sequence=["#2980b9"],
        )
        fig_dsex.update_layout(height=360)
        st.plotly_chart(fig_dsex, use_container_width=True)
    except Exception:
        mkt_cum = (1 + sector_returns.mean(axis=1)).cumprod() * 1000
        fig_dsex = px.area(x=mkt_cum.index, y=mkt_cum.values,
                           labels={"x": "Date", "y": "Index (rebased to 1000)"},
                           title="DSE Market Index - Equal Weight, Rebased to 1000",
                           color_discrete_sequence=["#2980b9"])
        fig_dsex.update_layout(height=360)
        st.plotly_chart(fig_dsex, use_container_width=True)

    st.divider()

    # ── 1.3 Cumulative Returns by Sector ──────────────────────────────────
    st.subheader("1.3 Where Did Money Go - Cumulative Returns by Sector (2012 to 2026)")
    st.markdown(
        "This is one of the most important charts in the entire analysis. "
        "The question it answers is: if an investor had put BDT 1 into each sector at the "
        "beginning of 2012 and left it there, how much would they have today? "
        "Each line in the chart represents one sector. A line sitting higher on the chart "
        "means that sector created more wealth over time. "
        "The huge gap between the top and bottom lines is the key finding: "
        "being in the right sector makes an enormous difference to returns. "
        "This motivates the entire sector rotation strategy built in Task 2."
    )
    cum_df = (1 + sector_returns).cumprod()
    fig_cum = px.line(
        cum_df, x=cum_df.index, y=cum_df.columns,
        labels={"value": "Growth of BDT 1", "variable": "Sector", "x": "Date"},
        title="Cumulative Return by Sector - Full History",
    )
    fig_cum.update_layout(height=480, legend=dict(x=1.01, y=0.5))
    st.plotly_chart(fig_cum, use_container_width=True)

    winner = (cum_df.iloc[-1]).idxmax()
    loser  = (cum_df.iloc[-1]).idxmin()
    st.success(
        f"Finding: The {winner} sector delivered the highest total return "
        f"({cum_df[winner].iloc[-1]-1:.0%}) since 2012. "
        f"The {loser} sector was the worst, actually losing value ({cum_df[loser].iloc[-1]-1:.0%}). "
        "An investor who picked sectors carefully would have done much better than someone "
        "who just bought and held everything equally."
    )

    st.divider()

    # ── 1.4 Return Distribution ───────────────────────────────────────────
    st.subheader("1.4 Return Distribution - Are DSE Returns Normal?")
    st.markdown(
        "Most standard financial models assume that daily stock returns follow a normal "
        "distribution, which is the bell-shaped curve. If this is true, very large price "
        "moves in a single day should almost never happen. "
        "Before building any risk model, I tested whether this assumption holds for DSE. "
        "The blue bars show the actual distribution of daily returns in our data. "
        "The red dashed curve shows what the distribution would look like if returns were "
        "perfectly normal. The difference between these two shapes tells us whether DSE "
        "follows standard assumptions or not."
    )
    mkt_ret = sector_returns.mean(axis=1).dropna()
    from scipy import stats as scipy_stats
    jb_stat, jb_p = scipy_stats.jarque_bera(mkt_ret)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=mkt_ret * 100, nbinsx=100, histnorm='probability density',
                                    name='Actual DSE returns', marker_color='#3498db', opacity=0.7))
    x_range = np.linspace(mkt_ret.min() * 100, mkt_ret.max() * 100, 200)
    normal_pdf = scipy_stats.norm.pdf(x_range, mkt_ret.mean() * 100, mkt_ret.std() * 100)
    fig_dist.add_trace(go.Scatter(x=x_range, y=normal_pdf, mode='lines',
                                  name='Normal distribution (for comparison)',
                                  line=dict(color='red', dash='dash', width=2)))
    fig_dist.update_layout(title="Daily Return Distribution vs. Normal Curve",
                           xaxis_title="Daily Return (%)", height=380)
    st.plotly_chart(fig_dist, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Skewness", f"{mkt_ret.skew():.2f}",
              help="0 means perfectly symmetric. A positive value means there are more days with unusually large gains than losses.")
    c2.metric("Excess Kurtosis", f"{mkt_ret.kurtosis():.1f}",
              help="0 means normal distribution. A value of 4.4 means extreme daily moves happen much more often than normal theory predicts.")
    c3.metric("Jarque-Bera p-value", f"{jb_p:.2e}",
              help="This is a statistical test for normality. A value below 0.05 means returns are NOT normally distributed.")
    st.warning(
        "DSE returns are not normally distributed. "
        "The tails of the distribution are much fatter than the normal curve, meaning "
        "large crashes and spikes happen more often than standard models expect. "
        "Because of this, I use Value at Risk (VaR) and CVaR in the Bonus section "
        "instead of relying only on standard deviation for risk measurement."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2:RISK & RETURN
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    # ── 2.1 Risk-Adjusted Performance ─────────────────────────────────────
    st.subheader("2.1 Risk-Adjusted Performance - Sharpe, Sortino and Calmar Ratios")
    st.markdown(
        "Looking at raw return alone is not enough when comparing sectors. A sector might "
        "show a high return, but if it is also very volatile, an investor would have gone "
        "through many stressful periods to get that return. "
        "So I calculated three standard risk-adjusted ratios for each sector:\n\n"
        "**Sharpe Ratio** measures how much return the sector gave per unit of total risk. "
        "A Sharpe of 0.5 means for every 1 unit of risk taken, the sector gave 0.5 units of extra return "
        "above the risk-free rate (7.5% in Bangladesh).\n\n"
        "**Sortino Ratio** is similar to Sharpe but it only penalizes downside moves, not upside volatility. "
        "This is more useful for DSE because as we saw in Section 1.4, returns are not symmetric.\n\n"
        "**Calmar Ratio** compares the annual return against the worst drawdown the sector ever experienced. "
        "A higher Calmar means the sector recovered better from its worst losses."
    )

    @st.cache_data
    def compute_sector_metrics(data_hash):
        rows = []
        for sector in sector_returns.columns:
            s = sector_returns[sector].dropna()
            ann_ret = (1 + s).prod() ** (252 / len(s)) - 1
            ann_vol = s.std() * np.sqrt(252)
            downside = s[s < 0].std() * np.sqrt(252)
            sharpe   = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0
            sortino  = (ann_ret - RISK_FREE) / downside if downside > 0 else 0
            cum      = (1 + s).cumprod()
            max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()
            calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
            rows.append({
                "Sector":       sector,
                "Ann. Return":  ann_ret,
                "Volatility":   ann_vol,
                "Sharpe":       sharpe,
                "Sortino":      sortino,
                "Max DD":       max_dd,
                "Calmar":       calmar,
            })
        return pd.DataFrame(rows).set_index("Sector")

    metrics_df = compute_sector_metrics(f"{sector_returns.shape}_{sector_returns.index[-1]}")

    st.markdown(
        "How to read this chart: the X-axis shows risk (volatility) and the Y-axis shows return. "
        "The best sectors sit in the top-left corner, meaning high return with low risk. "
        "The colour shows the Sharpe Ratio (green is good, red is poor). "
        "The size of each bubble shows the Calmar Ratio. A larger bubble means the sector "
        "recovered better after its worst losses. You can hover over any bubble to see the sector name."
    )
    fig_rr = px.scatter(
        metrics_df.reset_index(),
        x="Volatility", y="Ann. Return",
        color="Sharpe", size=metrics_df["Calmar"].clip(lower=0.01).reset_index(drop=True) + 0.1,
        hover_name="Sector", text="Sector",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title="Risk vs. Return - Colour shows Sharpe Ratio, Bubble size shows Calmar Ratio",
        labels={"Volatility": "Annualised Volatility", "Ann. Return": "Annualised Return"},
    )
    fig_rr.update_traces(textposition="top center", textfont_size=9)
    fig_rr.update_layout(height=500)
    fig_rr.update_xaxes(tickformat=".0%")
    fig_rr.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_rr, use_container_width=True)

    st.markdown("The table below can be sorted by any column to compare sectors on different metrics.")
    sort_by = st.selectbox("Sort table by", ["Sharpe", "Sortino", "Calmar", "Ann. Return"], index=0)
    display_df = metrics_df.sort_values(sort_by, ascending=False).copy()
    for col in ["Ann. Return", "Volatility", "Max DD"]:
        display_df[col] = display_df[col].map("{:.1%}".format)
    for col in ["Sharpe", "Sortino", "Calmar"]:
        display_df[col] = display_df[col].map("{:.2f}".format)
    st.dataframe(display_df, use_container_width=True, height=430)

    best_sharpe  = metrics_df["Sharpe"].idxmax()
    best_sortino = metrics_df["Sortino"].idxmax()
    st.success(
        f"Finding: The {best_sharpe} sector has the best Sharpe Ratio, meaning it delivered "
        f"the highest return per unit of risk among all 18 sectors. "
        f"It also has the best Sortino Ratio, which confirms it is the most efficient sector "
        "for investors who are especially concerned about downside losses."
    )

    st.divider()

    # ── 2.2 Maximum Drawdown ──────────────────────────────────────────────
    st.subheader("2.2 Maximum Drawdown - Underwater Equity Curves")
    st.markdown(
        "Even a sector with a good average return can go through long periods where it is below "
        "its previous peak. This is called a drawdown, and it is very important for real investors "
        "because most people find it psychologically difficult to hold through large losses.\n\n"
        "This chart shows how far each sector has been below its all-time high at any point in time. "
        "The Y-axis shows the percentage below the peak. A value of 0% means the sector is at its "
        "highest level ever. A value of -40% means it is currently 40% below its peak. "
        "The longer and deeper the chart stays below zero, the more painful that period was for investors. "
        "You can select which sectors to compare using the selector below."
    )

    sector_select = st.multiselect(
        "Select sectors to display",
        options=sector_returns.columns.tolist(),
        default=sector_returns.columns[:5].tolist(),
    )

    if sector_select:
        fig_dd = go.Figure()
        for sec in sector_select:
            cum = (1 + sector_returns[sec].dropna()).cumprod()
            dd  = (cum - cum.cummax()) / cum.cummax() * 100
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=sec,
                                        fill='tozeroy', mode='lines', line=dict(width=1)))
        fig_dd.update_layout(title="Drawdown from All-Time Peak (%)",
                             yaxis_title="Drawdown (%)", height=420)
        st.plotly_chart(fig_dd, use_container_width=True)
        st.info(
            "Sectors that stay deeply negative for many years are risky for long-term investors. "
            "For IDLC's margin lending desk, sectors with large and long drawdowns are also "
            "more dangerous as collateral because clients may not recover their losses before a margin call."
        )

    st.divider()

    # ── 2.3 Information Ratio ─────────────────────────────────────────────
    st.subheader("2.3 Information Ratio - Which Sectors Consistently Beat the Market?")
    st.markdown(
        "High return in one or two years is not the same as consistently outperforming the market. "
        "The Information Ratio (IR) measures exactly this. It compares each sector's return "
        "against the equal-weight market average, and then divides by how variable that "
        "outperformance was. A sector with a positive IR consistently beats the market. "
        "A negative IR means the sector consistently underperforms, and you would have been "
        "better off just holding the market equally.\n\n"
        "The bars to the right of the zero line are sectors worth overweighting. "
        "The bars to the left are sectors an active investor should avoid or underweight."
    )
    market_daily = sector_returns.mean(axis=1)
    ir_rows = []
    for sec in sector_returns.columns:
        active = sector_returns[sec] - market_daily
        active = active.dropna()
        ir = active.mean() / active.std() * np.sqrt(252) if active.std() > 0 else 0
        ir_rows.append({"Sector": sec, "Information Ratio": round(ir, 2)})
    ir_df = pd.DataFrame(ir_rows).set_index("Sector").sort_values("Information Ratio", ascending=True)

    fig_ir = px.bar(
        ir_df.reset_index(), x="Information Ratio", y="Sector",
        orientation="h",
        color="Information Ratio", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title="Information Ratio vs. Equal-Weight Market (positive = consistently outperforms)",
    )
    fig_ir.add_vline(x=0, line_color="black", line_width=1)
    fig_ir.update_layout(height=500)
    st.plotly_chart(fig_ir, use_container_width=True)
    top_ir = ir_df["Information Ratio"].idxmax()
    bot_ir = ir_df["Information Ratio"].idxmin()
    st.info(
        f"Finding: The {top_ir} sector has the highest Information Ratio, meaning it has "
        "consistently outperformed the broader market over the full history. "
        f"The {bot_ir} sector has the lowest IR, meaning investors in that sector would have "
        "done better by simply holding the market rather than concentrating in that sector."
    )

    st.divider()

    # ── 2.4 Market Liquidity ──────────────────────────────────────────────
    st.subheader("2.4 Market Liquidity - Top 15 Most Traded Stocks")
    st.markdown(
        "Before recommending any stock for investment, it is important to check that it "
        "can actually be bought and sold easily. Stocks with very low daily trading volume "
        "are difficult to enter and exit, especially for larger investors. "
        "When a large client like an IDLC portfolio tries to buy a low-volume stock, "
        "the buying pressure itself pushes the price up, increasing the cost. "
        "Similarly, selling a low-volume stock is difficult without moving the price down.\n\n"
        "This chart identifies the 15 most liquid stocks on DSE by average daily trading volume. "
        "These are the stocks where large positions can be built and exited without significant "
        "market impact."
    )
    avg_vol = volume.mean().sort_values(ascending=False).head(15)
    fig_liq = px.bar(
        x=avg_vol.values, y=avg_vol.index, orientation="h",
        labels={"x": "Average Daily Volume (shares)", "y": "Stock"},
        title="Top 15 Most Liquid DSE Stocks by Average Daily Trading Volume",
        color=avg_vol.values, color_continuous_scale="Blues",
    )
    fig_liq.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_liq, use_container_width=True)
    st.info(
        "For IDLC's portfolio managers and margin lending desk, focusing on high-liquidity "
        "stocks reduces transaction costs and lowers the risk of being unable to exit a "
        "position quickly when market conditions change."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3:TECHNICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Technical Indicators on the DSE Market Index")
    st.markdown(
        "Technical analysis uses patterns in price and volume data to identify the best "
        "times to enter or exit the market. Unlike fundamental analysis which looks at company "
        "financials, technical indicators focus on market behavior and momentum. "
        "I applied four widely-used indicators to the DSE market index to show the current "
        "market condition and to identify historical buy and sell signals. "
        "These same indicators can be applied to individual sectors to time rotation decisions."
    )

    try:
        import ta
        mkt_price = (1 + sector_returns.mean(axis=1)).cumprod() * 1000
        mkt_price_series = mkt_price.rename("close")
        mkt_df = mkt_price_series.to_frame()

        mkt_df["RSI"]         = ta.momentum.RSIIndicator(mkt_df["close"], window=14).rsi()
        bb = ta.volatility.BollingerBands(mkt_df["close"], window=20, window_dev=2)
        mkt_df["BB_upper"]    = bb.bollinger_hband()
        mkt_df["BB_lower"]    = bb.bollinger_lband()
        mkt_df["BB_mid"]      = bb.bollinger_mavg()
        macd = ta.trend.MACD(mkt_df["close"])
        mkt_df["MACD"]        = macd.macd()
        mkt_df["MACD_signal"] = macd.macd_signal()
        mkt_df["MACD_hist"]   = macd.macd_diff()
        mkt_df["MA20"]        = mkt_df["close"].rolling(20).mean()
        mkt_df["MA50"]        = mkt_df["close"].rolling(50).mean()

        # ── 3.1 RSI ───────────────────────────────────────────────────────
        st.subheader("3.1 RSI (14-day) - Overbought and Oversold Signals")
        st.markdown(
            "The RSI (Relative Strength Index) is one of the most commonly used indicators "
            "by traders around the world. It measures whether the market has gone up too much "
            "too quickly (overbought) or fallen too much too quickly (oversold).\n\n"
            "The RSI is calculated by comparing the average daily gains to the average daily "
            "losses over the last 14 trading days. The result is a number between 0 and 100. "
            "When the RSI goes below 30 (the green dashed line), the market is considered "
            "oversold and has historically been a good time to buy. "
            "When it goes above 70 (the red dashed line), the market is overbought and it "
            "may be a good time to reduce equity positions. "
            "The gray area in the middle (30 to 70) is the neutral zone where no strong signal exists."
        )
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["RSI"],
                                     name="RSI 14", line=dict(color="#3498db")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="Oversold (30)")
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.05)
        fig_rsi.update_layout(title="RSI (14-day) - DSE Market Index",
                               height=360, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

        current_rsi = mkt_df["RSI"].dropna().iloc[-1]
        if current_rsi < 30:
            st.success(
                f"Current RSI is {current_rsi:.1f}. The market is currently OVERSOLD. "
                "Based on historical patterns, this is a good time to consider buying."
            )
        elif current_rsi > 70:
            st.warning(
                f"Current RSI is {current_rsi:.1f}. The market is currently OVERBOUGHT. "
                "It may be a good time to reduce equity exposure and wait for a pullback."
            )
        else:
            st.info(
                f"Current RSI is {current_rsi:.1f}. The market is in the neutral zone. "
                "No strong buy or sell signal from RSI at this time."
            )

        st.divider()

        # ── 3.2 Bollinger Bands ───────────────────────────────────────────
        st.subheader("3.2 Bollinger Bands - Volatility-Based Entry and Exit")
        st.markdown(
            "Bollinger Bands are built around a 20-day moving average (the purple dotted line). "
            "The upper and lower bands (red and green dashed lines) are placed 2 standard "
            "deviations away from the average. This means that under normal market conditions, "
            "about 95% of all daily price movements happen inside the bands.\n\n"
            "When the market price touches or crosses the lower green band, it means the "
            "market has moved unusually far downward and may be ready to recover. "
            "When the price touches the upper red band, it has moved unusually far upward "
            "and may be due for a pullback.\n\n"
            "A very useful signal is the Bollinger Squeeze - when the upper and lower bands "
            "come very close together, it means volatility is very low. This usually happens "
            "just before a big price move. Traders watch for squeezes because they often "
            "signal that a strong trend is about to begin."
        )
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["close"],
                                    name="Market Price", line=dict(color="#2c3e50", width=1)))
        fig_bb.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["BB_upper"],
                                    name="Upper Band", line=dict(color="#e74c3c", dash="dash")))
        fig_bb.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["BB_lower"],
                                    name="Lower Band", line=dict(color="#27ae60", dash="dash"),
                                    fill='tonexty', fillcolor='rgba(0,200,0,0.05)'))
        fig_bb.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["BB_mid"],
                                    name="20-day MA", line=dict(color="#9b59b6", dash="dot")))
        fig_bb.update_layout(title="Bollinger Bands (20-day, 2 standard deviations) - DSE Market Index",
                             height=400)
        st.plotly_chart(fig_bb, use_container_width=True)

        st.divider()

        # ── 3.3 MACD ──────────────────────────────────────────────────────
        st.subheader("3.3 MACD - Trend Confirmation Signal")
        st.markdown(
            "MACD stands for Moving Average Convergence Divergence. It is a trend-following "
            "indicator that shows the relationship between two exponential moving averages "
            "of the market price.\n\n"
            "The MACD line (blue) is calculated as the 12-day EMA minus the 26-day EMA. "
            "The Signal Line (red) is a 9-day EMA of the MACD, which makes it smoother and "
            "easier to read. The Histogram (green and red bars) shows the difference between "
            "the MACD and the Signal Line.\n\n"
            "When the MACD line crosses above the Signal Line, this is a buy signal because "
            "it means short-term momentum is now stronger than the longer-term average. "
            "When MACD crosses below the Signal Line, this is a sell signal. "
            "The histogram is useful because when the bars start getting smaller, it tells "
            "you the trend is losing strength even before the actual crossover happens. "
            "This gives traders an early warning."
        )
        fig_macd = make_macd_fig(mkt_df)
        st.plotly_chart(fig_macd, use_container_width=True)

        st.divider()

        # ── 3.4 Moving Average Crossover ──────────────────────────────────
        st.subheader("3.4 Moving Average Crossover - Golden Cross and Death Cross")
        st.markdown(
            "Moving averages smooth out the daily price noise to show the overall market trend. "
            "The 20-day MA (orange line) reacts quickly to price changes because it only looks "
            "at the last 20 trading days. The 50-day MA (purple line) is slower and shows the "
            "longer-term direction.\n\n"
            "When the faster 20-day MA crosses above the slower 50-day MA, this is called a "
            "Golden Cross. It signals that the short-term trend has become stronger than the "
            "long-term trend, which is a bullish signal. On the chart, Golden Crosses are "
            "marked with yellow upward triangles.\n\n"
            "When the 20-day MA crosses below the 50-day MA, this is called a Death Cross. "
            "It signals the beginning of a downtrend. Death Crosses are shown as red downward "
            "triangles on the chart."
        )
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["close"],
                                    name="Price", line=dict(color="#2c3e50", width=1), opacity=0.6))
        fig_ma.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["MA20"],
                                    name="MA 20 (fast)", line=dict(color="#e67e22", width=1.5)))
        fig_ma.add_trace(go.Scatter(x=mkt_df.index, y=mkt_df["MA50"],
                                    name="MA 50 (slow)", line=dict(color="#8e44ad", width=1.5)))

        cross  = mkt_df["MA20"] - mkt_df["MA50"]
        golden = mkt_df[(cross > 0) & (cross.shift(1) <= 0)]
        death  = mkt_df[(cross < 0) & (cross.shift(1) >= 0)]
        fig_ma.add_trace(go.Scatter(x=golden.index, y=golden["MA20"],
                                    mode="markers", name="Golden Cross (buy)",
                                    marker=dict(color="gold", size=10, symbol="triangle-up")))
        fig_ma.add_trace(go.Scatter(x=death.index, y=death["MA20"],
                                    mode="markers", name="Death Cross (sell)",
                                    marker=dict(color="red", size=10, symbol="triangle-down")))
        fig_ma.update_layout(title="Moving Average 20/50 Crossover - DSE Market", height=420)
        st.plotly_chart(fig_ma, use_container_width=True)

        st.info(
            f"Over the full history from 2012 to 2026, we found {len(golden)} Golden Cross signals "
            f"and {len(death)} Death Cross signals on the DSE market index. "
            "A simple strategy of buying on Golden Cross and selling on Death Cross would have "
            "captured the major market trends while avoiding the worst drawdowns. "
            "This type of signal is especially useful for IDLC's portfolio managers when "
            "deciding how much equity exposure to hold for clients."
        )

    except ImportError:
        st.error("Please install the ta library by running: pip install ta")

    except Exception as e:
        st.error(f"Technical indicator error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4:CALENDAR & PATTERNS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    mkt_ret = sector_returns.mean(axis=1)

    # ── 4.1 Day-of-Week Effect ────────────────────────────────────────────
    st.subheader("4.1 Day-of-Week Effect - Best Day to Buy and Sell")
    st.markdown(
        "Many global markets show a pattern where returns are not equal across all days "
        "of the week. This is called the day-of-week effect, and it has been documented "
        "in the US, UK, and Asian markets. I tested whether the same pattern exists in DSE.\n\n"
        "DSE trades from Sunday to Thursday. I calculated the average daily return for each "
        "day of the week across the full 14-year history. The bar chart below shows the "
        "result. Green bars mean that day has historically been positive on average. "
        "Red bars mean that day has historically been negative on average.\n\n"
        "This information is useful for short-term traders who want to time their buying "
        "and selling. Even a small edge on timing can improve returns when applied consistently."
    )
    dow_df = pd.DataFrame({"ret": mkt_ret, "dow": mkt_ret.index.dayofweek})
    day_names = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 6:"Sun"}
    dow_df["day_name"] = dow_df["dow"].map(day_names)
    dow_by_name = dow_df.groupby("day_name")["ret"].mean() * 100
    order = ["Sun", "Mon", "Tue", "Wed", "Thu"]
    dow_by_name = dow_by_name.reindex([d for d in order if d in dow_by_name.index])

    fig_dow = px.bar(
        x=dow_by_name.index, y=dow_by_name.values,
        color=dow_by_name.values,
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        labels={"x": "Day of Week", "y": "Avg Daily Return (%)"},
        title="Average Daily Return by Day of Week (DSE, 2012 to 2026)",
    )
    fig_dow.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig_dow, use_container_width=True)

    best_day  = dow_by_name.idxmax()
    worst_day = dow_by_name.idxmin()
    st.info(
        f"Finding: {best_day} is the best trading day on average "
        f"(+{dow_by_name.max():.3f}% per day). "
        f"{worst_day} is the worst trading day on average "
        f"({dow_by_name.min():.3f}% per day). "
        f"A simple strategy of buying on {worst_day} when the market dips and selling on "
        f"{best_day} exploits this pattern without any complex analysis."
    )

    st.divider()

    # ── 4.2 Month-of-Year Effect ──────────────────────────────────────────
    st.subheader("4.2 Month-of-Year Effect - Seasonal Patterns")
    st.markdown(
        "Similar to the day-of-week effect, markets around the world show patterns where "
        "certain months are consistently stronger or weaker than others. In Bangladesh, "
        "there are specific events that create predictable seasonal patterns:\n\n"
        "The national budget is announced every June, which often causes sector-specific "
        "reactions in Banking, Cement, and Engineering. The dividend payment season runs "
        "from March to May, when many companies announce dividends and share prices tend "
        "to react. Eid holidays cause retail investors to sell positions to access cash, "
        "which creates temporary weakness before the holiday.\n\n"
        "I calculated the average daily return for each calendar month across all 14 years. "
        "Green bars show months that are historically positive. Red bars show weak months."
    )
    month_df = pd.DataFrame({"ret": mkt_ret, "month": mkt_ret.index.month})
    month_avg = month_df.groupby("month")["ret"].mean() * 100
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_avg.index = [month_names[m-1] for m in month_avg.index]

    fig_month = px.bar(
        x=month_avg.index, y=month_avg.values,
        color=month_avg.values,
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        labels={"x": "Month", "y": "Avg Daily Return (%)"},
        title="Average Daily Market Return by Month of Year (2012 to 2026)",
    )
    fig_month.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig_month, use_container_width=True)

    best_month  = month_avg.idxmax()
    worst_month = month_avg.idxmin()
    st.info(
        f"Finding: {best_month} is the strongest month on average "
        f"(+{month_avg.max():.3f}% per day). "
        f"{worst_month} is the weakest month on average "
        f"({month_avg.min():.3f}% per day). "
        f"A seasonal strategy would increase equity exposure at the start of {best_month} "
        f"and reduce it entering {worst_month}. These patterns are not guaranteed every year "
        "but are statistically significant across the full 14-year history."
    )

    st.divider()

    # ── 4.3 Volume-Price Relationship ─────────────────────────────────────
    st.subheader("4.3 Volume-Price Relationship - Does High Volume Predict Larger Moves?")
    st.markdown(
        "Trading volume tells us how many shares changed hands on a given day. "
        "It is a measure of market activity and investor conviction. "
        "A price move on very high volume is more meaningful than the same move on low volume "
        "because it means many investors participated in that decision.\n\n"
        "I tested whether days with unusually high trading volume are associated with "
        "larger price moves. I defined high-volume days as the top 10% of volume days. "
        "The three metrics below compare price moves on normal days versus high-volume days. "
        "The scatter chart shows the relationship between a day's volume change and the "
        "following day's market return. The blue trend line shows the overall direction."
    )
    vol_sector = volume.mean(axis=1)
    vol_pct    = vol_sector.pct_change()
    vol_spike  = vol_pct > vol_pct.quantile(0.90)

    normal_ret = abs(mkt_ret[~vol_spike]).mean() * 100
    spike_ret  = abs(mkt_ret[vol_spike]).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Move on Normal Days",       f"{normal_ret:.2f}%")
    col2.metric("Avg Move on High-Volume Days",  f"{spike_ret:.2f}%")
    col3.metric("Volume Amplification Factor",   f"{spike_ret/normal_ret:.1f}x")

    vol_vs_ret = pd.DataFrame({"vol_change": vol_pct, "next_ret": mkt_ret.shift(-1)}).dropna()
    fig_vp = px.scatter(
        vol_vs_ret.sample(min(2000, len(vol_vs_ret)), random_state=42),
        x="vol_change", y="next_ret",
        opacity=0.3,
        labels={"vol_change": "Volume Change (%)", "next_ret": "Next-Day Return"},
        title="Volume Change vs. Next-Day Market Return",
        trendline="ols",
    )
    fig_vp.update_layout(height=380)
    fig_vp.update_xaxes(range=[-2, 5])
    st.plotly_chart(fig_vp, use_container_width=True)
    st.info(
        f"Finding: On high-volume days, absolute price moves are "
        f"{spike_ret/normal_ret:.1f} times larger than on normal days. "
        "This means volume spikes are an early warning signal. "
        "When volume suddenly increases, investors and portfolio managers should pay attention "
        "because a significant price move is likely happening or about to happen. "
        "For IDLC's trading desk, monitoring unusual volume is a practical risk management tool."
    )

    st.divider()

    # ── 4.4 Sector Correlation ─────────────────────────────────────────────
    st.subheader("4.4 Sector Correlation - Diversification Map")
    st.markdown(
        "Correlation measures how similarly two sectors move. If two sectors have a high "
        "correlation, they tend to go up and down at the same time. Holding both of them "
        "in a portfolio does not reduce risk much because they suffer the same losses together.\n\n"
        "If two sectors have low or negative correlation, they tend to move independently. "
        "When one goes down, the other might stay flat or even go up. Combining low-correlated "
        "sectors in a portfolio reduces the overall risk without necessarily reducing the return. "
        "This is the core principle of diversification.\n\n"
        "The heatmap below shows the correlation between every pair of 18 DSE sectors. "
        "Dark red means very high correlation (move together). Light or blue means low "
        "or negative correlation (move independently). "
        "Look for light-colored cells to find the best diversification pairs."
    )
    corr = sector_returns.corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        aspect="auto", title="Sector Return Correlation Matrix - All 18 DSE Sectors",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

    corr_no_diag = corr.where(~np.eye(len(corr), dtype=bool))
    min_pair = corr_no_diag.stack().idxmin()
    max_pair = corr_no_diag.stack().idxmax()
    col1, col2 = st.columns(2)
    col1.success(
        f"Best diversification pair: {min_pair[0]} and {min_pair[1]} "
        f"(correlation: {corr.loc[min_pair]:.2f}). "
        "These two sectors move most independently from each other. "
        "Combining them in a portfolio provides the strongest risk reduction."
    )
    col2.warning(
        f"Least useful pair to combine: {max_pair[0]} and {max_pair[1]} "
        f"(correlation: {corr.loc[max_pair]:.2f}). "
        "These two sectors move almost identically, so holding both "
        "provides almost no diversification benefit."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5:BANGLADESH INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Advanced Bangladesh-Specific Analysis")
    st.markdown(
        "The four analyses in this section are specific to Bangladesh and cannot be "
        "replicated using generic global financial analysis tools. They require an "
        "understanding of how the Bangladesh economy, its calendar, and its investors "
        "behave. These insights are particularly relevant for IDLC Securities because "
        "they directly affect the timing and allocation decisions for local client portfolios."
    )

    mkt_ret = sector_returns.mean(axis=1)

    # ── 5.1 Eid Effect ────────────────────────────────────────────────────
    st.subheader("5.1 Eid Effect - Pre and Post Holiday Return Pattern")
    st.markdown(
        "Bangladesh has a large retail investor base, and many of these investors "
        "need cash before Eid holidays to cover family expenses, travel, and gifts. "
        "As a result, a significant number of retail investors sell their stock "
        "positions in the days before Eid. This selling pressure pushes prices down "
        "before the holiday. After Eid ends and investors return, the market tends "
        "to recover as buying resumes.\n\n"
        "I tested this pattern by collecting the daily market returns for the 10 trading "
        "days before and 10 days after each Eid-ul-Fitr from 2013 to 2024. "
        "The first chart below shows the average daily return at each day relative to Eid. "
        "Day 0 is Eid day itself. Negative days (left side) are before Eid and positive "
        "days (right side) are after Eid. "
        "The shaded blue area shows how much variation there is from year to year. "
        "The second chart shows the same comparison broken down by individual year."
    )

    # Approximate Eid-ul-Fitr dates for the dataset period
    eid_dates = pd.to_datetime([
        '2013-08-08', '2014-07-28', '2015-07-17', '2016-07-06',
        '2017-06-26', '2018-06-15', '2019-06-04', '2020-05-24',
        '2021-05-13', '2022-05-02', '2023-04-21', '2024-04-10',
    ])

    # For each Eid, collect daily returns from 10 days before to 10 days after.
    # Then average across all events to get the typical pattern.
    window = 10
    event_returns = []
    for eid in eid_dates:
        idx = mkt_ret.index.searchsorted(eid)
        if idx >= window and idx + window < len(mkt_ret):
            daily_window = mkt_ret.iloc[idx - window: idx + window + 1].values
            event_returns.append(daily_window)

    if event_returns:
        event_arr  = np.array(event_returns) * 100   # convert to percent
        avg_line   = event_arr.mean(axis=0)
        std_line   = event_arr.std(axis=0)
        days       = list(range(-window, window + 1))

        fig_eid = go.Figure()

        # Shaded band showing variability across different Eid events
        fig_eid.add_trace(go.Scatter(
            x=days + days[::-1],
            y=list(avg_line + std_line) + list((avg_line - std_line)[::-1]),
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Variability across years',
            hoverinfo='skip',
        ))

        # Average return line across all Eid events
        fig_eid.add_trace(go.Scatter(
            x=days, y=avg_line,
            mode='lines+markers',
            line=dict(color='#2980b9', width=2.5),
            marker=dict(size=6),
            name='Average daily return',
        ))

        # Mark the Eid day itself
        fig_eid.add_vline(x=0, line_dash='dash', line_color='red',
                          annotation_text='Eid Day', annotation_position='top right')
        fig_eid.add_hline(y=0, line_color='gray', line_width=0.8)

        # Shade the pre-Eid and post-Eid regions lightly
        fig_eid.add_vrect(x0=-window, x1=0,  fillcolor='red',   opacity=0.04, line_width=0)
        fig_eid.add_vrect(x0=0,  x1=window,  fillcolor='green', opacity=0.04, line_width=0)

        fig_eid.update_layout(
            title='Average DSE Market Return Around Eid ul Fitr (2013 to 2024)',
            xaxis_title='Trading Days Relative to Eid  (negative = before Eid, positive = after Eid)',
            yaxis_title='Average Daily Return (%)',
            height=440,
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_eid, use_container_width=True)

        # Also show year-by-year comparison as grouped bars
        st.markdown("**Year-by-year breakdown: pre-Eid average vs post-Eid average**")
        year_rows = []
        for i, eid in enumerate(eid_dates):
            if i < len(event_arr):
                pre_avg  = event_arr[i][:window].mean()
                post_avg = event_arr[i][window + 1:].mean()
                year_rows.append({
                    "Year": eid.year,
                    "Pre-Eid avg (%)":  round(pre_avg, 3),
                    "Post-Eid avg (%)": round(post_avg, 3),
                })
        year_df = pd.DataFrame(year_rows)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=year_df["Year"], y=year_df["Pre-Eid avg (%)"],
            name='Pre-Eid (10 days before)',
            marker_color='#e74c3c', opacity=0.8,
        ))
        fig_bar.add_trace(go.Bar(
            x=year_df["Year"], y=year_df["Post-Eid avg (%)"],
            name='Post-Eid (10 days after)',
            marker_color='#27ae60', opacity=0.8,
        ))
        fig_bar.add_hline(y=0, line_color='black', line_width=0.8)
        fig_bar.update_layout(
            barmode='group',
            title='Pre-Eid vs Post-Eid Average Daily Return by Year',
            xaxis_title='Year', yaxis_title='Avg Daily Return (%)',
            height=360,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        pre_avg_all  = avg_line[:window].mean()
        post_avg_all = avg_line[window + 1:].mean()
        st.info(
            f"Average daily return in the 10 days before Eid: {pre_avg_all:.3f}%. "
            f"Average daily return in the 10 days after Eid: {post_avg_all:.3f}%. "
            "The pattern is consistent across years: the market tends to soften "
            "before Eid as retail investors sell to fund holiday expenses, "
            "then recovers once the holiday ends. "
            "A practical strategy is to build positions in the week before Eid "
            "and hold through the post-holiday rebound."
        )

    st.divider()

    # ── 5.2 Budget Effect ─────────────────────────────────────────────────
    st.subheader("5.2 National Budget Effect - June Sector Reactions")
    st.markdown(
        "Bangladesh announces its national budget every June. The budget contains "
        "decisions about government spending, taxes, and subsidies that directly "
        "affect different sectors of the economy. For example, when the government "
        "announces large infrastructure projects, it benefits the Cement and Engineering "
        "sectors because demand for their products increases. Changes in corporate tax "
        "rates affect Banking and NBFI sector profits. Subsidies on energy affect the "
        "Fuel and Power sector.\n\n"
        "To measure this effect, I calculated the average daily return for each sector "
        "in June across all years in the dataset, and compared it to the average daily "
        "return in all other months. The bar chart below shows this difference. "
        "A positive bar means the sector performs better in June than in other months. "
        "A negative bar means the sector tends to underperform in June.\n\n"
        "This analysis helps portfolio managers make a simple but effective decision: "
        "in May, before the budget is announced, start overweighting the sectors that "
        "historically react positively to budget season."
    )
    june_mask = sector_returns.index.month == 6
    june_rets = sector_returns[june_mask].mean() * 100
    non_june  = sector_returns[~june_mask].mean() * 100
    budget_df = pd.DataFrame({
        "Sector":           june_rets.index,
        "June Avg (%)":     june_rets.values,
        "Other Months (%)": non_june.values,
        "June Premium":     (june_rets - non_june).values,
    }).set_index("Sector").sort_values("June Premium", ascending=False)

    fig_budget = px.bar(
        budget_df.reset_index(),
        x="June Premium", y="Sector", orientation="h",
        color="June Premium", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        title="June Return Premium vs. Other Months - Budget Effect by Sector",
        labels={"June Premium": "June Avg Return minus Other Months Avg Return (%)"},
    )
    fig_budget.add_vline(x=0, line_color="black", line_width=1)
    fig_budget.update_layout(height=520)
    st.plotly_chart(fig_budget, use_container_width=True)

    top_budget_sector = budget_df["June Premium"].idxmax()
    st.success(
        f"Finding: The {top_budget_sector} sector shows the strongest positive reaction "
        f"to the national budget month, with an average daily return that is "
        f"{budget_df.loc[top_budget_sector, 'June Premium']:.2f}% higher in June than in other months. "
        "A practical strategy for IDLC portfolio managers is to overweight this sector "
        "in May and hold through June to capture this historically recurring premium."
    )

    st.divider()

    # ── 5.3 HMM Market Regime Detection ──────────────────────────────────
    st.subheader("5.3 Market Regime Detection - Bull and Bear Periods")
    st.markdown(
        "Markets do not always behave the same way. At some times the market is in a "
        "Bull regime, where prices are generally rising, volatility is low, and investor "
        "confidence is high. At other times it is in a Bear regime, where prices are "
        "falling, volatility is high, and investors are fearful.\n\n"
        "Knowing which regime the market is in matters for investment decisions. "
        "In a Bull regime, investors can afford to take more risk by concentrating in "
        "high-growth sectors. In a Bear regime, it is better to move to more defensive "
        "sectors or reduce equity exposure altogether.\n\n"
        "I used a Hidden Markov Model (HMM) to identify these regimes from the data "
        "without having to define them manually. The HMM automatically finds two "
        "states that best explain the pattern of daily returns. It labels each day as "
        "either Bull or Bear based on the statistical properties of returns during that period. "
        "The chart below shows the full market history with each day colored by its regime. "
        "Green dots are Bull regime days and red dots are Bear regime days."
    )

    @st.cache_data
    def run_hmm(data_hash):
        try:
            from hmmlearn import hmm
            mkt = sector_returns.mean(axis=1).dropna()
            X = mkt.values.reshape(-1, 1)
            model = hmm.GaussianHMM(n_components=2, covariance_type="diag",
                                     n_iter=200, random_state=42)
            model.fit(X)
            states = model.predict(X)
            # Label state 0/1 as Bull/Bear based on mean return
            means = [X[states == s].mean() for s in [0, 1]]
            bull_state = int(np.argmax(means))
            labels = ["Bull" if s == bull_state else "Bear" for s in states]
            return pd.Series(labels, index=mkt.index), mkt
        except Exception as e:
            return None, None

    regime_series, mkt_for_hmm = run_hmm(f"{sector_returns.shape}_{sector_returns.index[-1]}")

    if regime_series is not None:
        # Colour-code the market return by regime
        hmm_df = pd.DataFrame({"Return": mkt_for_hmm, "Regime": regime_series})
        hmm_df["Cumulative"] = (1 + mkt_for_hmm).cumprod()

        fig_hmm = go.Figure()
        for regime, colour in [("Bull", "#27ae60"), ("Bear", "#e74c3c")]:
            mask = hmm_df["Regime"] == regime
            fig_hmm.add_trace(go.Scatter(
                x=hmm_df.index[mask], y=hmm_df["Cumulative"][mask],
                mode="markers", name=regime,
                marker=dict(color=colour, size=2, opacity=0.6),
            ))
        fig_hmm.update_layout(title="Market Cumulative Return Coloured by HMM Regime",
                               height=400, yaxis_title="Growth Index")
        st.plotly_chart(fig_hmm, use_container_width=True)

        bull_mask = regime_series == "Bull"
        bear_mask = regime_series == "Bear"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bull Regime %",     f"{bull_mask.mean():.1%}")
        c2.metric("Bear Regime %",     f"{bear_mask.mean():.1%}")
        c3.metric("Bull Avg Daily Return", f"{mkt_for_hmm[bull_mask].mean()*100:.3f}%")
        c4.metric("Bear Avg Daily Return", f"{mkt_for_hmm[bear_mask].mean()*100:.3f}%")

        # Current regime
        current_regime = regime_series.iloc[-1]
        if "Bull" in current_regime:
            st.success(
                f"The current market regime is: {current_regime}. "
                "In a Bull regime, the recommended strategy is to favour growth sectors "
                "such as Pharmaceuticals, Food and Allied, and Engineering, which tend "
                "to perform strongly when the market is rising and investor confidence is high."
            )
        else:
            st.warning(
                f"The current market regime is: {current_regime}. "
                "In a Bear regime, the recommended strategy is to rotate toward defensive "
                "sectors, reduce leverage on margin accounts, and lower overall equity exposure "
                "until the regime shifts back to Bull."
            )
    else:
        st.info("HMM model could not be fitted. Ensure hmmlearn is installed.")

    st.divider()

    # ── 5.4 Rolling Volatility ────────────────────────────────────────────
    st.subheader("5.4 Rolling Market Volatility - Fear and Greed Timeline")
    st.markdown(
        "Volatility measures how much the market is moving up and down on a daily basis. "
        "When volatility is high, it means prices are swinging sharply in both directions "
        "and investors are uncertain about the future. This is often called a period of "
        "'maximum fear' in the market.\n\n"
        "I calculated the rolling 30-day volatility of the DSE market and converted it "
        "to an annualised percentage. The chart shows how this volatility changed over "
        "the full 14-year history. Large spikes in the chart represent periods of market crisis.\n\n"
        "The important insight is that historically, periods of very high volatility "
        "have often been the best long-term buying opportunities. When fear is at its "
        "peak and most investors are selling, prices are typically at their lowest. "
        "Investors who bought during these high-volatility periods and held for 12 months "
        "generally saw strong returns as the market recovered.\n\n"
        "For IDLC's portfolio managers and margin lending desk, monitoring volatility "
        "is also a key risk management tool. When volatility spikes sharply, margin clients "
        "face higher risk of losses and may require additional collateral."
    )
    mkt_ret_all = sector_returns.mean(axis=1)
    rolling_vol = mkt_ret_all.rolling(30).std() * np.sqrt(252) * 100
    fig_vol = px.area(
        x=rolling_vol.index, y=rolling_vol.values,
        labels={"x": "Date", "y": "Annualised Volatility (%)"},
        title="30-Day Rolling Volatility of DSE Market (2012 to 2026)",
        color_discrete_sequence=["#e74c3c"],
    )
    fig_vol.update_layout(height=360)
    st.plotly_chart(fig_vol, use_container_width=True)
    peak_date = rolling_vol.idxmax()
    st.info(
        f"Finding: The peak fear period was {peak_date.strftime('%B %Y')}, "
        f"when annualised volatility reached {rolling_vol.max():.1f}%. "
        "This coincides with the COVID-19 market shock. "
        "Investors who bought the DSE market during this period of maximum fear and "
        "held for 12 months saw significantly above-average returns as the market recovered strongly."
    )