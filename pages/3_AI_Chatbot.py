"""
Bonus:AI Signal Engine + Investment Chatbot
Risk management tools and a Groq-powered chatbot that answers
questions about the DSE analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from signal_engine import DSESectorSignalEngine

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("Bonus: AI Signal Engine and Investment Chatbot")
st.markdown(
    "This section includes risk management tools, position sizing guidance, "
    "margin risk classification for IDLC's lending desk, and an AI chatbot "
    "that answers investment questions based on the actual analysis data."
)
st.divider()

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    sector_returns = pd.read_parquet("data/processed/sector_returns.parquet")
    return sector_returns

sector_returns = load_data()

@st.cache_resource
def get_engine(sector_returns):
    engine = DSESectorSignalEngine(sector_returns, risk_free_rate=0.075)
    engine.train_ml(cutoff_date="2018-12-31")
    return engine

engine  = get_engine(sector_returns)
signals = engine.generate_signals()

# Identify top-3 BUY sectors for the portfolio calculations below
top3 = signals[signals["Signal"] == "BUY"].head(3).index.tolist()
if len(top3) < 3:
    top3 = signals.head(3).index.tolist()

# ── Section 1: Live Signal Summary ───────────────────────────────────────
st.subheader("3.1 Live Signal Engine Output")
st.markdown(
    f"Signals as of **{sector_returns.index[-1].strftime('%d %B %Y')}**. "
    "The engine retrains on data through 2018 and generates fresh signals each session.\n\n"
    "Each sector gets a composite score from 0 to 100 based on three inputs: "
    "35% momentum across multiple timeframes, 30% relative strength versus the market, "
    "and 35% XGBoost ML probability of being in the top 3 next month. "
    "A score above 65 is a BUY signal — all three components agree the sector is likely "
    "to continue outperforming. Below 35 is a SELL — momentum is weak, the sector is "
    "underperforming the market, and the ML model assigns low probability to it. "
    "Everything in between is HOLD. The Confidence column shows how decisive the signal is: "
    "a score of 74 is a stronger BUY than a score of 66, even if both are classified the same way."
)

col1, col2, col3 = st.columns(3)
buy_list  = signals[signals["Signal"] == "BUY"].index.tolist()
sell_list = signals[signals["Signal"] == "SELL"].index.tolist()
hold_list = signals[signals["Signal"] == "HOLD"].index.tolist()

col1.metric("BUY Sectors",  len(buy_list),  delta="overweight these")
col2.metric("HOLD Sectors", len(hold_list), delta="neutral")
col3.metric("SELL Sectors", len(sell_list), delta="underweight these", delta_color="inverse")

def colour_signal(val):
    if val == "BUY":  return "background-color:#d4edda;color:#155724;font-weight:bold"
    if val == "SELL": return "background-color:#f8d7da;color:#721c24;font-weight:bold"
    return "background-color:#fff3cd;color:#856404"

st.markdown(
    "**Score 0–100:** >65 = BUY (strong momentum + ML confirms), "
    "35–65 = HOLD (neutral, no clear edge), <35 = SELL (underperforming, capital leaving). "
    "**Confidence** reflects how far the score is from the neutral midpoint (50)."
)
st.dataframe(
    signals.style.map(colour_signal, subset=["Signal"]).format({"Composite Score": "{:.1f}"}),
    use_container_width=True,
    height=400,
)
st.divider()

# ── Section 2: Portfolio Risk Management ─────────────────────────────────
st.subheader("3.2 Portfolio Risk Management: VaR & CVaR")
st.markdown(
    f"The current recommended portfolio is **{', '.join(top3)}** — the top BUY sectors "
    "from the signal engine. Before investing, it is important to understand the "
    "downside: how bad can a single bad month actually get?\n\n"
    "Value at Risk (VaR) answers this at the 95% level. It means: in the worst 5% of "
    "months historically, losses were at least this large. CVaR (Conditional VaR) goes "
    "one step further — it is the average loss specifically in those worst months, giving "
    "a more complete picture of tail risk. Both numbers are calculated using actual "
    "historical monthly returns from the DSE data, not theoretical assumptions.\n\n"
    "The chart below shows the full return distribution for the portfolio versus the "
    "equal-weight market, so you can see both the typical range of returns and the "
    "left tail where the worst losses sit."
)

portfolio_daily = sector_returns[top3].mean(axis=1)
market_daily    = sector_returns.mean(axis=1)

monthly_port = (1 + portfolio_daily).resample("ME").prod() - 1
monthly_mkt  = (1 + market_daily).resample("ME").prod() - 1

var_port  = np.percentile(monthly_port.dropna(), 5)   # 95% VaR
cvar_port = monthly_port[monthly_port <= var_port].mean()
var_mkt   = np.percentile(monthly_mkt.dropna(), 5)
cvar_mkt  = monthly_mkt[monthly_mkt <= var_mkt].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio VaR 95%",  f"{var_port:.1%}",  help="Worst loss in 5% of months")
col2.metric("Portfolio CVaR",     f"{cvar_port:.1%}", help="Avg loss when VaR is breached")
col3.metric("Market VaR 95%",     f"{var_mkt:.1%}")
col4.metric("Market CVaR",        f"{cvar_mkt:.1%}")

# Return distribution chart
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=monthly_mkt * 100, name="Market", opacity=0.5,
                                marker_color="gray", nbinsx=50, histnorm="probability density"))
fig_dist.add_trace(go.Histogram(x=monthly_port * 100, name="Portfolio", opacity=0.7,
                                marker_color="#3498db", nbinsx=50, histnorm="probability density"))
fig_dist.add_vline(x=var_port * 100,  line_dash="dash", line_color="#e74c3c",
                   annotation_text=f"VaR: {var_port:.1%}")
fig_dist.add_vline(x=cvar_port * 100, line_dash="solid", line_color="#c0392b",
                   annotation_text=f"CVaR: {cvar_port:.1%}")
fig_dist.update_layout(title="Monthly Return Distribution:Portfolio vs Market",
                       xaxis_title="Monthly Return (%)", height=380, barmode="overlay")
st.markdown(
    "**How to read this chart:** "
    "**Blue bars** = portfolio monthly return distribution. "
    "**Gray bars** = equal-weight market. "
    "The tall peak in the middle = the most common return range (near zero). "
    "**Dashed red line (VaR)** = the threshold: in the worst 5% of months, losses are at least this bad. "
    "**Solid red line (CVaR)** = the average loss in those worst 5% of months:the true expected shortfall."
)
st.plotly_chart(fig_dist, use_container_width=True)

st.info(
    f"**For IDLC Margin Clients:** A client using 2× leverage faces a worst-month loss "
    f"of ~**{abs(var_port) * 2:.1%}** (their own money only). "
    "This is the key number for setting margin call thresholds."
)
st.divider()

# ── Section 3: Kelly Criterion ────────────────────────────────────────────
st.subheader("3.3 Kelly Criterion: Optimal Position Sizing")
st.markdown(
    "Knowing which sectors to buy is one thing. Knowing how much capital to put in each "
    "sector is equally important. The Kelly Criterion is a mathematical formula that "
    "answers exactly this question based on historical win rates and average payoffs.\n\n"
    "The formula is: **f = Win Rate − (Loss Rate ÷ Odds)** where Odds = average winning "
    "month divided by average losing month. A sector with a high win rate and large average "
    "gains relative to losses gets a higher Kelly fraction — meaning you should allocate "
    "more capital to it.\n\n"
    "**Why Half-Kelly?** Full Kelly maximises long-run growth in theory, but it assumes "
    "perfectly accurate historical estimates. In practice those estimates are noisy, and "
    "full Kelly can lead to very large drawdowns if estimates are even slightly off. "
    "Using half the Kelly fraction keeps about 75% of the theoretical growth benefit while "
    "cutting risk substantially. This is standard practice among professional fund managers."
)

kelly_rows = []
for sector in sector_returns.columns:
    s         = sector_returns[sector].dropna()
    monthly_s = (1 + s).resample("ME").prod() - 1
    wins      = monthly_s > 0
    p_win     = wins.mean()
    avg_win   = monthly_s[wins].mean() if wins.sum() > 0 else 0
    avg_loss  = abs(monthly_s[~wins].mean()) if (~wins).sum() > 0 else 1e-6
    odds      = avg_win / avg_loss if avg_loss > 0 else 0
    kelly     = max(0, p_win - (1 - p_win) / odds) if odds > 0 else 0

    kelly_rows.append({
        "Sector":              sector,
        "Win Rate":            round(p_win * 100, 1),
        "Avg Win (%)":         round(avg_win * 100, 2),
        "Avg Loss (%)":        round(avg_loss * 100, 2),
        "Full Kelly (%)":      round(kelly * 100, 1),
        "Half Kelly (%)":      round(kelly * 50, 1),   # practical recommendation
    })

kelly_df = pd.DataFrame(kelly_rows).set_index("Sector").sort_values("Half Kelly (%)", ascending=False)

fig_kelly = px.bar(
    kelly_df.reset_index(),
    x="Half Kelly (%)",
    y="Sector",
    orientation="h",
    color="Win Rate",
    color_continuous_scale="Greens",
    title="Half-Kelly Position Sizing:Practical Allocation per Sector",
    labels={"Half Kelly (%)": "Recommended Allocation (%)", "Win Rate": "Win Rate (%)"},
)
fig_kelly.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_kelly, use_container_width=True)

with st.expander("Show full Kelly Criterion table"):
    st.dataframe(kelly_df, use_container_width=True)

st.divider()

# ── Section 4: Margin Financing Risk Classification ───────────────────────
st.subheader("3.4 IDLC Margin Financing Risk Classification")
st.markdown(
    "From a lending desk perspective, not all sectors are equally safe as margin collateral. "
    "If a client borrows against their stock holdings and the sector crashes sharply, "
    "the stock value may fall below the loan amount before IDLC can issue a margin call "
    "and recover its money. The more volatile a sector, the greater this risk.\n\n"
    "This table classifies every DSE sector into three risk tiers based on annualised "
    "volatility, maximum historical drawdown, and monthly VaR. Each tier maps to a "
    "recommended Loan-to-Value (LTV) ratio for IDLC's lending desk.\n\n"
    "**LTV explained:** A 60% LTV means IDLC lends BDT 60 for every BDT 100 of stocks "
    "the client holds — safe because there is a 40% buffer before the loan is at risk. "
    "A 40% LTV means IDLC lends only BDT 40 per BDT 100 — applied to high-volatility "
    "sectors where a sudden crash could wipe out the client's equity before a margin "
    "call can be executed."
)

margin_rows = []
for sector in sector_returns.columns:
    s         = sector_returns[sector].dropna()
    monthly_s = (1 + s).resample("ME").prod() - 1
    ann_vol   = s.std() * np.sqrt(252)
    cum       = (1 + s).cumprod()
    max_dd    = ((cum - cum.cummax()) / cum.cummax()).min()
    var_95    = np.percentile(monthly_s.dropna(), 5)

    if ann_vol < 0.20:
        ltv    = "60%"
        rating = "Low Risk"
    elif ann_vol < 0.35:
        ltv    = "50%"
        rating = "Medium Risk"
    else:
        ltv    = "40%"
        rating = "High Risk"

    margin_rows.append({
        "Sector":           sector,
        "Ann. Volatility":  round(ann_vol * 100, 1),
        "Max Drawdown (%)": round(max_dd * 100, 1),
        "Monthly VaR 95%":  round(var_95 * 100, 1),
        "Loan-to-Value":    ltv,
        "IDLC Rating":      rating,
    })

margin_df = (pd.DataFrame(margin_rows)
               .set_index("Sector")
               .sort_values("Ann. Volatility"))

# Colour the rating column
def colour_rating(val):
    if val == "Low Risk":
        return "background-color: #d4edda; color: #155724"
    elif val == "High Risk":
        return "background-color: #f8d7da; color: #721c24"
    return "background-color: #fff3cd; color: #856404"

st.dataframe(
    margin_df.style.map(colour_rating, subset=["IDLC Rating"]),
    use_container_width=True,
    height=450,
)

low_risk_sectors = margin_df[margin_df["IDLC Rating"] == "Low Risk"].index.tolist()
st.success(
    f"**Safest sectors for margin lending:** {', '.join(low_risk_sectors)}. "
    "These qualify for 60% LTV — IDLC can lend BDT 60 for every BDT 100 of stocks the client holds as collateral."
)
st.divider()

# ── Section 5: AI Investment Chatbot ─────────────────────────────────────
st.subheader("3.5 DSE Investment Chatbot: Ask Anything")
st.markdown(
    "This chatbot is powered by **Llama 3.1 (Groq API)**. Unlike a general AI assistant, "
    "it is grounded in the actual numbers from this analysis — sector scores, VaR figures, "
    "Kelly allocations, and margin ratings. It will not make up numbers or give generic "
    "financial advice. Every answer it gives is based on the live data computed above.\n\n"
    "You can ask it sector-specific questions, compare two sectors, get a portfolio "
    "recommendation, or ask what the data says about a sector you already hold."
)

# Build the knowledge base from live analysis results
def build_knowledge_base():
    """
    Summarise the key analysis outputs into a compact text block.
    This gets sent as context with every user question.
    """
    buy_str  = ", ".join(signals[signals["Signal"] == "BUY"].index.tolist()) or "None"
    sell_str = ", ".join(signals[signals["Signal"] == "SELL"].index.tolist()) or "None"
    hold_str = ", ".join(signals[signals["Signal"] == "HOLD"].index.tolist()) or "None"

    # Top-3 Kelly sectors
    top_kelly = kelly_df.head(3)
    kelly_str = "; ".join([
        f"{sec}: {row['Half Kelly (%)']:.0f}% (win rate {row['Win Rate']:.0f}%)"
        for sec, row in top_kelly.iterrows()
    ])

    # Low-risk margin sectors
    low_risk = margin_df[margin_df["IDLC Rating"] == "Low Risk"].index.tolist()
    high_risk = margin_df[margin_df["IDLC Rating"] == "High Risk"].index.tolist()

    # Sector-by-sector signal scores
    signal_detail = "\n".join([
        f"  - {sec}: Score {row['Composite Score']:.0f}/100, {row['Signal']}, Confidence {row['Confidence']}"
        for sec, row in signals.iterrows()
    ])

    context = f"""
DSE SECTOR ANALYSIS:as of {sector_returns.index[-1].strftime('%B %Y')}

CURRENT SIGNALS:
  BUY  (overweight): {buy_str}
  HOLD (neutral):    {hold_str}
  SELL (underweight): {sell_str}

DETAILED SCORES:
{signal_detail}

PORTFOLIO RISK (top-3 BUY sectors: {', '.join(top3)}):
  VaR 95% (monthly): {var_port:.1%} :worst loss in 5% of months
  CVaR (expected shortfall): {cvar_port:.1%}
  For 2x leverage margin clients: worst month loss = {abs(var_port)*2:.1%}

KELLY CRITERION:top position sizes (half-Kelly):
  {kelly_str}

MARGIN FINANCING SAFETY:
  Low Risk (60% LTV eligible): {', '.join(low_risk) if low_risk else 'None'}
  High Risk (40% LTV max): {', '.join(high_risk) if high_risk else 'None'}

BACKTEST CONTEXT (2019–2026 bear market):
  - Buy & Hold equal weight: ~5% annual return
  - Momentum rotation underperformed due to sustained sector-wide declines
  - The rotation framework's primary value is risk classification, not return maximisation
  - Signal engine trained on 2010–2018 data (fully out-of-sample since 2019)

KEY DSE FACTS:
  - Risk-free rate: 7.5% (Bangladesh T-bill)
  - DSE transaction cost: 0.25% buy + 0.40% sell = 0.65% per round trip
  - BSEC minimum margin: 1:1 equity ratio
  - Bangladesh GDP growth target: >6%:drives sector expansion/contraction cycles
"""
    return context

knowledge_base = build_knowledge_base()

# Chatbot system prompt
SYSTEM_PROMPT = """You are the DSE Investment Assistant for IDLC Securities, Bangladesh.
You answer questions about DSE sector investments based ONLY on the analysis data provided to you.

Rules:
- Always cite specific numbers from the analysis (scores, percentages, ratios)
- Give actionable recommendations: which sector to buy, hold, or avoid
- Keep answers under 200 words:be direct and clear
- If the user asks about something not in the data, say so honestly
- Use BDT for currency, and refer to IDLC Securities as 'us' or 'our clients'
- Do not make up numbers:only use what is in the analysis context
"""

# Get Groq API key from Streamlit secrets (set in .streamlit/secrets.toml locally
# or in the Streamlit Cloud secrets panel for deployment)
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    st.warning(
        "Groq API key not found. To enable the chatbot:\n"
        "1. Create `.streamlit/secrets.toml` in your project root\n"
        "2. Add: `GROQ_API_KEY = \"your_key_here\"`\n"
        "3. Get a free key at https://console.groq.com"
    )
else:
    # Initialise chat history in session state so messages persist across reruns
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box at the bottom
    user_input = st.chat_input("Ask about DSE sectors, risk, portfolio allocation...")

    if user_input:
        # Show the user's message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build the full prompt: system + analysis context + conversation history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nANALYSIS DATA:\n" + knowledge_base}
        ]
        # Add recent conversation history (last 6 messages to stay within token limits)
        for msg in st.session_state.chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Call Groq API and stream the response
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                stream = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.3,     # low temperature for factual financial answers
                    max_tokens=400,
                    stream=True,
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")  # typing cursor

                response_placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Chatbot error: {e}")

    # Suggested questions to get started
    if not st.session_state.get("chat_history"):
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        cols[0].markdown("- Which sector should I invest in right now?")
        cols[0].markdown("- Why should I invest in Food & Allied?")
        cols[1].markdown("- I'm already in Banking:should I hold or sell?")
        cols[1].markdown("- Which sector is safest for a margin loan?")