"""
Portfolio Analysis Dashboard
============================
A production-ready Streamlit application for quantitative portfolio analysis.

Run with:
    streamlit run app.py

Dependencies:
    streamlit, yfinance, pandas, numpy, plotly, scipy
"""

import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

TRADING_DAYS = 252

# =============================================================================
# PAGE CONFIG & THEME
# =============================================================================
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0E1117; }

    section[data-testid="stSidebar"] {
        background-color: #161A22;
        border-right: 1px solid #262730;
    }

    h1, h2, h3, h4 { color: #FAFAFA; }

    div[data-testid="stMetric"] {
        background-color: #161A22;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #9CA3AF; }

    div.stButton > button, div.stFormSubmitButton > button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55em 1.2em;
        width: 100%;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #E03A3A;
        color: white;
    }

    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-top: 2.2rem;
        margin-bottom: 0.4rem;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 6px;
    }
    .subtle-caption { color: #9CA3AF; font-size: 0.85rem; margin-bottom: 1rem; }

    .weight-warning {
        background-color: rgba(255, 75, 75, 0.12);
        border: 1px solid #FF4B4B;
        border-radius: 8px;
        padding: 10px 14px;
        color: #FF9B9B;
        font-weight: 600;
    }
    .weight-ok {
        background-color: rgba(60, 200, 120, 0.12);
        border: 1px solid #3CC878;
        border-radius: 8px;
        padding: 10px 14px;
        color: #8CF0B4;
        font-weight: 600;
    }

    thead tr th { background-color: #1B1F27 !important; color: #FAFAFA !important; }
    tbody tr td { color: #E5E7EB !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PORTFOLIO = "#FF4B4B"
COLOR_BENCHMARK = "#5B8DEF"
COLOR_POSITIVE = "#3CC878"
COLOR_NEGATIVE = "#FF4B4B"
PAPER_BG = "#0E1117"
PLOT_BG = "#0E1117"


def style_fig(fig, height=420, legend=True):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        if legend
        else dict(),
        font=dict(color="#E5E7EB"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#262730", zerolinecolor="#262730")
    fig.update_yaxes(gridcolor="#262730", zerolinecolor="#262730")
    return fig


# =============================================================================
# DATA FETCHING
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers, start, end):
    """Fetch adjusted close prices for a list of tickers individually,
    so a single bad ticker doesn't break the whole download."""
    series_dict = {}
    failed = []
    for t in tickers:
        t = t.strip()
        if not t:
            continue
        try:
            hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist.columns:
                failed.append(t)
                continue
            s = hist["Close"].copy()
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = pd.to_datetime(s.index).normalize()
            series_dict[t] = s
        except Exception:
            failed.append(t)

    if not series_dict:
        return pd.DataFrame(), failed

    prices = pd.DataFrame(series_dict)
    prices = prices.sort_index()
    prices = prices.ffill().dropna(how="all")
    # Drop tickers that are entirely NaN after alignment
    fully_nan = prices.columns[prices.isna().all()].tolist()
    for c in fully_nan:
        failed.append(c)
    prices = prices.drop(columns=fully_nan, errors="ignore")
    prices = prices.dropna(how="any")
    return prices, failed


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_benchmark(ticker, start, end):
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist is None or hist.empty:
            return pd.Series(dtype=float)
        s = hist["Close"].copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = pd.to_datetime(s.index).normalize()
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)


# =============================================================================
# METRIC HELPERS
# =============================================================================
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def cagr(cum_series: pd.Series) -> float:
    if len(cum_series) < 2:
        return np.nan
    n_days = (cum_series.index[-1] - cum_series.index[0]).days
    n_years = n_days / 365.25
    if n_years <= 0:
        return np.nan
    total_return = cum_series.iloc[-1] / cum_series.iloc[0]
    if total_return <= 0:
        return np.nan
    return total_return ** (1 / n_years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float) -> float:
    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = annualized_vol(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - rf) / ann_vol


def sortino_ratio(returns: pd.Series, rf: float) -> float:
    ann_ret = returns.mean() * TRADING_DAYS
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.nan
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS)
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return (ann_ret - rf) / downside_dev


def max_drawdown(cum_series: pd.Series) -> float:
    running_max = cum_series.cummax()
    dd = cum_series / running_max - 1
    return dd.min()


def drawdown_series(cum_series: pd.Series) -> pd.Series:
    running_max = cum_series.cummax()
    return cum_series / running_max - 1


def beta_vs_benchmark(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    if cov[1, 1] == 0:
        return np.nan
    return cov[0, 1] / cov[1, 1]


def portfolio_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return (returns * weights).sum(axis=1)


def cumulative_from_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## ⚙️ Configuration")

default_tickers = "RELIANCE.NS, VBL.NS, TCS.NS"
tickers_raw = st.sidebar.text_input(
    "Stock Tickers (comma-separated)",
    value=default_tickers,
    help="Use Yahoo Finance ticker symbols, e.g. RELIANCE.NS, TCS.NS, AAPL",
)
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))  # de-duplicate, preserve order

benchmark_options = {
    "Nifty 50 (^NSEI)": "^NSEI",
    "Sensex (^BSESN)": "^BSESN",
    "Nifty Bank (^NSEBANK)": "^NSEBANK",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
}
benchmark_label = st.sidebar.selectbox("Benchmark Index", list(benchmark_options.keys()), index=0)
benchmark_ticker = benchmark_options[benchmark_label]

st.sidebar.markdown("#### 📅 Date Range")
col_d1, col_d2 = st.sidebar.columns(2)
default_end = datetime.date(2025, 1, 1)
default_start = default_end - datetime.timedelta(days=365 * 4)
with col_d1:
    start_date = st.date_input("Start Date", value=default_start, max_value=datetime.date.today())
with col_d2:
    end_date = st.date_input("End Date", value=default_end, max_value=datetime.date.today())

risk_free_rate_pct = st.sidebar.slider(
    "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=6.50, step=0.05
)
risk_free_rate = risk_free_rate_pct / 100.0

st.sidebar.markdown("#### ⚖️ Portfolio Weights (%)")
weights_pct = {}
if tickers:
    n = len(tickers)
    base = round(100.0 / n, 2)
    for i, t in enumerate(tickers):
        default_w = round(100.0 - base * (n - 1), 2) if i == n - 1 else base
        weights_pct[t] = st.sidebar.number_input(
            f"{t}", min_value=0.0, max_value=100.0, value=float(default_w), step=1.0, key=f"w_{t}"
        )

    total_weight = sum(weights_pct.values())
    if abs(total_weight - 100.0) > 0.01:
        st.sidebar.markdown(
            f'<div class="weight-warning">⚠️ Weights sum to {total_weight:.2f}% — must be 100%</div>',
            unsafe_allow_html=True,
        )
        weights_valid = False
    else:
        st.sidebar.markdown(
            f'<div class="weight-ok">✅ Weights sum to {total_weight:.2f}%</div>',
            unsafe_allow_html=True,
        )
        weights_valid = True
else:
    weights_valid = False
    st.sidebar.warning("Enter at least one ticker.")

st.sidebar.markdown("---")
mc_start_value = st.sidebar.number_input(
    "Monte Carlo Starting Value (₹)", min_value=1000.0, value=100000.0, step=1000.0
)
mc_n_sims = st.sidebar.slider("Monte Carlo Simulations", min_value=200, max_value=5000, value=1000, step=100)
mc_n_days = st.sidebar.slider("Monte Carlo Horizon (trading days)", min_value=30, max_value=500, value=250, step=10)

run_clicked = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

if run_clicked:
    if not tickers:
        st.sidebar.error("Please provide at least one valid ticker.")
    elif not weights_valid:
        st.sidebar.error("Fix portfolio weights so they sum to 100% before running.")
    elif start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
    else:
        st.session_state["run"] = True
        st.session_state["tickers"] = tickers
        st.session_state["weights_pct"] = weights_pct
        st.session_state["benchmark_ticker"] = benchmark_ticker
        st.session_state["benchmark_label"] = benchmark_label
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["risk_free_rate"] = risk_free_rate
        st.session_state["mc_start_value"] = mc_start_value
        st.session_state["mc_n_sims"] = mc_n_sims
        st.session_state["mc_n_days"] = mc_n_days


# =============================================================================
# MAIN TITLE
# =============================================================================
st.markdown("# 📊 Portfolio Analysis Dashboard")
st.markdown(
    '<p class="subtle-caption">Quantitative equity portfolio analytics — returns, risk, correlation, '
    "Monte Carlo simulation and efficient frontier optimisation.</p>",
    unsafe_allow_html=True,
)

if not st.session_state.get("run"):
    st.info("👈 Configure your tickers, weights and date range in the sidebar, then click **Run Analysis**.")
    st.stop()

# =============================================================================
# LOAD DATA
# =============================================================================
tickers = st.session_state["tickers"]
weights_pct = st.session_state["weights_pct"]
benchmark_ticker = st.session_state["benchmark_ticker"]
benchmark_label = st.session_state["benchmark_label"]
start_date = st.session_state["start_date"]
end_date = st.session_state["end_date"]
risk_free_rate = st.session_state["risk_free_rate"]
mc_start_value = st.session_state["mc_start_value"]
mc_n_sims = st.session_state["mc_n_sims"]
mc_n_days = st.session_state["mc_n_days"]

with st.spinner("Fetching market data from Yahoo Finance..."):
    prices, failed_tickers = fetch_prices(tickers, start_date, end_date)
    bench_prices = fetch_benchmark(benchmark_ticker, start_date, end_date)

if failed_tickers:
    st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}. They were excluded from the analysis.")

active_tickers = [t for t in tickers if t in prices.columns]

if prices.empty or len(active_tickers) == 0:
    st.error("No valid price data could be retrieved for the selected tickers and date range. Please check the tickers or widen the date range.")
    st.stop()

if len(prices) < 30:
    st.error("Not enough trading history in the selected date range (need at least ~30 trading days). Please widen the date range.")
    st.stop()

if bench_prices.empty:
    st.warning(f"⚠️ Benchmark data for {benchmark_ticker} could not be retrieved. Benchmark comparisons will be unavailable.")

# Re-normalise weights over the tickers that actually returned data
raw_weights = np.array([weights_pct[t] for t in active_tickers], dtype=float)
if raw_weights.sum() == 0:
    weights = np.ones(len(active_tickers)) / len(active_tickers)
else:
    weights = raw_weights / raw_weights.sum()
weights_series = pd.Series(weights, index=active_tickers)

# Align prices & benchmark on common dates
common_index = prices.index
if not bench_prices.empty:
    common_index = prices.index.intersection(bench_prices.index)
    prices_aligned = prices.loc[common_index]
    bench_aligned = bench_prices.loc[common_index]
else:
    prices_aligned = prices
    bench_aligned = pd.Series(dtype=float)

if len(common_index) < 30:
    st.warning("Very little overlapping history between portfolio and benchmark; benchmark comparisons may be unreliable.")

asset_returns = to_returns(prices_aligned)
port_returns = portfolio_daily_returns(asset_returns, weights)
port_cum = cumulative_from_returns(port_returns)

if not bench_aligned.empty:
    bench_returns = bench_aligned.pct_change().dropna()
    bench_returns = bench_returns.loc[bench_returns.index.intersection(port_returns.index)]
    bench_cum = cumulative_from_returns(bench_returns)
else:
    bench_returns = pd.Series(dtype=float)
    bench_cum = pd.Series(dtype=float)

# =============================================================================
# SECTION 1 — KPI CARDS
# =============================================================================
st.markdown('<div class="section-header">1️⃣ Portfolio Metrics</div>', unsafe_allow_html=True)

port_cagr = cagr(port_cum)
port_vol = annualized_vol(port_returns)
port_sharpe = sharpe_ratio(port_returns, risk_free_rate)
port_mdd = max_drawdown(port_cum)

if not bench_cum.empty and len(bench_cum) > 1:
    bench_cagr = cagr(bench_cum)
    bench_vol = annualized_vol(bench_returns)
    bench_sharpe = sharpe_ratio(bench_returns, risk_free_rate)
    bench_mdd = max_drawdown(bench_cum)
else:
    bench_cagr = bench_vol = bench_sharpe = bench_mdd = np.nan

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x * 100:.2f}%"


def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"


def delta_pct(port_val, bench_val):
    if pd.isna(port_val) or pd.isna(bench_val):
        return None
    return f"{(port_val - bench_val) * 100:+.2f}% vs benchmark"


def delta_num(port_val, bench_val):
    if pd.isna(port_val) or pd.isna(bench_val):
        return None
    return f"{(port_val - bench_val):+.2f} vs benchmark"


with kpi1:
    st.metric("CAGR", fmt_pct(port_cagr), delta_pct(port_cagr, bench_cagr))
with kpi2:
    st.metric("Volatility (Annualised)", fmt_pct(port_vol), delta_pct(bench_vol, port_vol) and delta_pct(port_vol, bench_vol))
with kpi3:
    st.metric("Sharpe Ratio", fmt_num(port_sharpe), delta_num(port_sharpe, bench_sharpe))
with kpi4:
    st.metric("Max Drawdown", fmt_pct(port_mdd), delta_pct(port_mdd, bench_mdd))

# =============================================================================
# SECTION 2 — CUMULATIVE RETURNS
# =============================================================================
st.markdown('<div class="section-header">2️⃣ Cumulative Returns</div>', unsafe_allow_html=True)

fig_cum = go.Figure()
fig_cum.add_trace(
    go.Scatter(
        x=port_cum.index, y=(port_cum - 1) * 100, name="Portfolio",
        line=dict(color=COLOR_PORTFOLIO, width=2.5),
    )
)
if not bench_cum.empty:
    fig_cum.add_trace(
        go.Scatter(
            x=bench_cum.index, y=(bench_cum - 1) * 100, name=benchmark_label,
            line=dict(color=COLOR_BENCHMARK, width=2),
        )
    )
fig_cum.update_layout(yaxis_title="Cumulative Return (%)", xaxis_title="Date")
st.plotly_chart(style_fig(fig_cum), use_container_width=True)

# =============================================================================
# SECTION 3 — DRAWDOWN
# =============================================================================
st.markdown('<div class="section-header">3️⃣ Drawdown Chart</div>', unsafe_allow_html=True)

port_dd = drawdown_series(port_cum) * 100
fig_dd = go.Figure()
fig_dd.add_trace(
    go.Scatter(
        x=port_dd.index, y=port_dd, name="Portfolio Drawdown", fill="tozeroy",
        line=dict(color=COLOR_NEGATIVE, width=1.5), fillcolor="rgba(255,75,75,0.25)",
    )
)
if not bench_cum.empty:
    bench_dd = drawdown_series(bench_cum) * 100
    fig_dd.add_trace(
        go.Scatter(
            x=bench_dd.index, y=bench_dd, name=f"{benchmark_label} Drawdown",
            line=dict(color=COLOR_BENCHMARK, width=1.5, dash="dot"),
        )
    )
fig_dd.update_layout(yaxis_title="Drawdown (%)", xaxis_title="Date")
st.plotly_chart(style_fig(fig_dd), use_container_width=True)

# =============================================================================
# SECTION 4 — CORRELATION & ALLOCATION
# =============================================================================
st.markdown('<div class="section-header">4️⃣ Correlation & Allocation</div>', unsafe_allow_html=True)
col_corr, col_alloc = st.columns(2)

with col_corr:
    st.markdown("**Correlation Matrix**")
    corr_matrix = asset_returns[active_tickers].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    st.plotly_chart(style_fig(fig_corr, height=420, legend=False), use_container_width=True)

with col_alloc:
    st.markdown("**Portfolio Allocation**")
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=active_tickers,
                values=weights * 100,
                hole=0.55,
                marker=dict(line=dict(color=PAPER_BG, width=2)),
                textinfo="label+percent",
            )
        ]
    )
    st.plotly_chart(style_fig(fig_pie, height=420, legend=True), use_container_width=True)

# =============================================================================
# SECTION 5 — INDIVIDUAL ASSET PERFORMANCE
# =============================================================================
st.markdown('<div class="section-header">5️⃣ Individual Asset Performance</div>', unsafe_allow_html=True)
col_ret, col_vol = st.columns(2)

with col_ret:
    st.markdown("**Annualised Return per Stock**")
    asset_cagrs = {}
    for t in active_tickers:
        asset_cum = cumulative_from_returns(asset_returns[t].dropna())
        asset_cagrs[t] = cagr(asset_cum)
    cagr_series = pd.Series(asset_cagrs).sort_values()
    colors_bar = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in cagr_series.values]
    fig_bar = go.Figure(
        go.Bar(
            x=cagr_series.values * 100, y=cagr_series.index, orientation="h",
            marker_color=colors_bar, text=[f"{v*100:.1f}%" for v in cagr_series.values],
            textposition="outside",
        )
    )
    fig_bar.update_layout(xaxis_title="Annualised Return (%)")
    st.plotly_chart(style_fig(fig_bar, height=420, legend=False), use_container_width=True)

with col_vol:
    st.markdown("**30-Day Rolling Volatility**")
    fig_rvol = go.Figure()
    for t in active_tickers:
        rvol = asset_returns[t].rolling(30).std() * np.sqrt(TRADING_DAYS) * 100
        fig_rvol.add_trace(go.Scatter(x=rvol.index, y=rvol, name=t, mode="lines"))
    fig_rvol.update_layout(yaxis_title="Annualised Volatility (%)")
    st.plotly_chart(style_fig(fig_rvol, height=420), use_container_width=True)

# =============================================================================
# SECTION 6 — MONTHLY RETURNS HEATMAP
# =============================================================================
st.markdown('<div class="section-header">6️⃣ Monthly Returns Heatmap</div>', unsafe_allow_html=True)

monthly_returns = (1 + port_returns).resample("ME").prod() - 1
monthly_df = monthly_returns.to_frame("ret")
monthly_df["Year"] = monthly_df.index.year
monthly_df["Month"] = monthly_df.index.strftime("%b")
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
pivot = monthly_df.pivot_table(index="Year", columns="Month", values="ret")
pivot = pivot.reindex(columns=month_order)

fig_heat = px.imshow(
    pivot * 100,
    text_auto=".1f",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    aspect="auto",
    labels=dict(color="Return (%)"),
)
fig_heat.update_yaxes(type="category", dtick=1)
st.plotly_chart(style_fig(fig_heat, height=max(320, 60 * len(pivot)), legend=False), use_container_width=True)

# =============================================================================
# SECTION 7 — ROLLING BETA
# =============================================================================
st.markdown('<div class="section-header">7️⃣ Rolling Beta (60-Day)</div>', unsafe_allow_html=True)

if not bench_returns.empty:
    aligned_rb = pd.concat([port_returns.rename("port"), bench_returns.rename("bench")], axis=1).dropna()
    roll_cov = aligned_rb["port"].rolling(60).cov(aligned_rb["bench"])
    roll_var = aligned_rb["bench"].rolling(60).var()
    rolling_beta = roll_cov / roll_var
    fig_beta = go.Figure()
    fig_beta.add_trace(
        go.Scatter(x=rolling_beta.index, y=rolling_beta, name="60-Day Rolling Beta", line=dict(color=COLOR_PORTFOLIO, width=2))
    )
    fig_beta.add_hline(y=1, line_dash="dash", line_color="#9CA3AF", annotation_text="Beta = 1")
    fig_beta.update_layout(yaxis_title="Beta")
    st.plotly_chart(style_fig(fig_beta), use_container_width=True)
else:
    st.info("Rolling beta requires benchmark data, which is currently unavailable.")

# =============================================================================
# SECTION 8 — MONTE CARLO SIMULATION
# =============================================================================
st.markdown('<div class="section-header">8️⃣ Monte Carlo Simulation</div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtle-caption">Simulating {mc_n_sims} paths over {mc_n_days} trading days, starting from ₹{mc_start_value:,.0f}, '
    "using the portfolio's historical daily mean & volatility (Geometric Brownian Motion).</p>",
    unsafe_allow_html=True,
)

mu = port_returns.mean()
sigma = port_returns.std()

rng = np.random.default_rng(seed=42)
daily_sim_returns = rng.normal(loc=mu, scale=sigma, size=(mc_n_days, mc_n_sims))
sim_growth = np.cumprod(1 + daily_sim_returns, axis=0)
sim_paths = mc_start_value * sim_growth
sim_paths = np.vstack([np.full(mc_n_sims, mc_start_value), sim_paths])  # prepend day-0

p5 = np.percentile(sim_paths, 5, axis=1)
p50 = np.percentile(sim_paths, 50, axis=1)
p95 = np.percentile(sim_paths, 95, axis=1)
days_axis = np.arange(0, mc_n_days + 1)

fig_mc = go.Figure()
fig_mc.add_trace(go.Scatter(x=days_axis, y=p95, name="95th Percentile (Best Case)", line=dict(color=COLOR_POSITIVE, width=2)))
fig_mc.add_trace(go.Scatter(x=days_axis, y=p50, name="Median", line=dict(color=COLOR_BENCHMARK, width=2.5)))
fig_mc.add_trace(go.Scatter(x=days_axis, y=p5, name="5th Percentile (Worst Case)", line=dict(color=COLOR_NEGATIVE, width=2)))
fig_mc.add_hline(y=mc_start_value, line_dash="dash", line_color="#9CA3AF", annotation_text="Starting Value")
fig_mc.update_layout(xaxis_title="Trading Days Ahead", yaxis_title="Portfolio Value (₹)")
st.plotly_chart(style_fig(fig_mc), use_container_width=True)

final_values = sim_paths[-1, :]
median_final = np.median(final_values)
best_case = np.percentile(final_values, 95)
worst_case = np.percentile(final_values, 5)
prob_loss = (final_values < mc_start_value).mean() * 100

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Median Final Value", f"₹{median_final:,.0f}")
mc2.metric("Best Case (95th pct)", f"₹{best_case:,.0f}")
mc3.metric("Worst Case (5th pct)", f"₹{worst_case:,.0f}")
mc4.metric("Probability of Loss", f"{prob_loss:.1f}%")

# =============================================================================
# SECTION 9 — EFFICIENT FRONTIER
# =============================================================================
st.markdown('<div class="section-header">9️⃣ Efficient Frontier Analysis</div>', unsafe_allow_html=True)

n_assets = len(active_tickers)
mean_daily = asset_returns[active_tickers].mean().values
cov_daily = asset_returns[active_tickers].cov().values
ann_mean = mean_daily * TRADING_DAYS
ann_cov = cov_daily * TRADING_DAYS


def port_perf(w, ann_mean, ann_cov, rf):
    ret = np.dot(w, ann_mean)
    vol = np.sqrt(np.dot(w.T, np.dot(ann_cov, w)))
    sr = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sr


if n_assets >= 2:
    n_portfolios = 4000
    rng2 = np.random.default_rng(seed=7)
    raw = rng2.random((n_portfolios, n_assets))
    random_weights = raw / raw.sum(axis=1, keepdims=True)

    rets = random_weights @ ann_mean
    vols = np.sqrt(np.einsum("ij,jk,ik->i", random_weights, ann_cov, random_weights))
    sharpes = (rets - risk_free_rate) / vols

    def neg_sharpe(w):
        r, v, s = port_perf(w, ann_mean, ann_cov, risk_free_rate)
        return -s if not np.isnan(s) else 1e6

    def port_vol_fn(w):
        return np.sqrt(np.dot(w.T, np.dot(ann_cov, w)))

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init_guess = np.repeat(1 / n_assets, n_assets)

    res_max_sharpe = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    res_min_var = minimize(port_vol_fn, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)

    w_max_sharpe = res_max_sharpe.x if res_max_sharpe.success else init_guess
    w_min_var = res_min_var.x if res_min_var.success else init_guess

    ret_ms, vol_ms, sr_ms = port_perf(w_max_sharpe, ann_mean, ann_cov, risk_free_rate)
    ret_mv, vol_mv, sr_mv = port_perf(w_min_var, ann_mean, ann_cov, risk_free_rate)
    ret_cur, vol_cur, sr_cur = port_perf(weights, ann_mean, ann_cov, risk_free_rate)

    fig_ef = go.Figure()
    fig_ef.add_trace(
        go.Scatter(
            x=vols * 100, y=rets * 100, mode="markers",
            marker=dict(size=5, color=sharpes, colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
            name="Random Portfolios", hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        )
    )
    fig_ef.add_trace(
        go.Scatter(
            x=[vol_cur * 100], y=[ret_cur * 100], mode="markers", name="Current Portfolio",
            marker=dict(size=16, color=COLOR_PORTFOLIO, symbol="star", line=dict(color="white", width=1)),
        )
    )
    fig_ef.add_trace(
        go.Scatter(
            x=[vol_mv * 100], y=[ret_mv * 100], mode="markers", name="Min Variance Portfolio",
            marker=dict(size=16, color=COLOR_BENCHMARK, symbol="diamond", line=dict(color="white", width=1)),
        )
    )
    fig_ef.add_trace(
        go.Scatter(
            x=[vol_ms * 100], y=[ret_ms * 100], mode="markers", name="Max Sharpe Portfolio",
            marker=dict(size=16, color=COLOR_POSITIVE, symbol="triangle-up", line=dict(color="white", width=1)),
        )
    )
    fig_ef.update_layout(xaxis_title="Annualised Volatility (%)", yaxis_title="Annualised Return (%)")
    st.plotly_chart(style_fig(fig_ef, height=520), use_container_width=True)

    st.markdown("**Allocation Breakdown: Current vs Min Variance vs Max Sharpe**")
    alloc_table = pd.DataFrame(
        {
            "Current Portfolio (%)": weights * 100,
            "Min Variance Portfolio (%)": w_min_var * 100,
            "Max Sharpe Portfolio (%)": w_max_sharpe * 100,
        },
        index=active_tickers,
    ).round(2)
    summary_row = pd.DataFrame(
        {
            "Current Portfolio (%)": [ret_cur * 100, vol_cur * 100, sr_cur],
            "Min Variance Portfolio (%)": [ret_mv * 100, vol_mv * 100, sr_mv],
            "Max Sharpe Portfolio (%)": [ret_ms * 100, vol_ms * 100, sr_ms],
        },
        index=["Expected Return (%)", "Volatility (%)", "Sharpe Ratio"],
    ).round(2)
    st.dataframe(alloc_table, use_container_width=True)
    st.dataframe(summary_row, use_container_width=True)
else:
    st.info("Efficient frontier analysis requires at least 2 assets in the portfolio.")

# =============================================================================
# SECTION 10 — FULL SUMMARY TABLE
# =============================================================================
st.markdown('<div class="section-header">🔟 Full Summary Table</div>', unsafe_allow_html=True)

port_sortino = sortino_ratio(port_returns, risk_free_rate)
if not bench_returns.empty:
    bench_sortino = sortino_ratio(bench_returns, risk_free_rate)
    port_beta = beta_vs_benchmark(port_returns, bench_returns)
    bench_beta = 1.0
else:
    bench_sortino = np.nan
    port_beta = np.nan
    bench_beta = np.nan

summary_df = pd.DataFrame(
    {
        "Portfolio": [
            port_cagr * 100 if pd.notna(port_cagr) else np.nan,
            port_vol * 100 if pd.notna(port_vol) else np.nan,
            port_sharpe,
            port_sortino,
            port_beta,
            port_mdd * 100 if pd.notna(port_mdd) else np.nan,
        ],
        benchmark_label: [
            bench_cagr * 100 if pd.notna(bench_cagr) else np.nan,
            bench_vol * 100 if pd.notna(bench_vol) else np.nan,
            bench_sharpe,
            bench_sortino,
            bench_beta,
            bench_mdd * 100 if pd.notna(bench_mdd) else np.nan,
        ],
    },
    index=["CAGR (%)", "Volatility (%)", "Sharpe Ratio", "Sortino Ratio", "Beta", "Max Drawdown (%)"],
).round(2)

st.dataframe(summary_df, use_container_width=True)

st.markdown("---")
st.markdown(
    '<p class="subtle-caption">Data sourced from Yahoo Finance via yfinance. '
    "This dashboard is for educational and informational purposes only and does not constitute investment advice.</p>",
    unsafe_allow_html=True,
)
