"""
Fundamental Valuation & Peer Multiples Dashboard
=================================================
A production-ready Streamlit application for DCF valuation and peer
multiples analysis, powered by live data from yfinance.

Run with:  streamlit run app.py
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Fundamental Valuation & Peer Multiples Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — DARK, EXECUTIVE-READY THEME
# ============================================================================
CUSTOM_CSS = """
<style>
    /* ---------- Global ---------- */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: #FFFFFF;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Force white text across all standard Streamlit text elements */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown a,
    .stText, .stCaption, .stDataFrame, .stTable,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricLabel"],
    div[data-testid="stExpander"] summary,
    .stSelectbox label, .stSlider label, .stTextInput label, .stNumberInput label,
    .stRadio label, .stCheckbox label {
        color: #FFFFFF !important;
    }
    .stCaption, small, [data-testid="stCaptionContainer"] {
        color: #D6D9E0 !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #262B3D;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input {
        background-color: #1B2030;
        color: #FFFFFF !important;
        border: 1px solid #2E3448;
        border-radius: 8px;
    }

    /* ---------- Headings ---------- */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .app-title {
        font-size: 2.05rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F8BFF 0%, #34D1BF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .app-subtitle {
        color: #D6D9E0;
        font-size: 0.95rem;
        margin-top: -6px;
        margin-bottom: 18px;
    }

    /* ---------- Card container ---------- */
    .metric-card {
        background: linear-gradient(160deg, #161B29 0%, #12161F 100%);
        border: 1px solid #262B3D;
        border-radius: 14px;
        padding: 18px 20px 14px 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
        height: 100%;
    }
    .metric-label {
        color: #D6D9E0 !important;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF !important;
    }
    .metric-delta-positive {
        color: #3ED598 !important;
        font-size: 0.92rem;
        font-weight: 700;
    }
    .metric-delta-negative {
        color: #FF5C7A !important;
        font-size: 0.92rem;
        font-weight: 700;
    }
    .metric-delta-neutral {
        color: #FFFFFF !important;
        font-size: 0.92rem;
        font-weight: 700;
    }

    /* ---------- Section banners ---------- */
    .section-banner {
        background: linear-gradient(90deg, #161B29 0%, #12161F 100%);
        border-left: 4px solid #4F8BFF;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 18px 0 14px 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .warn-banner {
        background: #2A1E12;
        border-left: 4px solid #F2A93B;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 10px 0;
        color: #F2C879;
        font-size: 0.88rem;
    }
    .error-banner {
        background: #2A1414;
        border-left: 4px solid #FF5C7A;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 10px 0;
        color: #FFA3B4;
        font-size: 0.88rem;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #12161F;
        border-radius: 10px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #8A93A6;
        font-weight: 600;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4F8BFF 0%, #34D1BF 100%) !important;
        color: #0E1117 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #4F8BFF 0%, #34D1BF 100%);
        color: #0E1117;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        opacity: 0.88;
        color: #0E1117;
    }

    /* ---------- Dataframe ---------- */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: #262B3D;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_MAIN_TICKER = "BAJFINANCE.NS"
DEFAULT_PEERS = "SHRIRAMFIN.NS, CHOLAFIN.NS, MUTHOOTFIN.NS, TATACAPITAL.NS"

INCOME_REVENUE_KEYS = ["Total Revenue", "TotalRevenue", "Revenue"]
INCOME_EBIT_KEYS = ["EBIT", "Operating Income", "OperatingIncome"]
INCOME_PRETAX_KEYS = ["Pretax Income", "PretaxIncome", "Income Before Tax"]
INCOME_TAX_KEYS = ["Tax Provision", "TaxProvision", "Income Tax Expense"]
INCOME_NET_INCOME_KEYS = ["Net Income", "NetIncome", "Net Income Common Stockholders"]

BS_CASH_KEYS = [
    "Cash And Cash Equivalents",
    "CashAndCashEquivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash",
]
BS_ST_DEBT_KEYS = ["Current Debt", "CurrentDebt", "Short Long Term Debt", "Short Term Debt"]
BS_LT_DEBT_KEYS = ["Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation"]
BS_TOTAL_DEBT_KEYS = ["Total Debt", "TotalDebt"]
BS_SHARES_KEYS = ["Ordinary Shares Number", "OrdinarySharesNumber", "Share Issued"]
BS_EQUITY_KEYS = [
    "Stockholders Equity",
    "StockholdersEquity",
    "Total Stockholder Equity",
    "Common Stock Equity",
]
BS_TOTAL_LIAB_KEYS = ["Total Liabilities Net Minority Interest", "TotalLiab", "Total Liabilities"]

CF_DA_KEYS = [
    "Depreciation And Amortization",
    "DepreciationAndAmortization",
    "Depreciation Amortization Depletion",
    "Depreciation",
]
CF_CAPEX_KEYS = ["Capital Expenditure", "CapitalExpenditure", "Purchase Of PPE"]
CF_NWC_KEYS = ["Change In Working Capital", "ChangeInWorkingCapital"]

# ============================================================================
# HELPER — FORMATTING
# ============================================================================
def currency_symbol_for(ticker: str, info: dict) -> str:
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "₹"
    curr = (info or {}).get("currency", "USD")
    mapping = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}
    return mapping.get(curr, (curr + " ") if curr else "$")


def fmt_money(value, symbol="$", decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{symbol}{value:,.{decimals}f}"
    except Exception:
        return "N/A"


def fmt_large(value, symbol="$"):
    """Format large numbers. Indian Rupee values use the Lakh/Crore convention;
    all other currencies use the standard K/M/B/T convention."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    abs_v = abs(value)

    if symbol == "₹":
        # Indian numbering system: 1 Crore = 1e7, 1 Lakh = 1e5
        if abs_v >= 1e7:
            return f"₹{value / 1e7:,.2f} Cr"
        if abs_v >= 1e5:
            return f"₹{value / 1e5:,.2f} L"
        if abs_v >= 1e3:
            return f"₹{value / 1e3:,.2f} K"
        return f"₹{value:,.2f}"

    if abs_v >= 1e12:
        return f"{symbol}{value / 1e12:,.2f}T"
    if abs_v >= 1e9:
        return f"{symbol}{value / 1e9:,.2f}B"
    if abs_v >= 1e6:
        return f"{symbol}{value / 1e6:,.2f}M"
    if abs_v >= 1e3:
        return f"{symbol}{value / 1e3:,.2f}K"
    return f"{symbol}{value:,.2f}"


def fmt_pct(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{value * 100:,.{decimals}f}%" if abs(value) < 5 else f"{value:,.{decimals}f}%"
    except Exception:
        return "N/A"


def fmt_ratio(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{value:,.{decimals}f}x"
    except Exception:
        return "N/A"


def safe_num(value, default=np.nan):
    try:
        if value is None:
            return default
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


# ============================================================================
# HELPER — ROBUST STATEMENT EXTRACTION
# ============================================================================
def sorted_columns(df):
    """Return columns of a yfinance financial statement sorted most-recent first."""
    cols = list(df.columns)
    try:
        cols = sorted(cols, key=lambda c: pd.to_datetime(c), reverse=True)
    except Exception:
        pass
    return cols


def get_row(df, key_candidates):
    """Return a Series (across periods, most-recent first) for the first matching row label."""
    if df is None or df.empty:
        return None
    cols = sorted_columns(df)
    for key in key_candidates:
        if key in df.index:
            row = df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.reindex(cols)
    return None


def latest_value(row):
    """First non-null value from a period-indexed Series (most-recent first)."""
    if row is None:
        return np.nan
    for v in row.values:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return safe_num(v)
    return np.nan


def series_values_recent_first(row, n):
    """Return up to n most-recent non-null values as a list, most-recent first."""
    if row is None:
        return [np.nan] * n
    vals = [safe_num(v) for v in row.values]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < n:
        vals = vals + [np.nan] * (n - len(vals))
    return vals[:n]


# ============================================================================
# CACHED DATA FETCHERS
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or "symbol" not in info and "shortName" not in info:
            return {}
        return info
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statements(ticker):
    try:
        t = yf.Ticker(ticker)
        income = t.financials if t.financials is not None else pd.DataFrame()
        balance = t.balance_sheet if t.balance_sheet is not None else pd.DataFrame()
        cashflow = t.cashflow if t.cashflow is not None else pd.DataFrame()
        return income, balance, cashflow
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker, period="5y"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        return hist
    except Exception:
        return pd.DataFrame()


def current_price(ticker, info):
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price:
        return safe_num(price)
    hist = fetch_history(ticker, "5d")
    if hist is not None and not hist.empty:
        return safe_num(hist["Close"].iloc[-1])
    return np.nan


# ============================================================================
# SIDEBAR — INPUTS
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    main_ticker = st.text_input(
        "Main Ticker",
        value=DEFAULT_MAIN_TICKER,
        help="NSE tickers use the .NS suffix (e.g. RELIANCE.NS, TCS.NS). BSE tickers use .BO (e.g. RELIANCE.BO).",
    ).strip().upper()

    peer_input = st.text_input(
        "Peer Tickers (comma-separated)",
        value=DEFAULT_PEERS,
        help="e.g. SHRIRAMFIN.NS, CHOLAFIN.NS, MUTHOOTFIN.NS, TATACAPITAL.NS",
    )
    peer_tickers = [p.strip().upper() for p in peer_input.split(",") if p.strip()]
    # Ensure main ticker is not duplicated inside the peer list for peer-only stats
    peer_only_tickers = [p for p in peer_tickers if p != main_ticker]

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Data source: Yahoo Finance via `yfinance`")
    st.caption("💡 NSE tickers: append **.NS** · BSE tickers: append **.BO**")

if not main_ticker:
    st.warning("Please enter a main ticker in the sidebar to begin.")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="app-title">📊 Fundamental Valuation & Peer Multiples Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">DCF intrinsic valuation, sensitivity analysis, and peer-relative multiples for NSE/BSE-listed (and global) equities — powered by live market data.</div>',
    unsafe_allow_html=True,
)

# ============================================================================
# FETCH MAIN TICKER DATA
# ============================================================================
main_info = fetch_info(main_ticker)
if not main_info:
    st.markdown(
        f'<div class="error-banner">⚠️ Could not retrieve data for <b>{main_ticker}</b>. '
        f'Please check the ticker symbol and try again.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

income_stmt, balance_sheet, cashflow_stmt = fetch_statements(main_ticker)
main_currency = currency_symbol_for(main_ticker, main_info)
main_price = current_price(main_ticker, main_info)
company_name = main_info.get("shortName") or main_info.get("longName") or main_ticker

if income_stmt.empty:
    st.markdown(
        '<div class="warn-banner">⚠️ Income statement data is unavailable for this ticker. '
        'DCF projections may be limited.</div>',
        unsafe_allow_html=True,
    )

st.markdown(f"### {company_name}  ·  `{main_ticker}`")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(
    ["📈  DCF Valuation Model", "📊  Peer Group & Multiples Analysis", "📄  Raw Financial Statements"]
)

# ============================================================================
# EXTRACT HISTORICAL FUNDAMENTALS FOR DCF (shared across tab)
# ============================================================================
revenue_row = get_row(income_stmt, INCOME_REVENUE_KEYS)
ebit_row = get_row(income_stmt, INCOME_EBIT_KEYS)
pretax_row = get_row(income_stmt, INCOME_PRETAX_KEYS)
tax_row = get_row(income_stmt, INCOME_TAX_KEYS)

cash_row = get_row(balance_sheet, BS_CASH_KEYS)
st_debt_row = get_row(balance_sheet, BS_ST_DEBT_KEYS)
lt_debt_row = get_row(balance_sheet, BS_LT_DEBT_KEYS)
total_debt_row = get_row(balance_sheet, BS_TOTAL_DEBT_KEYS)
shares_row = get_row(balance_sheet, BS_SHARES_KEYS)

da_row = get_row(cashflow_stmt, CF_DA_KEYS)
capex_row = get_row(cashflow_stmt, CF_CAPEX_KEYS)
nwc_row = get_row(cashflow_stmt, CF_NWC_KEYS)

# Revenue history (up to 4 periods) for CAGR
rev_hist = series_values_recent_first(revenue_row, 4)
rev_hist_clean = [v for v in rev_hist if not np.isnan(v)]
if len(rev_hist_clean) >= 2:
    n_periods = len(rev_hist_clean) - 1
    start_rev, end_rev = rev_hist_clean[-1], rev_hist_clean[0]
    if start_rev > 0 and n_periods > 0:
        hist_cagr = (end_rev / start_rev) ** (1 / n_periods) - 1
    else:
        hist_cagr = 0.06
else:
    hist_cagr = 0.06
hist_cagr = float(np.clip(hist_cagr, -0.20, 0.60))

latest_revenue = safe_num(latest_value(revenue_row))
latest_ebit = safe_num(latest_value(ebit_row))
if np.isnan(latest_ebit):
    # fall back: operating margin from info
    op_margin_info = safe_num(main_info.get("operatingMargins"))
    if not np.isnan(op_margin_info) and not np.isnan(latest_revenue):
        latest_ebit = latest_revenue * op_margin_info

hist_ebit_margin = (latest_ebit / latest_revenue) if (not np.isnan(latest_ebit) and not np.isnan(latest_revenue) and latest_revenue != 0) else 0.20
hist_ebit_margin = float(np.clip(hist_ebit_margin, 0.01, 0.60))

latest_pretax = safe_num(latest_value(pretax_row))
latest_tax = safe_num(latest_value(tax_row))
if not np.isnan(latest_pretax) and not np.isnan(latest_tax) and latest_pretax != 0:
    hist_tax_rate = float(np.clip(latest_tax / latest_pretax, 0.0, 0.45))
else:
    hist_tax_rate = 0.21

latest_cash = safe_num(latest_value(cash_row), 0.0)
latest_total_debt = safe_num(latest_value(total_debt_row))
if np.isnan(latest_total_debt):
    st_d = safe_num(latest_value(st_debt_row), 0.0)
    lt_d = safe_num(latest_value(lt_debt_row), 0.0)
    latest_total_debt = st_d + lt_d
latest_total_debt = safe_num(latest_total_debt, 0.0)

latest_shares = safe_num(latest_value(shares_row))
if np.isnan(latest_shares):
    latest_shares = safe_num(main_info.get("sharesOutstanding"))

# D&A / CapEx / NWC as % of revenue (average across available periods)
def ratio_to_revenue(row, rev_series_vals):
    vals = series_values_recent_first(row, len(rev_series_vals))
    ratios = []
    for v, r in zip(vals, rev_series_vals):
        if not np.isnan(v) and not np.isnan(r) and r != 0:
            ratios.append(v / r)
    return float(np.mean(ratios)) if ratios else np.nan

rev_series_for_ratio = series_values_recent_first(revenue_row, 4)
da_pct_rev = ratio_to_revenue(da_row, rev_series_for_ratio)
capex_pct_rev = ratio_to_revenue(capex_row, rev_series_for_ratio)
nwc_pct_rev = ratio_to_revenue(nwc_row, rev_series_for_ratio)

if np.isnan(da_pct_rev):
    da_pct_rev = 0.03
if np.isnan(capex_pct_rev):
    capex_pct_rev = -0.04  # yfinance reports capex as negative outflow typically
if np.isnan(nwc_pct_rev):
    nwc_pct_rev = -0.005

# ============================================================================
# TAB 1 — DCF VALUATION MODEL
# ============================================================================
with tab1:
    if np.isnan(latest_revenue) or np.isnan(latest_shares) or latest_shares == 0:
        st.markdown(
            '<div class="error-banner">⚠️ Insufficient fundamental data (revenue or shares outstanding) '
            'to run a DCF model for this ticker.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.expander("🎛️  DCF Assumptions & Configuration", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                horizon = st.slider("Forecast Horizon (Years)", 5, 10, 5, 1)
                growth_rate = st.slider(
                    "Revenue Growth Rate (%)",
                    -10.0, 40.0, round(hist_cagr * 100, 1), 0.5,
                    help="Defaults to historical revenue CAGR.",
                ) / 100.0
            with c2:
                ebit_margin = st.slider(
                    "Operating (EBIT) Margin (%)",
                    1.0, 60.0, round(hist_ebit_margin * 100, 1), 0.5,
                ) / 100.0
                tax_rate = st.slider("Tax Rate (%)", 0.0, 45.0, round(hist_tax_rate * 100, 1), 0.5) / 100.0
            with c3:
                wacc = st.slider("WACC (%)", 4.0, 15.0, 8.5, 0.1) / 100.0
                terminal_growth = st.slider("Terminal Growth Rate (%)", 1.0, 5.0, 2.5, 0.1) / 100.0

            if terminal_growth >= wacc:
                st.markdown(
                    '<div class="warn-banner">⚠️ Terminal growth rate must be below WACC. '
                    'Adjust the sliders to avoid a non-economic valuation.</div>',
                    unsafe_allow_html=True,
                )

        # ---------------- DCF Engine ----------------
        def run_dcf(rev0, growth, ebit_mgn, tax, wacc_, term_g, years,
                    da_pct, capex_pct, nwc_pct, cash, debt, shares):
            projections = []
            revenue = rev0
            for yr in range(1, years + 1):
                revenue = revenue * (1 + growth)
                ebit = revenue * ebit_mgn
                nopat = ebit * (1 - tax)
                da = revenue * da_pct
                capex = revenue * capex_pct  # already negative-oriented ratio
                d_nwc = revenue * nwc_pct
                fcff = nopat + da + capex + d_nwc
                disc_factor = 1 / ((1 + wacc_) ** yr)
                pv_fcff = fcff * disc_factor
                projections.append(
                    {
                        "Year": f"Y{yr}",
                        "Revenue": revenue,
                        "EBIT": ebit,
                        "NOPAT": nopat,
                        "FCFF": fcff,
                        "Discount Factor": disc_factor,
                        "PV of FCFF": pv_fcff,
                    }
                )
            df_proj = pd.DataFrame(projections)
            final_fcff = df_proj["FCFF"].iloc[-1]
            if wacc_ > term_g:
                terminal_value = final_fcff * (1 + term_g) / (wacc_ - term_g)
            else:
                terminal_value = np.nan
            pv_terminal_value = terminal_value * df_proj["Discount Factor"].iloc[-1] if not np.isnan(terminal_value) else np.nan

            enterprise_value = df_proj["PV of FCFF"].sum() + (pv_terminal_value if not np.isnan(pv_terminal_value) else 0)
            equity_value = enterprise_value + cash - debt
            intrinsic_value_per_share = equity_value / shares if shares else np.nan

            return df_proj, terminal_value, pv_terminal_value, enterprise_value, equity_value, intrinsic_value_per_share

        (df_proj, terminal_value, pv_terminal_value, enterprise_value,
         equity_value, intrinsic_value) = run_dcf(
            latest_revenue, growth_rate, ebit_margin, tax_rate, wacc, terminal_growth,
            horizon, da_pct_rev, capex_pct_rev, nwc_pct_rev, latest_cash, latest_total_debt, latest_shares
        )

        # ---------------- KPI Header Cards ----------------
        st.markdown('<div class="section-banner">Valuation Summary</div>', unsafe_allow_html=True)

        upside = np.nan
        margin_of_safety = np.nan
        if not np.isnan(intrinsic_value) and not np.isnan(main_price) and main_price != 0:
            upside = (intrinsic_value - main_price) / main_price * 100
        if not np.isnan(intrinsic_value) and intrinsic_value != 0 and not np.isnan(main_price):
            margin_of_safety = (intrinsic_value - main_price) / intrinsic_value * 100

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">Current Market Price</div>
                        <div class="metric-value">{fmt_money(main_price, main_currency)}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">Intrinsic Value / Share</div>
                        <div class="metric-value">{fmt_money(intrinsic_value, main_currency)}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with k3:
            delta_class = "metric-delta-neutral"
            if not np.isnan(margin_of_safety):
                delta_class = "metric-delta-positive" if margin_of_safety > 0 else "metric-delta-negative"
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">Margin of Safety</div>
                        <div class="metric-value {delta_class}">{fmt_pct(margin_of_safety/100) if not np.isnan(margin_of_safety) else "N/A"}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with k4:
            delta_class = "metric-delta-neutral"
            if not np.isnan(upside):
                delta_class = "metric-delta-positive" if upside > 0 else "metric-delta-negative"
            label = "Upside" if (not np.isnan(upside) and upside >= 0) else "Downside"
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">{label} Potential</div>
                        <div class="metric-value {delta_class}">{fmt_pct(upside/100) if not np.isnan(upside) else "N/A"}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">Enterprise Value</div>
                        <div class="metric-value">{fmt_large(enterprise_value, main_currency)}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with e2:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">Equity Value</div>
                        <div class="metric-value">{fmt_large(equity_value, main_currency)}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with e3:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">PV of Terminal Value</div>
                        <div class="metric-value">{fmt_large(pv_terminal_value, main_currency)}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        # ---------------- Projected Cash Flow Chart ----------------
        st.markdown('<div class="section-banner">Projected Free Cash Flow to Firm (FCFF)</div>', unsafe_allow_html=True)

        fig_fcff = go.Figure()
        fig_fcff.add_trace(
            go.Bar(
                x=df_proj["Year"],
                y=df_proj["FCFF"],
                name="Nominal FCFF",
                marker_color="#4F8BFF",
            )
        )
        fig_fcff.add_trace(
            go.Bar(
                x=df_proj["Year"],
                y=df_proj["PV of FCFF"],
                name="Present Value of FCFF",
                marker_color="#34D1BF",
            )
        )
        fig_fcff.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#E6E6E6"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            yaxis_title=f"Value ({main_currency})",
        )
        st.plotly_chart(fig_fcff, use_container_width=True)

        with st.expander("📋 View Detailed Projection Table"):
            display_df = df_proj.copy()
            for col in ["Revenue", "EBIT", "NOPAT", "FCFF", "PV of FCFF"]:
                display_df[col] = display_df[col].apply(lambda v: fmt_large(v, main_currency))
            display_df["Discount Factor"] = df_proj["Discount Factor"].apply(lambda v: f"{v:.3f}")
            st.dataframe(display_df.set_index("Year"), use_container_width=True)

        # ---------------- Sensitivity Analysis Matrix ----------------
        st.markdown('<div class="section-banner">Sensitivity Analysis — Intrinsic Value per Share</div>', unsafe_allow_html=True)
        st.caption("WACC (columns) vs. Terminal Growth Rate (rows)")

        wacc_range = np.round(np.linspace(max(wacc - 0.02, 0.03), wacc + 0.02, 5), 4)
        tg_range = np.round(np.linspace(max(terminal_growth - 0.01, 0.005), terminal_growth + 0.01, 5), 4)

        sensitivity_matrix = np.zeros((len(tg_range), len(wacc_range)))
        for i, tg in enumerate(tg_range):
            for j, w in enumerate(wacc_range):
                if w <= tg:
                    sensitivity_matrix[i, j] = np.nan
                    continue
                _, _, _, _, _, iv = run_dcf(
                    latest_revenue, growth_rate, ebit_margin, tax_rate, w, tg,
                    horizon, da_pct_rev, capex_pct_rev, nwc_pct_rev,
                    latest_cash, latest_total_debt, latest_shares,
                )
                sensitivity_matrix[i, j] = iv

        # Color scale: Red = undervalued relative to price(i.e. below price -> overvalued stock? )
        # Per spec: Red (Undervalued) to Green (Overvalued) -- color mapped directly to intrinsic value magnitude
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=sensitivity_matrix,
                x=[f"{w*100:.1f}%" for w in wacc_range],
                y=[f"{t*100:.1f}%" for t in tg_range],
                colorscale=[[0, "#FF5C7A"], [0.5, "#F2A93B"], [1, "#3ED598"]],
                text=np.round(sensitivity_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 12, "color": "#0E1117"},
                colorbar=dict(title=f"{main_currency}/share"),
                hovertemplate="WACC: %{x}<br>Terminal Growth: %{y}<br>Intrinsic Value: %{z:.2f}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#E6E6E6"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            xaxis_title="WACC",
            yaxis_title="Terminal Growth Rate",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption(
            f"Base case assumptions → Growth: {growth_rate*100:.1f}% · EBIT Margin: {ebit_margin*100:.1f}% · "
            f"Tax Rate: {tax_rate*100:.1f}% · D&A/Rev: {da_pct_rev*100:.1f}% · CapEx/Rev: {capex_pct_rev*100:.1f}% · "
            f"ΔNWC/Rev: {nwc_pct_rev*100:.1f}%"
        )

# ============================================================================
# TAB 2 — PEER GROUP & MULTIPLES ANALYSIS
# ============================================================================
with tab2:
    st.markdown('<div class="section-banner">Peer Comparison Table</div>', unsafe_allow_html=True)

    all_tickers = [main_ticker] + [p for p in peer_only_tickers]
    peer_rows = []
    fetch_errors = []

    for tk in all_tickers:
        info = fetch_info(tk) if tk != main_ticker else main_info
        if not info:
            fetch_errors.append(tk)
            continue
        row = {
            "Ticker": tk,
            "Company": info.get("shortName") or info.get("longName") or tk,
            "Market Cap": safe_num(info.get("marketCap")),
            "P/E (TTM)": safe_num(info.get("trailingPE")),
            "Forward P/E": safe_num(info.get("forwardPE")),
            "EV/EBITDA": safe_num(info.get("enterpriseToEbitda")),
            "P/B": safe_num(info.get("priceToBook")),
            "ROE (%)": safe_num(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") is not None else np.nan,
            "D/E": safe_num(info.get("debtToEquity")),
            "Op. Margin (%)": safe_num(info.get("operatingMargins")) * 100 if info.get("operatingMargins") is not None else np.nan,
        }
        peer_rows.append(row)

    if fetch_errors:
        st.markdown(
            f'<div class="warn-banner">⚠️ Could not retrieve data for: {", ".join(fetch_errors)}. '
            f'These tickers were excluded from the analysis.</div>',
            unsafe_allow_html=True,
        )

    if not peer_rows:
        st.markdown('<div class="error-banner">⚠️ No peer data could be retrieved.</div>', unsafe_allow_html=True)
    else:
        peer_df = pd.DataFrame(peer_rows).set_index("Ticker")
        numeric_cols = ["Market Cap", "P/E (TTM)", "Forward P/E", "EV/EBITDA", "P/B", "ROE (%)", "D/E", "Op. Margin (%)"]

        display_peer_df = peer_df.copy()
        display_peer_df["Market Cap"] = display_peer_df["Market Cap"].apply(lambda v: fmt_large(v, main_currency))
        for col in ["P/E (TTM)", "Forward P/E", "EV/EBITDA", "P/B", "D/E"]:
            display_peer_df[col] = display_peer_df[col].apply(lambda v: fmt_ratio(v))
        for col in ["ROE (%)", "Op. Margin (%)"]:
            display_peer_df[col] = display_peer_df[col].apply(lambda v: f"{v:,.2f}%" if not np.isnan(v) else "N/A")

        def highlight_main(row):
            if row.name == main_ticker:
                return ["background-color: #1B2C4A; color: #9FCBFF; font-weight: 700;"] * len(row)
            return [""] * len(row)

        styled = display_peer_df.style.apply(highlight_main, axis=1)
        st.dataframe(styled, use_container_width=True, height=min(48 * (len(display_peer_df) + 1), 420))

        # Peer average / median row (excluding main ticker where possible)
        peer_only_df = peer_df.drop(index=main_ticker, errors="ignore")
        if not peer_only_df.empty:
            stat_cols = st.columns(2)
            with stat_cols[0]:
                st.markdown("**Peer Group Averages** *(excluding main ticker)*")
                avg_row = peer_only_df[numeric_cols].mean(numeric_only=True)
                avg_display = {
                    "Market Cap": fmt_large(avg_row.get("Market Cap"), main_currency),
                    "P/E (TTM)": fmt_ratio(avg_row.get("P/E (TTM)")),
                    "Forward P/E": fmt_ratio(avg_row.get("Forward P/E")),
                    "EV/EBITDA": fmt_ratio(avg_row.get("EV/EBITDA")),
                    "P/B": fmt_ratio(avg_row.get("P/B")),
                    "ROE (%)": f"{avg_row.get('ROE (%)'):.2f}%" if not np.isnan(avg_row.get("ROE (%)", np.nan)) else "N/A",
                    "D/E": fmt_ratio(avg_row.get("D/E")),
                    "Op. Margin (%)": f"{avg_row.get('Op. Margin (%)'):.2f}%" if not np.isnan(avg_row.get("Op. Margin (%)", np.nan)) else "N/A",
                }
                st.table(pd.DataFrame(avg_display, index=["Peer Avg"]).T)
            with stat_cols[1]:
                st.markdown("**Peer Group Medians** *(excluding main ticker)*")
                med_row = peer_only_df[numeric_cols].median(numeric_only=True)
                med_display = {
                    "Market Cap": fmt_large(med_row.get("Market Cap"), main_currency),
                    "P/E (TTM)": fmt_ratio(med_row.get("P/E (TTM)")),
                    "Forward P/E": fmt_ratio(med_row.get("Forward P/E")),
                    "EV/EBITDA": fmt_ratio(med_row.get("EV/EBITDA")),
                    "P/B": fmt_ratio(med_row.get("P/B")),
                    "ROE (%)": f"{med_row.get('ROE (%)'):.2f}%" if not np.isnan(med_row.get("ROE (%)", np.nan)) else "N/A",
                    "D/E": fmt_ratio(med_row.get("D/E")),
                    "Op. Margin (%)": f"{med_row.get('Op. Margin (%)'):.2f}%" if not np.isnan(med_row.get("Op. Margin (%)", np.nan)) else "N/A",
                }
                st.table(pd.DataFrame(med_display, index=["Peer Median"]).T)
        else:
            med_row = pd.Series(dtype=float)

        # ---------------- Scatter Plot ----------------
        st.markdown('<div class="section-banner">Valuation Multiples — EV/EBITDA vs. ROE</div>', unsafe_allow_html=True)

        scatter_df = peer_df.reset_index().dropna(subset=["EV/EBITDA", "ROE (%)"])
        if scatter_df.empty:
            st.info("Not enough data available to render the scatter plot for this ticker/peer set.")
        else:
            colors = ["#4F8BFF" if tk != main_ticker else "#FF5C7A" for tk in scatter_df["Ticker"]]
            sizes = scatter_df["Market Cap"].fillna(scatter_df["Market Cap"].median() if not scatter_df["Market Cap"].dropna().empty else 1e9)
            sizes = np.clip(sizes / sizes.max() * 60, 18, 60) if sizes.max() > 0 else [30] * len(scatter_df)

            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scatter(
                    x=scatter_df["EV/EBITDA"],
                    y=scatter_df["ROE (%)"],
                    mode="markers+text",
                    text=scatter_df["Ticker"],
                    textposition="top center",
                    textfont=dict(color="#E6E6E6", size=12),
                    marker=dict(size=sizes, color=colors, line=dict(width=1.5, color="#0E1117"), opacity=0.9),
                    hovertemplate="<b>%{text}</b><br>EV/EBITDA: %{x:.2f}x<br>ROE: %{y:.2f}%<extra></extra>",
                )
            )
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(color="#E6E6E6"),
                xaxis_title="EV / EBITDA (x)",
                yaxis_title="Return on Equity (%)",
                margin=dict(l=10, r=10, t=30, b=10),
                height=460,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("🔴 Main ticker · 🔵 Peers · Marker size scaled by market capitalization.")

        # ---------------- Under/Overvaluation Bar Chart vs Peer Median ----------------
        st.markdown('<div class="section-banner">Relative Positioning vs. Peer Median</div>', unsafe_allow_html=True)

        if main_ticker in peer_df.index and not peer_only_df.empty:
            main_row = peer_df.loc[main_ticker]
            deviation = {}
            for col in numeric_cols:
                med_val = med_row.get(col, np.nan)
                main_val = main_row.get(col, np.nan)
                if not np.isnan(med_val) and med_val != 0 and not np.isnan(main_val):
                    deviation[col] = (main_val - med_val) / abs(med_val) * 100
                else:
                    deviation[col] = np.nan

            dev_series = pd.Series(deviation).dropna().sort_values()
            if dev_series.empty:
                st.info("Not enough overlapping data to compute relative positioning.")
            else:
                bar_colors = ["#3ED598" if v >= 0 else "#FF5C7A" for v in dev_series.values]
                fig_bar = go.Figure(
                    go.Bar(
                        x=dev_series.values,
                        y=dev_series.index,
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.1f}%" for v in dev_series.values],
                        textposition="outside",
                    )
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font=dict(color="#E6E6E6"),
                    xaxis_title="% Deviation from Peer Median",
                    margin=dict(l=10, r=40, t=30, b=10),
                    height=420,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption(
                    f"Bars show how {main_ticker} compares to the peer median across each metric. "
                    "Positive values indicate the metric is higher than peers (interpretation depends on the metric — "
                    "e.g. higher ROE is favorable, while higher P/E may indicate richer valuation)."
                )
        else:
            st.info("Main ticker or peer data unavailable to compute relative positioning.")

# ============================================================================
# TAB 3 — RAW FINANCIAL STATEMENTS
# ============================================================================
with tab3:
    st.markdown('<div class="section-banner">Annual Financial Statements</div>', unsafe_allow_html=True)
    st.caption(f"All figures in {main_currency}, as reported via Yahoo Finance.")

    def render_statement(df, title):
        st.markdown(f"#### {title}")
        if df is None or df.empty:
            st.markdown(
                f'<div class="warn-banner">⚠️ {title} data is not available for {main_ticker}.</div>',
                unsafe_allow_html=True,
            )
            return
        display = df.copy()
        display = display[sorted_columns(display)]
        display.columns = [pd.to_datetime(c).strftime("%Y-%m-%d") if not isinstance(c, str) else c for c in display.columns]
        formatted = display.applymap(lambda v: fmt_large(v, main_currency) if pd.notnull(v) else "N/A")
        st.dataframe(formatted, use_container_width=True, height=420)

    sub1, sub2, sub3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])
    with sub1:
        render_statement(income_stmt, "Income Statement")
    with sub2:
        render_statement(balance_sheet, "Balance Sheet")
    with sub3:
        render_statement(cashflow_stmt, "Cash Flow Statement")

    st.markdown("---")
    st.markdown('<div class="section-banner">Key Company Snapshot</div>', unsafe_allow_html=True)
    snap_cols = st.columns(4)
    snapshot_items = [
        ("Sector", main_info.get("sector", "N/A")),
        ("Industry", main_info.get("industry", "N/A")),
        ("Employees", f"{main_info.get('fullTimeEmployees'):,}" if main_info.get("fullTimeEmployees") else "N/A"),
        ("Exchange", main_info.get("exchange", "N/A")),
    ]
    for col, (label, value) in zip(snap_cols, snapshot_items):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="font-size:1.1rem;">{value}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "⚠️ This dashboard is for educational and informational purposes only and does not constitute investment advice. "
    "Data is sourced from Yahoo Finance via the `yfinance` library and may be delayed, incomplete, or subject to error."
)
