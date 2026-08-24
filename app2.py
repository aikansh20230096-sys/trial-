Act as a Senior Quant Finance & Full-Stack Python Developer. Build a production-ready, highly polished Streamlit application in a single file (`app.py`) titled "Fundamental Valuation & Peer Multiples Dashboard".

### 1. Requirements & Dependencies
- Use `streamlit`, `yfinance`, `pandas`, `numpy`, `plotly`.
- Theme & Styling: Dark UI (`#0E1117` background, sleek card containers, custom metric colors). Make tables and charts look modern and executive-ready.
- Data fetching: Use `yfinance` to automatically pull Income Statements, Balance Sheets, Cash Flow Statements, and live price/key metric data.

### 2. Layout Structure & Navigation
Use a clean sidebar for inputs and dynamic tabbed navigation (`st.tabs`) for the main workspace:
- **Sidebar**: 
  - Main Ticker Input (e.g., `AAPL`, `MSFT`, `RELIANCE.NS`).
  - Peer Tickers Input (multiselect/text input default: `AAPL, MSFT, GOOGL, NVDA, AMZN`).
  - Refresh Data Button.
- **Tabs**:
  1. 📈 **DCF Valuation Model**
  2. 📊 **Peer Group & Multiples Analysis**
  3. 📄 **Raw Financial Statements**

---

### 3. Detailed Tab Specifications

#### Tab 1: DCF Valuation Model
- **Inputs & Interactive Sliders (in sidebar or expandable config bar)**:
  - Forecast Horizon (Years): 5 to 10 years (default: 5).
  - Revenue Growth Rate (%): Slider defaulted to historical CAGR.
  - Operating Margin / EBIT Margin (%): Slider.
  - WACC (Weighted Average Cost of Capital %): Slider (4.0% to 15.0%, default 8.5%).
  - Terminal Growth Rate (%): Slider (1.0% to 5.0%, default 2.5%).
  - Tax Rate (%): Slider (default 21%).
- **Calculation Engine**:
  - Project Free Cash Flows to Firm (FCFF) = $EBIT \times (1 - t) + D\&A - CapEx - \Delta NWC$.
  - Calculate Terminal Value via Gordon Growth Model: $TV = \frac{FCFF_n \times (1 + g)}{WACC - g}$.
  - Discount all cash flows to present value using $WACC$.
  - Compute Enterprise Value, add Cash, subtract Total Debt to compute Equity Value.
  - Compute Intrinsic Value per Share = $\frac{\text{Equity Value}}{\text{Shares Outstanding}}$.
- **Visuals & Layout**:
  - **Header KPI Cards**: Current Market Price vs. Calculated Intrinsic Value, Margin of Safety (%), Upside/Downside (%).
  - **Projected Cash Flow Chart**: Interactive Plotly bar chart showing projected FCFF per year and discounted present value.
  - **Sensitivity Analysis Matrix**: 2D Heatmap table showing Intrinsic Value per share across varying WACC (x-axis) vs. Terminal Growth Rates (y-axis). Color scale from Red (Undervalued) to Green (Overvalued).

#### Tab 2: Peer Group & Multiples Analysis
- **Data Collection**:
  - Automatically extract metrics for main ticker + peer tickers: Market Cap, P/E Ratio, Forward P/E, EV/EBITDA, Price to Book (P/B), Return on Equity (ROE %), Debt to Equity (D/E), Operating Margin (%).
- **Visuals & Layout**:
  - **Peer Comparison Table**: Interactive table formatted nicely with color highlighting (e.g., highlighting peer averages vs min/max).
  - **Valuation Multiples Scatter Plot**: Interactive Plotly scatter plot (e.g., EV/EBITDA vs ROE or P/E vs Growth) with ticker labels.
  - **Under/Overvaluation Radar Chart or Bar Chart**: Visual ranking of how the main target stock compares against the peer group median across all multiples.

#### Tab 3: Raw Financial Statements
- Dynamic tables showing annual Income Statement, Balance Sheet, and Cash Flow metrics pulled via `yfinance` for deep-dive checks.

### 4. Code Standards & Error Handling
- Wrap all `yfinance` API calls in `@st.cache_data` for speed and responsiveness.
- Include fallback logic/error banners if historical metrics or balance sheet rows are missing for specific stocks.
- Ensure all numbers are formatted clearly (e.g., Currency formatting $ / ₹, Percentages %, Large numbers in Billions/Millions).
- Provide complete code with zero placeholders or truncation.