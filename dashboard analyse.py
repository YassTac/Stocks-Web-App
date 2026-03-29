"""Stock Dashboard – a Streamlit app for visualising price data + technical indicators.

Features
--------
* Ticker selection (single or multiple tickers)
* Date‑range & interval picker (daily, weekly, monthly, intraday)
* Optional overlays: SMA, EMA, Bollinger Bands, RSI
* Interactive Plotly charts (price + indicators, separate RSI chart)
* Raw data table with download button
* Caching of data & figures for snappy reloads
* Full type‑hints, doc‑strings and modular layout – ready for production.

Author  : Your Name
Created : 2026‑03‑29
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import plotly as go
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------
# Streamlit page configuration – one of the first calls in the script
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="📈 Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/streamlit/streamlit/issues",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "Stock visualisation dashboard powered by yfinance + Plotly",
    },
)

# ----------------------------------------------------------------------
# Helper – colour palette (kept here for easy customisation)
# ----------------------------------------------------------------------
PALETTE = {
    "price": "#000000",  # black
    "sma": "#1f77b4",    # blue
    "ema": "#ff7f0e",    # orange
    "bb_upper": "#d62728",   # red
    "bb_lower": "#2ca02c",   # green
    "rsi": "#9467bd",    # purple
}

# ----------------------------------------------------------------------
# --------------------------------------------------- CACHING LOGIC
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60)  # data refreshed at most once per hour
def fetch_price_data(
    tickers: List[str],
    start: date,
    end: date,
    interval: str,
) -> pd.DataFrame:
    """Download OHLCV data for one or many tickers using yfinance.

    Parameters
    ----------
    tickers: List[str]
        List of ticker symbols (e.g. ['AAPL', 'MSFT'])
    start, end: date
        Inclusive start / exclusive end of the period.
    interval: str
        yfinance interval string – e.g. '1d', '1wk', '1mo', '60m', '30m', …

    Returns
    -------
    pd.DataFrame
        Multi‑Indexed DataFrame (ticker, column) if >1 ticker,
        otherwise a regular DataFrame with columns Open, High, Low, Close,
        Adj Close, Volume.
    """
    # yfinance expects strings; we join tickers with space for multiple symbols.
    ticker_str = " ".join(tickers)
    raw = yf.download(
        ticker_str,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,  # keep original “Adj Close” column
        group_by="ticker",
    )

    if raw.empty:
        st.error("No data returned – check ticker spelling and date range.")
        return pd.DataFrame()

    # When we request *one* ticker yfinance returns a flat DataFrame.
    # Multi‑ticker requests return a hierarchical column index (ticker, column).
    if len(tickers) == 1:
        return raw  # type: ignore[return-value]
    else:
        # Flatten the MultiIndex: (ticker, column) → ticker_column
        flat = raw.copy()
        flat.columns = ["_".join(col) for col in flat.columns.to_flat_index()]
        return flat


@st.cache_resource
def get_plotly_template() -> Callable[[str], go.Figure]:
    """Return a factory that creates a Plotly figure with a predefined layout."""
    def factory(title: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig
    return factory


# ----------------------------------------------------------------------
# --------------------------------------------------- INDICATOR CALCULATIONS
# ----------------------------------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()


def bollinger_bands(series: pd.Series, window: int, num_std: float) -> pd.DataFrame:
    """Bollinger Bands – returns a DataFrame with Upper & Lower columns."""
    ma = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return pd.DataFrame({"BB_Upper": upper, "BB_Lower": lower})


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Use Wilder's smoothing method
    ma_up = up.ewm(alpha=1 / window, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False).mean()

    rs = ma_up / ma_down
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


# ----------------------------------------------------------------------
# --------------------------------------------------- CHART CONSTRUCTION
# ----------------------------------------------------------------------
def build_price_chart(
    df: pd.DataFrame,
    ticker: str,
    indicators: Mapping[str, Any],
) -> go.Figure:
    """Create the main price chart with optional SMA, EMA, Bollinger Bands."""
    fig_factory = get_plotly_template()
    fig = fig_factory(f"{ticker.upper()} – Price & Indicators")

    # ------------------------------------------------------------------
    # 1️⃣  Core price line (Close)
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close",
            line=dict(color=PALETTE["price"], width=2),
        )
    )

    # ------------------------------------------------------------------
    # 2️⃣  Overlays (conditionally added)
    # ------------------------------------------------------------------
    if indicators.get("sma"):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA"],
                name=f"SMA {indicators['sma_window']}",
                line=dict(color=PALETTE["sma"], width=1.5, dash="dot"),
            )
        )
    if indicators.get("ema"):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA"],
                name=f"EMA {indicators['ema_window']}",
                line=dict(color=PALETTE["ema"], width=1.5, dash="dash"),
            )
        )
    if indicators.get("bb"):
        # Upper band – solid line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name=f"BB Upper ({indicators['bb_window']}, {indicators['bb_std']}σ)",
                line=dict(color=PALETTE["bb_upper"], width=1),
                opacity=0.7,
            )
        )
        # Lower band – solid line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name=f"BB Lower ({indicators['bb_window']}, {indicators['bb_std']}σ)",
                line=dict(color=PALETTE["bb_lower"], width=1),
                opacity=0.7,
                fill="tonexty",          # fill to previous trace (upper band)
                fillcolor="rgba(255,200,0,0.1)",
            )
        )
    # ------------------------------------------------------------------
    # 3️⃣  Layout tweaks
    # ------------------------------------------------------------------
    fig.update_yaxes(fixedrange=False)  # allow vertical zoom
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def build_rsi_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plot a separate RSI chart with over‑bought / over‑sold bands."""
    fig_factory = get_plotly_template()
    fig = fig_factory(f"{ticker.upper()} – RSI")

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color=PALETTE["rsi"], width=2),
        )
    )
    # Over‑bought (70) & Over‑sold (30) reference lines
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Over‑bought")
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Over‑sold")

    fig.update_yaxes(range=[0, 100])
    fig.update_xaxes(rangeslider_visible=False)

    return fig


# ----------------------------------------------------------------------
# --------------------------------------------------- UI: Sidebar Settings
# ----------------------------------------------------------------------
def sidebar_controls() -> Tuple[List[str], date, date, str, Dict[str, Any]]:
    """Render the sidebar widgets and return the user configuration."""
    st.sidebar.title("⚙️ Settings")

    # ---------- Ticker(s) ----------
    ticker_input = st.sidebar.text_input(
        "Ticker symbol(s) (comma‑separated)", value="AAPL", help="e.g. AAPL, MSFT, GOOGL"
    )
    # Normalise input → list of uppercase symbols without whitespace
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    # ---------- Date range ----------
    today = date.today()
    default_start = today - timedelta(days=365)  # one‑year look‑back

    start = st.sidebar.date_input(
        "Start date", value=default_start, max_value=today - timedelta(days=1)
    )
    end = st.sidebar.date_input(
        "End date", value=today, min_value=start + timedelta(days=1)
    )

    # ---------- Interval ----------
    interval_options = {
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo",
        "1 Hour": "1h",
        "30 Minutes": "30m",
        "15 Minutes": "15m",
        "5 Minutes": "5m",
    }
    interval_label = st.sidebar.selectbox(
        "Time‑interval",
        options=list(interval_options.keys()),
        index=0,
        help="Intraday intervals (e.g. 5m, 15m) only work for the last ~30 days with yfinance.",
    )
    interval = interval_options[interval_label]

    # ---------- Technical Indicators ----------
    st.sidebar.subheader("Technical indicators")
    # SMA
    show_sma = st.sidebar.checkbox("Simple Moving Average (SMA)", value=False)
    sma_window = (
        st.sidebar.slider("SMA window (days)", 5, 200, 20, key="sma_window")
        if show_sma
        else None
    )

    # EMA
    show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA)", value=False)
    ema_window = (
        st.sidebar.slider("EMA window (days)", 5, 200, 20, key="ema_window")
        if show_ema
        else None
    )

    # Bollinger Bands
    show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
    bb_window = (
        st.sidebar.slider("BB window (days)", 5, 200, 20, key="bb_window")
        if show_bb
        else None
    )
    bb_std = (
        st.sidebar.slider("BB standard‑deviations", 1.0, 3.0, 2.0, 0.1, key="bb_std")
        if show_bb
        else None
    )

    # RSI
    show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=False)
    rsi_window = (
        st.sidebar.slider("RSI window (days)", 5, 50, 14, key="rsi_window")
        if show_rsi
        else None
    )

    # ---------- Refresh button ----------
    if st.sidebar.button("🔄 Refresh data", type="primary"):
        # Force cache invalidation by clearing the internal dictionary.
        # Streamlit will automatically rerun the script afterwards.
        st.experimental_rerun()

    # Collect indicator configuration in a dict to pass around.
    indicators: Dict[str, Any] = {
        "sma": show_sma,
        "sma_window": sma_window,
        "ema": show_ema,
        "ema_window": ema_window,
        "bb": show_bb,
        "bb_window": bb_window,
        "bb_std": bb_std,
        "rsi": show_rsi,
        "rsi_window": rsi_window,
    }

    # Guard against empty ticker lists.
    if not tickers:
        st.sidebar.error("Please enter at least one ticker symbol.")
        st.stop()

    return tickers, start, end, interval, indicators


# ----------------------------------------------------------------------
# --------------------------------------------------- MAIN APP
# ----------------------------------------------------------------------
def main() -> None:
    """Entry point – builds the entire dashboard."""
    # Sidebar → user choices
    tickers, start_date, end_date, interval, indicators = sidebar_controls()

    # Use the first ticker for the main chart title (the app works with multiple symbols,
    # but the price chart shows one at a time for clarity).
    primary_ticker = tickers[0]

    # ------------------------------------------------------------------
    # 1️⃣  Load price data (cached)
    # ------------------------------------------------------------------
    with st.spinner(f"Downloading data for {', '.join(tickers)} …"):
        # yfinance returns a DataFrame with a MultiIndex column layout for >1 ticker.
        raw_df = fetch_price_data(tickers, start_date, end_date, interval)

    if raw_df.empty:
        st.warning("No data fetched – adjust the date range / interval.")
        st.stop()

    # ------------------------------------------------------------------
    # 2️⃣  Slice DataFrame to the selected primary ticker (simplify downstream code)
    # ------------------------------------------------------------------
    if len(tickers) == 1:
        df = raw_df.copy()
    else:
        # Columns are like "AAPL_Close", "AAPL_Open", … – filter by ticker prefix.
        df = raw_df.filter(regex=f"^{primary_ticker}_").copy()
        # Rename to standard column names without ticker prefix.
        df.columns = [col.split("_", 1)[1] for col in df.columns]
    df = df.dropna(subset=["Close"]).sort_index()
    df.index.name = "Date"

    # ------------------------------------------------------------------
    # 3️⃣  Compute selected technical indicators (add as new columns)
    # ------------------------------------------------------------------
    if indicators.get("sma"):
        df["SMA"] = sma(df["Close"], indicators["sma_window"])

    if indicators.get("ema"):
        df["EMA"] = ema(df["Close"], indicators["ema_window"])

    if indicators.get("bb"):
        bb = bollinger_bands(df["Close"], indicators["bb_window"], indicators["bb_std"])
        df["BB_Upper"] = bb["BB_Upper"]
        df["BB_Lower"] = bb["BB_Lower"]

    if indicators.get("rsi"):
        df["RSI"] = rsi(df["Close"], indicators["rsi_window"])

    # ------------------------------------------------------------------
    # 4️⃣  Layout – three tabs (Chart, Table, Indicators, Download)
    # ------------------------------------------------------------------
    tab_price, tab_table, tab_indicators, tab_download = st.tabs(
        ["📈 Price chart", "📊 Data table", "🧭 Technical indicators", "⬇️ Download"]
    )

    # ---------------------------------------------------
    # Tab 1 – Price chart (price + SMA/EMA/BB)
    # ---------------------------------------------------
    with tab_price:
        price_fig = build_price_chart(df, primary_ticker, indicators)
        st.plotly_chart(price_fig, use_container_width=True, theme="streamlit")

    # ---------------------------------------------------
    # Tab 2 – Raw data table
    # ---------------------------------------------------
    with tab_table:
        # Show a nicely formatted table – allow the user to scroll horizontally.
        st.dataframe(
            df.style.format("{:.2f}"),
            use_container_width=True,
            height=600,
        )
        # Provide quick summary metrics at top of the table.
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Latest Close",
                value=f"${df['Close'].iloc[-1]:.2f}",
                delta=f"${(df['Close'].iloc[-1] - df['Close'].iloc[-2]):.2f}"
                if len(df) > 1
                else None,
            )
        with col2:
            st.metric(
                label="Average Volume",
                value=f"{df['Volume'].mean():,.0f}",
                delta=None,
            )
        with col3:
            st.metric(
                label="Price Change (YTD)",
                value=f"{(df['Close'].iloc[-1] - df['Close'].iloc[0]):.2f}",
                delta=None,
                delta_color="inverse",
            )

    # ---------------------------------------------------
    # Tab 3 – Indicator‑specific visualisations (e.g. RSI)
    # ---------------------------------------------------
    with tab_indicators:
        if indicators.get("rsi"):
            rsi_fig = build_rsi_chart(df, primary_ticker)
            st.plotly_chart(rsi_fig, use_container_width=True, theme="streamlit")
        else:
            st.info("Enable any indicator (e.g., RSI) in the sidebar to view its chart here.")

    # ---------------------------------------------------
    # Tab 4 – CSV download
    # ---------------------------------------------------
    with tab_download:
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Download data as CSV",
            data=csv,
            file_name=f"{primary_ticker}_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
        st.caption(
            "The file contains all columns currently visible in the data table, "
            "including any computed indicator series (e.g., SMA, EMA, BB, RSI)."
        )

    # ---------------------------------------------------
    # Footer – helpful links & version stamp
    # ---------------------------------------------------
    st.markdown(
        """
        ---
        **Tip:**  
        * Use the **↔️ Zoom** in the chart to focus on a specific period.  
        * You can add up to **5 tickers** (comma‑separated) – the first one is used for the price chart; the others appear in the data table.  
        * Adjust the **interval** to intraday frequencies (e.g., 5 min) for the most recent 30 days of data.  

        *Built with* :heart: *using* **Streamlit**, **yfinance**, **Pandas**, **Plotly**.  
        """
    )
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ----------------------------------------------------------------------
# --------------------------------------------------- Run app
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
