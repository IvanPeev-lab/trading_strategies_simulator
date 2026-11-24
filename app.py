"""
KCAP Strategy Lab: interactive back-tester for the Session 3 homework.
Run with:   streamlit run app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import engine as eng

st.set_page_config(page_title="KCAP Strategy Lab", page_icon="📈", layout="wide")

GREEN, RED, GREY, BLUE, ORANGE, PURPLE = (
    "#16a34a", "#dc2626", "#6b7280", "#2563eb", "#f59e0b", "#7c3aed")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance…")
def load_prices(tickers: tuple, start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(list(tickers), start=start, end=end + dt.timedelta(days=1),
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def money(x: float) -> str:
    return f"{'-' if x < 0 else ''}£{abs(x):,.0f}"


def pct(x: float) -> str:
    return "n/a" if x != x else f"{x:+.1%}"


def shade_oos(fig, oos_start, end):
    if oos_start is not None:
        fig.add_vrect(x0=oos_start, x1=end, fillcolor=ORANGE, opacity=0.10,
                      line_width=0, layer="below", row="all", col="all")


def pnl_chart(bt, bench_ret, capital, oos_start, title):
    fig = go.Figure()
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    if bench_ret is not None:
        bh = capital * (1 + bench_ret.fillna(0)).cumprod() - capital
        fig.add_scatter(x=bh.index, y=bh, name="Buy & hold (benchmark)",
                        line=dict(color=GREY, width=1.5, dash="dot"),
                        hovertemplate="%{x|%d %b %Y}<br>Buy & hold: £%{y:,.0f}<extra></extra>")
    pnl = bt.equity - capital
    fig.add_scatter(x=pnl.index, y=pnl, name="Strategy", line=dict(color=BLUE, width=2.5),
                    hovertemplate="%{x|%d %b %Y}<br>Strategy P/L: £%{y:,.0f}<extra></extra>")
    shade_oos(fig, oos_start, pnl.index[-1])
    fig.update_layout(title=title, yaxis_title="Profit / loss (£)", yaxis_tickprefix="£",
                      yaxis_tickformat=",.0f", hovermode="x unified", height=430,
                      margin=dict(t=60, b=30), legend=dict(orientation="h", y=1.08, x=0))
    return fig


def trade_markers(pos):
    chg = pos.ne(pos.shift()) & pos.shift().notna()
    ev = pos[chg]
    return ev[ev == 1].index, ev[ev == -1].index, ev[ev == 0].index


def signal_chart(sig, oos_start, entry=None, exit_=None):
    has_z = sig.z is not None
    fig = make_subplots(rows=2 if has_z else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35] if has_z else [1.0],
                        vertical_spacing=0.06)
    fig.add_scatter(x=sig.ref.index, y=sig.ref, name=sig.ref_name,
                    line=dict(color=BLUE, width=1.6), row=1, col=1)
    palette = [ORANGE, PURPLE, PURPLE]
    for (name, line), colour in zip(sig.lines.items(), palette):
        dash = "dot" if "σ" in name else "solid"
        fig.add_scatter(x=line.index, y=line, name=name,
                        line=dict(color=colour, width=1.2, dash=dash), row=1, col=1)
    long_in, short_in, flat = trade_markers(sig.pos)
    for idx, name, colour, symbol in [(long_in, "Go long", GREEN, "triangle-up"),
                                       (short_in, "Go short", RED, "triangle-down"),
                                       (flat, "Close", GREY, "x")]:
        if len(idx):
            fig.add_scatter(x=idx, y=sig.ref.loc[idx], mode="markers", name=name,
                            marker=dict(color=colour, symbol=symbol, size=10,
                                        line=dict(width=1, color="white")),
                            row=1, col=1)
    if has_z:
        fig.add_scatter(x=sig.z.index, y=sig.z, name="z-score",
                        line=dict(color=PURPLE, width=1.3), row=2, col=1)
        for lvl, dash in [(entry, "dash"), (-entry, "dash"), (exit_, "dot")]:
            fig.add_hline(y=lvl, line_color=GREY, line_dash=dash, line_width=1, row=2, col=1)
        fig.update_yaxes(title_text="z-score", row=2, col=1)
    fig.update_yaxes(title_text=sig.ref_name, row=1, col=1)
    shade_oos(fig, oos_start, sig.ref.index[-1])
    fig.update_layout(height=620 if has_z else 470, hovermode="x unified",
                      margin=dict(t=40, b=30), legend=dict(orientation="h", y=1.06, x=0),
                      title="Signals: where the rules bought, sold and closed")
    return fig


def drawdown_chart(bt, oos_start):
    dd = eng.drawdown(bt.equity) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=RED, width=1),
                               name="Drawdown",
                               hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}% below peak<extra></extra>"))
    shade_oos(fig, oos_start, dd.index[-1])
    fig.update_layout(title="Drawdown: how far below the previous peak (slide 23)",
                      yaxis_title="%", height=260, margin=dict(t=50, b=30))
    return fig


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.title("📈 KCAP Strategy Lab")
st.caption("King's Capital Academy · Session 3 homework · change the variables on the "
           "left and the profit/loss graph updates.")

with st.sidebar:
    st.header("1 · Strategy")
    strategy = st.selectbox("Choose a strategy", ["Momentum", "Mean reversion", "Pairs trading"])

    st.header("2 · Data")
    if strategy == "Pairs trading":
        c1, c2 = st.columns(2)
        tk_a = c1.text_input("Stock A", "PEP", key="a").strip().upper()
        tk_b = c2.text_input("Stock B", "KO", key="b").strip().upper()
        tickers = (tk_a, tk_b)
        st.caption("Try similar firms: PEP/KO, V/MA, XOM/CVX, HD/LOW.")
    else:
        default = "AAPL" if strategy == "Momentum" else "KO"
        tickers = (st.text_input("Ticker (Yahoo Finance)", default, key=f"t_{strategy}")
                   .strip().upper(),)
    d1, d2 = st.columns(2)
    start_date = d1.date_input("From", dt.date(2015, 1, 1), min_value=dt.date(2000, 1, 1))
    end_date = d2.date_input("To", dt.date.today(), max_value=dt.date.today())

    st.header("3 · Strategy variables")
    allow_short = True
    if strategy == "Momentum":
        fast = st.slider("Fast average (days)", 2, 100, 8)
        slow = st.slider("Slow average (days)", 5, 250, 25)
        ma_kind = st.radio("Average type", ["SMA", "EMA"], horizontal=True,
                           help="SMA: every day counts equally. EMA: recent days count more.")
        allow_short = st.checkbox("SELL signal = go short", True,
                                  help="Untick to go to cash on a SELL instead.")
        max_window = slow
    else:
        window = st.slider("Look-back window (days)", 10, 250, 30,
                           help="Days used to measure the average and the 'typical wiggle'.")
        entry = st.slider("Entry: |z| beyond", 0.5, 4.0, 2.0, 0.1)
        exit_ = st.slider("Close when z returns to", 0.0, 2.0, 0.0, 0.1)
        if strategy == "Mean reversion":
            allow_short = st.checkbox("Allow shorting when z is high", True)
        else:
            spread_kind = st.radio("Spread", ["Ratio (A ÷ B)", "Difference (A − B)"], horizontal=True)
        max_window = window

    st.header("4 · Realism")
    capital = st.number_input("Starting capital (£)", 1_000, 10_000_000, 10_000, 1_000)
    cost_bps = st.slider("Trading cost (basis points per trade)", 0.0, 50.0, 5.0, 0.5,
                         help="1 bp = 0.01%. Charged on every buy/sell (slide 21: costs add up).")
    oos_pct = st.slider("Hold out last X% as out-of-sample", 0, 50, 0, 5,
                        help="Slide 30: judge the rule on data it has not 'seen'.")

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()
if strategy == "Momentum" and fast >= slow:
    st.error("The fast average must be shorter than the slow average.")
    st.stop()
if strategy != "Momentum" and exit_ >= entry:
    st.error("The exit level must be smaller than the entry level.")
    st.stop()
if "" in tickers:
    st.info("Enter a ticker symbol to begin.")
    st.stop()
if strategy == "Pairs trading" and tickers[0] == tickers[1]:
    st.error("Pick two different stocks.")
    st.stop()

# --------------------------------------------------------------------------
# Data + signals
# --------------------------------------------------------------------------
warmup = dt.timedelta(days=int(max_window * 1.6) + 10)   # so indicators are ready on day 1
try:
    data = load_prices(tickers, start_date - warmup, end_date)
except Exception as exc:  # network / yfinance problems
    st.error(f"Could not download prices: {exc}")
    st.stop()

missing = [t for t in tickers if t not in data.columns or data[t].dropna().empty]
if missing:
    st.error(f"No price data found for: {', '.join(missing)}. Check the ticker symbol "
             "(Yahoo format, e.g. AAPL, BP.L, ^GSPC) and the date range.")
    st.stop()
data = data[list(tickers)].dropna()
start_ts = pd.Timestamp(start_date)

if strategy == "Momentum":
    sig = eng.momentum_signals(data[tickers[0]], fast, slow, ma_kind, allow_short, start_ts)
elif strategy == "Mean reversion":
    sig = eng.mean_reversion_signals(data[tickers[0]], window, entry, exit_, allow_short, start_ts)
else:
    sig = eng.pairs_signals(data[tickers[0]], data[tickers[1]], window, entry, exit_,
                            spread_kind, start_ts)

if len(sig.pos) < 60:
    st.error("Not enough price history in this range. Widen the dates.")
    st.stop()

bt = eng.run_backtest(sig, cost_bps, capital)
full = eng.metrics(bt.strat_ret, capital)

oos_start = None
if oos_pct > 0:
    cut = int(len(bt.strat_ret) * (1 - oos_pct / 100))
    oos_start = bt.strat_ret.index[cut]

name = (f"{tickers[0]} vs {tickers[1]} pairs trade" if strategy == "Pairs trading"
        else f"{strategy} on {tickers[0]}")

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
st.subheader(name)
m = st.columns(6)
m[0].metric("Final value", money(full["Final value"]), money(full["Profit / loss"]))
m[1].metric("Total return", pct(full["Total return"]))
m[2].metric("Return / year", pct(full["Return / year"]))
m[3].metric("Volatility", f"{full['Volatility']:.1%}")
m[4].metric("Worst fall", f"{full['Worst fall (max drawdown)']:.1%}")
m[5].metric("Sharpe (rf 4%)", f"{full['Sharpe']:.2f}", eng.sharpe_label(full["Sharpe"]),
            delta_color="off")

bench_note = ""
if sig.bench_ret is not None:
    bm = eng.metrics(sig.bench_ret, capital)
    bench_note = (f" Buy & hold over the same dates: {pct(bm['Total return'])} total, "
                  f"Sharpe {bm['Sharpe']:.2f}, worst fall {bm['Worst fall (max drawdown)']:.1%}.")
st.caption(f"{len(bt.trades)} trades.{bench_note}")

if oos_start is not None:
    st.warning("Orange shading = out-of-sample (the last "
               f"{oos_pct}%). Tune your variables on the unshaded part only, then check "
               "the table below once.")

st.plotly_chart(pnl_chart(bt, sig.bench_ret, capital, oos_start,
                          f"Profit / loss over time (starting capital {money(capital)})"),
                width="stretch")

if oos_start is not None:
    ins = eng.metrics(bt.strat_ret.loc[:oos_start].iloc[:-1], capital)
    oos = eng.metrics(bt.strat_ret.loc[oos_start:], capital)
    table = pd.DataFrame({"In-sample (tuned on)": ins, "Out-of-sample (fresh)": oos}).T
    st.markdown("**In-sample vs out-of-sample**: a big drop is a sign of overfitting (slide 30).")
    st.dataframe(table.style.format({
        "Final value": "£{:,.0f}", "Profit / loss": "£{:,.0f}", "Total return": "{:+.1%}",
        "Return / year": "{:+.1%}", "Volatility": "{:.1%}",
        "Worst fall (max drawdown)": "{:.1%}", "Sharpe": "{:.2f}"}),
        width="stretch")

st.plotly_chart(signal_chart(sig, oos_start,
                             entry if strategy != "Momentum" else None,
                             exit_ if strategy != "Momentum" else None),
                width="stretch")
st.plotly_chart(drawdown_chart(bt, oos_start), width="stretch")

with st.expander("Trade list"):
    if bt.trades.empty:
        st.write("No trades in this period. Try a smaller entry level or a longer date range.")
    else:
        t = bt.trades
        lvl = [c for c in t.columns if c.startswith(("Entry ", "Exit ")) and "date" not in c]
        st.dataframe(t.style.format({"Return": "{:+.2%}", **{c: "{:,.3f}" for c in lvl}}),
                     width="stretch", hide_index=True)
        closed = t[t["Status"] == "Closed"]
        if len(closed):
            st.caption(f"Win rate on closed trades: {(closed['Return'] > 0).mean():.0%} "
                       f"({len(closed)} trades).")

with st.expander("How this strategy works (from the slides)"):
    if strategy == "Momentum":
        st.markdown(
            "- **Bet:** trends continue (slide 8).\n"
            "- **Signal:** fast average crosses **above** slow → BUY; **below** → SELL (slide 10).\n"
            "- **SMA vs EMA:** the EMA weights recent days more using k = 2 ÷ (n+1) (slide 13): "
            "earlier signals, more false alarms.\n"
            "- **Weakness:** quick reversals, called whipsaws, lose money. Try shrinking the "
            "gap between fast and slow and watch the trade count and costs jump.")
    elif strategy == "Mean reversion":
        st.markdown(
            "- **Bet:** stretched prices snap back (slide 14).\n"
            "- **Signal:** z = (price − rolling average) ÷ rolling standard deviation. "
            "z above +entry → short; z below −entry → long; close when z returns to the exit "
            "level (slides 15-16).\n"
            "- **Weakness:** a real change in the company keeps the price away from 'normal' "
            "and the losses on a short have no ceiling (slides 6, 19).")
    else:
        st.markdown(
            "- **Bet:** the gap between two similar stocks closes (slide 18).\n"
            "- **Signal:** z-score of the spread. z above +entry → A looks dear: **short A, long B**; "
            "z below −entry → **long A, short B**. Close both when z returns to the exit level.\n"
            "- **Market neutral:** equal £ on each side, so a market-wide fall roughly cancels.\n"
            "- **Weakness:** the link can break, e.g. one firm has real news (slide 19).")

with st.expander("Assumptions (read before trusting any number)"):
    st.markdown(
        "- **No peeking:** a signal uses the closing price of day *t*; the trade earns the "
        "return from day *t+1* onwards (slide 21).\n"
        "- **Prices** are Yahoo Finance daily closes adjusted for splits and dividends. "
        "Returns are percentages, so the stock's currency does not matter and your "
        "capital is shown in £.\n"
        "- **Position size:** always 100% of capital (no leverage). Pairs: 50% on each leg.\n"
        "- **Pairs** are rebalanced to equal £ each day while the trade is open, a "
        "simplification of 'equal £ at the start'.\n"
        "- **Costs:** the basis points are charged per unit of position change; no borrowing "
        "fee for shorts and no tax.\n"
        "- **Sharpe** = (annualised return − 4%) ÷ annualised volatility, as on slide 25. "
        "**Worst fall** = largest drop from a previous peak (slide 23).\n"
        "- A good result here is **not proof** (slide 30). Keep some data aside and test once.")
