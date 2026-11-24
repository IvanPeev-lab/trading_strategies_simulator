"""
Strategy + back-test engine for the KCAP Academy Session 3 homework.

Everything here follows the definitions on the lecture slides:
  * Moving average (slide 9), crossover (slide 10), EMA (slide 13)
  * z-score (slide 15), mean-reversion rule (slide 16)
  * Pairs trading on the z-score of a spread (slide 18)
  * Drawdown (slide 23) and Sharpe ratio with a 4% risk-free rate (slide 25)
  * Costs and "no peeking" (slide 21)

Conventions (important, and the reason results are honest):
  * A signal is computed from the CLOSING price of day t.
  * The trade happens at that same close, so the position first earns the
    return of day t+1.  Code: held = pos.shift(1).  This is what stops the
    back-test "peeking" at tomorrow's price.
  * Position +1 = fully long, -1 = fully short, 0 = in cash.
  * For a pair, +1 = long A / short B and the capital is split 50/50, so the
    gross exposure is 1 (same size of bet as a single stock).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE = 0.04  # slide 25


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def moving_average(prices: pd.Series, n: int, kind: str = "SMA") -> pd.Series:
    """SMA: average of the last n prices.  EMA: k = 2/(n+1) (slide 13)."""
    if kind == "EMA":
        # span=n  ->  alpha = 2/(n+1) = k on the slide.
        return prices.ewm(span=n, adjust=False, min_periods=n).mean()
    return prices.rolling(n).mean()


def zscore(series: pd.Series, window: int):
    """z = (today - rolling average) / rolling standard deviation (slide 15)."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z, mean, std


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
@dataclass
class Signals:
    pos: pd.Series                 # desired position at the close of each day
    unit_ret: pd.Series            # return earned per +1 of position, per day
    ref: pd.Series                 # price (or spread) shown on the chart
    lines: dict = field(default_factory=dict)   # overlay lines for the chart
    z: pd.Series | None = None     # z-score (mean reversion / pairs only)
    bench_ret: pd.Series | None = None  # buy-and-hold daily return, if any
    ref_name: str = "Price"


def momentum_signals(prices: pd.Series, fast: int, slow: int, kind: str,
                     allow_short: bool, start: pd.Timestamp) -> Signals:
    """Fast/slow moving-average crossover (slides 8-13).

    BUY when fast crosses ABOVE slow, SELL when it crosses BELOW.  We only
    act on an actual crossover, so the first valid day (where we have not
    seen a cross yet) is flat.  SELL = go short if allow_short, else cash.
    """
    fast_ma = moving_average(prices, fast, kind).loc[start:]
    slow_ma = moving_average(prices, slow, kind).loc[start:]
    px = prices.loc[start:]

    side = np.sign(fast_ma - slow_ma).replace(0, np.nan).ffill()
    crossed = side.ne(side.shift()) & side.notna() & side.shift().notna()
    pos = side.where(crossed.cumsum() > 0).fillna(0.0)
    if not allow_short:
        pos = pos.clip(lower=0.0)

    return Signals(
        pos=pos,
        unit_ret=px.pct_change().fillna(0.0),
        ref=px,
        lines={f"Fast {kind} ({fast})": fast_ma, f"Slow {kind} ({slow})": slow_ma},
        bench_ret=px.pct_change().fillna(0.0),
    )


def band_positions(z: pd.Series, entry: float, exit_: float,
                   allow_short: bool = True) -> pd.Series:
    """Rule from slide 16: z > +entry -> short, z < -entry -> long,
    close when z comes back to `exit_` (0 on the slide)."""
    out = np.zeros(len(z))
    cur = 0
    for i, v in enumerate(z.to_numpy()):
        if np.isnan(v):
            cur = 0
        elif cur == 0:
            if v > entry and allow_short:
                cur = -1
            elif v < -entry:
                cur = 1
        elif cur == 1 and v >= -exit_:
            cur = 0
        elif cur == -1 and v <= exit_:
            cur = 0
        out[i] = cur
    return pd.Series(out, index=z.index)


def mean_reversion_signals(prices: pd.Series, window: int, entry: float,
                           exit_: float, allow_short: bool,
                           start: pd.Timestamp) -> Signals:
    z, mean, std = zscore(prices, window)
    z, mean, std, px = (x.loc[start:] for x in (z, mean, std, prices))
    return Signals(
        pos=band_positions(z, entry, exit_, allow_short),
        unit_ret=px.pct_change().fillna(0.0),
        ref=px,
        lines={f"{window}-day average": mean,
               f"+{entry:g} σ": mean + entry * std,
               f"−{entry:g} σ": mean - entry * std},
        z=z,
        bench_ret=px.pct_change().fillna(0.0),
    )


def pairs_signals(price_a: pd.Series, price_b: pd.Series, window: int,
                  entry: float, exit_: float, spread_kind: str,
                  start: pd.Timestamp) -> Signals:
    """Trade the z-score of the spread A/B (ratio) or A-B (difference).

    z > +entry : A looks dear, B cheap  -> SHORT A, LONG B   (pos = -1)
    z < -entry : A looks cheap, B dear  -> LONG A, SHORT B   (pos = +1)
    Close both when z returns to the exit level (slide 18).
    """
    spread = price_a / price_b if spread_kind == "Ratio (A ÷ B)" else price_a - price_b
    z, mean, std = zscore(spread, window)
    z, mean, std, sp = (x.loc[start:] for x in (z, mean, std, spread))
    ret_a = price_a.pct_change().loc[start:].fillna(0.0)
    ret_b = price_b.pct_change().loc[start:].fillna(0.0)
    return Signals(
        pos=band_positions(z, entry, exit_, allow_short=True),
        unit_ret=0.5 * ret_a - 0.5 * ret_b,   # £ split 50/50 across the legs
        ref=sp,
        lines={f"{window}-day average": mean,
               f"+{entry:g} σ": mean + entry * std,
               f"−{entry:g} σ": mean - entry * std},
        z=z,
        bench_ret=None,
        ref_name="Spread",
    )


# --------------------------------------------------------------------------
# Back-test
# --------------------------------------------------------------------------
@dataclass
class Backtest:
    strat_ret: pd.Series
    equity: pd.Series
    held: pd.Series
    trades: pd.DataFrame


def run_backtest(sig: Signals, cost_bps: float, capital: float) -> Backtest:
    """Replay the rules.  Cost = bps charged on every unit of position change
    (so a flip from long to short costs twice as much as opening a trade)."""
    held = sig.pos.shift(1).fillna(0.0)               # no peeking
    turnover = held.diff().abs().fillna(held.abs())
    strat_ret = held * sig.unit_ret - turnover * cost_bps / 1e4
    strat_ret.iloc[0] = 0.0                           # day 0 is the baseline
    equity = capital * (1 + strat_ret).cumprod()
    return Backtest(strat_ret, equity, held, extract_trades(sig, cost_bps))


def extract_trades(sig: Signals, cost_bps: float) -> pd.DataFrame:
    pos, ret, ref = sig.pos, sig.unit_ret, sig.ref
    seg = pos.ne(pos.shift()).cumsum()
    idx = pos.index
    rows = []
    for _, g in pos.groupby(seg):
        direction = g.iloc[0]
        if direction == 0:
            continue
        i0 = idx.get_loc(g.index[0])
        i1_next = i0 + len(g)                 # first day of the next segment
        is_open = i1_next >= len(idx)
        i1 = len(idx) - 1 if is_open else i1_next
        # Entered at close of day i0, earns returns of days i0+1 .. i1.
        gross = (1 + direction * ret.iloc[i0 + 1:i1 + 1]).prod() - 1
        cost = (1 if is_open else 2) * cost_bps / 1e4
        rows.append({
            "Entry date": idx[i0].date(),
            "Exit date": None if is_open else idx[i1].date(),
            "Direction": "Long" if direction > 0 else "Short",
            f"Entry {sig.ref_name.lower()}": float(ref.iloc[i0]),
            f"Exit {sig.ref_name.lower()}": float(ref.iloc[i1]),
            "Return": gross - cost,
            "Status": "Open" if is_open else "Closed",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Scoring (slides 22-25)
# --------------------------------------------------------------------------
def sharpe_ratio(ann_return: float, ann_vol: float, rf: float = RISK_FREE) -> float:
    """(return - risk-free rate) / volatility  (slide 25)."""
    return (ann_return - rf) / ann_vol if ann_vol and ann_vol > 0 else float("nan")


def drawdown(equity: pd.Series) -> pd.Series:
    """(trough - peak) / peak, measured against the highest value so far."""
    return equity / equity.cummax() - 1


def metrics(strat_ret: pd.Series, capital: float) -> dict:
    strat_ret = strat_ret.copy()
    if len(strat_ret) < 2:
        return {}
    strat_ret.iloc[0] = 0.0
    equity = capital * (1 + strat_ret).cumprod()
    total = equity.iloc[-1] / capital - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    ann_ret = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    ann_vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    return {
        "Final value": equity.iloc[-1],
        "Profit / loss": equity.iloc[-1] - capital,
        "Total return": total,
        "Return / year": ann_ret,
        "Volatility": ann_vol,
        "Worst fall (max drawdown)": drawdown(equity).min(),
        "Sharpe": sharpe_ratio(ann_ret, ann_vol),
    }


def sharpe_label(s: float) -> str:
    """Rough guide from slide 25."""
    if s != s:  # NaN
        return "n/a"
    return "Weak" if s < 0.5 else "Decent" if s <= 1 else "Strong (rare)"
