"""Run with:  pytest -q   (checks the engine against numbers printed on the slides)."""
import numpy as np
import pandas as pd
import pytest

import engine as e


def series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


# ---- Slide 9: moving average -------------------------------------------------
def test_sma_slide9():
    s = series([100, 102, 101, 104, 103, 105])
    ma = e.moving_average(s, 5)
    assert ma.iloc[4] == pytest.approx(102)   # Mon-Fri window
    assert ma.iloc[5] == pytest.approx(103)   # window slides, 100 drops out


# ---- Slide 13: EMA -------------------------------------------------------------
def test_ema_formula_slide13():
    s = series(np.arange(100, 130, dtype=float))
    n = 10
    k = 2 / (n + 1)
    assert k == pytest.approx(0.1818, abs=1e-3)
    ema = e.moving_average(s, n, "EMA")
    manual = s.iloc[0]
    for p in s.iloc[1:]:
        manual = p * k + manual * (1 - k)
    assert ema.iloc[-1] == pytest.approx(manual)
    assert ema.iloc[:n - 1].isna().all()      # warm-up masked


# ---- Slide 15: z-score ---------------------------------------------------------
def test_zscore_matches_formula():
    s = series([100, 95, 105, 98, 102, 112])
    z, m, sd = e.zscore(s, 5)
    expected = (s.iloc[-1] - s.iloc[-5:].mean()) / s.iloc[-5:].std()
    assert z.iloc[-1] == pytest.approx(expected)
    assert (112 - 100) / 5 == pytest.approx(2.4)  # slide arithmetic


# ---- Slide 16: entry / exit rule ----------------------------------------------
def test_band_rule():
    z = series([0, 1, 2.4, 1.0, 0.0, -0.5, -2.3, -1.0, 0.1, 0.5])
    pos = e.band_positions(z, entry=2, exit_=0).tolist()
    #           0  1   2.4  1.0  0.0  -0.5 -2.3  -1.0 0.1  0.5
    assert pos == [0, 0, -1, -1, 0, 0, 1, 1, 0, 0]


def test_no_short_mode():
    z = series([0, 2.5, 1, 0, -2.5, 0])
    assert e.band_positions(z, 2, 0, allow_short=False).tolist() == [0, 0, 0, 0, 1, 0]


# ---- Slide 10 / 12: crossover signals alternate -------------------------------
def test_momentum_alternates():
    rng = np.random.default_rng(1)
    px = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))))
    sig = e.momentum_signals(px, 8, 25, "SMA", True, px.index[0])
    changes = sig.pos[sig.pos.ne(sig.pos.shift())].iloc[1:]
    nz = changes[changes != 0]
    assert len(nz) > 3
    assert (nz.values[1:] == -nz.values[:-1]).all()   # BUY, SELL, BUY, ...


# ---- Slide 21: no peeking ---------------------------------------------------------
def test_no_lookahead():
    rng = np.random.default_rng(2)
    px = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    base = e.run_backtest(e.mean_reversion_signals(px, 30, 2, 0, True, px.index[0]), 0, 1000)
    px2 = px.copy()
    px2.iloc[-1] *= 1.5                      # change only the LAST price
    alt = e.run_backtest(e.mean_reversion_signals(px2, 30, 2, 0, True, px2.index[0]), 0, 1000)
    # Everything before the last day must be identical.
    pd.testing.assert_series_equal(base.equity.iloc[:-1], alt.equity.iloc[:-1])
    assert base.held.iloc[0] == 0            # first day can never hold anything


# ---- Slide 18: market neutral ----------------------------------------------------
def test_pairs_market_neutral():
    # Both stocks fall 10% on the same day: long one, short the other nets £0.
    a = series([100, 100, 90, 90]); b = series([50, 50, 45, 45])
    sig = e.pairs_signals(a, b, 2, 2, 0, "Ratio (A ÷ B)", a.index[0])
    assert sig.unit_ret.abs().max() == pytest.approx(0, abs=1e-12)


# ---- Costs reduce profit ---------------------------------------------------------
def test_costs_hurt():
    rng = np.random.default_rng(3)
    px = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))))
    sig = e.momentum_signals(px, 8, 25, "SMA", True, px.index[0])
    free = e.run_backtest(sig, 0, 1000).equity.iloc[-1]
    paid = e.run_backtest(sig, 20, 1000).equity.iloc[-1]
    assert paid < free


# ---- Slides 23-25: drawdown and Sharpe ------------------------------------------
def test_drawdown_slide23():
    assert e.drawdown(series([100, 70])).min() == pytest.approx(-0.30)


def test_sharpe_slide25_and_27():
    assert e.sharpe_ratio(0.12, 0.10) == pytest.approx(0.80)   # Fund A
    assert e.sharpe_ratio(0.18, 0.25) == pytest.approx(0.56)   # Fund B
    assert e.sharpe_ratio(0.15, 0.20) == pytest.approx(0.55)   # momentum fund
    assert e.sharpe_ratio(0.09, 0.06) == pytest.approx(0.833, abs=1e-3)
    assert e.sharpe_ratio(0.07, 0.04) == pytest.approx(0.75)


def test_trade_returns_roughly_match_equity():
    rng = np.random.default_rng(4)
    px = series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, 1500))))
    sig = e.mean_reversion_signals(px, 30, 2, 0, True, px.index[0])
    bt = e.run_backtest(sig, 0, 1000)
    closed_open = (1 + bt.trades["Return"]).prod()
    assert closed_open == pytest.approx(bt.equity.iloc[-1] / 1000, rel=0.02)
