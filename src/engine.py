"""Put-hedge backtest, ported from the original FDRM VIX project (same pricing and trading rules).

The only change: the decision "buy a new 3-month put today?" on each roll date comes from a
signal Series instead of the fixed test VIX < 13, so any model's forecast can drive it.
Options are priced with Black-Scholes using India VIX plus a put skew and a bid/ask spread.
"""
import math

import numpy as np
import pandas as pd

R, Q = 0.065, 0.013     # risk-free rate, Nifty dividend yield
SKEW_PER_PCT = 0.6      # extra put IV (vol points) per 1% out of the money
HALF_SPREAD = 0.5       # bid/ask half-spread in vol points
SPIKE = 22.0            # sell half the puts the first day VIX closes at or above this
TRAIL = 0.20            # sell the rest once VIX is 20% below its peak since the spike
TENOR = 3               # months
MONEYNESS = 0.95        # 5% out of the money
TD = 252


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, T, sigma, r=R, q=Q):
    T, sigma = max(T, 1e-8), max(sigma, 1e-4)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def roll_dates(index):
    return sorted(pd.Series(index, index=index).groupby([index.year, index.month]).min())


def run(close, vix, buy, monetise=True, tenor=TENOR, moneyness=MONEYNESS, spike=SPIKE, trail=TRAIL):
    """close, vix: daily Series; buy: boolean Series indexed by roll dates (True = buy a new put).

    Returns (daily DataFrame with portfolio value, trade ledger).
    """
    dates = close.index
    rolls = roll_dates(dates)
    roll_set, roll_pos = set(rolls), {d: i for i, d in enumerate(rolls)}
    units = 1.0 / close.iloc[0]
    pos, trades, rec = [], [], []

    def iv(v, S, K, side):
        otm = max(0.0, (1 - K / S) * 100)
        return max((v + SKEW_PER_PCT * otm + side * HALF_SPREAD) / 100, 0.01)

    def price(p, d, S, v, side=0):
        T = (p["exp"] - d).days / 365
        return max(p["K"] - S, 0.0) if T <= 0 else bs_put(S, p["K"], T, iv(v, S, p["K"], side))

    def close_pos(p, d, S, v, frac, why):
        nonlocal units
        qty = p["qty"] * frac
        px = price(p, d, S, v, side=-1)
        units += qty * px / S
        p["qty"] -= qty
        trades.append(dict(date=d, trade_id=p["tid"], action=why, strike=p["K"], qty=qty, price=px,
                           cash=qty * px, vix=v))

    for d, S, v in zip(dates, close.values, vix.values):
        for p in pos:
            if p["exp"] <= d:
                close_pos(p, d, S, v, 1.0, "expiry")
        pos = [p for p in pos if p["qty"] > 1e-12]
        if monetise:
            for p in pos:
                if v >= spike:
                    p["peak"] = max(p.get("peak", 0.0), v)
                    if not p["sold1"]:
                        close_pos(p, d, S, v, 0.5, "sell half (spike)")
                        p["sold1"] = True
                if p.get("peak") and v <= (1 - trail) * p["peak"] and p["qty"] > 0:
                    close_pos(p, d, S, v, 1.0, "sell rest (spike fading)")
            pos = [p for p in pos if p["qty"] > 1e-12]
        if d in roll_set and bool(buy.get(d, False)):
            i = roll_pos[d] + tenor
            if i < len(rolls):
                for p in pos:
                    close_pos(p, d, S, v, 1.0, "roll (sell old put)")
                pos = []
                value = units * S
                K = round(S * moneyness / 50) * 50
                p = dict(K=K, exp=rolls[i], qty=value / S, tid=len(trades), sold1=False)
                px = price(p, d, S, v, side=+1)
                units -= p["qty"] * px / S
                pos.append(p)
                trades.append(dict(date=d, trade_id=p["tid"], action="buy", strike=K, qty=p["qty"], price=px,
                                   cash=-p["qty"] * px, vix=v))
        hedge = sum(price(p, d, S, v) * p["qty"] for p in pos)
        rec.append((d, units * S + hedge, hedge > 0))
    out = pd.DataFrame(rec, columns=["date", "value", "hedged"]).set_index("date")
    return out, pd.DataFrame(trades)


def metrics(value, ledger=None, rf=R):
    r = value.pct_change().dropna()
    yrs = (value.index[-1] - value.index[0]).days / 365.25
    cagr = (value.iloc[-1] / value.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * math.sqrt(TD)
    dd = value / value.cummax() - 1
    m = dict(final=value.iloc[-1], CAGR=cagr * 100, vol=vol * 100, sharpe=(cagr - rf) / vol, maxDD=dd.min() * 100,
             worst_21d=(value / value.shift(21) - 1).min() * 100)
    if ledger is not None:
        if len(ledger):
            # each cashflow as % of the portfolio's value that day, as in the original project
            pct = ledger.cash.values / value.reindex(ledger.date).values * 100
            paid, got = -pct[pct < 0].sum(), pct[pct > 0].sum()
            m.update(puts_bought=int((ledger.action == "buy").sum()), premium_paid_pa=paid / yrs,
                     net_cost_pa=(paid - got) / yrs)
        else:
            m.update(puts_bought=0, premium_paid_pa=0.0, net_cost_pa=0.0)
    return m
