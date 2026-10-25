"""Load Nifty 50 and India VIX, build features (known at the close of day t) and targets (after t)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TD = 252
H_RV = 21          # forecast horizon for realised volatility (one month)
H_DD = 63          # horizon for the drawdown target (three months, the life of the put)
DD_LEVEL = -0.07   # "stress": Nifty closes at least 7% below today's level within 63 days

# features that only use Nifty prices (available from 2007)
NIFTY_FEATURES = ["lrv5", "lrv10", "lrv21", "lrv63", "lrv126", "lpk5", "lpk21", "semi21",
                  "ret5", "ret21", "ret63", "dd252", "ma50", "ma200", "absret1"]
# extra features from India VIX (available from 2020)
VIX_FEATURES = ["lvix", "dvix5", "dvix21", "vrp", "vixpct"]


def load():
    nifty = pd.read_csv(DATA / "nifty50_daily_raw.csv", parse_dates=["Date"]).set_index("Date")
    nifty = nifty[["Open", "High", "Low", "Close"]].astype(float)
    nifty = nifty[~nifty.index.duplicated()].sort_index()
    vix = pd.read_csv(DATA / "india_vix_daily_raw.csv", encoding="utf-8-sig", parse_dates=["Date"])
    vix = vix.rename(columns={"Sum of Close": "VIX"}).set_index("Date")[["VIX"]].astype(float).sort_index()
    vix = vix[~vix.index.duplicated()]
    return nifty.join(vix, how="left")


def build(df):
    df = df.copy()
    c = df.Close
    r = np.log(c).diff()
    df["ret"] = r
    ann = math.sqrt(TD) * 100
    for w in (5, 10, 21, 63, 126):
        df[f"rv{w}"] = r.rolling(w).std() * ann
        df[f"lrv{w}"] = np.log(df[f"rv{w}"])
    # Parkinson range-based volatility uses the daily high and low
    pk = np.log(df.High / df.Low) ** 2 / (4 * math.log(2))
    for w in (5, 21):
        df[f"lpk{w}"] = np.log(np.sqrt(pk.rolling(w).mean()) * ann + 1e-6)
    df["semi21"] = np.log(np.sqrt((np.minimum(r, 0) ** 2).rolling(21).mean()) * ann + 1e-6)
    for w in (5, 21, 63):
        df[f"ret{w}"] = c / c.shift(w) - 1
    df["dd252"] = c / c.rolling(252).max() - 1
    df["ma50"] = c / c.rolling(50).mean() - 1
    df["ma200"] = c / c.rolling(200).mean() - 1
    df["absret1"] = r.abs()
    # India VIX features (NaN before 2020)
    df["lvix"] = np.log(df.VIX)
    df["dvix5"] = df.lvix - df.lvix.shift(5)
    df["dvix21"] = df.lvix - df.lvix.shift(21)
    df["vrp"] = df.VIX - df.rv21
    df["vixpct"] = df.VIX.rolling(252, min_periods=120).rank(pct=True)

    # ---- targets: everything here uses data AFTER day t ----
    fwd_rv = r[::-1].rolling(H_RV).std()[::-1].shift(-1) * ann     # realised vol over t+1 .. t+21
    df["rv_fwd"] = fwd_rv
    df["y_rv"] = np.log(fwd_rv)
    fmin = c[::-1].rolling(H_DD).min()[::-1].shift(-1)              # lowest close over t+1 .. t+63
    worst = fmin / c - 1
    df["worst_fwd"] = worst
    df["y_dd"] = np.where(worst.isna(), np.nan, (worst <= DD_LEVEL).astype(float))
    return df


def dataset():
    return build(load())


if __name__ == "__main__":
    d = dataset()
    print(d[NIFTY_FEATURES + VIX_FEATURES + ["y_rv", "y_dd"]].describe().T.round(3))
