"""Step 1: walk-forward forecasts of next-month volatility and of a 7% fall within three months.

Writes output/predictions.csv (one row per trading day, one column per model) and
output/forecast_metrics.csv / output/classifier_metrics.csv.
Run: python forecast.py            (about 10-15 minutes on a laptop CPU, mostly the LSTM)
     python forecast.py --no-lstm  (under a minute; keeps any LSTM columns already saved)
"""
import sys
import time
from pathlib import Path

if "--no-lstm" not in sys.argv:
    import torch  # noqa: F401  (load torch before scikit-learn/LightGBM to avoid a library clash on some Linux setups)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import features as F   # noqa: E402
import models as M     # noqa: E402

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
TEST_START = pd.Timestamp("2020-01-01")   # first out-of-sample day for models that use only Nifty
VIX_START = pd.Timestamp("2020-01-01")    # India VIX history starts here
COMMON_START = pd.Timestamp("2022-01-01") # window where every model, including the VIX ones, has forecasts

t0 = time.time()
df = F.dataset()
NF, VF = F.NIFTY_FEATURES, F.NIFTY_FEATURES + F.VIX_FEATURES
HAR = ["lrv5", "lrv21", "lrv63"]
P = pd.DataFrame(index=df.index)
P["actual_rv"] = df.rv_fwd
P["actual_dd"] = df.y_dd
P["worst_fwd"] = df.worst_fwd
P["VIX"] = df.VIX
P["Close"] = df.Close

# ---- volatility forecasts (log scale; converted to vol points at the end) ----
wf = dict(start=TEST_START, horizon=F.H_RV)
P["rv_trailing"] = df.lrv21                                    # "next month looks like last month"
P["rv_vix"] = np.log(df.VIX)                                   # the market's own forecast
P["rv_har"] = M.walk_forward(df, HAR, "y_rv", M.ols, **wf)     # classic HAR model (Corsi 2009)
P["rv_lgbm"] = M.walk_forward(df, NF, "y_rv", M.lgb_reg, **wf)
P["rv_har_vix"] = M.walk_forward(df, HAR + ["lvix"], "y_rv", M.ols, train_from=VIX_START, min_train=300, **wf)
P["rv_lgbm_vix"] = M.walk_forward(df, VF, "y_rv", M.lgb_reg, train_from=VIX_START, min_train=300, **wf)
print(f"volatility models done ({time.time() - t0:.0f}s)")

# ---- probability of a 7% fall within 63 trading days ----
wf = dict(start=TEST_START, horizon=F.H_DD)
P["dd_base"] = M.walk_forward(df, ["lrv21"], "y_dd", M.base_rate, **wf)
P["dd_logit_rv"] = M.walk_forward(df, ["lrv21"], "y_dd", M.logit, **wf)
P["dd_logit"] = M.walk_forward(df, NF, "y_dd", M.logit, **wf)
P["dd_lgbm"] = M.walk_forward(df, NF, "y_dd", M.lgb_clf, **wf)
P["dd_logit_vix"] = M.walk_forward(df, ["lvix"], "y_dd", M.logit, train_from=VIX_START, min_train=300, **wf)
P["dd_lgbm_vix"] = M.walk_forward(df, VF, "y_dd", M.lgb_clf, train_from=VIX_START, min_train=300, **wf)
print(f"drawdown models done ({time.time() - t0:.0f}s)")

prev = OUT / "predictions.csv"
if "--no-lstm" in sys.argv:
    old = pd.read_csv(prev, index_col=0, parse_dates=True) if prev.exists() else pd.DataFrame(index=P.index)
    P["rv_lstm"] = np.log(old["rv_lstm"]) if "rv_lstm" in old else np.nan
    P["dd_lstm"] = old["dd_lstm"] if "dd_lstm" in old else np.nan
else:
    import lstm
    P["rv_lstm"], P["dd_lstm"] = lstm.walk_forward(df, NF, start=TEST_START)
    print(f"LSTM done ({time.time() - t0:.0f}s)")

# back from logs to volatility points
for c in [c for c in P if c.startswith("rv_")]:
    P[c] = np.exp(P[c])
P = P[P.index >= TEST_START]
P.to_csv(OUT / "predictions.csv")

# ---- scoring ----
RV_NAMES = {"rv_trailing": "Last month's volatility", "rv_vix": "India VIX", "rv_har": "HAR regression",
            "rv_har_vix": "HAR + VIX regression", "rv_lgbm": "Gradient boosting (Nifty only)",
            "rv_lgbm_vix": "Gradient boosting (Nifty + VIX)", "rv_lstm": "LSTM (Nifty only)"}
DD_NAMES = {"dd_base": "Base rate", "dd_logit_rv": "Logistic on last month's vol",
            "dd_logit_vix": "Logistic on India VIX", "dd_logit": "Logistic (all Nifty features)",
            "dd_lgbm": "Gradient boosting (Nifty only)", "dd_lgbm_vix": "Gradient boosting (Nifty + VIX)",
            "dd_lstm": "LSTM (Nifty only)"}


def dm_t(loss_bench, loss_model, lags=F.H_RV):
    """Diebold-Mariano t-statistic with Newey-West errors (forecast windows overlap).
    Positive = the model's squared errors are smaller than the benchmark's."""
    d = np.asarray(loss_bench) - np.asarray(loss_model)
    n, m = len(d), d.mean()
    e = d - m
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(v / n)


def rv_scores(p, start):
    s = p[(p.index >= start)].dropna(subset=["actual_rv"])
    rows = {}
    for c, name in RV_NAMES.items():
        x = s[list(dict.fromkeys(["actual_rv", c, "rv_vix"]))].dropna()   # same days as VIX, for a fair comparison
        if x.empty:
            continue
        e = np.log(x[c]) - np.log(x.actual_rv)
        bench = np.log(s.loc[x.index, "rv_vix"]) - np.log(x.actual_rv)
        rows[name] = dict(model=c, days=len(x), MAE_vol_pts=(x[c] - x.actual_rv).abs().mean(),
                          RMSE_log=np.sqrt((e ** 2).mean()), bias_vol_pts=(x[c] - x.actual_rv).mean(),
                          corr=np.corrcoef(x[c], x.actual_rv)[0, 1],
                          R2_vs_VIX=1 - (e ** 2).sum() / (bench ** 2).sum(),
                          DM_t_vs_VIX=dm_t(bench ** 2, e ** 2) if c != "rv_vix" else np.nan)
    return pd.DataFrame(rows).T


def dd_scores(p, start):
    s = p[(p.index >= start)].dropna(subset=["actual_dd"])
    rows = {}
    for c, name in DD_NAMES.items():
        x = s[["actual_dd", c]].dropna()
        if x.empty:
            continue
        rows[name] = dict(model=c, days=len(x), base_rate=x.actual_dd.mean(), AUC=roc_auc_score(x.actual_dd, x[c]),
                          Brier=brier_score_loss(x.actual_dd, x[c]), mean_prob=x[c].mean())
    return pd.DataFrame(rows).T


fm = pd.concat({"2020-2026": rv_scores(P, TEST_START), "2022-2026": rv_scores(P, COMMON_START)})
cm = pd.concat({"2020-2026": dd_scores(P, TEST_START), "2022-2026": dd_scores(P, COMMON_START)})
fm.to_csv(OUT / "forecast_metrics.csv")
cm.to_csv(OUT / "classifier_metrics.csv")
pd.set_option("display.width", 200)
print(fm.drop(columns="model").astype(float).round(3))
print(cm.drop(columns="model").astype(float).round(3))
print(f"total {time.time() - t0:.0f}s")
