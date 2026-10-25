"""Step 2: use the walk-forward forecasts to decide when to buy the 3-month put, and compare
with the original VIX < 13 rule. Every strategy uses the same option pricing and the same
selling rule (sell half at VIX 22, the rest when VIX falls 20% from its peak).

Reads output/predictions.csv, writes output/backtest_*.csv and output/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import engine as E   # noqa: E402

OUT = ROOT / "output"
END = pd.Timestamp("2026-04-13")          # same last day as the original project
P = pd.read_csv(OUT / "predictions.csv", index_col=0, parse_dates=True)
P = P[(P.index <= END) & P.VIX.notna()]
rolls = E.roll_dates(P.index)
R = P.loc[rolls]


def signal(rule):
    return pd.Series(rule.values.astype(bool), index=rule.index)


# A strategy = a yes/no decision on each first trading day of the month. Thresholds are fixed in
# advance (no tuning on the test period): "cheap" = the model expects next month's volatility to be
# at least what the option market charges (India VIX); "risky" = the model's chance of a 7% fall is
# above the long-run base rate it was trained on.
STRATEGIES = {
    "Unhedged": signal(R.VIX < 0),
    "Original rule: buy when VIX < 13": signal(R.VIX < 13),
    "Buy every month": signal(R.VIX > 0),
    "Gradient boosting: insurance looks cheap": signal(R.rv_lgbm >= R.VIX),
    "LSTM: insurance looks cheap": signal(R.rv_lstm >= R.VIX),
    "Gradient boosting: fall risk above normal": signal(R.dd_lgbm > R.dd_base),
    "LSTM: fall risk above normal": signal(R.dd_lstm > R.dd_base),
    "VIX < 13 or LSTM insurance cheap": signal((R.VIX < 13) | (R.rv_lstm >= R.VIX)),
}
GROUP = {"Unhedged": "Reference", "Original rule: buy when VIX < 13": "Reference",
         "Buy every month": "Reference", "VIX < 13 or LSTM insurance cheap": "Combined"}

vals, perf, ledgers = {}, {}, []
for name, buy in STRATEGIES.items():
    v, led = E.run(P.Close, P.VIX, buy)
    vals[name] = v.value
    perf[name] = dict(E.metrics(v.value, led), hedged_pct=v.hedged.mean() * 100,
                      buy_months_pct=buy.mean() * 100, group=GROUP.get(name, "Machine learning"))
    if len(led):
        ledgers.append(led.assign(strategy=name))
vals = pd.DataFrame(vals)
perf = pd.DataFrame(perf).T
vals.to_csv(OUT / "backtest_values.csv")
perf.to_csv(OUT / "backtest_performance.csv")
pd.concat(ledgers).to_csv(OUT / "backtest_trades.csv", index=False)
pd.DataFrame({k: v.astype(int) for k, v in STRATEGIES.items()}).to_csv(OUT / "monthly_signals.csv")

# sub-periods: the same daily values, measured from a later start (removes the COVID crash)
sub = {}
for start in ["2020-07-01", "2022-01-01"]:
    w = vals[vals.index >= start]
    sub[start] = pd.DataFrame({k: E.metrics(w[k] / w[k].iloc[0]) for k in w}).T[["CAGR", "vol", "maxDD", "sharpe"]]
subp = pd.concat(sub)
subp.to_csv(OUT / "backtest_subperiods.csv")

# how sensitive is the result to the "cheap" threshold? (forecast vol / VIX >= k)
grid = []
for model in ["rv_lgbm", "rv_lstm", "rv_har"]:
    for k in np.round(np.arange(0.7, 1.31, 0.05), 2):
        v, led = E.run(P.Close, P.VIX, signal(R[model] >= k * R.VIX))
        m = E.metrics(v.value, led)
        grid.append(dict(model=model, k=k, CAGR=m["CAGR"], maxDD=m["maxDD"], sharpe=m["sharpe"],
                         puts=m["puts_bought"]))
for t in np.round(np.arange(0.05, 0.61, 0.05), 2):
    for model in ["dd_lgbm", "dd_lstm"]:
        v, led = E.run(P.Close, P.VIX, signal(R[model] >= t))
        m = E.metrics(v.value, led)
        grid.append(dict(model=model, k=t, CAGR=m["CAGR"], maxDD=m["maxDD"], sharpe=m["sharpe"],
                         puts=m["puts_bought"]))
pd.DataFrame(grid).to_csv(OUT / "threshold_sensitivity.csv", index=False)

pd.set_option("display.width", 220)
print(perf.drop(columns="group").astype(float).round(2).to_string())
print(subp.astype(float).round(2).to_string())

fm = pd.read_csv(OUT / "forecast_metrics.csv", index_col=[0, 1])
cm = pd.read_csv(OUT / "classifier_metrics.csv", index_col=[0, 1])
summary = dict(period=dict(start=str(P.index[0].date()), end=str(P.index[-1].date()), days=len(P)),
               performance=perf.reset_index(names="strategy").to_dict(orient="records"),
               subperiods=subp.reset_index(names=["from", "strategy"]).to_dict(orient="records"),
               forecast=fm.reset_index(names=["window", "name"]).to_dict(orient="records"),
               classifier=cm.reset_index(names=["window", "name"]).to_dict(orient="records"))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
