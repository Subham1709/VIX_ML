"""Step 3: charts for the README and the PDF summary (saved to figures/)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "output", ROOT / "figures"
FIG.mkdir(exist_ok=True)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         violet="#4a3aa7", red="#e34948", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", grey="#9a9893")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9.5, "axes.edgecolor": C["ink2"],
    "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": C["grid"], "grid.linewidth": 0.6, "lines.linewidth": 1.4,
    "axes.titleweight": "bold", "axes.titlesize": 10.5, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


P = pd.read_csv(OUT / "predictions.csv", index_col=0, parse_dates=True)
fm = pd.read_csv(OUT / "forecast_metrics.csv", index_col=[0, 1])
cm = pd.read_csv(OUT / "classifier_metrics.csv", index_col=[0, 1])
vals = pd.read_csv(OUT / "backtest_values.csv", index_col=0, parse_dates=True)
perf = pd.read_csv(OUT / "backtest_performance.csv", index_col=0)
grid = pd.read_csv(OUT / "threshold_sensitivity.csv")

# 1. forecasts against what happened
s = P.dropna(subset=["actual_rv"])
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(s.index, s.actual_rv, color=C["ink"], lw=1.6, label="What actually happened (next 21 days)")
ax.plot(s.index, s.VIX, color=C["blue"], lw=1, label="India VIX")
ax.plot(s.index, s.rv_lgbm, color=C["orange"], lw=1, label="Gradient boosting")
ax.plot(s.index, s.rv_lstm, color=C["aqua"], lw=1, label="LSTM")
ax.set_ylim(0, 70)
ax.set_ylabel("Volatility per year (%)")
ax.set_title("Next-month Nifty volatility: forecasts made each day vs what followed")
ax.legend(ncol=2, loc="upper right")
save(fig, "fig1_vol_forecasts.png")

# 2. accuracy table as bars (2022-2026: every model has forecasts)
w = fm.loc["2022-2026"].sort_values("MAE_vol_pts")
fig, ax = plt.subplots(figsize=(7.5, 3.2))
cols = [C["blue"] if n == "India VIX" else (C["orange"] if ("boost" in n or "LSTM" in n) else C["grey"])
        for n in w.index]
ax.barh(w.index[::-1], w.MAE_vol_pts[::-1], color=cols[::-1])
for i, v in enumerate(w.MAE_vol_pts[::-1]):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8.5)
ax.set_xlabel("Average forecast error, volatility points (lower is better)")
ax.set_title("Forecast error, Jan 2022 to Mar 2026")
ax.grid(axis="y", visible=False)
save(fig, "fig2_forecast_error.png")

# 3. probability of a 7% fall
s = P.dropna(subset=["dd_lstm"])
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.fill_between(s.index, 0, s.actual_dd, step="post", color=C["red"], alpha=0.15, lw=0,
                label="A 7% fall did follow within 3 months")
ax.plot(s.index, s.dd_lgbm, color=C["orange"], lw=1, label="Gradient boosting")
ax.plot(s.index, s.dd_lstm, color=C["aqua"], lw=1, label="LSTM")
ax.plot(s.index, s.dd_base, color=C["ink2"], lw=1, ls="--", label="Long-run base rate")
ax.set_ylim(0, 1)
ax.set_ylabel("Predicted chance")
ax.set_title("Predicted chance of a 7% fall within 3 months")
ax.legend(ncol=2, loc="upper right")
save(fig, "fig3_fall_probability.png")

# 4. backtest
show = {"Unhedged": C["ink"], "Original rule: buy when VIX < 13": C["blue"],
        "LSTM: insurance looks cheap": C["aqua"], "Gradient boosting: insurance looks cheap": C["orange"],
        "LSTM: fall risk above normal": C["violet"]}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
for n, col in show.items():
    v = vals[n]
    a1.plot(v.index, v, color=col, lw=1.6 if n.startswith("Original") else 1.1, label=n)
    a2.plot(v.index, (v / v.cummax() - 1) * 100, color=col, lw=1)
a1.set_ylabel("Value of 1 invested")
a1.set_title("Hedge backtest: growth of 1 and falls from the previous high")
a1.legend(ncol=2, fontsize=8)
a2.set_ylabel("Below high (%)")
save(fig, "fig4_backtest.png")

# 5. threshold sensitivity
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
names = {"rv_lgbm": "Gradient boosting", "rv_lstm": "LSTM", "rv_har": "HAR regression"}
for (m, n), col in zip(names.items(), [C["orange"], C["aqua"], C["grey"]]):
    g = grid[grid.model == m]
    a1.plot(g.k, g.CAGR, marker="o", ms=3, color=col, label=n)
o = perf.loc["Original rule: buy when VIX < 13"]
u = perf.loc["Unhedged"]
for a in (a1,):
    a.axhline(o.CAGR, color=C["blue"], ls="--", lw=1, label="Original rule")
    a.axhline(u.CAGR, color=C["ink"], ls=":", lw=1, label="Unhedged")
a1.set_xlabel("Buy when forecast vol / VIX is at least")
a1.set_ylabel("Return a year (%)")
a1.set_title("'Insurance looks cheap' rule")
a1.legend(fontsize=7, loc="lower right", ncol=2)
for m, n, col in [("dd_lgbm", "Gradient boosting", C["orange"]), ("dd_lstm", "LSTM", C["aqua"])]:
    g = grid[grid.model == m]
    a2.plot(g.k, g.CAGR, marker="o", ms=3, color=col, label=n)
a2.axhline(o.CAGR, color=C["blue"], ls="--", lw=1)
a2.axhline(u.CAGR, color=C["ink"], ls=":", lw=1)
a2.set_xlabel("Buy when predicted chance of a 7% fall is at least")
a2.set_title("'Fall risk' rule")
a2.legend(fontsize=7.5)
save(fig, "fig5_threshold_sensitivity.png")
print("figures saved")
