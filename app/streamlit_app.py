"""Machine learning on top of the FDRM India VIX project: a web app.

Forecasts and backtest results are read from files that forecast.py and backtest.py saved in
output/. The models are not retrained here. The "Try it yourself" tab re-runs the hedge
backtest live (same engine as backtest.py) for whatever rule and threshold you choose.

Run locally from the repo root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT / "src"))
import engine as E   # noqa: E402

REPO_URL = "https://github.com/Subham1709/VIX_ML"
ORIGINAL_APP = "https://subham-fdrm-vix.streamlit.app"
ORIGINAL_REPO = "https://github.com/Subham1709/FDRM_VIX"

ORIGINAL = "Original rule: buy when VIX < 13"
COLOURS = {"Unhedged": "#222222", ORIGINAL: "#1f6fd1", "Buy every month": "#c2185b",
           "Gradient boosting: insurance looks cheap": "#e8772e", "LSTM: insurance looks cheap": "#1baf7a",
           "Gradient boosting: fall risk above normal": "#e8b02e", "LSTM: fall risk above normal": "#4a3aa7",
           "VIX < 13 or LSTM insurance cheap": "#7b8a8b", "Your rule": "#d81b60"}
RV_MODELS = {"rv_vix": "India VIX", "rv_trailing": "Last month's volatility", "rv_har": "HAR regression",
             "rv_har_vix": "HAR + VIX regression", "rv_lgbm": "Gradient boosting", "rv_lstm": "LSTM"}
DD_MODELS = {"dd_base": "Long-run base rate", "dd_logit_rv": "Logistic on last month's vol",
             "dd_logit": "Logistic (all features)", "dd_lgbm": "Gradient boosting", "dd_lstm": "LSTM"}
METRICS = {"CAGR": "Return a year (%)", "vol": "Volatility (%)", "sharpe": "Sharpe", "maxDD": "Worst fall (%)",
           "net_cost_pa": "Insurance cost a year (%)", "hedged_pct": "Time insured (%)",
           "puts_bought": "Puts bought"}

st.set_page_config(page_title="Group 8 · ML and India VIX", layout="wide")


@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(OUT / name, **kw)


pred = load_csv("predictions.csv", index_col=0, parse_dates=True)
values = load_csv("backtest_values.csv", index_col=0, parse_dates=True)
perf = load_csv("backtest_performance.csv", index_col=0)
subp = load_csv("backtest_subperiods.csv", index_col=[0, 1])
fm = load_csv("forecast_metrics.csv", index_col=[0, 1])
cm = load_csv("classifier_metrics.csv", index_col=[0, 1])
trades = load_csv("backtest_trades.csv", parse_dates=["date"])
summary = json.loads((OUT / "summary.json").read_text())

o, u = perf.loc[ORIGINAL], perf.loc["Unhedged"]
ml_rows = perf[perf.group == "Machine learning"]
best_ml = ml_rows.CAGR.astype(float).idxmax()
f22 = fm.loc["2022-2026"]


def line_chart(df, title, fmt=".2f", height=320, colours=None, legend=True):
    long = df.rename_axis("date").reset_index().melt("date", var_name="Series", value_name="v").dropna()
    names = list(df.columns)
    scale = alt.Scale(domain=names, range=[(colours or {}).get(n, "#888888") for n in names]) if colours else \
        alt.Scale(domain=names)
    return alt.Chart(long).mark_line(strokeWidth=1.4).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("v:Q", title=title, scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=scale,
                        legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=400) if legend else None),
        tooltip=["date:T", "Series", alt.Tooltip("v:Q", format=fmt)],
    ).properties(height=height)


# ---------- header ----------
st.title("Can machine learning time crash insurance better than India VIX?")
st.markdown(
    "FDRM course project, **Group 8**, IIT Jodhpur · "
    f"[Code on GitHub]({REPO_URL}) · An extension of our [India VIX hedging project]({ORIGINAL_APP})  \n"
    f"Daily Nifty 50 data from 2007 and India VIX from 2020. Every forecast below was made using only data "
    f"available at the time. Backtest from {summary['period']['start']} to {summary['period']['end']}."
)
c = st.columns(4)
c[0].metric("Original rule, return a year", f"{o.CAGR:.1f}%", f"{o.CAGR - u.CAGR:+.1f} pts vs no hedge")
c[1].metric("Best machine learning rule", f"{perf.loc[best_ml, 'CAGR']:.1f}%",
            f"{perf.loc[best_ml, 'CAGR'] - o.CAGR:+.1f} pts vs original rule", delta_color="off",
            help=best_ml)
c[2].metric("LSTM volatility forecast vs VIX", f"{f22.loc['LSTM (Nifty only)', 'MAE_vol_pts']:.1f} pts error",
            f"VIX: {f22.loc['India VIX', 'MAE_vol_pts']:.1f} pts", delta_color="off",
            help="Average gap between the forecast and the volatility that followed, Jan 2022 to Mar 2026")
c[3].metric("Best model at predicting a 7% fall", f"AUC {cm.loc['2022-2026'].AUC.max():.2f}",
            "0.5 = coin toss", delta_color="off")

tabs = st.tabs(["Introduction", "Forecasting volatility", "Predicting falls", "Did ML time the hedge better?",
                "Try it yourself", "How it was done"])

# ---------- 0. introduction ----------
with tabs[0]:
    st.subheader("Where this comes from")
    st.markdown(
        "In our main project we protected a Nifty 50 portfolio with a **put option** (insurance that pays if the "
        "market falls more than 5% within three months). The rule was simple: buy the put only when **India VIX is "
        "below 13**, because that is when insurance is cheap, and sell it when VIX jumps above 22. It returned "
        f"{o.CAGR:.1f}% a year against {u.CAGR:.1f}% with no protection, and the worst fall was "
        f"{abs(o.maxDD):.0f}% instead of {abs(u.maxDD):.0f}%."
    )
    st.subheader("The question here")
    st.markdown(
        "VIX is the option market's own forecast of the next month's volatility. Could a machine learning model, "
        "trained on years of price history, forecast that better, and would a better forecast lead to better "
        "timing of the insurance? We tried two ideas:\n\n"
        "1. **Insurance looks cheap.** Forecast next month's volatility. If the model expects at least as much "
        "movement as VIX is charging for, the put looks cheap, so buy it.\n"
        "2. **Fall risk above normal.** Predict the chance that Nifty falls 7% or more within three months. If it "
        "is above the usual rate, buy the put.\n\n"
        "Each idea was tested with a **gradient boosting** model (many small decision trees, a standard machine "
        "learning method for tables of numbers) and an **LSTM** (a neural network that reads the last three "
        "months of data in order, a standard deep learning method for time series). Simple models were run "
        "alongside as benchmarks."
    )
    st.subheader("What we found")
    lstm, vixr = f22.loc["LSTM (Nifty only)"], f22.loc["India VIX"]
    st.markdown(
        f"- **Volatility forecasts improved a little.** From 2022 the LSTM's average error was "
        f"{lstm.MAE_vol_pts:.1f} volatility points against {vixr.MAE_vol_pts:.1f} for VIX, mainly because VIX "
        f"is on average {vixr.bias_vol_pts:.1f} points too high. A plain regression that mixes recent volatility "
        "with VIX did as well. Gradient boosting did not beat VIX. None of the differences passes a "
        "significance test.\n"
        f"- **Nobody predicts falls.** The best model's AUC was {cm.loc['2022-2026'].AUC.max():.2f}, where 0.5 is "
        "a coin toss. Falls after 2021 mostly started from calm markets, which is exactly when the original "
        "rule buys.\n"
        f"- **The simple VIX rule was hard to beat.** Of the machine learning rules, the best returned "
        f"{perf.loc[best_ml, 'CAGR']:.1f}% a year ({best_ml}) against {o.CAGR:.1f}% for the original rule. "
        "Small changes to the thresholds move these results by several points either way, so the honest "
        "conclusion is that machine learning added no reliable value here.\n\n"
        "This is a useful result. It supports the main project: the cheapness of insurance, read straight from "
        "VIX, matters more than trying to predict the next crash."
    )
    with st.expander("Glossary"):
        st.markdown(
            "- **Volatility:** how much Nifty moves, per year, in %.\n"
            "- **India VIX:** the market's forecast of Nifty's volatility over the next 30 days, from option prices.\n"
            "- **Put option:** insurance that pays if Nifty ends below a set level (the strike).\n"
            "- **Walk-forward test:** retrain the model each month using only the past, then forecast the next "
            "month. This is how the model would have been used in real time.\n"
            "- **HAR regression:** a standard volatility model that averages last week's, last month's and last "
            "quarter's volatility.\n"
            "- **Gradient boosting:** many small decision trees, each correcting the previous ones' errors.\n"
            "- **LSTM:** long short-term memory network, a neural network that reads a sequence and remembers "
            "what matters.\n"
            "- **AUC:** the chance that the model gives a higher score to a day before a fall than to a day "
            "before no fall. 0.5 is guessing, 1 is perfect.\n"
            "- **CAGR:** average growth per year, compounded. **Worst fall:** the biggest drop from a previous high."
        )

# ---------- 1. volatility ----------
with tabs[1]:
    st.markdown(
        "The black line is the volatility Nifty actually had over the next 21 trading days. Each coloured line is "
        "a forecast made on that day. Pick the forecasts to compare. Drag on the chart to zoom."
    )
    pick = st.multiselect("Forecasts", list(RV_MODELS.values()), default=["India VIX", "Gradient boosting", "LSTM"])
    inv = {v: k for k, v in RV_MODELS.items()}
    df = pred[["actual_rv"] + [inv[p] for p in pick]].rename(columns={"actual_rv": "What happened", **RV_MODELS})
    df = df.dropna(subset=["What happened"])
    col = dict(zip(["What happened", "India VIX", "Last month's volatility", "HAR regression",
                    "HAR + VIX regression", "Gradient boosting", "LSTM"],
                   ["#111111", "#1f6fd1", "#9a9893", "#7b57c8", "#c2185b", "#e8772e", "#1baf7a"]))
    st.altair_chart(line_chart(df, "Volatility per year (%)", ".1f", colours=col).interactive(bind_y=False),
                    width="stretch")
    st.subheader("How accurate was each forecast?")
    win = st.radio("Period", ["2022-2026", "2020-2026"], horizontal=True,
                   help="From 2022 every model has forecasts. Models that use VIX start in 2021.")
    t = fm.loc[win].drop(columns="model").astype(float)
    t["days"] = t.days.astype(int)
    st.dataframe(t.round(3).rename(columns={
        "days": "Days", "MAE_vol_pts": "Average error (vol pts)", "RMSE_log": "Error, log scale",
        "bias_vol_pts": "Average bias (vol pts)", "corr": "Correlation", "R2_vs_VIX": "Improvement on VIX",
        "DM_t_vs_VIX": "Significance vs VIX (t)"}), width="stretch")
    st.caption(
        "Improvement on VIX: share of VIX's squared log error that the model removes (0.10 means 10% smaller "
        "errors; negative means worse). Significance: a Diebold-Mariano t-statistic with Newey-West errors; "
        "it needs to be above about 2 to count. VIX's positive bias is the volatility risk premium from the main "
        "project: options usually charge for more movement than follows."
    )

# ---------- 2. falls ----------
with tabs[2]:
    st.markdown(
        "Each day, each model gives the chance that Nifty closes at least 7% below today's level at some point in "
        "the next three months (roughly when a 5% out-of-the-money put starts paying well). The shaded bars show "
        "the days when that did happen."
    )
    pick = st.multiselect("Models", list(DD_MODELS.values()), default=["Long-run base rate", "Gradient boosting", "LSTM"])
    inv = {v: k for k, v in DD_MODELS.items()}
    df = pred[[inv[p] for p in pick]].rename(columns=DD_MODELS)
    long = df.rename_axis("date").reset_index().melt("date", var_name="Series", value_name="v").dropna()
    hap = pred[["actual_dd"]].dropna().rename_axis("date").reset_index()
    bars = alt.Chart(hap[hap.actual_dd == 1]).mark_rule(color="#e34948", opacity=0.18).encode(x="date:T")
    lines = alt.Chart(long).mark_line(strokeWidth=1.3).encode(
        x=alt.X("date:T", title=None), y=alt.Y("v:Q", title="Predicted chance", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Series:N", legend=alt.Legend(orient="top", title=None, labelLimit=400)),
        tooltip=["date:T", "Series", alt.Tooltip("v:Q", format=".2f")])
    st.altair_chart((bars + lines).properties(height=320), width="stretch")
    win = st.radio("Period ", ["2022-2026", "2020-2026"], horizontal=True)
    t = cm.loc[win].drop(columns="model").astype(float)
    t["days"] = t.days.astype(int)
    st.dataframe(t.round(3).rename(columns={"days": "Days", "base_rate": "Share of days before a fall",
                                            "AUC": "AUC (0.5 = guessing)", "Brier": "Brier score (lower is better)",
                                            "mean_prob": "Average predicted chance"}), width="stretch")
    st.markdown(
        "No model ranks risky days much better than guessing. The logistic model on India VIX has an AUC well "
        "below 0.5: trained on 2020-21, it learned that high VIX comes before falls (true in the COVID crash), "
        "but after that most falls started when VIX was low. Patterns in markets change, and six years of VIX "
        "data contain very few independent falls to learn from."
    )

# ---------- 3. backtest ----------
with tabs[3]:
    st.markdown(
        "Every strategy holds Nifty and uses the same 3-month put, 5% below the market, priced the same way as in "
        "the main project. They differ only in **when they buy** (decided on the first trading day of each "
        "month). All of them sell half the put when VIX reaches 22 and the rest once VIX falls 20% from its peak."
    )
    left, right = st.columns([1, 3])
    with left:
        start = st.radio("Start from", ["2 Jan 2020 (full sample)", "1 Jul 2020 (after COVID)", "3 Jan 2022"])
    with right:
        picked = st.multiselect("Strategies", list(values.columns),
                                default=["Unhedged", ORIGINAL, "Gradient boosting: insurance looks cheap",
                                         "LSTM: insurance looks cheap"])
    key = {"1 Jul": "2020-07-01", "3 Jan": "2022-01-01"}.get(start[:5])
    v = values.loc[key:] if key else values
    if picked:
        v = v[picked] / v[picked].iloc[0]
        st.altair_chart(line_chart(v, "Value of ₹1 invested", ".3f", colours=COLOURS), width="stretch")
        st.altair_chart(line_chart((v / v.cummax() - 1) * 100, "Below previous high (%)", ".1f", 170,
                                   colours=COLOURS, legend=False), width="stretch")
    if key:
        table = subp.loc[key].astype(float)[["CAGR", "vol", "sharpe", "maxDD"]]
    else:
        table = perf.drop(columns=["group", "final", "worst_21d", "premium_paid_pa", "buy_months_pct"]).astype(float)
    st.dataframe(table.round(2).rename(columns=METRICS), width="stretch")
    st.caption("A negative insurance cost means the puts earned more than they cost.")
    st.markdown(
        "The original rule's lead comes almost entirely from the put it bought in January 2020, before the COVID "
        "crash. Gradient boosting's 'cheap insurance' rule happened to buy in February 2020 and caught the crash "
        "too. The LSTM's rule did not buy before the crash and ended level with no hedge. After July 2020, all "
        "the original rule and both 'cheap insurance' rules are within about half a point a year of holding "
        "Nifty with no protection, while buying on fall risk cost 2 to 3 points a year."
    )
    with st.expander("Every option trade"):
        s = st.selectbox("Strategy", trades.strategy.unique())
        tr = trades[trades.strategy == s]
        st.dataframe(tr.assign(date=tr.date.dt.date)[["date", "action", "strike", "price", "vix"]].round(2)
                     .rename(columns={"date": "Date", "action": "What happened", "strike": "Strike",
                                      "price": "Price (index points)", "vix": "VIX"}),
                     width="stretch", hide_index=True)

# ---------- 4. try it ----------
with tabs[4]:
    st.markdown(
        "Build your own buying rule from the saved forecasts and see how it would have done. The backtest runs "
        "live with the same pricing and selling rule as the other strategies."
    )
    a, b = st.columns([1, 2])
    with a:
        kind = st.radio("Buy the put when", ["Insurance looks cheap (forecast vol vs VIX)",
                                             "Predicted chance of a 7% fall is high", "VIX is low (original idea)"])
        R = pred.loc[E.roll_dates(values.index)]   # same trading days as the saved backtest
        if kind.startswith("Insurance"):
            m = st.selectbox("Forecast", ["Gradient boosting", "LSTM", "HAR regression", "HAR + VIX regression"],
                             key="m_rv")
            k = st.slider("Buy if forecast ÷ VIX is at least", 0.6, 1.4, 1.0, 0.05, key="k_rv")
            col = {v: k_ for k_, v in RV_MODELS.items()}[m]
            buy = (R[col] >= k * R.VIX).fillna(False)
            label = f"{m}: forecast ≥ {k:.2f} × VIX"
        elif kind.startswith("Predicted"):
            m = st.selectbox("Model", ["Gradient boosting", "LSTM", "Logistic (all features)",
                                       "Logistic on last month's vol"], key="m_dd")
            k = st.slider("Buy if predicted chance is at least", 0.05, 0.7, 0.35, 0.05, key="k_dd")
            col = {v: k_ for k_, v in DD_MODELS.items()}[m]
            buy = (R[col] >= k).fillna(False)
            label = f"{m}: chance ≥ {k:.2f}"
        else:
            k = st.slider("Buy if VIX is below", 10.0, 20.0, 13.0, 0.5, key="k_vix")
            buy = R.VIX < k
            label = f"VIX < {k:.1f}"
        also = st.checkbox("Also buy whenever VIX < 13", value=False)
        if also:
            buy = buy | (R.VIX < 13)
            label += " or VIX < 13"
    val, led = E.run(values.index.to_series().map(pred.Close), values.index.to_series().map(pred.VIX),
                     buy.astype(bool))
    mine = dict(E.metrics(val.value, led), hedged_pct=val.hedged.mean() * 100)
    with b:
        m3 = st.columns(4)
        m3[0].metric("Return a year", f"{mine['CAGR']:.1f}%", f"{mine['CAGR'] - o.CAGR:+.1f} vs original rule")
        m3[1].metric("Worst fall", f"{mine['maxDD']:.0f}%", f"{mine['maxDD'] - o.maxDD:+.0f} vs original rule")
        m3[2].metric("Puts bought", f"{mine['puts_bought']}", f"original: {int(o.puts_bought)}", delta_color="off")
        m3[3].metric("Time insured", f"{mine['hedged_pct']:.0f}%")
        cmp = pd.DataFrame({"Your rule": val.value, ORIGINAL: values[ORIGINAL], "Unhedged": values["Unhedged"]})
        st.altair_chart(line_chart(cmp, "Value of ₹1 invested", ".3f", 300, colours=COLOURS), width="stretch")
        months = [d.strftime("%b %Y") for d in buy.index[buy.values]]
        st.caption(f"Your rule ({label}) bought in: " + (", ".join(months) if months else "no months"))
    st.markdown(
        "Try moving the slider a little: the result often jumps by several points because one more or one fewer "
        "put in early 2020 changes everything. That is the main warning of this project. Choosing the threshold "
        "after seeing these results would be fitting the past, not forecasting."
    )

# ---------- 5. method ----------
with tabs[5]:
    st.subheader("Data")
    st.markdown(
        "Nifty 50 daily prices from September 2007 and India VIX from January 2020 to April 2026, the same files "
        "as the main project. Using Nifty back to 2007 gives the models twelve years, including the 2008 crisis, "
        "to learn from before the test starts in 2020."
    )
    st.subheader("What the models see")
    st.markdown(
        "Fifteen numbers per day, all known at that day's close: realised volatility over the last 5, 10, 21, 63 "
        "and 126 days; volatility from the daily high-low range; downside volatility; returns over 1 week, 1 "
        "month and 3 months; distance from the 1-year high and from the 50- and 200-day averages; and the size "
        "of today's move. The 'Nifty + VIX' versions add the VIX level, its 1-week and 1-month change, VIX minus "
        "recent volatility, and where VIX sits in its past year's range. The LSTM sees the last 63 days of the "
        "fifteen Nifty numbers."
    )
    st.subheader("No peeking at the future")
    st.markdown(
        "- **Walk-forward training.** On the first trading day of each month the model is retrained on all "
        "earlier data and then forecasts every day of that month. The LSTM is retrained every three months.\n"
        "- **Gap before the test.** A day only enters training once its answer is fully known: 21 days later for "
        "the volatility target and 63 days later for the fall target.\n"
        "- **Fixed settings.** Model settings and the two buying thresholds (forecast at least equal to VIX; "
        "chance above the base rate) were set before running the backtest. The threshold chart on the "
        "'Try it yourself' tab is shown to expose how fragile the results are, not to choose a better one.\n"
        "- **Same trading rules.** Option prices, the bid/ask spread, the selling rule and the dates are exactly "
        "those of the main project. Our backtest engine reproduces its original-rule figures to the decimal."
    )
    st.subheader("Models")
    st.markdown(
        "- **Benchmarks:** India VIX itself; last month's volatility; the HAR regression; HAR plus VIX; "
        "logistic regression.\n"
        "- **Gradient boosting:** LightGBM, 300 to 400 shallow trees (8 leaves), learning rate 0.03.\n"
        "- **LSTM:** one layer of 32 units with two outputs (volatility and chance of a fall) trained together, "
        "PyTorch, early stopping on the most recent 15% of the training data, average of three random starts."
    )
    st.subheader("Limits")
    st.markdown(
        "Option prices are modelled from VIX, not real trades. Six years of VIX contain one real crash, so any "
        "comparison of hedging rules rests on a handful of decisions. Overlapping 3-month windows mean the "
        "1,500 test days hold far fewer independent observations. Models with VIX features have only one to "
        "four years of training data."
    )

st.divider()
st.caption(
    "FDRM course project, Group 8, IIT Jodhpur. An extension of our India VIX project "
    f"([app]({ORIGINAL_APP}), [code]({ORIGINAL_REPO})). Code, data and method for this app: [GitHub]({REPO_URL})."
)
