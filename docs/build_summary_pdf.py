"""Builds docs/fdrm_vix_ml_summary.pdf from the saved results (python docs/build_summary_pdf.py)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "output", ROOT / "figures"
perf = pd.read_csv(OUT / "backtest_performance.csv", index_col=0)
fm = pd.read_csv(OUT / "forecast_metrics.csv", index_col=[0, 1]).loc["2022-2026"]
cm_ = pd.read_csv(OUT / "classifier_metrics.csv", index_col=[0, 1]).loc["2022-2026"]
subp = pd.read_csv(OUT / "backtest_subperiods.csv", index_col=[0, 1])
S = json.loads((OUT / "summary.json").read_text())

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.4, alignment=TA_LEFT,
                      spaceAfter=5)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, alignment=TA_LEFT,
                    spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=8, spaceAfter=4)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#52514e"))
cell = ParagraphStyle("c", parent=body, fontSize=8.3, leading=10, spaceAfter=0)


def P(t, st=body):
    return Paragraph(t, st)


def table(rows, widths, bold_row=None):
    data = [[P(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecebe7")),
             ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#52514e")),
             ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d9d8d4")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if bold_row is not None:
        style.append(("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#e3eefb")))
    t.setStyle(TableStyle(style))
    return t


def img(name, width=17):
    from PIL import Image as PI
    w, h = PI.open(FIG / name).size
    return Image(str(FIG / name), width=width * cm, height=width * cm * h / w)


o, u = perf.loc["Original rule: buy when VIX < 13"], perf.loc["Unhedged"]
gb, ls = perf.loc["Gradient boosting: insurance looks cheap"], perf.loc["LSTM: insurance looks cheap"]
story = [
    P("Machine learning and India VIX: can a model time crash insurance better than VIX?", h1),
    P("FDRM course project, Group 8, IIT Jodhpur · github.com/Subham1709/VIX_ML · app: subham-vix-ml.streamlit.app", small),
    Spacer(1, 6),
    P("Question", h2),
    P("Our main project protected a Nifty 50 portfolio with a 3-month put, 5% below the market, bought only when "
      f"India VIX was below 13 and sold into volatility spikes. It returned {o.CAGR:.1f}% a year against "
      f"{u.CAGR:.1f}% unhedged, with a worst fall of {abs(o.maxDD):.0f}% instead of {abs(u.maxDD):.0f}%. This "
      "extension asks whether machine learning (gradient boosting) or deep learning (an LSTM) can time that "
      "insurance better. We tried two ideas: buy when a model expects next month's volatility to be at least "
      "what VIX charges (<i>insurance looks cheap</i>), and buy when a model's chance of a 7% fall within three "
      "months is above its long-run rate (<i>fall risk above normal</i>)."),
    P("Data and method", h2),
    P("Nifty 50 daily prices from September 2007 and India VIX from January 2020 to 13 April 2026 (the main "
      "project's files). Each day the models see 15 Nifty features known at the close (realised volatility over "
      "5 to 126 days, high-low range volatility, downside volatility, 1-week to 3-month returns, distance from "
      "the 1-year high and moving averages); VIX versions add 5 VIX features. The LSTM reads the last 63 days."),
    P("<b>No lookahead.</b> Forecasts are walk-forward: on the first trading day of each month every model is "
      "retrained on earlier data only (the LSTM every three months), and a day enters training only after its "
      "answer is known (21 days later for volatility, 63 for falls). The test runs from January 2020, after "
      "twelve years of training data. Model settings and the two buying thresholds were fixed before the "
      "backtest. Option pricing and trading rules are those of the main project; our engine reproduces its "
      "original-rule results exactly."),
    P("Forecasting volatility", h2),
]
rows = [["Forecast of next-month volatility (Jan 2022 to Mar 2026)", "Average error (vol pts)", "Bias (vol pts)",
         "Improvement on VIX", "Significance (t)"]]
for n in ["India VIX", "Last month's volatility", "HAR regression", "HAR + VIX regression",
          "Gradient boosting (Nifty only)", "Gradient boosting (Nifty + VIX)", "LSTM (Nifty only)"]:
    r = fm.loc[n]
    rows.append([n, f"{r.MAE_vol_pts:.2f}", f"{r.bias_vol_pts:+.2f}", f"{r.R2_vs_VIX:+.0%}" if n != "India VIX" else "",
                 f"{r.DM_t_vs_VIX:.2f}" if n != "India VIX" else ""])
story += [table(rows, [6.6 * cm, 2.6 * cm, 2.2 * cm, 2.8 * cm, 2.6 * cm]),
          P("Improvement on VIX: share of VIX's squared log error removed. Significance: Diebold-Mariano t-statistic "
            "with Newey-West errors (about 2 is needed).", small),
          P(f"The LSTM's error was {fm.loc['LSTM (Nifty only)', 'MAE_vol_pts']:.1f} points against "
            f"{fm.loc['India VIX', 'MAE_vol_pts']:.1f} for VIX, largely because VIX is on average "
            f"{fm.loc['India VIX', 'bias_vol_pts']:.1f} points too high (the volatility risk premium). A simple "
            "regression mixing recent volatility with VIX did as well; gradient boosting did not beat VIX. None of "
            "the improvements is statistically significant."),
          img("fig1_vol_forecasts.png", 16.5),
          P("Predicting falls", h2)]
rows = [["Chance of a 7% fall within 3 months (Jan 2022 to Mar 2026)", "AUC (0.5 = guessing)", "Brier score"]]
for n in ["Base rate", "Logistic on last month's vol", "Logistic on India VIX", "Logistic (all Nifty features)",
          "Gradient boosting (Nifty only)", "LSTM (Nifty only)"]:
    rows.append([n, f"{cm_.loc[n, 'AUC']:.2f}", f"{cm_.loc[n, 'Brier']:.3f}"])
story += [table(rows, [9.4 * cm, 3.6 * cm, 3.6 * cm]),
          Spacer(1, 4),
          P("No model ranks the days before a fall much better than chance. The logistic model on VIX, trained on "
            "2020-21, learned that high VIX precedes falls (true in the COVID crash) and was then wrong: after 2021 "
            "most falls began in calm markets, exactly when the original rule buys insurance.")]
rows = [["Strategy (Jan 2020 to Apr 2026)", "Return a year", "Worst fall", "Sharpe", "Insurance cost a year",
         "Puts", "Return from Jul 2020"]]
order = ["Unhedged", "Original rule: buy when VIX < 13", "Buy every month", "Gradient boosting: insurance looks cheap",
         "LSTM: insurance looks cheap", "Gradient boosting: fall risk above normal", "LSTM: fall risk above normal",
         "VIX < 13 or LSTM insurance cheap"]
for n in order:
    r = perf.loc[n]
    rows.append([n, f"{float(r.CAGR):.1f}%", f"{float(r.maxDD):.1f}%", f"{float(r.sharpe):.2f}",
                 f"{float(r.net_cost_pa):+.1f}%", f"{int(float(r.puts_bought))}",
                 f"{subp.loc[('2020-07-01', n), 'CAGR']:.1f}%"])
story += [KeepTogether([P("Timing the hedge", h2), table(rows, [6.3 * cm, 1.8 * cm, 1.8 * cm, 1.4 * cm, 2.2 * cm,
                                                                1.2 * cm, 2.1 * cm], bold_row=2),
                        P("All strategies use the same put and sell half at VIX 22, the rest when VIX falls 20% "
                          "from its peak. Negative cost: the puts earned more than they cost.", small)]),
          img("fig4_backtest.png", 15.5),
          P(f"The best pre-set ML rule, gradient boosting <i>insurance looks cheap</i>, returned {gb.CAGR:.1f}% "
            f"against {o.CAGR:.1f}% for the original rule, with a deeper worst fall ({gb.maxDD:.0f}% vs "
            f"{o.maxDD:.0f}%). The LSTM version did not buy before the COVID crash and ended level with no hedge "
            f"({ls.CAGR:.1f}%). Rules based on fall risk did worse than no hedge. Combining the original rule with "
            "the LSTM signal added 0.3 points a year, too little to trust given the fragility below."),
          img("fig5_threshold_sensitivity.png", 16),
          P("Moving a threshold by 0.05 changes returns by up to 4 points, because one put more or fewer in early "
            "2020 decides the result. From July 2020 the original and both 'cheap insurance' rules are within about "
            "half a point a year of holding Nifty unhedged."),
          P("Conclusion", h2),
          P("Machine learning produced slightly better volatility forecasts than VIX, but no reliable improvement in "
            "predicting falls or in timing the insurance. The simple rule of buying puts when VIX says they are "
            "cheap remains the best performer, which supports the main project's conclusion: what matters is the "
            "price of insurance, not a forecast of the next crash."),
          P("Limitations", h2),
          P("Option prices are modelled from VIX rather than traded prices. VIX starts in 2020, so VIX-based models "
            "have one to four years of training data. Six years contain one real crash, and overlapping 3-month "
            "windows hold far fewer independent observations than the 1,500 test days suggest. Trades are at the "
            "close on the day of the forecast, as in the main project."),
          ]

doc = SimpleDocTemplate(str(ROOT / "docs" / "fdrm_vix_ml_summary.pdf"), pagesize=A4, leftMargin=2 * cm,
                        rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                        title="Machine learning and India VIX", author="Group 8, IIT Jodhpur")


def footer(canvas, doc_):
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#8a8984"))
    canvas.drawString(2 * cm, 1 * cm, "FDRM course project, Group 8, IIT Jodhpur")
    canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, str(doc_.page))


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("pdf written")
