"""Write reports/results.md and the charts in reports/figures/ from a training run."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from churn import model

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GREY = "#a3a29e"
LABELS = {"logistic": "Logistic regression", "xgboost": "XGBoost (default)", "xgboost_tuned": "XGBoost (tuned)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


def fig_roc(r):
    y = r["test"].Churn
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"logistic": GREY, "xgboost": ORANGE, "xgboost_tuned": BLUE}
    for run in r["runs"]:
        fpr, tpr, _ = roc_curve(y, run["p_test"])
        ax.plot(fpr, tpr, color=colors[run["name"]],
                label=f"{LABELS[run['name']]}  AUC {run['metrics']['test_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (churners caught)", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    ax.set_title("All three models rank churners about equally well", pad=12)
    save(fig, "roc_curves")


def fig_value(r):
    test, p = r["test"], r["chosen"]["p_test"]
    curve = model.value_curve(test.Churn, p, test.MonthlyCharges)
    pol = r["policies"].set_index("policy")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(curve.share_targeted * 100, curve.value / 1000, color=BLUE, label="Model ranking: target the top x%")
    ax.axhline(0, color=INK2, lw=1)
    marks = [("Model: expected value > 0", BLUE, "Model rule\n(expected value > 0)", (8, 8)),
             ("All month-to-month contracts", ORANGE, "All month-to-month", (8, -22)),
             ("Everyone", GREY, "Everyone", (-62, -18))]
    for name, color, text, offset in marks:
        row = pol.loc[name]
        ax.plot(row.share_targeted * 100, row.net_value / 1000, "o", ms=9, color=color, mec=SURFACE, mew=2)
        ax.annotate(text, (row.share_targeted * 100, row.net_value / 1000), xytext=offset,
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set(xlabel="Customers targeted (% of test set, highest churn risk first)",
           ylabel="Net campaign value (thousands)", xlim=(0, 100),
           ylim=(-2, max(curve.value.max(), pol.net_value.max()) / 1000 * 1.15))
    ax.legend(loc="lower left")
    ax.set_title(f"Model targeting earns {r['uplift']['relative']:.0%} more than the contract rule", pad=12)
    save(fig, "campaign_value")


def fig_importance(r):
    imp = r["importance"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(imp.feature, imp.drop_in_auc, xerr=imp["std"], color=BLUE, height=0.6, ecolor=INK2,
            error_kw={"lw": 1})
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Drop in test ROC-AUC when the column is shuffled")
    ax.set_title("Contract, tenure and internet type drive the predictions", pad=12)
    save(fig, "feature_importance")


def fig_calibration(r):
    y = r["test"].Churn
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for run, color in [(r["chosen"], BLUE)]:
        frac, mean = calibration_curve(y, run["p_test"], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", color=color, ms=6, label=LABELS[run["name"]])
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Perfect calibration")
    ax.set(xlabel="Predicted churn probability (decile mean)", ylabel="Observed churn rate", xlim=(0, 1),
           ylim=(0, 1))
    ax.legend(loc="upper left")
    ax.set_title("Predicted probabilities match observed churn", pad=12)
    save(fig, "calibration")


def table(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.get(c, str)(v) for c, v in zip(cols, row)) + " |")
    return "\n".join(lines)


def write(r):
    FIG.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    models = pd.DataFrame([{"model": LABELS[x["name"]], **{k: v for k, v in x["metrics"].items()}}
                           for x in r["runs"]])
    f4 = lambda v: f"{v:.4f}"
    pct = lambda v: f"{v:.1%}"
    money = lambda v: f"{v:,.0f}"
    meta = r["meta"]
    tuned = {k.replace("clf__", ""): round(v, 4) if isinstance(v, float) else v
             for k, v in r["chosen"]["params"].items()}
    out = ["# Results (generated by `python -m churn.train`)", "",
           f"- Customers: {len(r['df']):,} ({len(r['train']):,} train, {len(r['test']):,} test); "
           f"churn rate {r['df'].Churn.mean():.1%}",
           f"- Chosen model: **{LABELS[meta['model']]}** (highest cross-validated ROC-AUC), "
           f"MLflow run `{meta['mlflow_run_id']}`",
           f"- Tuned parameters: {tuned}", "",
           "## Model comparison (5-fold CV on train, then the untouched test set)", "",
           table(models, {k: f4 for k in models.columns if k != "model"}), "",
           "## Retention campaign on the test set", "",
           f"Assumptions: offer costs {model.OFFER_COST:.0f} per customer, keeps {model.SAVE_RATE:.0%} of "
           f"would-be churners, each saved customer is worth {model.MONTHS_SAVED} months of their bill.", "",
           table(r["policies"], {"share_targeted": pct, "churners_caught": pct,
                                 "precision": lambda v: "–" if v != v else pct(v), "net_value": money}), "",
           f"Model rule minus month-to-month rule: {money(r['uplift']['difference'])} "
           f"({r['uplift']['relative']:+.0%}), bootstrap 95% CI {money(r['uplift']['ci_low'])} to "
           f"{money(r['uplift']['ci_high'])}; positive in {r['uplift']['share_positive']:.0%} of resamples", "",
           "## Sensitivity to the campaign assumptions", "",
           table(r["sensitivity"], {"save_rate": pct, "offer_cost": money, "model_rule": money,
                                    "month_to_month_rule": money, "model_minus_rule": money}), "",
           "## Permutation importance (test set, drop in ROC-AUC)", "",
           table(r["importance"].head(10), {"drop_in_auc": f4, "std": f4}), ""]
    (ROOT / "reports" / "results.md").write_text("\n".join(out), encoding="utf-8")
    fig_roc(r)
    fig_value(r)
    fig_importance(r)
    fig_calibration(r)
    print("Wrote reports/results.md and", len(list(FIG.glob("*.png"))), "figures")
