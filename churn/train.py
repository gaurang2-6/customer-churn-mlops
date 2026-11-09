"""Train and compare churn models, track every run in MLflow, save the chosen model for the API.

    python -m churn.train            # full run: baseline, XGBoost, tuned XGBoost (about a minute)
    python -m churn.train --quick    # smaller search, used in CI
"""
import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from churn import data, model

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
EXPERIMENT = "telco-churn"
# Our own feature function and XGBoost classes, which MLflow's safe (skops) format must be told to load.
TRUSTED_TYPES = ["churn.model.add_features", "xgboost.sklearn.XGBClassifier", "xgboost.core.Booster"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SEARCH_SPACE = {
    "clf__n_estimators": randint(150, 700),
    "clf__learning_rate": loguniform(0.01, 0.15),
    "clf__max_depth": randint(2, 6),
    "clf__min_child_weight": randint(1, 10),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.5, 0.5),
    "clf__reg_lambda": loguniform(0.1, 20),
}


def metrics(y, p, prefix):
    return {f"{prefix}_roc_auc": roc_auc_score(y, p), f"{prefix}_pr_auc": average_precision_score(y, p),
            f"{prefix}_brier": brier_score_loss(y, p), f"{prefix}_log_loss": log_loss(y, p)}


def evaluate(name, pipe, params, train, test):
    """Cross-validate on train, refit on all of train, score the test set once; log it all to MLflow."""
    X, y = train[data.FEATURES], train[data.TARGET]
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": name, **{k.replace("clf__", ""): v for k, v in params.items()}})
        oof = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]
        fitted = clone(pipe).fit(X, y)
        p_test = fitted.predict_proba(test[data.FEATURES])[:, 1]
        m = {**metrics(y, oof, "cv"), **metrics(test[data.TARGET], p_test, "test")}
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(fitted, name="model", input_example=X.head(3),
                                 skops_trusted_types=TRUSTED_TYPES)
        run_id = mlflow.active_run().info.run_id
    print(f"{name:<16} CV ROC-AUC {m['cv_roc_auc']:.4f}   test ROC-AUC {m['test_roc_auc']:.4f}")
    return {"name": name, "pipe": fitted, "params": params, "oof": oof, "p_test": p_test, "metrics": m,
            "run_id": run_id}


def tune(train, n_iter):
    search = RandomizedSearchCV(model.build("xgboost"), SEARCH_SPACE, n_iter=n_iter, scoring="roc_auc", cv=CV,
                                random_state=42, n_jobs=-1)
    search.fit(train[data.FEATURES], train[data.TARGET])
    return {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in search.best_params_.items()}


def policies(test, p):
    """Campaign value on the test set for simple targeting rules vs. the model's expected-value rule."""
    y, mc = test[data.TARGET].values, test.MonthlyCharges.values
    ev = model.expected_value(p, mc)
    rules = {
        "Nobody (do nothing)": np.zeros(len(test), bool),
        "Everyone": np.ones(len(test), bool),
        "All month-to-month contracts": (test.Contract == "Month-to-month").values,
        "Model: top 20% by churn risk": p >= np.quantile(p, 0.8),
        "Model: expected value > 0": ev > 0,
    }
    rows = [{"policy": k, "customers_targeted": int(t.sum()), "share_targeted": t.mean(),
             "churners_caught": y[t].sum() / y.sum(), "precision": y[t].mean() if t.any() else np.nan,
             "net_value": model.campaign_value(y, mc, t)} for k, t in rules.items()]
    return pd.DataFrame(rows)


def uplift(test, p, n_boot=2000, seed=0):
    """Bootstrap the test customers: how much more the model rule earns than the month-to-month rule."""
    y, mc = test[data.TARGET].values, test.MonthlyCharges.values
    per_model = (model.expected_value(p, mc) > 0) * (y * model.SAVE_RATE * model.MONTHS_SAVED * mc - model.OFFER_COST)
    m2m = (test.Contract == "Month-to-month").values
    per_rule = m2m * (y * model.SAVE_RATE * model.MONTHS_SAVED * mc - model.OFFER_COST)
    diff = per_model - per_rule
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(n_boot)])
    return {"difference": diff.sum(), "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975),
            "share_positive": (boots > 0).mean(), "relative": diff.sum() / per_rule.sum()}


def sensitivity(test, p):
    """Does the model rule still beat the month-to-month rule if the campaign assumptions are wrong?"""
    y, mc = test[data.TARGET].values, test.MonthlyCharges.values
    m2m = (test.Contract == "Month-to-month").values
    rows = []
    for save_rate in (0.15, 0.30, 0.45):
        for cost in (30.0, 60.0, 120.0):
            kw = {"save_rate": save_rate, "offer_cost": cost}
            target = model.expected_value(p, mc, **kw) > 0
            rows.append({"save_rate": save_rate, "offer_cost": cost, "targeted": int(target.sum()),
                         "model_rule": model.campaign_value(y, mc, target, **kw),
                         "month_to_month_rule": model.campaign_value(y, mc, m2m, **kw)})
    out = pd.DataFrame(rows)
    out["model_minus_rule"] = out.model_rule - out.month_to_month_rule
    return out


def importance(pipe, test):
    r = permutation_importance(pipe, test[data.FEATURES], test[data.TARGET], scoring="roc_auc", n_repeats=10,
                               random_state=42, n_jobs=-1)
    return (pd.DataFrame({"feature": data.FEATURES, "drop_in_auc": r.importances_mean, "std": r.importances_std})
            .sort_values("drop_in_auc", ascending=False).reset_index(drop=True))


def main(quick=False):
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    df = data.load()
    train, test = data.split(df)
    print(f"{len(train):,} train / {len(test):,} test customers, churn rate {df.Churn.mean():.1%}")

    best_params = tune(train, n_iter=6 if quick else 30)
    runs = [
        evaluate("logistic", model.build("logistic"), {"C": 1.0}, train, test),
        evaluate("xgboost", model.build("xgboost"), {}, train, test),
        evaluate("xgboost_tuned", model.build("xgboost").set_params(**best_params), best_params, train, test),
    ]
    # Choose on cross-validation only; the test set is reported, never used to pick.
    chosen = max(runs, key=lambda r: r["metrics"]["cv_roc_auc"])
    print(f"Chosen: {chosen['name']} (run {chosen['run_id']})")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(chosen["pipe"], MODEL_DIR / "model.joblib")
    meta = {"model": chosen["name"], "mlflow_run_id": chosen["run_id"],
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_rows": len(train), "features": data.FEATURES,
            "metrics": {k: round(v, 4) for k, v in chosen["metrics"].items()},
            "economics": {"offer_cost": model.OFFER_COST, "save_rate": model.SAVE_RATE,
                          "months_saved": model.MONTHS_SAVED}}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"df": df, "train": train, "test": test, "runs": runs, "chosen": chosen, "meta": meta,
            "policies": policies(test, chosen["p_test"]), "uplift": uplift(test, chosen["p_test"]),
            "sensitivity": sensitivity(test, chosen["p_test"]), "importance": importance(chosen["pipe"], test)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    from churn import report
    report.write(main(quick=args.quick))
