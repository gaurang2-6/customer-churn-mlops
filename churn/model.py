"""Feature engineering, model pipelines and the retention-campaign profit model."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.data import CATEGORIES, INTERNET_ADDONS, NUMERIC

ENGINEERED = ["services", "charge_ratio", "new_customer"]

# Retention offer economics (assumptions, stated in the README and editable here).
OFFER_COST = 60.0         # cost of one retention offer: a discount plus the agent's call
SAVE_RATE = 0.30          # share of would-be churners the offer keeps
MONTHS_SAVED = 12         # revenue kept per saved customer, in months of their current bill


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["services"] = (df[INTERNET_ADDONS] == "Yes").sum(axis=1) + (df.PhoneService == "Yes")
    # Current bill vs. their average bill so far: > 1 means the price has gone up.
    df["charge_ratio"] = df.MonthlyCharges / (df.TotalCharges / df.tenure.clip(lower=1)).replace(0, np.nan)
    df["charge_ratio"] = df.charge_ratio.fillna(1.0)
    df["new_customer"] = (df.tenure <= 6).astype(int)
    return df


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric = NUMERIC + ENGINEERED
    return ColumnTransformer([
        ("cat", OneHotEncoder(categories=[CATEGORIES[c] for c in CATEGORIES], handle_unknown="ignore",
                              sparse_output=False), list(CATEGORIES)),
        ("num", StandardScaler() if scale else "passthrough", numeric),
    ], verbose_feature_names_out=False)


def build(kind: str, **params) -> Pipeline:
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, C=params.get("C", 1.0))
    elif kind == "xgboost":
        defaults = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
                    "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}
        clf = XGBClassifier(**{**defaults, **params}, eval_metric="logloss", random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("prep", preprocessor(scale=kind == "logistic")),
                     ("clf", clf)])


def campaign_value(y_true, monthly_charges, targeted, save_rate=SAVE_RATE, offer_cost=OFFER_COST) -> float:
    """Net value of sending the offer to `targeted` customers, scored against who actually churned.

    Each targeted churner is saved with probability SAVE_RATE, keeping MONTHS_SAVED months of their bill;
    every targeted customer costs OFFER_COST whether or not they would have left.
    """
    y, mc, t = (np.asarray(a, dtype=float) for a in (y_true, monthly_charges, targeted))
    return float(np.sum(t * (y * save_rate * MONTHS_SAVED * mc - offer_cost)))


def expected_value(proba, monthly_charges, save_rate=SAVE_RATE, offer_cost=OFFER_COST):
    """Expected value of targeting each customer, given the model's churn probability."""
    return np.asarray(proba) * save_rate * MONTHS_SAVED * np.asarray(monthly_charges) - offer_cost


def value_curve(y_true, proba, monthly_charges, steps=101) -> pd.DataFrame:
    """Campaign value when targeting the top x% of customers ranked by churn probability."""
    order = np.argsort(-np.asarray(proba))
    y, mc = np.asarray(y_true)[order], np.asarray(monthly_charges)[order]
    per_customer = y * SAVE_RATE * MONTHS_SAVED * mc - OFFER_COST
    cum = np.r_[0, np.cumsum(per_customer)]
    share = np.linspace(0, 1, steps)
    k = np.round(share * len(y)).astype(int)
    return pd.DataFrame({"share_targeted": share, "customers": k, "value": cum[k],
                         "churners_caught": np.r_[0, np.cumsum(y)][k] / max(y.sum(), 1)})
