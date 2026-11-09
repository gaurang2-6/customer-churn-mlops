import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from churn import data, model
from churn.train import MODEL_DIR

# The deployed model must clear this on the held-out test set, or CI fails.
MIN_TEST_AUC = 0.82


@pytest.fixture(scope="module")
def df():
    return data.load()


def test_data_passes_validation_and_has_no_text_charges(df):
    assert len(df) == 7043 and df.TotalCharges.dtype.kind == "f"
    assert df.loc[df.tenure == 0, "TotalCharges"].eq(0).all()


@pytest.mark.parametrize("breakage, message", [
    (lambda d: d.assign(Contract="Weekly"), "Contract has unexpected values"),
    (lambda d: d.assign(tenure=-1), "negative"),
    (lambda d: d.drop(columns="tenure"), "missing columns"),
    (lambda d: pd.concat([d, d.head(1)]), "duplicate customerID"),
])
def test_validation_rejects_bad_data(df, breakage, message):
    with pytest.raises(ValueError, match=message):
        data.validate(breakage(df))


def test_split_is_stratified(df):
    train, test = data.split(df)
    assert abs(train.Churn.mean() - test.Churn.mean()) < 0.005 and len(test) == round(0.2 * len(df))


def test_saved_model_clears_the_quality_gate(df):
    _, test = data.split(df)
    pipe = joblib.load(MODEL_DIR / "model.joblib")
    auc = roc_auc_score(test.Churn, pipe.predict_proba(test[data.FEATURES])[:, 1])
    assert auc >= MIN_TEST_AUC
    assert np.isclose(auc, json.loads((MODEL_DIR / "metadata.json").read_text())["metrics"]["test_roc_auc"],
                      atol=1e-3)


def test_features_handle_new_customers_without_dividing_by_zero(df):
    f = model.add_features(df[df.tenure == 0])
    assert np.isfinite(f.charge_ratio).all()


def test_campaign_value_by_hand():
    # One churner at 50/month and one loyal customer, both targeted:
    # 0.3 × 12 × 50 − 60 for the churner, −60 for the other.
    got = model.campaign_value([1, 0], [50, 80], [1, 1])
    assert np.isclose(got, model.SAVE_RATE * model.MONTHS_SAVED * 50 - 2 * model.OFFER_COST)


def test_value_curve_ends_at_targeting_everyone():
    y, p, mc = np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.6, 0.1]), np.array([70, 20, 90, 30])
    curve = model.value_curve(y, p, mc, steps=5)
    assert curve.value.iloc[0] == 0
    assert np.isclose(curve.value.iloc[-1], model.campaign_value(y, mc, np.ones(4)))
