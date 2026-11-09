"""Download, clean and validate the IBM Telco Customer Churn dataset (cached in data/)."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# IBM's own copy of the Kaggle "Telco Customer Churn" file.
SOURCE = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
          "data/Telco-Customer-Churn.csv")
FILE = "telco_churn.csv"

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
YES_NO = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                   "StreamingMovies"]

# Allowed values per categorical column. The same lists drive the data checks and the API schema.
CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    **{c: ["No", "Yes"] for c in YES_NO},
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: ["No", "No internet service", "Yes"] for c in INTERNET_ADDONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check",
                      "Mailed check"],
}
FEATURES = list(CATEGORIES) + NUMERIC


def fetch() -> Path:
    path = DATA_DIR / FILE
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Downloading {FILE}")
        urlretrieve(SOURCE, path)
    return path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the one known defect: TotalCharges is text, blank for the 11 customers with tenure 0."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df.TotalCharges.astype(str).str.strip().replace("", "0"))
    if TARGET in df:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on data that would silently break the model (the checks a data contract would run)."""
    problems = []
    missing = [c for c in FEATURES if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[FEATURES].isna().any().any():
        problems.append(f"nulls in {df[FEATURES].columns[df[FEATURES].isna().any()].tolist()}")
    for col, allowed in CATEGORIES.items():
        bad = set(df[col].unique()) - set(allowed)
        if bad:
            problems.append(f"{col} has unexpected values {sorted(map(str, bad))}")
    if (df[NUMERIC] < 0).any().any():
        problems.append("negative tenure or charges")
    if (df.tenure > 120).any():
        problems.append("tenure above 120 months")
    if "customerID" in df and df.customerID.duplicated().any():
        problems.append("duplicate customerID")
    if TARGET in df and not set(df[TARGET].unique()) <= {0, 1}:
        problems.append("target is not 0/1")
    if problems:
        raise ValueError("data validation failed: " + "; ".join(problems))
    return df


def load() -> pd.DataFrame:
    return validate(clean(pd.read_csv(fetch())))


def split(df: pd.DataFrame, seed: int = 42):
    """80/20 stratified split. The test set is touched once, for the final numbers."""
    return train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=seed)
