"""FastAPI service: score customers for churn risk and say whether a retention offer pays.

    uvicorn churn.api:app --reload      # then open http://localhost:8000/docs
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from churn import model

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

YesNo = Literal["No", "Yes"]
Addon = Literal["No", "No internet service", "Yes"]


class Customer(BaseModel):
    """One customer, in the same columns as the Telco file. Invalid values are rejected with a 422."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 2,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0}})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Addon
    OnlineBackup: Addon
    DeviceProtection: Addon
    TechSupport: Addon
    StreamingTV: Addon
    StreamingMovies: Addon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check",
                           "Mailed check"]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    risk_band: Literal["low", "medium", "high"]
    expected_value_of_offer: float = Field(description="Expected net value of sending the retention offer")
    send_offer: bool


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


@lru_cache
def load():
    return joblib.load(MODEL_DIR / "model.joblib"), json.loads((MODEL_DIR / "metadata.json").read_text())


def score(customers: list[Customer]) -> list[Prediction]:
    pipe, _ = load()
    df = pd.DataFrame([c.model_dump() for c in customers])
    proba = pipe.predict_proba(df)[:, 1]
    ev = model.expected_value(proba, df.MonthlyCharges)
    return [Prediction(churn_probability=round(float(p), 4),
                       risk_band="high" if p >= 0.5 else "medium" if p >= 0.2 else "low",
                       expected_value_of_offer=round(float(v), 2), send_offer=bool(v > 0))
            for p, v in zip(proba, ev)]


app = FastAPI(title="Telco churn API", version="1.0.0",
              description="Churn probability and a send/don't-send retention decision per customer.")


@app.get("/health")
def health():
    _, meta = load()
    return {"status": "ok", "model": meta["model"], "trained_at": meta["trained_at"],
            "test_roc_auc": meta["metrics"]["test_roc_auc"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch):
    return score(batch.customers)
