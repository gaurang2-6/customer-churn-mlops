from fastapi.testclient import TestClient

from churn.api import Customer, app

client = TestClient(app)
EXAMPLE = Customer.model_config["json_schema_extra"]["example"]
LOYAL = {**EXAMPLE, "tenure": 60, "Contract": "Two year", "InternetService": "DSL", "OnlineSecurity": "Yes",
         "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 60.0,
         "TotalCharges": 3600.0}


def test_health_reports_the_model():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_a_probability_and_a_decision():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["send_offer"] == (body["expected_value_of_offer"] > 0)


def test_new_month_to_month_fibre_customer_is_riskier_than_a_two_year_one():
    risky, loyal = (client.post("/predict", json=c).json() for c in (EXAMPLE, LOYAL))
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["risk_band"] == "high" and loyal["risk_band"] == "low"


def test_batch_matches_single_predictions():
    batch = client.post("/predict/batch", json={"customers": [EXAMPLE, LOYAL]}).json()
    assert batch == [client.post("/predict", json=c).json() for c in (EXAMPLE, LOYAL)]


def test_invalid_input_is_rejected_with_422():
    assert client.post("/predict", json={**EXAMPLE, "Contract": "Weekly"}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "tenure": -3}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in EXAMPLE.items() if k != "gender"}).status_code == 422
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
