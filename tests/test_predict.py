# tests/test_predict.py
import os
# Ensure CI mode before importing the app so startup uses DummyModel/scaler
os.environ["CI"] = "true"

from fastapi.testclient import TestClient
from api.main import app, load_model_and_meta

load_model_and_meta()

client = TestClient(app)

single_payload = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
}

def test_predict_single():
    r = client.post("/predict", json=single_payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data and "probability" in data
    assert isinstance(data["probability"], float)

def test_predict_batch():
    payload = [single_payload, single_payload]
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "count" in data and data["count"] == 2
    assert "predictions" in data and len(data["predictions"]) == 2
