# tests/test_health.py
import os
# Ensure CI mode before importing the app so startup uses DummyModel/scaler
os.environ["CI"] = "true"

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
