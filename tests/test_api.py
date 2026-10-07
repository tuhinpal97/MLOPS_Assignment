from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from fastapi.testclient import TestClient
import src.api.main as api


def test_predict_contract(monkeypatch):
    fake = MagicMock()
    fake.predict.return_value = np.array([1])
    fake.predict_proba.return_value = np.array([[0.2, 0.8]])
    monkeypatch.setattr(api, "model", fake)
    monkeypatch.setattr(api, "metadata", {"model_name": "test-model"})
    # Avoid startup loading a real model artifact.
    api.app.router.on_startup.clear()
    client = TestClient(api.app)
    payload = {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
               "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
               "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1
    assert body["confidence"] == 0.8
