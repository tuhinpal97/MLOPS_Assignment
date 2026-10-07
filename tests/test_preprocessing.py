import pandas as pd
from src.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_and_categorical_values():
    X = pd.DataFrame([
        {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
         "ca": None, "thal": 6},
        {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0,
         "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1,
         "ca": 0, "thal": 3},
    ])
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == 2
    assert Xt.shape[1] > X.shape[1]
