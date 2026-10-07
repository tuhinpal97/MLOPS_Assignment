from __future__ import annotations

from pathlib import Path
import urllib.request
import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
FEATURES = COLUMNS[:-1]
TARGET = "target"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
MIRROR_URL = "https://raw.githubusercontent.com/SKawsar/Machine-Learning-With-Python/main/processed.cleveland.data"


def download_raw(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    last_error = None
    for url in (UCI_URL, MIRROR_URL):
        try:
            urllib.request.urlretrieve(url, output)
            return output
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
    raise RuntimeError(f"Unable to download dataset from UCI or mirror: {last_error}")


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS, na_values="?")
    if list(df.columns) != COLUMNS or df.shape[1] != 14:
        raise ValueError("Unexpected UCI Heart Disease schema")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned[TARGET] = (cleaned["num"] > 0).astype(int)
    cleaned = cleaned.drop(columns=["num"])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def prepare_dataset(raw_path: str | Path, processed_path: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    download_raw(raw_path)
    cleaned = clean_dataframe(load_raw(raw_path))
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_path, index=False)
    return cleaned
