from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402

if __name__ == "__main__":
    df = prepare_dataset("data/raw/processed.cleveland.data", "data/processed/heart_clean.csv")
    print(f"Prepared {len(df)} rows at data/processed/heart_clean.csv")
