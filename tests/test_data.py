import pandas as pd
from src.data import COLUMNS, clean_dataframe


def test_clean_dataframe_binarizes_target_and_preserves_missing_features():
    row = [63, 1, 4, 145, 233, 1, 2, 150, 0, 2.3, 3, None, 6, 2]
    df = pd.DataFrame([row], columns=COLUMNS)
    out = clean_dataframe(df)
    assert out.loc[0, "target"] == 1
    assert "num" not in out.columns
    assert pd.isna(out.loc[0, "ca"])


def test_no_disease_maps_to_zero():
    row = [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 3, 0]
    out = clean_dataframe(pd.DataFrame([row], columns=COLUMNS))
    assert out.loc[0, "target"] == 0
