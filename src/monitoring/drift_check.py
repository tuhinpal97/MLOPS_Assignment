from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05):
    report = {}
    for col in NUMERIC:
        a, b = reference[col].dropna(), current[col].dropna()
        stat, p = ks_2samp(a, b)
        report[col] = {"ks_statistic": float(stat), "p_value": float(p), "drift": bool(p < alpha)}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="data/processed/heart_clean.csv")
    p.add_argument("--current", required=True)
    p.add_argument("--out", default="artifacts/metrics/drift_report.json")
    args = p.parse_args()
    report = drift_report(pd.read_csv(args.reference), pd.read_csv(args.current))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
