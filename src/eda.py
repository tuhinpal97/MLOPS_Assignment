from __future__ import annotations

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from src.data import prepare_dataset


def run_eda(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histograms for clinically interpretable continuous variables.
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna(), bins=15, edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    axes.flat[-1].axis("off")
    fig.suptitle("UCI Heart Disease - Feature Distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=180)
    plt.close(fig)

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No disease (0)", "Disease (1)"], counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center")
    ax.set_ylabel("Patients")
    ax.set_title("Class Balance")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=180)
    plt.close(fig)

    missing = df.isna().sum().sort_values(ascending=False)
    missing.to_csv(out_dir / "missing_values.csv", header=["missing_count"])
    df.describe(include="all").transpose().to_csv(out_dir / "summary_statistics.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/processed.cleveland.data")
    parser.add_argument("--processed", default="data/processed/heart_clean.csv")
    parser.add_argument("--out", default="artifacts/plots")
    args = parser.parse_args()
    df = prepare_dataset(args.raw, args.processed)
    run_eda(df, Path(args.out))
    print(f"EDA complete: {len(df)} rows, target prevalence={df.target.mean():.3f}")


if __name__ == "__main__":
    main()
