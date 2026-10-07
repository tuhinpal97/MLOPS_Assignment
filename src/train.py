from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data import prepare_dataset
from src.preprocessing import build_preprocessor


def _mlflow():
    import mlflow
    import mlflow.sklearn

    return mlflow


def build_searches(random_state: int = 42):
    pre = build_preprocessor()
    return {
        "logistic_regression": GridSearchCV(
            Pipeline(
                [
                    ("preprocess", clone(pre)),
                    ("model", LogisticRegression(max_iter=3000, random_state=random_state)),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
            scoring="roc_auc",
            cv=5,
            n_jobs=1,
            refit=True,
        ),
        "random_forest": GridSearchCV(
            Pipeline(
                [
                    ("preprocess", clone(pre)),
                    ("model", RandomForestClassifier(random_state=random_state, n_jobs=1)),
                ]
            ),
            {
                "model__n_estimators": [200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced"],
            },
            scoring="roc_auc",
            cv=5,
            n_jobs=1,
            refit=True,
        ),
    }


def evaluate_cv(model, X, y) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
    return {f"cv_{m}_mean": float(np.mean(scores[f"test_{m}"])) for m in scoring}


def evaluate_holdout(model, X_test, y_test) -> dict[str, float]:
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
    }


def save_plots(model, X_test, y_test, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve - {prefix}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_roc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"Confusion Matrix - {prefix}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/processed.cleveland.data")
    p.add_argument("--processed", default="data/processed/heart_clean.csv")
    p.add_argument("--model-dir", default="artifacts/model")
    p.add_argument("--metrics-dir", default="artifacts/metrics")
    p.add_argument("--plots-dir", default="artifacts/plots")
    p.add_argument("--experiment", default="heart-disease-classification")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    args = p.parse_args()

    df = prepare_dataset(args.raw, args.processed)
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model_dir, metrics_dir, plots_dir = map(
        Path,
        [args.model_dir, args.metrics_dir, args.plots_dir],
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    mlflow = _mlflow()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    summaries = {}
    champion = None
    champion_auc = -1.0

    for name, search in build_searches().items():
        started = time.time()
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            best = search.best_estimator_
            cv_metrics = evaluate_cv(best, X_train, y_train)
            test_metrics = evaluate_holdout(best, X_test, y_test)
            metrics = {
                **cv_metrics,
                **test_metrics,
                "fit_seconds": time.time() - started,
            }
            summaries[name] = {"best_params": search.best_params_, **metrics}
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_param("dataset_rows", len(df))
            mlflow.log_param("random_state", 42)
            save_plots(best, X_test, y_test, plots_dir, name)
            for plot in plots_dir.glob(f"{name}_*.png"):
                mlflow.log_artifact(str(plot), artifact_path="plots")
            mlflow.sklearn.log_model(best, artifact_path="model")
            if metrics["cv_roc_auc_mean"] > champion_auc:
                champion_auc = metrics["cv_roc_auc_mean"]
                champion = (name, best, search.best_params_, metrics)

    assert champion is not None
    champ_name, champ_model, champ_params, champ_metrics = champion

    # After model selection, refit the chosen pipeline on all cleaned data for serving.
    production_model = clone(champ_model).fit(X, y)
    joblib.dump(production_model, model_dir / "model.joblib")

    metadata = {
        "model_name": champ_name,
        "selection_metric": "5-fold cross-validation ROC-AUC on training split",
        "best_params": champ_params,
        "metrics": champ_metrics,
        "refit_for_production": True,
        "features": list(X.columns),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (metrics_dir / "all_models.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
