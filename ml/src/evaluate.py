"""
HeatWave Guardian — Hyperlocal ML Model Evaluation Module
=========================================================
Performs out-of-sample evaluation on temporal holdouts and baseline comparisons.
Reports:
- MAE, RMSE, Mean Bias, and R² evaluated directly on the localized temperature residual.
- Comparative delta improvements over Zero-Anomaly and Climatological Baselines.
- Calibrated 90% empirical prediction intervals.
- Per-city holdout performance and feature importances.
"""

import os
import sys
import json
import joblib
from typing import Dict, Any
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.src.data_loader import load_historical_dataset, DEFAULT_CSV_PATH
from ml.src.features import prepare_feature_matrix, FEATURE_NAMES
from ml.src.baseline import (
    calculate_metrics,
    calculate_baseline_improvements,
    ZeroAnomalyBaseline,
    ClimatologicalBiasBaseline,
    LinearRegressionBaseline,
)

MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "ml", "reports")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "hyperlocal_rf_model.joblib")
BASELINE_FILE_PATH = os.path.join(MODELS_DIR, "baseline_model.joblib")
METADATA_FILE_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
EVALUATION_REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_metrics.json")


def run_evaluation(csv_path: str = DEFAULT_CSV_PATH) -> Dict[str, Any]:
    """Load model and dataset, re-evaluate on test set and return full metrics report."""
    if not os.path.exists(MODEL_FILE_PATH):
        from ml.src.train import train_and_validate
        return train_and_validate(csv_path)

    df = load_historical_dataset(csv_path)
    X, y_residual, y_absolute = prepare_feature_matrix(df)

    # Temporal holdout test set (2024 heat season)
    train_mask = df["date"] < "2024-04-01"
    test_mask = df["date"] >= "2024-04-01"
    X_train, X_test = X[train_mask], X[test_mask]
    y_train_res, y_test_res = y_residual[train_mask], y_residual[test_mask]
    y_test_abs = y_absolute[test_mask]
    df_test = df[test_mask]
    bg_test = X_test["background_temp_c"].to_numpy()

    rf_model = joblib.load(MODEL_FILE_PATH)
    linear_baseline = joblib.load(BASELINE_FILE_PATH)
    zero_baseline = ZeroAnomalyBaseline()
    climatology_baseline = ClimatologicalBiasBaseline().fit(X_train, y_train_res)

    pred_res_rf = rf_model.predict(X_test)
    pred_res_linear = linear_baseline.predict_residual(X_test)
    pred_res_zero = zero_baseline.predict_residual(X_test)
    pred_res_clim = climatology_baseline.predict_residual(X_test)

    # Metrics on the primary target: localized temperature residual (Delta T)
    metrics_res_rf = calculate_metrics(y_test_res, pred_res_rf)
    metrics_res_linear = calculate_metrics(y_test_res, pred_res_linear)
    metrics_res_zero = calculate_metrics(y_test_res, pred_res_zero)
    metrics_res_clim = calculate_metrics(y_test_res, pred_res_clim)

    # Metrics on absolute temperature (T_bg + Delta T)
    y_pred_abs_rf = bg_test + pred_res_rf
    y_pred_abs_linear = bg_test + pred_res_linear
    y_pred_abs_zero = bg_test + pred_res_zero
    y_pred_abs_clim = bg_test + pred_res_clim

    metrics_abs_rf = calculate_metrics(y_test_abs, y_pred_abs_rf)
    metrics_abs_linear = calculate_metrics(y_test_abs, y_pred_abs_linear)
    metrics_abs_zero = calculate_metrics(y_test_abs, y_pred_abs_zero)
    metrics_abs_clim = calculate_metrics(y_test_abs, y_pred_abs_clim)

    # Baseline improvements
    impr_over_zero = calculate_baseline_improvements(metrics_res_rf, metrics_res_zero)
    impr_over_clim = calculate_baseline_improvements(metrics_res_rf, metrics_res_clim)
    impr_over_linear = calculate_baseline_improvements(metrics_res_rf, metrics_res_linear)

    # Empirical uncertainty intervals
    abs_errors = np.abs(y_test_res.to_numpy() - pred_res_rf)
    ci_90_half_width = round(float(np.percentile(abs_errors, 90.0)), 3)
    measured_coverage = round(float(np.mean(abs_errors <= ci_90_half_width) * 100.0), 1)

    # Per-city breakdowns on the test set
    per_city_metrics = {}
    for city in sorted(df_test["city"].unique()):
        city_mask = df_test["city"] == city
        y_c = y_test_res[city_mask]
        pred_c = pred_res_rf[city_mask]
        per_city_metrics[city] = calculate_metrics(y_c, pred_c)

    # Feature importances
    importances = rf_model.feature_importances_
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    ]

    report = {
        "model_type": "RandomForestRegressor",
        "model_version": "v1.1.0-rf-residual",
        "scientific_status": "PROTOTYPE_BENCHMARK_SYNTHETIC_LABELS_NOT_FOR_SCIENTIFIC_CLAIMS",
        "evaluation_dataset": "Indian Paired Dataset (2024 Test Set)",
        "sample_count": len(y_test_res),
        # Legacy-compatible and residual-accurate metrics block
        "metrics": {
            "random_forest": metrics_abs_rf,
            "linear_baseline": metrics_abs_linear,
            "reanalysis_transfer_baseline": metrics_abs_zero,
            "climatological_bias_baseline": metrics_abs_clim,
        },
        "residual_target_metrics": {
            "random_forest": metrics_res_rf,
            "linear_baseline": metrics_res_linear,
            "reanalysis_transfer_baseline": metrics_res_zero,
            "climatological_bias_baseline": metrics_res_clim,
        },
        "baseline_comparisons": {
            "over_zero_anomaly": impr_over_zero,
            "over_climatological_bias": impr_over_clim,
            "over_linear_regression": impr_over_linear,
        },
        "per_city_metrics": per_city_metrics,
        "feature_importances": feat_imp,
        "empirical_uncertainty": {
            "method": "Empirical Conformal Quantile of Held-Out Residuals",
            "ci_90_half_width_c": ci_90_half_width,
            "measured_test_coverage_pct": measured_coverage,
        },
    }

    with open(EVALUATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_evaluation()
    print("Evaluation Report Summary (Residual Target):")
    print(json.dumps(rep["residual_target_metrics"], indent=2))
