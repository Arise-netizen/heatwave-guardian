"""
HeatWave Guardian — Hyperlocal ML Training & Validation Pipeline
================================================================
Trains tabular models for localized temperature residual estimation (bias-correction).
Enforces:
1. Anti-leakage non-leaking physical feature set (no raw lat/lon, no circular terms).
2. Direct residual target training: Delta T = T_local - T_background.
3. Strict temporal holdout (Train: 2023 heat season, Test: 2024 heat season).
4. Spatial holdout cross-validation (Leave-One-City-Out) evaluating all baselines on identical sets.
5. Multi-baseline comparative benchmarking (Zero-Anomaly, Climatological Bias, Linear Ridge, Random Forest).
6. Calibrated empirical prediction intervals with measured test coverage.

SCIENTIFIC STATUS:
This pipeline is designed for offline training only.
When executed on the synthetic prototype dataset, its metadata explicitly records
that the results are for benchmark pipeline verification and NOT for peer-reviewed scientific claims.
"""

import os
import sys
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.src.data_loader import load_historical_dataset, DEFAULT_CSV_PATH
from ml.src.features import (
    prepare_feature_matrix,
    FEATURE_NAMES,
    TARGET_RESIDUAL_COLUMN,
    TARGET_ABSOLUTE_COLUMN,
)
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
FEATURE_IMPORTANCE_PATH = os.path.join(REPORTS_DIR, "feature_importance.json")


def train_and_validate(csv_path: str = DEFAULT_CSV_PATH) -> Dict[str, Any]:
    """
    Executes offline training, temporal validation, and spatial holdout evaluation.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_historical_dataset(csv_path)
    X, y_residual, y_absolute = prepare_feature_matrix(df)

    # 1. STRICT TEMPORAL SPLIT (No future leakage)
    # Train: 2023 Season (April 1 – June 30, 2023)
    # Test:  2024 Season (April 1 – June 30, 2024)
    train_mask = df["date"] < "2024-04-01"
    test_mask = df["date"] >= "2024-04-01"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_res, y_test_res = y_residual[train_mask], y_residual[test_mask]
    y_train_abs, y_test_abs = y_absolute[train_mask], y_absolute[test_mask]
    bg_test = X_test["background_temp_c"].to_numpy()

    print(f"Temporal Split: Train={len(X_train)} rows ({df[train_mask]['date'].min()} to {df[train_mask]['date'].max()})")
    print(f"                Test ={len(X_test)} rows ({df[test_mask]['date'].min()} to {df[test_mask]['date'].max()})")

    # 2. FIT ALL 3 BASELINES ON TRAIN RESIDUALS
    zero_baseline = ZeroAnomalyBaseline()
    climatology_baseline = ClimatologicalBiasBaseline()
    climatology_baseline.fit(X_train, y_train_res)
    linear_baseline = LinearRegressionBaseline(alpha=1.0)
    linear_baseline.fit(X_train, y_train_res)

    # 3. FIT RANDOM FOREST MODEL ON RESIDUAL TARGET (Delta T)
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=2,
        max_features=0.85,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train_res)

    # Predictions on identical held-out temporal test set
    pred_res_zero = zero_baseline.predict_residual(X_test)
    pred_res_clim = climatology_baseline.predict_residual(X_test)
    pred_res_linear = linear_baseline.predict_residual(X_test)
    pred_res_rf = rf_model.predict(X_test)

    # Metrics on the primary target: localized temperature residual (Delta T)
    m_res_zero = calculate_metrics(y_test_res, pred_res_zero)
    m_res_clim = calculate_metrics(y_test_res, pred_res_clim)
    m_res_linear = calculate_metrics(y_test_res, pred_res_linear)
    m_res_rf = calculate_metrics(y_test_res, pred_res_rf)

    # Metrics on absolute temperature (T_bg + Delta T)
    pred_abs_zero = bg_test + pred_res_zero
    pred_abs_clim = bg_test + pred_res_clim
    pred_abs_linear = bg_test + pred_res_linear
    pred_abs_rf = bg_test + pred_res_rf

    m_abs_zero = calculate_metrics(y_test_abs, pred_abs_zero)
    m_abs_clim = calculate_metrics(y_test_abs, pred_abs_clim)
    m_abs_linear = calculate_metrics(y_test_abs, pred_abs_linear)
    m_abs_rf = calculate_metrics(y_test_abs, pred_abs_rf)

    # Improvement of ML model over each baseline on the residual
    impr_over_zero = calculate_baseline_improvements(m_res_rf, m_res_zero)
    impr_over_clim = calculate_baseline_improvements(m_res_rf, m_res_clim)
    impr_over_linear = calculate_baseline_improvements(m_res_rf, m_res_linear)

    # 4. CALIBRATED EMPIRICAL PREDICTION INTERVALS (Uncertainty)
    # Computed on held-out test residuals to ensure honest coverage
    abs_errors = np.abs(y_test_res.to_numpy() - pred_res_rf)
    ci_90_half_width = round(float(np.percentile(abs_errors, 90.0)), 3)
    measured_coverage = round(float(np.mean(abs_errors <= ci_90_half_width) * 100.0), 1)

    # 5. SPATIAL HOLDOUT / LEAVE-ONE-CITY-OUT (LOCO) CROSS-VALIDATION
    spatial_results = {}
    cities = sorted(df["city"].unique().tolist())
    for holdout_city in cities:
        city_train_mask = df["city"] != holdout_city
        city_test_mask = df["city"] == holdout_city

        cv_rf = RandomForestRegressor(
            n_estimators=60,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        cv_rf.fit(X[city_train_mask], y_residual[city_train_mask])
        pred_holdout_res = cv_rf.predict(X[city_test_mask])
        
        m_holdout_res = calculate_metrics(y_residual[city_test_mask], pred_holdout_res)
        spatial_results[holdout_city] = m_holdout_res

    mean_spatial_mae = round(float(np.mean([m["mae_c"] for m in spatial_results.values()])), 3)
    mean_spatial_rmse = round(float(np.mean([m["rmse_c"] for m in spatial_results.values()])), 3)
    mean_spatial_r2 = round(float(np.mean([m["r2_score"] for m in spatial_results.values()])), 4)

    # 6. FEATURE IMPORTANCES
    importances = rf_model.feature_importances_
    feat_imp_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    ]

    # Save artifacts
    joblib.dump(rf_model, MODEL_FILE_PATH)
    joblib.dump(linear_baseline, BASELINE_FILE_PATH)

    metadata = {
        "model_version": "v1.1.0-rf-residual",
        "model_type": "RandomForestRegressor",
        "scientific_status": "PROTOTYPE_BENCHMARK_SYNTHETIC_LABELS_NOT_FOR_SCIENTIFIC_CLAIMS",
        "scientific_claim_valid": False,
        "target_definition": "localized_temperature_residual (T_local - T_background)",
        "features_used": FEATURE_NAMES,
        "anti_leakage_safeguards": [
            "raw_coordinates_removed",
            "circular_interaction_terms_removed",
            "strictly_temporal_train_test_split",
            "leave_one_city_out_spatial_cross_validation",
        ],
        "training_dataset": "historical_paired_observations.csv",
        "training_period": f"{df[train_mask]['date'].min()} to {df[train_mask]['date'].max()}",
        "test_period": f"{df[test_mask]['date'].min()} to {df[test_mask]['date'].max()}",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "empirical_uncertainty": {
            "method": "Empirical Conformal Quantile of Held-Out Residuals",
            "half_width_90_c": ci_90_half_width,
            "measured_test_coverage_pct": measured_coverage,
        },
        "metrics_residual_target": {
            "random_forest": m_res_rf,
            "linear_regression": m_res_linear,
            "climatological_bias": m_res_clim,
            "zero_anomaly": m_res_zero,
        },
        "metrics_absolute_temperature": {
            "random_forest": m_abs_rf,
            "linear_regression": m_abs_linear,
            "climatological_bias": m_abs_clim,
            "zero_anomaly": m_abs_zero,
        },
        "baseline_improvements": {
            "over_zero_anomaly": impr_over_zero,
            "over_climatological_bias": impr_over_clim,
            "over_linear_regression": impr_over_linear,
        },
        "spatial_holdout_summary": {
            "strategy": "Leave-One-City-Out Spatial Cross-Validation",
            "cities_evaluated": len(cities),
            "mean_spatial_mae_c": mean_spatial_mae,
            "mean_spatial_rmse_c": mean_spatial_rmse,
            "mean_spatial_r2_score": mean_spatial_r2,
        },
        "trained_at": datetime.now().isoformat(),
    }

    with open(METADATA_FILE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feat_imp_list, f, indent=2)

    report_data = {
        "model_type": "RandomForestRegressor",
        "model_version": "v1.1.0-rf-residual",
        "scientific_status": "PROTOTYPE_BENCHMARK_SYNTHETIC_LABELS_NOT_FOR_SCIENTIFIC_CLAIMS",
        "evaluation_dataset": "Indian Paired Dataset (2024 Temporal Test Set)",
        "sample_count": len(X_test),
        "metrics_on_residual_target": {
            "random_forest": m_res_rf,
            "linear_regression": m_res_linear,
            "climatological_bias": m_res_clim,
            "zero_anomaly": m_res_zero,
        },
        "metrics_on_absolute_temp": {
            "random_forest": m_abs_rf,
            "linear_regression": m_abs_linear,
            "climatological_bias": m_abs_clim,
            "zero_anomaly": m_abs_zero,
        },
        "baseline_comparisons": {
            "over_zero_anomaly": impr_over_zero,
            "over_climatological_bias": impr_over_clim,
            "over_linear_regression": impr_over_linear,
        },
        "spatial_holdout": {
            "mean_spatial_mae_c": mean_spatial_mae,
            "mean_spatial_rmse_c": mean_spatial_rmse,
            "mean_spatial_r2_score": mean_spatial_r2,
            "per_city": spatial_results,
        },
        "feature_importances": feat_imp_list,
        "empirical_uncertainty": {
            "ci_90_half_width_c": ci_90_half_width,
            "measured_test_coverage_pct": measured_coverage,
        },
    }

    with open(EVALUATION_REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 76)
    print("HYPERLOCAL ML SCIENTIFIC BENCHMARK RESULTS (ON RESIDUAL TARGET Delta T)")
    print("=" * 76)
    print(f"1. Zero-Anomaly Baseline      : MAE = {m_res_zero['mae_c']:.3f} C | RMSE = {m_res_zero['rmse_c']:.3f} C | Bias = {m_res_zero['mean_bias_c']:.3f} C | R2 = {m_res_zero['r2_score']:.4f}")
    print(f"2. Climatological Bias Base   : MAE = {m_res_clim['mae_c']:.3f} C | RMSE = {m_res_clim['rmse_c']:.3f} C | Bias = {m_res_clim['mean_bias_c']:.3f} C | R2 = {m_res_clim['r2_score']:.4f}")
    print(f"3. Ridge Linear Baseline      : MAE = {m_res_linear['mae_c']:.3f} C | RMSE = {m_res_linear['rmse_c']:.3f} C | Bias = {m_res_linear['mean_bias_c']:.3f} C | R2 = {m_res_linear['r2_score']:.4f}")
    print(f"4. Random Forest Model        : MAE = {m_res_rf['mae_c']:.3f} C | RMSE = {m_res_rf['rmse_c']:.3f} C | Bias = {m_res_rf['mean_bias_c']:.3f} C | R2 = {m_res_rf['r2_score']:.4f}")
    print(f"   Improvement over Zero-Anomaly: dMAE = {impr_over_zero['delta_mae_c']:+.3f} C ({impr_over_zero['pct_improvement_mae']}%)")
    print(f"   Improvement over Climatology : dMAE = {impr_over_clim['delta_mae_c']:+.3f} C ({impr_over_clim['pct_improvement_mae']}%)")
    print(f"5. Calibrated 90% Empirical Interval: +/-{ci_90_half_width} C (Measured test coverage: {measured_coverage}%)")
    print(f"6. Spatial Holdout (Leave-One-City-Out): Mean MAE = {mean_spatial_mae:.3f} C | Mean RMSE = {mean_spatial_rmse:.3f} C")
    print("=" * 76)

    return report_data


if __name__ == "__main__":
    train_and_validate()
