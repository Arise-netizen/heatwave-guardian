"""
HeatWave Guardian — Hyperlocal ML Baseline Models & Benchmark Module
====================================================================
Implements standard reference baselines for microclimate downscaling comparison:
1. Zero-Anomaly Baseline (Status-quo Reanalysis): Assumes Delta T = 0 (T_local = T_background).
2. Climatological Bias Baseline: Assumes Delta T = historical training mean residual.
3. Ordinary Least Squares / Ridge Linear Regression Baseline: Standard linear fit.

All baselines and the ML model are evaluated on the EXACT SAME held-out observations
using the localized temperature residual as the primary target.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate MAE, RMSE, Mean Bias, and R² for a set of predictions.
    When evaluated on residuals, reflects true downscaling residual error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))

    # Guard against zero-variance edge cases in R2
    if np.var(y_true) < 1e-9:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))

    return {
        "mae_c": round(mae, 3),
        "rmse_c": round(rmse, 3),
        "mean_bias_c": round(bias, 3),
        "r2_score": round(r2, 4),
        "sample_count": len(y_true),
    }


def calculate_baseline_improvements(metrics_model: Dict[str, float], metrics_baseline: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate absolute delta improvement and percentage improvement of model over baseline.
    """
    base_mae = metrics_baseline["mae_c"]
    model_mae = metrics_model["mae_c"]
    delta_mae = round(base_mae - model_mae, 3)
    
    pct_mae = round((delta_mae / base_mae * 100.0), 1) if base_mae > 0 else 0.0
    
    base_rmse = metrics_baseline["rmse_c"]
    model_rmse = metrics_model["rmse_c"]
    delta_rmse = round(base_rmse - model_rmse, 3)
    pct_rmse = round((delta_rmse / base_rmse * 100.0), 1) if base_rmse > 0 else 0.0

    meaningful = bool(delta_mae >= 0.05 and pct_mae >= 5.0)

    return {
        "delta_mae_c": delta_mae,
        "pct_improvement_mae": pct_mae,
        "delta_rmse_c": delta_rmse,
        "pct_improvement_rmse": pct_rmse,
        "meaningful_improvement": meaningful,
    }


class ZeroAnomalyBaseline:
    """
    Status-quo atmospheric baseline (Zero-Anomaly / Reanalysis Direct Transfer):
    Assumes localized residual Delta T = 0, so T_local = T_background.
    """
    def __init__(self):
        self.name = "Zero-Anomaly Baseline (Reanalysis Transfer)"

    def fit(self, X: pd.DataFrame, y_residual: Optional[pd.Series] = None):
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts absolute temperature (T_bg + 0)."""
        if isinstance(X, pd.DataFrame):
            return X["background_temp_c"].to_numpy()
        return np.asarray(X)[:, 0]


class ClimatologicalBiasBaseline:
    """
    Station/Climatological Bias Baseline:
    Predicts constant mean residual Delta T = mean(y_train_residual).
    """
    def __init__(self):
        self.name = "Climatological Bias Baseline"
        self.mean_residual: float = 0.0

    def fit(self, X: pd.DataFrame, y_residual: pd.Series):
        self.mean_residual = float(np.mean(y_residual))
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_residual, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts absolute temperature (T_bg + mean_residual)."""
        bg = X["background_temp_c"].to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)[:, 0]
        return bg + self.mean_residual


class LinearRegressionBaseline:
    """
    Linear Ridge Regression Baseline:
    Fits localized temperature residual (Delta T) from non-leaking features.
    """
    def __init__(self, alpha: float = 1.0):
        self.name = "Ridge Linear Regression Baseline"
        self.model = Ridge(alpha=alpha, random_state=42)

    def fit(self, X: pd.DataFrame, y_residual: pd.Series):
        self.model.fit(X, y_residual)
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts absolute temperature (T_bg + predicted_residual)."""
        bg = X["background_temp_c"].to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)[:, 0]
        return bg + self.predict_residual(X)


# Backward-compatible alias for existing code
ReanalysisDirectTransferBaseline = ZeroAnomalyBaseline
