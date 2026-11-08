"""
HeatWave Guardian — Hyperlocal ML Feature Engineering Module
============================================================
Transforms background atmospheric reanalysis and municipal spatial morphology
into standardized, non-leaking feature vectors for microclimate downscaling.

ANTI-LEAKAGE SCIENTIFIC AUDIT COMPLIANCE:
- Removed: raw latitude/longitude coordinates (prevents spatial memorization).
- Removed: circular engineered interaction terms (heat_stagnation_index, solar_thermal_loading)
  that leaked the synthetic empirical generating formula.
- Target Definition: Localized temperature residual (observed local temperature - background temperature).
"""

import math
from typing import Dict, Any, List, Union, Tuple, Optional
import pandas as pd
import numpy as np


# Standard non-leaking physical feature set (12 features)
FEATURE_NAMES = [
    "background_temp_c",
    "background_rh_pct",
    "background_wind_ms",
    "background_solar_wm2",
    "elevation_m",
    "urban_built_up_ratio",
    "vegetation_canopy_deficit",
    "population_density_k",
    "distance_to_coast_km",
    "is_coastal",
    "day_of_year_sin",
    "day_of_year_cos",
]

# Target Definitions
TARGET_RESIDUAL_COLUMN = "local_anomaly_c"        # Primary target: localized temperature residual (Delta T)
TARGET_ABSOLUTE_COLUMN = "local_observed_temp_c"  # Observed local temperature
TARGET_COLUMN = TARGET_RESIDUAL_COLUMN
ANOMALY_COLUMN = TARGET_RESIDUAL_COLUMN


def compute_day_of_year_cyclical(date_str: str) -> Tuple[float, float]:
    """Compute sine and cosine cyclical encodings for day of year."""
    try:
        dt = pd.to_datetime(date_str)
        day_of_year = dt.dayofyear
    except Exception:
        day_of_year = 150  # Mid-summer default (May/June)
    
    rad = 2.0 * math.pi * day_of_year / 365.25
    return round(math.sin(rad), 4), round(math.cos(rad), 4)


def extract_features_from_dict(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract and compute a standardized non-leaking feature vector from a dictionary
    of background weather and spatial morphology parameters.
    """
    bg_t = float(row.get("background_temp_c", row.get("temp_c", 35.0)))
    bg_rh = float(row.get("background_rh_pct", row.get("rh_pct", row.get("humidity", 50.0))))
    bg_wind = float(row.get("background_wind_ms", row.get("wind_ms", 2.0)))
    bg_solar = float(row.get("background_solar_wm2", row.get("solar_wm2", 600.0)))
    
    elevation = float(row.get("elevation_m", 200.0))
    built_up = float(row.get("urban_built_up_ratio", 0.70))
    canopy_def = float(row.get("vegetation_canopy_deficit", 0.75))
    pop_k = float(row.get("population_density_k", 8.0))
    dist_coast = float(row.get("distance_to_coast_km", 300.0))
    is_coastal = int(row.get("is_coastal", 1 if dist_coast < 30.0 else 0))

    # Temporal cyclical features
    date_str = str(row.get("date", "2024-05-15"))
    sin_doy, cos_doy = compute_day_of_year_cyclical(date_str)

    return {
        "background_temp_c": round(bg_t, 2),
        "background_rh_pct": round(bg_rh, 1),
        "background_wind_ms": round(bg_wind, 2),
        "background_solar_wm2": round(bg_solar, 1),
        "elevation_m": round(elevation, 1),
        "urban_built_up_ratio": round(built_up, 3),
        "vegetation_canopy_deficit": round(canopy_def, 3),
        "population_density_k": round(pop_k, 2),
        "distance_to_coast_km": round(dist_coast, 1),
        "is_coastal": is_coastal,
        "day_of_year_sin": sin_doy,
        "day_of_year_cos": cos_doy,
    }


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Transforms a raw paired DataFrame into feature matrix X (12 features),
    target y_residual (local residual Delta T), and y_absolute (local temp).
    """
    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append(extract_features_from_dict(row.to_dict()))

    X = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    y_residual = df[TARGET_RESIDUAL_COLUMN].astype(float)
    y_absolute = df[TARGET_ABSOLUTE_COLUMN].astype(float)

    return X, y_residual, y_absolute


if __name__ == "__main__":
    sample = {
        "background_temp_c": 41.5,
        "background_rh_pct": 28.0,
        "background_wind_ms": 1.8,
        "background_solar_wm2": 820.0,
        "urban_built_up_ratio": 0.85,
        "vegetation_canopy_deficit": 0.78,
        "population_density_k": 11.3,
        "distance_to_coast_km": 1050.0,
        "elevation_m": 216.0,
        "is_coastal": 0,
        "date": "2024-05-28",
    }
    feats = extract_features_from_dict(sample)
    print(f"Non-leaking features ({len(feats)}):")
    for k, v in feats.items():
        print(f"  {k:28s} : {v}")
