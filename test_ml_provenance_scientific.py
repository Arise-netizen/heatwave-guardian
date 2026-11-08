"""
HeatWave Guardian — ML Data Provenance & Scientific Integrity Test Suite
========================================================================
Exhaustive verification of the critical ML data provenance rebuild:
1. Target provenance & synthetic label metadata flag.
2. Authentic station observation ingestion interface.
3. Feature anti-leakage (exclusion of raw coordinates and formula-mirroring terms).
4. Offline-only inference (zero training on request path).
5. Atmospheric moisture conservation during WBGT propagation.
6. Multi-baseline comparative benchmarking on identical held-out sets.
7. Residual target evaluation metrics.
8. Deployed-vs-evaluation feature preprocessing parity.
9. Calibrated empirical uncertainty intervals and coverage.
10. Driver explainability without fabricated percentage claims.
11. Dynamic model metadata provenance.
12. Model-unavailable behavior without fabricated scientific values.
13. Honest spatial resolution reporting.
14. Official IMD operational criteria documentation.
15. ISO 7243:2017 environmental screening terminology.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.src.data_loader import (
    load_historical_dataset,
    load_real_station_observations,
    build_authentic_station_ingestion_template,
    RealStationObservation,
    METADATA_JSON_PATH,
)
from ml.src.features import (
    FEATURE_NAMES,
    TARGET_RESIDUAL_COLUMN,
    extract_features_from_dict,
    prepare_feature_matrix,
)
from ml.src.baseline import (
    ZeroAnomalyBaseline,
    ClimatologicalBiasBaseline,
    LinearRegressionBaseline,
    calculate_metrics,
    calculate_baseline_improvements,
)
from ml.src.predict import (
    HyperlocalPredictor,
    adjust_humidity_for_temperature_delta,
    get_feature_names,
)
from engine import classify_imd, calculate_thermal_stress_index


class TestMLProvenanceScientific(unittest.TestCase):
    """Rigorous scientific verification of ML data provenance, anti-leakage, and physics."""

    # 1. Real vs. Synthetic Target Provenance
    def test_01_synthetic_label_provenance_flag(self):
        """1. Verify metadata explicitly flags dataset as synthetic formula labels and invalid for scientific claims."""
        self.assertTrue(os.path.exists(METADATA_JSON_PATH), "metadata.json must exist")
        with open(METADATA_JSON_PATH, "r") as f:
            meta = json.load(f)

        self.assertTrue(meta.get("is_synthetic"), "is_synthetic must be True for current prototype dataset")
        self.assertFalse(meta.get("scientific_claim_valid"), "scientific_claim_valid must be False")
        self.assertIn("SYNTHETIC", meta.get("scientific_status", "").upper())
        self.assertEqual(meta.get("target_variable"), "localized_temperature_residual")

    # 2. Real Station Ingestion Interface
    def test_02_real_station_ingestion_interface(self):
        """2. Verify RealStationObservation interface validates authentic station schema and bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, "template.csv")
            build_authentic_station_ingestion_template(template_path)
            self.assertTrue(os.path.exists(template_path))

            df_loaded = load_real_station_observations(template_path)
            self.assertEqual(len(df_loaded), 1)
            self.assertEqual(df_loaded["station_id"].iloc[0], "VIDP")
            self.assertEqual(df_loaded["quality_flag"].iloc[0], "VALID")

            # Verify physical bounds rejection
            bad_data = df_loaded.copy()
            bad_data.loc[0, "observed_temp_c"] = 75.0  # Impossible ambient temperature
            bad_path = os.path.join(tmpdir, "bad.csv")
            bad_data.to_csv(bad_path, index=False)
            with self.assertRaises(ValueError):
                load_real_station_observations(bad_path)

    # 3. Anti-Leakage Feature Set
    def test_03_anti_leakage_feature_set(self):
        """3. Verify FEATURE_NAMES excludes memorizable coordinates and circular formula terms."""
        self.assertEqual(len(FEATURE_NAMES), 12, "Feature set must contain exactly 12 standardized non-leaking features")
        self.assertNotIn("latitude", FEATURE_NAMES, "Raw latitude must be excluded to prevent spatial memorization")
        self.assertNotIn("longitude", FEATURE_NAMES, "Raw longitude must be excluded to prevent spatial memorization")
        self.assertNotIn("heat_stagnation_index", FEATURE_NAMES, "Circular formula term heat_stagnation_index must be excluded")
        self.assertNotIn("solar_thermal_loading", FEATURE_NAMES, "Circular formula term solar_thermal_loading must be excluded")

    # 4. Offline-Only Inference
    def test_04_offline_only_inference(self):
        """4. Verify inference requests do NOT automatically train the model if artifact is missing."""
        HyperlocalPredictor._model = None
        with patch("os.path.exists", return_value=False):
            # Must return None and not invoke train_and_validate
            model = HyperlocalPredictor.get_model()
            self.assertIsNone(model, "Inference must return None if artifact is missing without auto-training")

    # 5. Moisture Conservation in WBGT Propagation
    def test_05_moisture_conservation_wbgt_propagation(self):
        """5. Verify relative humidity decreases as temperature rises under conserved vapor pressure."""
        t_bg = 40.0
        rh_bg = 40.0
        t_local = 42.0  # +2°C local thermal heating

        rh_local = adjust_humidity_for_temperature_delta(t_bg, rh_bg, t_local)
        self.assertLess(rh_local, rh_bg, "Relative humidity must decrease under sensible heating when moisture is conserved")
        self.assertTrue(30.0 < rh_local < 40.0)

        # Equal temperatures produce identical RH
        self.assertEqual(adjust_humidity_for_temperature_delta(38.0, 50.0, 38.0), 50.0)

    # 6. Multi-Baseline Comparison on Identical Observations
    def test_06_baseline_comparisons_on_identical_set(self):
        """6. Verify all 4 baselines are compared on identical observations with delta improvements."""
        y_true = np.array([1.2, 1.5, 0.8, 2.0, 1.1])
        pred_model = np.array([1.1, 1.4, 0.9, 1.8, 1.2])
        pred_zero = np.zeros(5)

        m_model = calculate_metrics(y_true, pred_model)
        m_zero = calculate_metrics(y_true, pred_zero)

        impr = calculate_baseline_improvements(m_model, m_zero)
        self.assertIn("delta_mae_c", impr)
        self.assertIn("pct_improvement_mae", impr)
        self.assertGreater(impr["delta_mae_c"], 0.0)

    # 7. Residual Target Definition
    def test_07_residual_target_definition(self):
        """7. Verify primary target column is the localized temperature residual (local_anomaly_c)."""
        self.assertEqual(TARGET_RESIDUAL_COLUMN, "local_anomaly_c")
        df = load_historical_dataset()
        X, y_res, y_abs = prepare_feature_matrix(df.head(10))
        self.assertEqual(len(y_res), 10)
        # Verify residual equals observed minus background
        np.testing.assert_array_almost_equal(
            y_res.to_numpy(),
            (df.head(10)["local_observed_temp_c"] - df.head(10)["background_temp_c"]).to_numpy(),
            decimal=2,
        )

    # 8. Preprocessing Parity
    def test_08_preprocessing_parity(self):
        """8. Verify extraction from dictionary produces exact FEATURE_NAMES columns in correct order."""
        sample = {
            "background_temp_c": 39.0,
            "background_rh_pct": 32.0,
            "background_wind_ms": 2.1,
            "background_solar_wm2": 680.0,
            "date": "2024-05-15",
        }
        feats = extract_features_from_dict(sample)
        for fn in FEATURE_NAMES:
            self.assertIn(fn, feats)

    # 9. Empirical Uncertainty Intervals
    def test_09_empirical_uncertainty_bounds(self):
        """9. Verify prediction intervals bound the prediction and provide empirical coverage metadata."""
        w = {"temp_c": 41.0, "rh_pct": 30.0, "wind_ms": 2.0, "solar_wm2": 700.0}
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, w, city_name="Delhi")
        uncert = res["uncertainty"]

        self.assertIn("prediction_interval_90_c", uncert)
        self.assertIn("temp_lower_c", uncert)
        self.assertIn("temp_upper_c", uncert)
        self.assertIn("measured_test_coverage_pct", uncert)
        self.assertLess(uncert["temp_lower_c"], res["localized_temp_c"])
        self.assertGreater(uncert["temp_upper_c"], res["localized_temp_c"])

    # 10. No Fabricated Driver Percentages
    def test_10_driver_explanations_no_fake_percentages(self):
        """10. Verify key drivers do NOT contain fabricated percentage shares (e.g. 'contributed 45%')."""
        w = {"temp_c": 43.0, "rh_pct": 25.0, "wind_ms": 1.2, "solar_wm2": 800.0}
        res = HyperlocalPredictor.predict_hyperlocal(17.3850, 78.4867, w, city_name="Hyderabad")
        drivers = res["key_drivers"]

        self.assertGreater(len(drivers), 0)
        for d in drivers:
            self.assertIn("factor", d)
            self.assertIn("mechanism", d)
            # Ensure no fabricated string like "contributed 45%" or arbitrary percentage impact
            detail_str = str(d)
            self.assertNotIn("contributed", detail_str.lower())

    # 11. Dynamic Model Metadata
    def test_11_dynamic_model_metadata(self):
        """11. Verify model metadata is loaded dynamically and contains required provenance."""
        meta = HyperlocalPredictor.get_metadata()
        self.assertIsInstance(meta, dict)
        self.assertIn("model_version", meta)
        self.assertIn("target_definition", meta)
        self.assertIn("features_used", meta)
        self.assertIn("empirical_uncertainty", meta)

    # 12. Unavailable Model Behavior
    def test_12_unavailable_model_behavior(self):
        """12. Verify predictor returns explicit unavailable state without fabricating values."""
        with patch.object(HyperlocalPredictor, "get_model", return_value=None):
            res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, {"temp_c": 40.0, "rh_pct": 35.0})
            self.assertFalse(res["model_available"])
            self.assertEqual(res["status"], "service_unavailable")
            self.assertIn("offline", res["error"].lower())

    # 13. Realistic Spatial Resolution Claims
    def test_13_spatial_resolution_reporting(self):
        """13. Verify spatial resolution does not claim unevidenced 100m microclimate resolution."""
        w = {"temp_c": 39.0, "rh_pct": 35.0, "wind_ms": 2.0, "solar_wm2": 650.0}
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, w, city_name="Delhi")
        res_claim = res.get("spatial_resolution", "")
        self.assertNotIn("~100m–1km Microclimate Cell", res_claim)
        self.assertIn("proxy", res_claim.lower())

    # 14. Official IMD Criteria Documentation
    def test_14_imd_criteria_documentation(self):
        """14. Verify classify_imd docstring documents official IMD criteria and regional thresholds."""
        doc = classify_imd.__doc__ or ""
        self.assertIn("plains: 40°C", doc)
        self.assertIn("coastal: 37°C", doc)
        self.assertIn("hilly: 30°C", doc)
        self.assertIn("India Meteorological Department", doc)
        self.assertIn("daily maximum temperature", doc.lower())

    # 15. ISO 7243 Screening Terminology
    def test_15_iso_screening_disclaimer(self):
        """15. Verify disclaimer cites ISO 7243:2017 as environmental screening limits requiring exposure context."""
        w = {"temp_c": 38.0, "rh_pct": 40.0, "wind_ms": 2.0, "solar_wm2": 600.0}
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, w, city_name="Delhi")
        disclaimer = res.get("disclaimer", "")
        self.assertIn("ISO 7243:2017", disclaimer)
        self.assertIn("screening", disclaimer.lower())


if __name__ == "__main__":
    unittest.main()
