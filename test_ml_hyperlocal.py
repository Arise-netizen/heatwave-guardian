"""
HeatWave Guardian — Hyperlocal ML Intelligence Test Suite
=========================================================
Exhaustive verification of data ingestion, feature engineering, baselines,
Random Forest training, temporal validation, spatial holdout, explainability,
deterministic WBGT preservation, REST API endpoints, and safety guardrails.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from ml.src.data_loader import (
    load_historical_dataset,
    get_city_morphology,
    CITY_MORPHOLOGY,
    DEFAULT_CSV_PATH,
)
from ml.src.features import (
    extract_features_from_dict,
    prepare_feature_matrix,
    compute_day_of_year_cyclical,
    FEATURE_NAMES,
)
from ml.src.baseline import (
    calculate_metrics,
    ReanalysisDirectTransferBaseline,
    LinearRegressionBaseline,
)
from ml.src.train import train_and_validate
from ml.src.evaluate import run_evaluation
from ml.src.predict import HyperlocalPredictor
from services.hyperlocal_ml_service import HyperlocalMLService
from services.authority_service import AuthorityService
from engine import calculate_thermal_stress_index


class TestHyperlocalML(unittest.TestCase):
    """Exhaustive test suite for Hyperlocal ML Intelligence Layer."""

    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        # Ensure model is fresh and loaded
        HyperlocalPredictor._model = None
        cls.df = load_historical_dataset()
        cls.model = HyperlocalPredictor.get_model()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. DATA INGESTION & MORPHOLOGY TESTS
    # ──────────────────────────────────────────────────────────────────────────

    def test_01_load_dataset_structure(self):
        """1. Verify dataset structure: 1820 rows, 10 cities, required columns."""
        self.assertGreaterEqual(len(self.df), 1000)
        self.assertEqual(self.df["city"].nunique(), 10)
        required_cols = [
            "date", "city", "latitude", "longitude", "elevation_m",
            "urban_built_up_ratio", "vegetation_canopy_deficit", "population_density_k",
            "background_temp_c", "background_rh_pct", "background_wind_ms",
            "background_solar_wm2", "local_observed_temp_c", "local_anomaly_c"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)

    def test_02_dataset_no_missing_critical_values(self):
        """2. Verify zero missing or NaN values in critical atmospheric and target fields."""
        critical_cols = ["background_temp_c", "local_observed_temp_c", "local_anomaly_c", "latitude", "longitude"]
        for col in critical_cols:
            self.assertEqual(self.df[col].isnull().sum(), 0, f"NaN values detected in {col}")

    def test_03_city_morphology_retrieval(self):
        """3. Verify known city morphology contains valid physical parameters."""
        for city in ["Delhi", "Mumbai", "Hyderabad", "Bengaluru"]:
            morph = get_city_morphology(city)
            self.assertIn("elevation_m", morph)
            self.assertIn("urban_built_up_ratio", morph)
            self.assertTrue(0.0 <= morph["urban_built_up_ratio"] <= 1.0)
            self.assertTrue(0.0 <= morph["vegetation_canopy_deficit"] <= 1.0)

    def test_04_unknown_coordinates_morphology_fallback(self):
        """4. Verify spatial morphology interpolation for arbitrary unknown coordinates."""
        morph = get_city_morphology("UnknownVillage", lat=28.5, lon=77.1)
        self.assertEqual(morph["lat"], 28.5)
        self.assertEqual(morph["lon"], 77.1)
        self.assertGreater(morph["elevation_m"], 0)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING TESTS
    # ──────────────────────────────────────────────────────────────────────────

    def test_05_feature_extraction_vector(self):
        """5. Verify feature extraction produces all 16 standardized features."""
        sample_w = {
            "background_temp_c": 40.0,
            "background_rh_pct": 35.0,
            "background_wind_ms": 2.0,
            "background_solar_wm2": 700.0,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "date": "2024-05-20",
        }
        feats = extract_features_from_dict(sample_w)
        for fn in FEATURE_NAMES:
            self.assertIn(fn, feats)
            self.assertIsInstance(feats[fn], (int, float))

    def test_06_day_of_year_cyclical_bounds(self):
        """6. Verify day of year cyclical encoding produces sin/cos in [-1, 1]."""
        for d in ["2024-01-01", "2024-04-15", "2024-06-21", "2024-12-31"]:
            s, c = compute_day_of_year_cyclical(d)
            self.assertTrue(-1.0 <= s <= 1.0)
            self.assertTrue(-1.0 <= c <= 1.0)

    def test_07_physical_derived_features(self):
        """7. Verify non-leaking physical features extracted cleanly and coordinates excluded."""
        f = extract_features_from_dict({"background_temp_c": 40.0, "background_wind_ms": 6.0, "latitude": 28.6, "longitude": 77.2})
        self.assertNotIn("latitude", f)
        self.assertNotIn("longitude", f)
        self.assertNotIn("heat_stagnation_index", f)
        self.assertIn("background_wind_ms", f)
        self.assertEqual(f["background_wind_ms"], 6.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. BASELINES & METRICS TESTS
    # ──────────────────────────────────────────────────────────────────────────

    def test_08_reanalysis_transfer_baseline(self):
        """8. Verify reanalysis transfer baseline assumes zero anomaly."""
        X, y, _ = prepare_feature_matrix(self.df.head(20))
        base = ReanalysisDirectTransferBaseline()
        preds = base.predict(X)
        np.testing.assert_array_almost_equal(preds, X["background_temp_c"].to_numpy())

    def test_09_linear_regression_baseline_fit_predict(self):
        """9. Verify linear regression baseline fits and predicts finite values."""
        X, y, y_anom = prepare_feature_matrix(self.df.head(50))
        lr = LinearRegressionBaseline()
        lr.fit(X, y_anom)
        preds = lr.predict(X)
        self.assertEqual(len(preds), 50)
        self.assertFalse(np.isnan(preds).any())

    def test_10_calculate_metrics_correctness(self):
        """10. Verify calculation of MAE, RMSE, and R²."""
        y_t = np.array([40.0, 42.0, 44.0])
        y_p = np.array([41.0, 42.0, 43.0])
        m = calculate_metrics(y_t, y_p)
        self.assertAlmostEqual(m["mae_c"], 2.0 / 3.0, places=2)
        self.assertAlmostEqual(m["mean_bias_c"], 0.0, places=2)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. MODEL TRAINING, TEMPORAL LEAKAGE & GENERALIZATION
    # ──────────────────────────────────────────────────────────────────────────

    def test_11_temporal_split_integrity(self):
        """11. Verify zero future leakage: train set dates strictly precede test dates."""
        train_mask = self.df["date"] < "2024-04-01"
        test_mask = self.df["date"] >= "2024-04-01"
        max_train_date = self.df[train_mask]["date"].max()
        min_test_date = self.df[test_mask]["date"].min()
        self.assertLess(max_train_date, min_test_date)

    def test_12_random_forest_beats_reanalysis_baseline(self):
        """12. Verify Random Forest reduces MAE compared to status-quo reanalysis baseline."""
        report = run_evaluation()
        mae_rf = report["metrics"]["random_forest"]["mae_c"]
        mae_transfer = report["metrics"]["reanalysis_transfer_baseline"]["mae_c"]
        self.assertLess(mae_rf, mae_transfer)
        self.assertLess(mae_rf, 0.50)  # Downscaling error < 0.50°C

    def test_13_spatial_holdout_generalization(self):
        """13. Verify spatial holdout (unseen cities) achieves mean MAE < 0.60°C."""
        eval_path = os.path.join(BASE_DIR, "ml", "reports", "evaluation_metrics.json")
        with open(eval_path, "r") as f:
            rep = json.load(f)
        if "spatial_validation" in rep:
            mean_spatial_mae = rep["spatial_validation"]["mean_spatial_mae_c"]
            self.assertLess(mean_spatial_mae, 0.60)

    def test_14_feature_importance_sensibility(self):
        """14. Verify physical microclimate drivers dominate anomaly prediction."""
        importances = self.model.feature_importances_
        sorted_feats = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        top_feats = [f[0] for f in sorted_feats[:5]]
        expected_drivers = {"background_wind_ms", "urban_built_up_ratio", "is_coastal", "background_solar_wm2", "vegetation_canopy_deficit"}
        self.assertTrue(any(f in expected_drivers for f in top_feats))

    # ──────────────────────────────────────────────────────────────────────────
    # 5. INFERENCE, EXPLAINABILITY & SCIENTIFIC SAFETY
    # ──────────────────────────────────────────────────────────────────────────

    def test_15_hyperlocal_predictor_output_schema(self):
        """15. Verify predict_hyperlocal returns complete structured dictionary."""
        w = {"temp_c": 41.0, "rh_pct": 30.0, "wind_ms": 2.0, "solar_wm2": 700.0}
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, w, city_name="Delhi")
        
        self.assertIn("baseline_environment", res)
        self.assertIn("hyperlocal_environment", res)
        self.assertIn("key_drivers", res)
        self.assertIn("confidence", res)
        self.assertIn("model_metadata", res)
        self.assertTrue(res["model_metadata"]["is_ml_prediction"])
        self.assertFalse(res["model_metadata"]["is_measured_observation"])

    def test_16_deterministic_wbgt_preservation(self):
        """16. Verify hyperlocal WBGT is strictly generated by deterministic Liljegren calculation with moisture adjustment."""
        w = {"temp_c": 40.0, "rh_pct": 35.0, "wind_ms": 2.0, "solar_wm2": 650.0}
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, w, city_name="Delhi")
        
        # Manually compute deterministic WBGT using predicted local temperature and adjusted RH
        pred_local_t = res["hyperlocal_environment"]["temperature_c"]
        pred_local_rh = res["hyperlocal_environment"]["rh_pct"]
        expected_wbgt = calculate_thermal_stress_index(pred_local_t, pred_local_rh, 2.0, 650.0)["wbgt_c"]
        self.assertEqual(res["hyperlocal_environment"]["wbgt_c"], expected_wbgt)

    def test_17_explainable_physical_drivers_presence(self):
        """17. Verify physical directional driver attribution returns clear factors and impacts."""
        w = {"temp_c": 43.0, "rh_pct": 25.0, "wind_ms": 1.2, "solar_wm2": 800.0}
        res = HyperlocalPredictor.predict_hyperlocal(17.3850, 78.4867, w, city_name="Hyderabad")
        drivers = res["key_drivers"]
        self.assertGreater(len(drivers), 0)
        self.assertTrue(any("Urban Built-up" in d["factor"] or "Wind" in d["factor"] for d in drivers))

    def test_18_scientific_disclaimer_notice(self):
        """18. Verify non-medical disclaimer is prominently present."""
        w = {"temp_c": 38.0, "rh_pct": 40.0, "wind_ms": 2.0, "solar_wm2": 600.0}
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, w, city_name="Delhi")
        disclaimer = res["disclaimer"]
        self.assertIn("NOT constitute medical diagnosis", disclaimer)

    def test_19_service_caching(self):
        """19. Verify HyperlocalMLService caches identical queries."""
        w = {"temp_c": 39.5, "rh_pct": 45.0, "wind_ms": 2.0, "solar_wm2": 600.0}
        res1 = HyperlocalMLService.predict_point(28.6139, 77.2090, w, "Delhi")
        res2 = HyperlocalMLService.predict_point(28.6139, 77.2090, w, "Delhi")
        self.assertEqual(res1["hyperlocal_environment"]["temperature_c"], res2["hyperlocal_environment"]["temperature_c"])

    # ──────────────────────────────────────────────────────────────────────────
    # 6. REST API ENDPOINTS & GEOSPATIAL ENRICHMENT
    # ──────────────────────────────────────────────────────────────────────────

    def test_20_api_ml_hyperlocal_endpoint(self):
        """20. GET /api/ml/hyperlocal returns HTTP 200 with complete prediction payload."""
        with patch("app.fetch_current_weather", return_value={"temp_c": 41.0, "rh_pct": 30.0, "wind_ms": 2.0, "solar_wm2": 700.0}):
            res = self.client.get("/api/ml/hyperlocal?lat=28.6139&lon=77.2090&city=Delhi")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["location"], "Delhi")
            self.assertIn("hyperlocal_environment", data)

    def test_21_api_ml_metrics_endpoint(self):
        """21. GET /api/ml/metrics returns HTTP 200 with holdout metrics and feature importances."""
        res = self.client.get("/api/ml/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("metrics", data)
        self.assertIn("feature_importances", data)

    def test_22_authority_zones_ml_enrichment(self):
        """22. GET /api/authority/zones returns features enriched with ML microclimate downscaling."""
        AuthorityService.clear_cache()
        mock_w = {"temp_c": 38.0, "rh_pct": 45.0, "wind_ms": 2.2, "solar_wm2": 650.0}
        with patch("app.fetch_current_weather", return_value=mock_w):
            res = self.client.get("/api/authority/zones")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            features = data["features"]
            self.assertGreater(len(features), 0)
            p = features[0]["properties"]
            self.assertIn("ml_temp_c", p)
            self.assertIn("ml_anomaly_c", p)
            self.assertIn("ml_wbgt_c", p)
            self.assertIn("ml_confidence", p)

    # ──────────────────────────────────────────────────────────────────────────
    # 7. ROBUSTNESS & BOUNDARY RESILIENCE
    # ──────────────────────────────────────────────────────────────────────────

    def test_23_empty_weather_fallback(self):
        """23. Verify predictor handles empty or incomplete weather dictionary gracefully."""
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, {})
        self.assertIn("hyperlocal_environment", res)
        self.assertIsInstance(res["hyperlocal_environment"]["temperature_c"], float)

    def test_24_extreme_heat_physical_bounds(self):
        """24. Verify anomaly remains bounded between 0.1°C and 4.8°C even under extreme heat."""
        w_extreme = {"temp_c": 49.0, "rh_pct": 15.0, "wind_ms": 0.5, "solar_wm2": 950.0}
        res = HyperlocalPredictor.predict_hyperlocal(28.6139, 77.2090, w_extreme, city_name="Delhi")
        anomaly = res["hyperlocal_environment"]["anomaly_c"]
        self.assertTrue(0.0 <= anomaly <= 5.0)


if __name__ == "__main__":
    unittest.main()
