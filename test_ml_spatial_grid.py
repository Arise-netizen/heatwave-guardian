"""
HeatWave Guardian — ML Spatial Grid & Authority Map Integration Test Suite
==========================================================================
Tests the complete pipeline:
Historical/Live Weather + Spatial Features
                ↓
           ML Prediction
                ↓
      Localized Temperature
                ↓
        Liljegren WBGT
                ↓
        Existing Risk Engine
                ↓
         Map Visualization
"""

import unittest
from unittest.mock import patch
import json
import numpy as np

from app import app
from services.hyperlocal_ml_service import HyperlocalMLService
from services.authority_service import AuthorityService
from engine import calculate_thermal_stress_index, classify_imd, fuse_risk, RISK_COLORS
from ml.src.predict import HyperlocalPredictor


class TestMLSpatialGridIntegration(unittest.TestCase):
    """Exhaustive test suite for ML Hyperlocal Spatial Grid integration."""

    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        HyperlocalPredictor._model = None
        AuthorityService.clear_cache()

    def setUp(self):
        AuthorityService.clear_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. PIPELINE: ML PREDICTION → SPATIAL CELLS
    # ──────────────────────────────────────────────────────────────────────────

    def test_01_ml_prediction_to_spatial_cells(self):
        """1. Verify ML prediction generates ~1km microclimate sector cells."""
        weather = {"temp_c": 39.0, "rh_pct": 40.0, "wind_ms": 2.0, "solar_wm2": 700.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells(
            city_name="Delhi",
            center_lat=28.6139,
            center_lon=77.2090,
            background_weather=weather,
        )
        self.assertEqual(len(cells), 5)
        for cell in cells:
            p = cell["properties"]
            self.assertIn("cell_id", p)
            self.assertIn("ml_temp_c", p)
            self.assertIn("ml_anomaly_c", p)
            self.assertIsInstance(p["ml_temp_c"], float)
            self.assertIsInstance(p["ml_anomaly_c"], float)
            self.assertEqual(p["city"], "Delhi")

    def test_02_spatial_cell_to_deterministic_wbgt(self):
        """2. Verify localized temperature drives deterministic Liljegren WBGT recalculation."""
        weather = {"temp_c": 38.0, "rh_pct": 45.0, "wind_ms": 1.8, "solar_wm2": 720.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells(
            city_name="Hyderabad",
            center_lat=17.3850,
            center_lon=78.4867,
            background_weather=weather,
        )
        for cell in cells:
            p = cell["properties"]
            ml_t = p["ml_temp_c"]
            from ml.src.predict import adjust_humidity_for_temperature_delta
            adj_rh = adjust_humidity_for_temperature_delta(38.0, 45.0, ml_t)
            expected_wbgt = calculate_thermal_stress_index(ml_t, adj_rh, 1.8, 720.0)["wbgt_c"]
            self.assertEqual(p["wbgt_c"], expected_wbgt)

    def test_03_wbgt_to_existing_risk_classification(self):
        """3. Verify cell risk category strictly adheres to core engine fuse_risk()."""
        weather = {"temp_c": 42.0, "rh_pct": 35.0, "wind_ms": 1.5, "solar_wm2": 800.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells(
            city_name="Ahmedabad",
            center_lat=23.0225,
            center_lon=72.5714,
            background_weather=weather,
        )
        for cell in cells:
            p = cell["properties"]
            band = p["wbgt_band"]
            imd = classify_imd(p["ml_temp_c"], "plains")
            expected_risk = fuse_risk(band, imd)
            self.assertEqual(p["risk_category"], expected_risk)
            self.assertEqual(p["risk_color"], RISK_COLORS.get(expected_risk, "#2f9e44"))

    # ──────────────────────────────────────────────────────────────────────────
    # 2. GEOJSON STRUCTURE & STANDARDS
    # ──────────────────────────────────────────────────────────────────────────

    def test_04_geojson_polygon_structure(self):
        """4. Verify each cell geometry is a closed GeoJSON Polygon with 5 vertices."""
        weather = {"temp_c": 36.0, "rh_pct": 50.0, "wind_ms": 2.2, "solar_wm2": 600.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells(
            city_name="Pune",
            center_lat=18.5204,
            center_lon=73.8567,
            background_weather=weather,
        )
        for cell in cells:
            self.assertEqual(cell["type"], "Feature")
            geom = cell["geometry"]
            self.assertEqual(geom["type"], "Polygon")
            coords = geom["coordinates"]
            self.assertIsInstance(coords, list)
            ring = coords[0]
            self.assertEqual(len(ring), 5)  # Closed polygon: 4 corners + return to start
            self.assertEqual(ring[0], ring[-1])  # Start point equals end point

    def test_05_cell_id_and_sector_attributes(self):
        """5. Verify cell_id formatting and sector morphology attribution."""
        weather = {"temp_c": 35.0, "rh_pct": 55.0, "wind_ms": 2.0, "solar_wm2": 650.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells(
            city_name="Delhi",
            center_lat=28.6139,
            center_lon=77.2090,
            background_weather=weather,
        )
        cell_ids = [c["properties"]["cell_id"] for c in cells]
        self.assertEqual(len(set(cell_ids)), 5)  # All cell IDs unique
        self.assertTrue(any("CORE" in cid for cid in cell_ids))
        self.assertTrue(any("INDUS" in cid for cid in cell_ids))
        self.assertTrue(any("GREEN" in cid for cid in cell_ids))

    # ──────────────────────────────────────────────────────────────────────────
    # 3. ROBUSTNESS, FALLBACKS & PROPAGATION
    # ──────────────────────────────────────────────────────────────────────────

    def test_06_missing_ml_prediction_fallback(self):
        """6. Verify graceful fallback if ML inference encounters an unexpected exception."""
        weather = {"temp_c": 37.0, "rh_pct": 40.0, "wind_ms": 2.0, "solar_wm2": 600.0}
        with patch.object(HyperlocalPredictor, "predict_hyperlocal", side_effect=RuntimeError("Model inference engine offline")):
            cells = HyperlocalMLService.generate_spatial_grid_cells(
                city_name="Kolkata",
                center_lat=22.5726,
                center_lon=88.3639,
                background_weather=weather,
            )
            self.assertEqual(len(cells), 5)
            for cell in cells:
                p = cell["properties"]
                self.assertEqual(p["ml_temp_c"], 37.0)
                self.assertEqual(p["ml_anomaly_c"], 0.0)
                self.assertEqual(p["confidence"], "Fallback")

    def test_07_invalid_coordinates_handling(self):
        """7. Verify invalid coordinates return empty feature list without crashing."""
        weather = {"temp_c": 38.0}
        cells_out_of_range = HyperlocalMLService.generate_spatial_grid_cells("Invalid", 999.0, 77.0, weather)
        self.assertEqual(cells_out_of_range, [])

        cells_non_numeric = HyperlocalMLService.generate_spatial_grid_cells("Invalid", "bad_lat", 77.0, weather)
        self.assertEqual(cells_non_numeric, [])

    def test_08_unavailable_spatial_features_fallback(self):
        """8. Verify unknown city coordinates use default morphology safely."""
        weather = {"temp_c": 34.0, "rh_pct": 60.0, "wind_ms": 2.0, "solar_wm2": 500.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells(
            city_name="UnknownRemoteLocation",
            center_lat=24.5000,
            center_lon=81.2000,
            background_weather=weather,
            base_morphology=None,
        )
        self.assertEqual(len(cells), 5)
        self.assertGreaterEqual(cells[0]["properties"]["ml_temp_c"], 30.0)

    def test_09_confidence_and_data_quality_propagation(self):
        """9. Verify confidence state and data quality provenance are correctly propagated."""
        verified_weather = {"temp_c": 40.0, "rh_pct": 30.0, "wind_ms": 2.5, "solar_wm2": 750.0}
        cells_v = HyperlocalMLService.generate_spatial_grid_cells("Jaipur", 26.9124, 75.7873, verified_weather)
        self.assertEqual(cells_v[0]["properties"]["data_quality"], "Verified Reanalysis Input")
        self.assertEqual(cells_v[0]["properties"]["confidence"], "High")

        fallback_weather = {"temp_c": 40.0, "rh_pct": 30.0, "wind_ms": 0.0, "solar_wm2": 0.0}
        cells_fb = HyperlocalMLService.generate_spatial_grid_cells("Jaipur", 26.9124, 75.7873, fallback_weather)
        self.assertEqual(cells_fb[0]["properties"]["data_quality"], "Fallback Meteorological Synthesis")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. EXPLANATION, HONEST LABELS & SAFETY DISCLAIMERS
    # ──────────────────────────────────────────────────────────────────────────

    def test_10_three_tier_labels_presence(self):
        """10. Verify OBSERVED, REANALYSIS, and ML-PREDICTED labels are explicitly present."""
        weather = {"temp_c": 38.5, "rh_pct": 42.0, "wind_ms": 2.1, "solar_wm2": 680.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells("Lucknow", 26.8467, 80.9462, weather)
        labels = cells[0]["properties"]["labels"]
        self.assertIn("OBSERVED", labels["observed_label"])
        self.assertIn("REANALYSIS", labels["reanalysis_label"])
        self.assertIn("ML-PREDICTED", labels["ml_predicted_label"])

    def test_11_spatial_precision_disclaimer(self):
        """11. Verify spatial precision disclaimer confirms ~1km statistical downscaling."""
        weather = {"temp_c": 39.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells("Khammam", 17.2473, 80.1514, weather)
        p = cells[0]["properties"]
        self.assertEqual(p["spatial_resolution"], "~1km Microclimate Cell")
        self.assertIn("not ward-level sensor networks", p["spatial_precision_disclaimer"])

    def test_12_no_medical_claims_in_cell_metadata(self):
        """12. Verify ML metadata explicitly avoids medical or illness diagnosis."""
        weather = {"temp_c": 44.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells("Delhi", 28.6139, 77.2090, weather)
        disclaimer = cells[0]["properties"]["disclaimer"]
        self.assertIn("never declares medical diagnosis or mortality prediction", disclaimer)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. REST API ENDPOINTS
    # ──────────────────────────────────────────────────────────────────────────

    def test_13_api_authority_grid_endpoint(self):
        """13. GET /api/authority/grid returns HTTP 200 with FeatureCollection of grid cells."""
        mock_w = {"temp_c": 39.5, "rh_pct": 38.0, "wind_ms": 2.0, "solar_wm2": 710.0}
        with patch("app.fetch_current_weather", return_value=mock_w):
            res = self.client.get("/api/authority/grid?city=Delhi")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["type"], "FeatureCollection")
            self.assertEqual(data["total_cells"], 5)
            self.assertEqual(len(data["features"]), 5)
            self.assertIn("disclaimer", data)

    def test_14_api_authority_zones_view_grid_compatibility(self):
        """14. GET /api/authority/zones?view=grid returns spatial grid in backward-compatible mode."""
        mock_w = {"temp_c": 41.0, "rh_pct": 32.0, "wind_ms": 1.9, "solar_wm2": 760.0}
        with patch("app.fetch_current_weather", return_value=mock_w):
            res = self.client.get("/api/authority/zones?view=grid&city=Hyderabad")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["type"], "FeatureCollection")
            self.assertGreaterEqual(len(data["features"]), 5)

    def test_15_api_authority_grid_all_cities(self):
        """15. GET /api/authority/grid without city param returns aggregated cells for monitored areas."""
        mock_w = {"temp_c": 37.0, "rh_pct": 45.0, "wind_ms": 2.0, "solar_wm2": 650.0}
        with patch("app.fetch_current_weather", return_value=mock_w):
            res = self.client.get("/api/authority/grid")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["type"], "FeatureCollection")
            self.assertGreaterEqual(len(data["features"]), 50)  # 14 cities * 5 cells = 70 cells


if __name__ == "__main__":
    unittest.main()
