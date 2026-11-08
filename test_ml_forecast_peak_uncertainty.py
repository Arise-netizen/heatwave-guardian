"""
HeatWave Guardian — ML Heat Forecast, Peak-Risk & Uncertainty Test Suite
========================================================================
Validates:
1. Forecast feature compatibility (Open-Meteo variables → 16-feature vector)
2. ML future inference across hourly timesteps (localized temp, Liljegren WBGT)
3. Dynamic peak-risk period detection (strictly derived from data, not hardcoded)
4. Technically defensible uncertainty estimation (RF 100-tree ensemble spread)
5. Candidate lower-risk periods labeled "Lower forecast heat-stress window"
6. Dynamic risk explanation ("Why is this period highest risk?")
7. Missing forecast values resilience
8. Model unavailable graceful fallback
9. Invalid feature inputs boundary handling
10. API schema backward compatibility (POST /api/forecast, GET /api/forecast)
11. Existing risk engine and spatial map integration compatibility
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import numpy as np

from app import app
from services.hyperlocal_ml_service import HyperlocalMLService
from services.weather_service import WeatherService
from engine import (
    calculate_thermal_stress_index,
    classify_imd,
    fuse_risk,
    RISK_COLORS,
)
from ml.src.predict import HyperlocalPredictor, predict_hyperlocal, get_feature_names


class TestMLForecastPeakUncertainty(unittest.TestCase):
    """Exhaustive test suite for Step 4 ML Heat Forecast, Peak Risk, and Uncertainty."""

    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def setUp(self):
        # Ensure fresh model state
        HyperlocalPredictor._model = None

    # ──────────────────────────────────────────────────────────────────────────
    # 1. FORECAST FEATURE COMPATIBILITY
    # ──────────────────────────────────────────────────────────────────────────

    def test_01_forecast_feature_compatibility(self):
        """Verify that Open-Meteo forecast variables map cleanly to ML non-leaking features."""
        feature_names = get_feature_names()
        self.assertEqual(len(feature_names), 12)
        expected_features = [
            "background_temp_c", "background_rh_pct", "background_wind_ms", "background_solar_wm2",
            "elevation_m", "urban_built_up_ratio", "vegetation_canopy_deficit", "population_density_k",
            "distance_to_coast_km", "is_coastal", "day_of_year_sin", "day_of_year_cos"
        ]
        self.assertEqual(feature_names, expected_features)

        # Build feature vector from future forecast timestep
        weather_point = {
            "temp_c": 38.5,
            "rh_pct": 42.0,
            "wind_ms": 2.5,
            "solar_wm2": 750.0,
            "hour": 14,
            "month": 6,
        }
        pred = predict_hyperlocal(
            lat=17.3850,
            lon=78.4867,
            temp_c=weather_point["temp_c"],
            rh_pct=weather_point["rh_pct"],
            wind_ms=weather_point["wind_ms"],
            solar_wm2=weather_point["solar_wm2"],
            hour=weather_point["hour"],
            month=weather_point["month"],
            city_name="Hyderabad",
        )

        self.assertIn("localized_temp_c", pred)
        self.assertIn("temp_anomaly_c", pred)
        self.assertIn("uncertainty", pred)
        self.assertIsInstance(pred["localized_temp_c"], float)
        self.assertIsInstance(pred["temp_anomaly_c"], float)
        # Verify physical sanity
        self.assertGreater(pred["localized_temp_c"], 20.0)
        self.assertLess(pred["localized_temp_c"], 55.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ML FUTURE INFERENCE ACROSS HOURLY TIMESTEPS
    # ──────────────────────────────────────────────────────────────────────────

    def test_02_ml_future_hourly_inference(self):
        """Verify ML localized temperature and Liljegren WBGT calculation per forecast timestep."""
        mock_hourly = []
        base_temp = 32.0
        for h in range(24):
            temp = base_temp + 8.0 * np.sin(np.pi * (h - 6) / 12) if 6 <= h <= 18 else base_temp - 2.0
            solar = max(0.0, 800.0 * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
            mock_hourly.append({
                "time": f"2026-09-22T{h:02d}:00",
                "hour": h,
                "hour_str": f"{h:02d}:00",
                "temp_c": round(float(temp), 1),
                "rh_pct": 45.0,
                "wind_ms": 2.0,
                "solar_wm2": round(float(solar), 1),
            })

        mock_day = {
            "date": "2026-09-22",
            "max_temp_c": 40.0,
            "hourly_profile": mock_hourly,
        }

        enhanced_days = HyperlocalMLService.enhance_forecast_with_ml([mock_day], lat=17.3850, lon=78.4867, city_name="Hyderabad")
        self.assertEqual(len(enhanced_days), 1)
        d = enhanced_days[0]

        # Verify hourly enhancement
        self.assertEqual(len(d["hourly_profile"]), 24)
        for pt in d["hourly_profile"]:
            self.assertIn("ml_localized_temp_c", pt)
            self.assertIn("background_temp_c", pt)
            self.assertIn("ml_anomaly_c", pt)
            self.assertIn("wbgt_c", pt)
            self.assertIn("wbgt_band", pt)
            self.assertIn("risk_category", pt)
            self.assertIn("uncertainty", pt)
            # ML localized temp must differ from or equal background temp + anomaly
            self.assertAlmostEqual(pt["ml_localized_temp_c"], pt["background_temp_c"] + pt["ml_anomaly_c"], places=2)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. DYNAMIC PEAK-RISK PERIOD DETECTION (NOT HARDCODED)
    # ──────────────────────────────────────────────────────────────────────────

    def test_03_dynamic_peak_risk_detection_non_fixed_morning(self):
        """Verify peak risk detection dynamically shifts when peak heat occurs in morning (e.g. 10:00)."""
        mock_hourly = []
        for h in range(24):
            # Create artificial morning peak at 10:00
            if h == 10:
                temp = 44.0
                solar = 950.0
            elif 9 <= h <= 11:
                temp = 42.0
                solar = 850.0
            else:
                temp = 28.0
                solar = 100.0 if 6 <= h <= 18 else 0.0

            mock_hourly.append({
                "time": f"2026-09-22T{h:02d}:00",
                "hour": h,
                "hour_str": f"{h:02d}:00",
                "temp_c": temp,
                "rh_pct": 50.0,
                "wind_ms": 1.0,
                "solar_wm2": solar,
            })

        mock_day = {"date": "2026-09-22", "max_temp_c": 44.0, "hourly_profile": mock_hourly}
        enhanced = HyperlocalMLService.enhance_forecast_with_ml([mock_day], lat=17.3850, lon=78.4867, city_name="Hyderabad")
        d = enhanced[0]

        # Peak must be 10:00, NOT the default 13:00 or 14:00
        self.assertEqual(d["peak_wbgt_timestamp"], "2026-09-22T10:00")
        self.assertEqual(d["peak_ml_temp_timestamp"], "2026-09-22T10:00")
        self.assertIn("10:00", d["peak_risk_window"]["peak_period"])
        self.assertEqual(d["main_elevated_risk_window"]["start"], "09:00")
        self.assertEqual(d["main_elevated_risk_window"]["end"], "11:00")

    def test_04_dynamic_peak_risk_detection_non_fixed_late_afternoon(self):
        """Verify peak risk detection dynamically shifts when peak heat occurs in late afternoon (e.g. 17:00)."""
        mock_hourly = []
        for h in range(24):
            # Artificial late afternoon peak at 17:00
            if h == 17:
                temp = 43.5
                solar = 600.0
            elif 16 <= h <= 18:
                temp = 41.0
                solar = 500.0
            else:
                temp = 30.0
                solar = 200.0 if 6 <= h <= 18 else 0.0

            mock_hourly.append({
                "time": f"2026-09-22T{h:02d}:00",
                "hour": h,
                "hour_str": f"{h:02d}:00",
                "temp_c": temp,
                "rh_pct": 45.0,
                "wind_ms": 1.2,
                "solar_wm2": solar,
            })

        mock_day = {"date": "2026-09-22", "max_temp_c": 43.5, "hourly_profile": mock_hourly}
        enhanced = HyperlocalMLService.enhance_forecast_with_ml([mock_day], lat=17.3850, lon=78.4867, city_name="Hyderabad")
        d = enhanced[0]

        self.assertEqual(d["peak_wbgt_timestamp"], "2026-09-22T17:00")
        self.assertEqual(d["peak_ml_temp_timestamp"], "2026-09-22T17:00")
        self.assertIn("17:00", d["peak_risk_window"]["peak_period"])

    # ──────────────────────────────────────────────────────────────────────────
    # 4. UNCERTAINTY QUANTIFICATION (RF ENSEMBLE TREE SPREAD)
    # ──────────────────────────────────────────────────────────────────────────

    def test_05_technically_defensible_uncertainty(self):
        """Verify ensemble tree spread is derived across estimators, not fabricated."""
        pred = predict_hyperlocal(
            lat=28.6139,
            lon=77.2090,
            temp_c=40.0,
            rh_pct=35.0,
            wind_ms=2.0,
            solar_wm2=800.0,
            hour=14,
            month=5,
            city_name="Delhi",
        )

        uncert = pred.get("uncertainty")
        self.assertIsNotNone(uncert)
        self.assertEqual(uncert["method"], "Random Forest ensemble tree spread")
        self.assertIn("ensemble_tree_std_c", uncert)
        self.assertIn("ci_90_half_width_c", uncert)
        self.assertIn("confidence_state", uncert)
        self.assertIn("data_quality", uncert)

        # Tree std must be strictly positive and realistic (typically 0.05 to 1.5°C)
        std_val = uncert["ensemble_tree_std_c"]
        self.assertGreater(std_val, 0.0)
        self.assertLess(std_val, 3.0)

        # 90% CI must equal 1.645 * std
        expected_ci = round(1.645 * std_val, 2)
        self.assertAlmostEqual(uncert["ci_90_half_width_c"], expected_ci, places=2)

        # Upper and lower bounds must bound the localized temp
        t_loc = pred["localized_temp_c"]
        self.assertAlmostEqual(uncert["temp_lower_c"], round(t_loc - expected_ci, 2), places=2)
        self.assertAlmostEqual(uncert["temp_upper_c"], round(t_loc + expected_ci, 2), places=2)

        # Ensure NO arbitrary strings like "90% confidence" as a probability
        self.assertNotEqual(uncert["confidence_state"], "90% confidence")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. SAFER TIME WINDOWS ("Lower forecast heat-stress window")
    # ──────────────────────────────────────────────────────────────────────────

    def test_06_safer_time_windows_labeling(self):
        """Verify candidate lower-risk periods are clearly labeled as 'Lower forecast heat-stress window'."""
        mock_hourly = []
        for h in range(24):
            # Cool morning 06:00-09:00, hot afternoon 12:00-16:00
            temp = 24.0 if h < 9 else (42.0 if 12 <= h <= 15 else 32.0)
            solar = 200.0 if 6 <= h < 9 else (850.0 if 12 <= h <= 15 else 400.0)
            mock_hourly.append({
                "time": f"2026-09-22T{h:02d}:00",
                "hour": h,
                "hour_str": f"{h:02d}:00",
                "temp_c": temp,
                "rh_pct": 40.0,
                "wind_ms": 2.0,
                "solar_wm2": solar if 6 <= h <= 18 else 0.0,
            })

        mock_day = {"date": "2026-09-22", "max_temp_c": 42.0, "hourly_profile": mock_hourly}
        enhanced = HyperlocalMLService.enhance_forecast_with_ml([mock_day], lat=17.3850, lon=78.4867, city_name="Hyderabad")
        d = enhanced[0]

        low_windows = d.get("lower_risk_candidate_windows", [])
        self.assertGreater(len(low_windows), 0)
        for win in low_windows:
            self.assertEqual(win["label"], "Lower forecast heat-stress window")
            self.assertIn("start", win)
            self.assertIn("end", win)
            self.assertIn("wbgt_range_c", win)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. DYNAMIC EXPLANATION ("Why is this period highest risk?")
    # ──────────────────────────────────────────────────────────────────────────

    def test_07_peak_risk_explanation_drivers(self):
        """Verify dynamic explanation identifies supported environmental and ML drivers."""
        mock_hourly = []
        for h in range(24):
            temp = 43.0 if 13 <= h <= 15 else 28.0
            solar = 900.0 if 13 <= h <= 15 else 100.0
            wind = 0.5 if 13 <= h <= 15 else 3.0  # Stagnant wind during peak
            rh = 60.0 if 13 <= h <= 15 else 40.0  # Elevated humidity
            mock_hourly.append({
                "time": f"2026-09-22T{h:02d}:00",
                "hour": h,
                "hour_str": f"{h:02d}:00",
                "temp_c": temp,
                "rh_pct": rh,
                "wind_ms": wind,
                "solar_wm2": solar if 6 <= h <= 18 else 0.0,
            })

        mock_day = {"date": "2026-09-22", "max_temp_c": 43.0, "hourly_profile": mock_hourly}
        enhanced = HyperlocalMLService.enhance_forecast_with_ml([mock_day], lat=17.3850, lon=78.4867, city_name="Hyderabad")
        expl = enhanced[0].get("peak_explanation")

        self.assertIsNotNone(expl)
        self.assertEqual(expl["question"], "Why is this period highest risk?")
        self.assertIn("drivers", expl)
        drivers = expl["drivers"]
        self.assertGreater(len(drivers), 1)

        # Check for presence of supported drivers
        driver_text = " ".join(drivers)
        self.assertTrue(any("temperature" in d.lower() or "°c" in d.lower() for d in drivers))
        self.assertTrue(any("solar" in d.lower() or "irradiance" in d.lower() for d in drivers))
        self.assertTrue(any("wind" in d.lower() or "stagnation" in d.lower() for d in drivers))

    # ──────────────────────────────────────────────────────────────────────────
    # 7. MISSING FORECAST VALUES HANDLING
    # ──────────────────────────────────────────────────────────────────────────

    def test_08_missing_forecast_values_resilience(self):
        """Verify robust fallbacks when forecast inputs contain None or missing fields."""
        incomplete_hourly = [
            {
                "time": "2026-09-22T12:00",
                "hour": 12,
                "hour_str": "12:00",
                "temp_c": 36.0,
                # Missing rh_pct, wind_ms, solar_wm2
            },
            {
                "time": "2026-09-22T13:00",
                "hour": 13,
                "hour_str": "13:00",
                "temp_c": 37.0,
                "rh_pct": None,
                "wind_ms": None,
                "solar_wm2": None,
            }
        ]
        mock_day = {"date": "2026-09-22", "max_temp_c": 37.0, "hourly_profile": incomplete_hourly}
        # Should execute cleanly without raising KeyError or TypeError
        enhanced = HyperlocalMLService.enhance_forecast_with_ml([mock_day], lat=17.3850, lon=78.4867, city_name="Hyderabad")
        self.assertEqual(len(enhanced), 1)
        self.assertEqual(len(enhanced[0]["hourly_profile"]), 2)
        for pt in enhanced[0]["hourly_profile"]:
            self.assertIn("ml_localized_temp_c", pt)
            self.assertIsInstance(pt["ml_localized_temp_c"], float)

    # ──────────────────────────────────────────────────────────────────────────
    # 8. MODEL UNAVAILABLE FALLBACK
    # ──────────────────────────────────────────────────────────────────────────

    def test_09_model_unavailable_fallback(self):
        """Verify system falls back gracefully to deterministic weather if model cannot be loaded."""
        with patch.object(HyperlocalPredictor, "get_model", return_value=None):
            pred = predict_hyperlocal(
                lat=17.3850,
                lon=78.4867,
                temp_c=38.0,
                rh_pct=45.0,
                wind_ms=2.0,
                solar_wm2=700.0,
            )
            self.assertFalse(pred["model_available"])
            self.assertEqual(pred["temp_anomaly_c"], 0.0)
            self.assertEqual(pred["localized_temp_c"], 38.0)
            self.assertEqual(pred["uncertainty"]["confidence_state"], "Model unavailable (fallback to background forecast)")

    # ──────────────────────────────────────────────────────────────────────────
    # 9. INVALID FEATURE INPUTS BOUNDARY HANDLING
    # ──────────────────────────────────────────────────────────────────────────

    def test_10_invalid_feature_inputs_handling(self):
        """Verify out-of-range temperatures and negative humidities do not crash inference."""
        # Extreme temperatures and invalid humidity
        pred_extreme = predict_hyperlocal(
            lat=17.3850,
            lon=78.4867,
            temp_c=120.0,  # Extreme
            rh_pct=-15.0,  # Invalid negative
            wind_ms=-5.0,  # Invalid negative
            solar_wm2=2000.0,
        )
        self.assertIn("localized_temp_c", pred_extreme)
        self.assertIsInstance(pred_extreme["localized_temp_c"], float)

    # ──────────────────────────────────────────────────────────────────────────
    # 10. API SCHEMA & BACKWARD COMPATIBILITY
    # ──────────────────────────────────────────────────────────────────────────

    def test_11_api_forecast_post_schema_and_fields(self):
        """Verify POST /api/forecast returns both legacy fields and new Step 4 ML fields."""
        payload = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "18–40",
            "work_intensity": "moderate",
            "clothing": "light",
            "experience": "very_used",
            "recovery_environment": "shade",
        }
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 200)
        days = res.get_json()
        self.assertIsInstance(days, list)
        self.assertGreater(len(days), 0)

        for d in days:
            # 1. Existing backward-compatible fields
            self.assertIn("date", d)
            self.assertIn("max_temp_c", d)
            self.assertIn("wbgt_c", d)
            self.assertIn("risk_category", d)
            self.assertIn("work_minutes", d)
            self.assertIn("rest_minutes", d)
            self.assertIn("hourly_profile", d)

            # 2. Step 4 New ML and Peak-Risk fields
            self.assertIn("peak_timestamp", d)
            self.assertIn("peak_wbgt_timestamp", d)
            self.assertIn("peak_ml_temp_timestamp", d)
            self.assertIn("peak_ml_temp_c", d)
            self.assertIn("background_max_temp_c", d)
            self.assertIn("ml_anomaly_c", d)
            self.assertIn("main_elevated_risk_window", d)
            self.assertIn("lower_risk_candidate_windows", d)
            self.assertIn("peak_explanation", d)
            self.assertIn("uncertainty_summary", d)

            # Check hourly schema
            if d.get("hourly_profile"):
                pt = d["hourly_profile"][0]
                self.assertIn("ml_localized_temp_c", pt)
                self.assertIn("background_temp_c", pt)
                self.assertIn("ml_anomaly_c", pt)
                self.assertIn("wbgt_c", pt)
                self.assertIn("wbgt_band", pt)
                self.assertIn("risk_category", pt)
                self.assertIn("uncertainty", pt)
                self.assertIn("is_peak", pt)
                self.assertIn("is_lower_heat_stress_window", pt)

    def test_12_api_forecast_get_schema_and_fields(self):
        """Verify GET /api/forecast also returns Step 4 ML fields."""
        res = self.client.get("/api/forecast?location=Hyderabad&lat=17.3850&lon=78.4867")
        self.assertEqual(res.status_code, 200)
        days = res.get_json()
        self.assertIsInstance(days, list)
        self.assertGreater(len(days), 0)
        d0 = days[0]
        self.assertIn("peak_ml_temp_c", d0)
        self.assertIn("uncertainty_summary", d0)
        self.assertIn("peak_explanation", d0)

    # ──────────────────────────────────────────────────────────────────────────
    # 11. RISK ENGINE & MAP INTEGRATION COMPATIBILITY
    # ──────────────────────────────────────────────────────────────────────────

    def test_13_existing_risk_engine_compatibility(self):
        """Verify ML localized temp seamlessly drives Liljegren WBGT and core risk engine."""
        # Using localized temperature with existing engine functions
        ml_temp = 41.5
        ts = calculate_thermal_stress_index(ml_temp, rh_pct=40.0, wind_ms=2.0, solar_wm2=750.0)
        wbgt_val = ts["wbgt_c"]
        self.assertGreater(wbgt_val, 25.0)

        imd_status = classify_imd(ml_temp, region_type="plains")
        fused = fuse_risk(ts["wbgt_band"], imd_status)
        self.assertIn(fused, ["Low", "Moderate", "High", "Very High", "Extreme"])
        self.assertIn(fused, RISK_COLORS)

    def test_14_existing_spatial_grid_compatibility(self):
        """Verify existing /api/spatial-grid endpoint continues to function perfectly."""
        res = self.client.get("/api/spatial-grid?city=Hyderabad&lat=17.3850&lon=78.4867")
        self.assertEqual(res.status_code, 200)
        grid = res.get_json()
        self.assertEqual(grid.get("type"), "FeatureCollection")
        self.assertIn("features", grid)
        self.assertGreater(len(grid["features"]), 0)


if __name__ == "__main__":
    unittest.main()
