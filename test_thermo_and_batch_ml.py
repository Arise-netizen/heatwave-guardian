"""
HeatWave Guardian — Verification Suite for Scientific Bounds, Caching & Batch ML
================================================================================
Validates:
1. Thermodynamic Psychrometric Bounds: Tw <= Ta across extreme humidity regimes
2. Black Globe Thermodynamic Bounds & Numerical Stability (Tg >= Ta, non-divergent)
3. LRU Memoization & Caching for Liljegren WBGT solver
4. Vectorized HyperlocalPredictor.predict_batch() vs single-point predict_hyperlocal()
5. Batch ML spatial grid generation correctness and integrity
"""

import unittest
import math
import time
from typing import Dict, Any

from engine import (
    estimate_wbgt,
    calculate_wbgt,
    _stull_wet_bulb,
    _globe_temperature,
    clear_wbgt_cache,
    _estimate_wbgt_cached,
)
from ml.src.predict import HyperlocalPredictor, adjust_humidity_for_temperature_delta
from services.hyperlocal_ml_service import HyperlocalMLService


class TestThermodynamicBounds(unittest.TestCase):
    """Test thermodynamic boundaries and physical consistency of engine calculations."""

    def setUp(self):
        clear_wbgt_cache()

    def test_wet_bulb_strictly_bounded_by_dry_bulb(self):
        """Wet bulb temperature Tw can never exceed dry bulb temperature Ta for RH <= 100%."""
        temperatures = [-10.0, 0.0, 15.0, 25.0, 35.0, 42.0, 48.0, 52.0]
        humidities = [5.0, 20.0, 50.0, 80.0, 95.0, 99.0, 100.0]

        for t in temperatures:
            for rh in humidities:
                tw = _stull_wet_bulb(t, rh)
                self.assertLessEqual(
                    tw,
                    t,
                    f"Thermodynamic violation: Tw ({tw:.2f}°C) > Ta ({t:.2f}°C) at RH={rh}%"
                )

    def test_wet_bulb_at_saturation_equals_dry_bulb(self):
        """At 100% RH (complete saturation), Tw approaches Ta and cannot exceed it."""
        for t in [20.0, 30.0, 40.0, 50.0]:
            tw = _stull_wet_bulb(t, 100.0)
            self.assertLessEqual(tw, t)
            # Wet bulb depression at saturation is within empirical tolerance (<0.5°C)
            self.assertAlmostEqual(tw, t, delta=0.5)

    def test_globe_temperature_greater_than_or_equal_to_ambient_in_sun(self):
        """Under positive downward solar radiation, black globe temperature Tg >= Ta."""
        temp = 38.0
        for solar in [100.0, 400.0, 800.0, 1200.0]:
            for wind in [0.2, 1.0, 3.0, 10.0]:
                tg = _globe_temperature(temp, wind, solar)
                self.assertGreaterEqual(
                    tg,
                    temp,
                    f"Globe temp Tg ({tg:.2f}°C) < Ta ({temp:.2f}°C) under solar {solar} W/m²"
                )

    def test_globe_temperature_equals_ambient_in_dark(self):
        """With zero solar radiation, black globe temperature Tg == Ta."""
        for temp in [15.0, 28.0, 42.0]:
            tg = _globe_temperature(temp, 2.0, 0.0)
            self.assertEqual(tg, temp)

    def test_globe_temperature_low_wind_extreme_solar_stability(self):
        """Newton-Raphson solver does not diverge or oscillate under extreme stagnation."""
        temp = 48.0
        extreme_solar = 1400.0
        min_wind = 0.05  # Below standard minimum to test floor
        tg = _globe_temperature(temp, min_wind, extreme_solar)
        self.assertIsInstance(tg, float)
        self.assertFalse(math.isnan(tg))
        self.assertFalse(math.isinf(tg))
        self.assertGreater(tg, temp)
        self.assertLessEqual(tg, temp + 45.0)


class TestWBGTCaching(unittest.TestCase):
    """Test LRU memoization of Liljegren WBGT solver."""

    def setUp(self):
        clear_wbgt_cache()

    def test_cache_hits_and_consistency(self):
        """Repeated calls with identical conditions hit LRU cache and yield exact same values."""
        t, rh, w, s = 38.5, 45.0, 2.2, 700.0
        val1 = estimate_wbgt(t, rh, w, s)
        info_before = _estimate_wbgt_cached.cache_info()

        val2 = estimate_wbgt(t, rh, w, s)
        info_after = _estimate_wbgt_cached.cache_info()

        self.assertEqual(val1, val2)
        self.assertEqual(info_after.hits, info_before.hits + 1)

    def test_cache_clearing(self):
        """clear_wbgt_cache resets the LRU cache."""
        estimate_wbgt(35.0, 50.0, 2.0, 600.0)
        self.assertGreater(_estimate_wbgt_cached.cache_info().currsize, 0)
        clear_wbgt_cache()
        self.assertEqual(_estimate_wbgt_cached.cache_info().currsize, 0)


class TestPredictBatch(unittest.TestCase):
    """Test HyperlocalPredictor.predict_batch and verify 1-to-1 equivalence with predict_hyperlocal."""

    def test_batch_matches_single_prediction(self):
        """Batch prediction returns identical values to individual single-point predictions."""
        weather = {"temp_c": 39.0, "rh_pct": 42.0, "wind_ms": 2.5, "solar_wm2": 650.0}
        locations = [
            {"lat": 28.6139, "lon": 77.2090, "city_name": "Delhi"},
            {"lat": 26.9124, "lon": 75.7873, "city_name": "Jaipur"},
            {"lat": 17.3850, "lon": 78.4867, "city_name": "Hyderabad"},
        ]

        batch_items = [
            {"lat": loc["lat"], "lon": loc["lon"], "background_weather": weather, "city_name": loc["city_name"]}
            for loc in locations
        ]

        batch_results = HyperlocalPredictor.predict_batch(batch_items)
        self.assertEqual(len(batch_results), len(locations))

        for idx, loc in enumerate(locations):
            single_result = HyperlocalPredictor.predict_hyperlocal(
                lat=loc["lat"],
                lon=loc["lon"],
                background_weather=weather,
                city_name=loc["city_name"],
            )
            batch_item = batch_results[idx]

            self.assertEqual(batch_item["localized_temp_c"], single_result["localized_temp_c"])
            self.assertEqual(batch_item["temp_anomaly_c"], single_result["temp_anomaly_c"])
            self.assertEqual(
                batch_item["hyperlocal_environment"]["wbgt_c"],
                single_result["hyperlocal_environment"]["wbgt_c"]
            )
            self.assertEqual(
                batch_item["hyperlocal_environment"]["base_risk"],
                single_result["hyperlocal_environment"]["base_risk"]
            )
            self.assertEqual(
                batch_item["uncertainty"]["ci_90_half_width_c"],
                single_result["uncertainty"]["ci_90_half_width_c"]
            )

    def test_empty_batch(self):
        """Empty batch returns empty list."""
        self.assertEqual(HyperlocalPredictor.predict_batch([]), [])

    def test_single_item_batch(self):
        """Batch of 1 returns a single valid prediction."""
        weather = {"temp_c": 36.0, "rh_pct": 55.0, "wind_ms": 1.8, "solar_wm2": 500.0}
        res = HyperlocalPredictor.predict_batch([{
            "lat": 28.6139,
            "lon": 77.2090,
            "background_weather": weather,
            "city_name": "Delhi",
        }])
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0]["model_available"])
        self.assertIn("localized_temp_c", res[0])

    def test_spatial_grid_cells_using_batch(self):
        """HyperlocalMLService.generate_spatial_grid_cells produces valid GeoJSON using batch."""
        weather = {"temp_c": 41.0, "rh_pct": 35.0, "wind_ms": 2.0, "solar_wm2": 750.0}
        cells = HyperlocalMLService.generate_spatial_grid_cells(
            city_name="Nagpur",
            center_lat=21.1458,
            center_lon=79.0882,
            background_weather=weather,
        )
        self.assertEqual(len(cells), 5)
        for cell in cells:
            self.assertEqual(cell["type"], "Feature")
            self.assertEqual(cell["geometry"]["type"], "Polygon")
            props = cell["properties"]
            self.assertIn("ml_temp_c", props)
            self.assertIn("wbgt_c", props)
            self.assertIn("risk_category", props)
            self.assertGreater(props["ml_temp_c"], 0)


if __name__ == "__main__":
    unittest.main()
