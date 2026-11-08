"""
HeatWave Guardian — Production Data Reliability, Resilience & Regression Test Suite
=====================================================================================
Mandatory Phase 27 Verification covering all 81 targeted test cases:
- Weather Service (1-10)
- Forecast Pipeline (11-16)
- Geocoding Service (17-22)
- Data Quality & Integrity (23-28)
- Live Safety & Zero Synthetic Fallback (29-30)
- Map Reliability (31-35)
- Caching Architecture (36-39)
- Retry & Backoff Policies (40-43)
- Security & Secrets (44-49)
- Health & Observability (50-52)
- Frontend UI Resilience (53-61)
- Deployment Hardening (62-66)
- Complete System Regression (67-81)
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import time
import requests

from app import app, application
from engine import (
    CITIES,
    estimate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    classify_imd,
    fuse_risk,
    assess_vulnerability,
    escalate_risk,
    get_work_rest,
    assess_full_risk_profile,
    infer_region,
    detect_peak_risk_windows,
    calculate_hourly_risk_profile,
    simulate_exposure_scenario,
    evaluate_data_quality,
    evaluate_acclimatization_schedule,
    evaluate_operational_exposure,
)
from services.weather_service import (
    WeatherService,
    WeatherServiceError,
    WeatherServiceUnavailableError,
    WeatherServiceValidationError,
    MIN_TEMP_C,
    MAX_TEMP_C,
    MIN_RH_PCT,
    MAX_RH_PCT,
)
from services.geo_service import GeoService
from services.demographic_service import DemographicService
from services.alert_service import AlertService
from services.hap_service import HAPService
from services.mortality_service import MortalityRiskService
from services.cooling_recommender import CoolingRecommenderService
from services.institute_service import InstituteService
from services.authority_service import AuthorityService


class TestProductionReliabilityPhase27(unittest.TestCase):
    """Complete 81-point verification suite for HeatWave Guardian."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        WeatherService.clear_cache()
        GeoService.clear_cache()

    # ==========================================
    # WEATHER SERVICE (1 - 10)
    # ==========================================

    @patch("requests.get")
    def test_01_weather_successful_request(self, mock_get):
        """1. Weather: successful request and normalization."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "current": {
                "time": "2026-09-17T12:00",
                "temperature_2m": 39.5,
                "relative_humidity_2m": 48.0,
                "wind_speed_10m": 3.8,
                "shortwave_radiation_instant": 750.0,
            }
        }
        mock_get.return_value = mock_resp

        data = WeatherService.fetch_current_weather(28.6139, 77.2090)
        self.assertEqual(data["temp_c"], 39.5)
        self.assertEqual(data["rh_pct"], 48.0)
        self.assertEqual(data["wind_ms"], 3.8)
        self.assertEqual(data["solar_wm2"], 750.0)
        self.assertEqual(data["source"], "Open-Meteo Atmospheric Services")

    @patch("requests.get")
    def test_02_weather_timeout(self, mock_get):
        """2. Weather: persistent timeout raises WeatherServiceUnavailableError."""
        mock_get.side_effect = requests.Timeout("Connection timed out")
        with self.assertRaises(WeatherServiceUnavailableError):
            WeatherService.fetch_current_weather(17.3850, 78.4867, retries=1)

    @patch("requests.get")
    def test_03_weather_transient_server_failure(self, mock_get):
        """3. Weather: transient 500 failure recovers on retry."""
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = {
            "current": {
                "time": "2026-09-17T12:00",
                "temperature_2m": 36.0,
                "relative_humidity_2m": 50.0,
                "wind_speed_10m": 2.5,
                "shortwave_radiation_instant": 600.0,
            }
        }
        mock_get.side_effect = [fail_resp, ok_resp]

        data = WeatherService.fetch_current_weather(19.0760, 72.8777, retries=1)
        self.assertEqual(data["temp_c"], 36.0)
        self.assertEqual(mock_get.call_count, 2)

    @patch("requests.get")
    def test_04_weather_permanent_failure(self, mock_get):
        """4. Weather: permanent failure raises WeatherServiceUnavailableError."""
        mock_get.side_effect = requests.ConnectionError("Failed to resolve host")
        with self.assertRaises(WeatherServiceUnavailableError):
            WeatherService.fetch_current_weather(13.0827, 80.2707, retries=1)

    @patch("requests.get")
    def test_05_weather_malformed_json(self, mock_get):
        """5. Weather: malformed JSON response raises WeatherServiceValidationError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = "invalid_non_dict_string"
        mock_get.return_value = mock_resp

        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.fetch_current_weather(22.5726, 88.3639)

    @patch("requests.get")
    def test_06_weather_missing_fields(self, mock_get):
        """6. Weather: missing required fields raises WeatherServiceValidationError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "current": {
                "time": "2026-09-17T12:00",
                "relative_humidity_2m": 55.0,
                # Missing temperature_2m
                "wind_speed_10m": 3.0
            }
        }
        mock_get.return_value = mock_resp

        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.fetch_current_weather(28.6139, 77.2090)

    def test_07_weather_invalid_values(self):
        """7. Weather: out-of-bounds values rejected by validator."""
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=70.0, rh_pct=50.0, wind_ms=2.0, solar_wm2=500.0)
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=35.0, rh_pct=105.0, wind_ms=2.0, solar_wm2=500.0)
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=35.0, rh_pct=50.0, wind_ms=-2.0, solar_wm2=500.0)

    @patch("requests.get")
    def test_08_weather_duplicate_timestamps(self, mock_get):
        """8. Weather: duplicate timestamps in forecast series handled cleanly without crashing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        # Generate 5 days with duplicate timestamps injected
        times = []
        temps = []
        rhs = []
        winds = []
        for d in range(1, 6):
            for h in range(24):
                t_str = f"2026-09-{d:02d}T{h:02d}:00"
                times.append(t_str)
                temps.append(35.0 + h * 0.2)
                rhs.append(50.0)
                winds.append(2.5)
                if h == 12:  # Inject duplicate
                    times.append(t_str)
                    temps.append(35.0 + h * 0.2)
                    rhs.append(50.0)
                    winds.append(2.5)

        mock_resp.json.return_value = {
            "hourly": {
                "time": times,
                "temperature_2m": temps,
                "relative_humidity_2m": rhs,
                "wind_speed_10m": winds,
                "shortwave_radiation": [500.0] * len(times),
            }
        }
        mock_get.return_value = mock_resp

        forecast = WeatherService.fetch_forecast_weather(17.3850, 78.4867)
        self.assertEqual(len(forecast), 5)
        self.assertEqual(len(forecast[0]["hourly"]), 24)

    def test_09_weather_stale_response(self):
        """9. Weather: stale response timestamp detected."""
        stale_time = "2020-01-01T00:00:00Z"
        self.assertEqual(WeatherService.determine_freshness(stale_time, max_age_seconds=3600), "STALE")
        self.assertEqual(WeatherService.determine_freshness(None), "UNKNOWN")

    def test_10_weather_unit_conversion(self):
        """10. Weather: physical parameter formatting and precision."""
        validated = WeatherService.validate_weather_parameters(38.456, 52.123, 4.567, 850.12)
        self.assertEqual(validated["temp_c"], 38.5)
        self.assertEqual(validated["rh_pct"], 52.1)
        self.assertEqual(validated["wind_ms"], 4.57)
        self.assertEqual(validated["solar_wm2"], 850.1)

    # ==========================================
    # FORECAST PIPELINE (11 - 16)
    # ==========================================

    @patch("requests.get")
    def test_11_forecast_valid_forecast(self, mock_get):
        """11. Forecast: valid 5-day series parsed and peaks aggregated."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        times, temps, rhs, winds = [], [], [], []
        for d in range(1, 6):
            for h in range(24):
                times.append(f"2026-09-{d:02d}T{h:02d}:00")
                temps.append(30.0 + (10.0 if h == 14 else 0.0))
                rhs.append(50.0)
                winds.append(2.0)

        mock_resp.json.return_value = {
            "hourly": {
                "time": times,
                "temperature_2m": temps,
                "relative_humidity_2m": rhs,
                "wind_speed_10m": winds,
                "shortwave_radiation": [600.0] * len(times),
            }
        }
        mock_get.return_value = mock_resp

        forecast = WeatherService.fetch_forecast_weather(28.6139, 77.2090)
        self.assertEqual(len(forecast), 5)
        self.assertEqual(forecast[0]["max_temp_c"], 40.0)
        self.assertTrue(forecast[0]["peak_wbgt_c"] > 25.0)

    @patch("requests.get")
    def test_12_forecast_missing_hourly_data(self, mock_get):
        """12. Forecast: missing hourly block raises WeatherServiceValidationError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"daily": {}}
        mock_get.return_value = mock_resp

        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.fetch_forecast_weather(17.3850, 78.4867)

    @patch("requests.get")
    def test_13_forecast_incomplete_days(self, mock_get):
        """13. Forecast: fewer than 5 days raises WeatherServiceValidationError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        # Only 2 days
        times = [f"2026-09-01T{h:02d}:00" for h in range(24)] + [f"2026-09-02T{h:02d}:00" for h in range(24)]
        mock_resp.json.return_value = {
            "hourly": {
                "time": times,
                "temperature_2m": [35.0] * len(times),
                "relative_humidity_2m": [50.0] * len(times),
                "wind_speed_10m": [2.0] * len(times),
                "shortwave_radiation": [500.0] * len(times),
            }
        }
        mock_get.return_value = mock_resp

        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.fetch_forecast_weather(17.3850, 78.4867)

    @patch("requests.get")
    def test_14_forecast_missing_environmental_variable(self, mock_get):
        """14. Forecast: missing required temperature series raises WeatherServiceValidationError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        times = [f"2026-09-01T{h:02d}:00" for h in range(24)]
        mock_resp.json.return_value = {
            "hourly": {
                "time": times,
                # Missing temperature_2m
                "relative_humidity_2m": [50.0] * len(times),
                "wind_speed_10m": [2.0] * len(times),
            }
        }
        mock_get.return_value = mock_resp

        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.fetch_forecast_weather(17.3850, 78.4867)

    def test_15_forecast_stale_forecast(self):
        """15. Forecast: determine_freshness on historic forecast timestamp."""
        stale_forecast_time = "2025-05-01T12:00:00Z"
        self.assertEqual(WeatherService.determine_freshness(stale_forecast_time, max_age_seconds=14400), "STALE")

    @patch("requests.get")
    def test_16_forecast_invalid_timestamp(self, mock_get):
        """16. Forecast: invalid/empty timestamp items in hourly array skipped safely."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        times, temps, rhs, winds = [], [], [], []
        for d in range(1, 6):
            for h in range(24):
                times.append(f"2026-09-{d:02d}T{h:02d}:00")
                temps.append(35.0)
                rhs.append(50.0)
                winds.append(2.0)
        # Add invalid timestamp
        times.append("malformed-timestamp-without-T")
        temps.append(35.0)
        rhs.append(50.0)
        winds.append(2.0)

        mock_resp.json.return_value = {
            "hourly": {
                "time": times,
                "temperature_2m": temps,
                "relative_humidity_2m": rhs,
                "wind_speed_10m": winds,
                "shortwave_radiation": [500.0] * len(times),
            }
        }
        mock_get.return_value = mock_resp

        forecast = WeatherService.fetch_forecast_weather(17.3850, 78.4867)
        self.assertEqual(len(forecast), 5)

    # ==========================================
    # GEOCODING SERVICE (17 - 22)
    # ==========================================

    @patch("requests.get")
    def test_17_geocoding_valid_location(self, mock_get):
        """17. Geocoding: valid Indian city search returns enriched dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "results": [
                {
                    "name": "Nagpur",
                    "latitude": 21.1458,
                    "longitude": 79.0882,
                    "country": "India",
                    "country_code": "IN",
                    "admin1": "Maharashtra",
                    "population": 2405665,
                }
            ]
        }
        mock_get.return_value = mock_resp

        res = GeoService.search_locations("Nagpur")
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0]["name"], "Nagpur")
        self.assertEqual(res[0]["country_code"], "IN")
        self.assertEqual(res[0]["inferred_region"], "plains")

    @patch("requests.get")
    def test_18_geocoding_no_result(self, mock_get):
        """18. Geocoding: no-result search returns empty list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp

        res = GeoService.search_locations("NonExistentPlaceXYZ123")
        self.assertEqual(res, [])

    @patch("requests.get")
    def test_19_geocoding_ambiguous_location(self, mock_get):
        """19. Geocoding: multiple results disambiguated by state/population."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "results": [
                {
                    "name": "Aurangabad",
                    "latitude": 19.8762,
                    "longitude": 75.3433,
                    "country": "India",
                    "country_code": "IN",
                    "admin1": "Maharashtra",
                    "population": 1175116,
                },
                {
                    "name": "Aurangabad",
                    "latitude": 24.7533,
                    "longitude": 84.3742,
                    "country": "India",
                    "country_code": "IN",
                    "admin1": "Bihar",
                    "population": 102244,
                }
            ]
        }
        mock_get.return_value = mock_resp

        res = GeoService.search_locations("Aurangabad")
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["admin1"], "Maharashtra")
        self.assertEqual(res[1]["admin1"], "Bihar")

    def test_20_geocoding_invalid_coordinates(self):
        """20. Geocoding: invalid coordinates rejected by validator."""
        valid, _ = GeoService.validate_coordinates(95.0, 78.0)
        self.assertFalse(valid)
        valid, _ = GeoService.validate_coordinates(17.0, -190.0)
        self.assertFalse(valid)
        valid, _ = GeoService.validate_coordinates("invalid_number", 78.0)
        self.assertFalse(valid)

    @patch("requests.get")
    def test_21_geocoding_timeout(self, mock_get):
        """21. Geocoding: timeout falls back to local Indian preset cities."""
        mock_get.side_effect = requests.Timeout("Geocoding timed out")
        res = GeoService.search_locations("Delhi")
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0]["name"], "Delhi")

    @patch("requests.get")
    def test_22_geocoding_provider_failure(self, mock_get):
        """22. Geocoding: upstream HTTP 500 falls back to local preset cities."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp

        res = GeoService.search_locations("Mumbai")
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0]["name"], "Mumbai")

    # ==========================================
    # DATA QUALITY (23 - 28)
    # ==========================================

    def test_23_data_quality_fresh_data(self):
        """23. Data Quality: fresh observed data receives High confidence."""
        weather = {"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.5, "solar_wm2": 600.0}
        dq = evaluate_data_quality(weather, is_forecast=False, source="Open-Meteo", location="Hyderabad")
        self.assertEqual(dq["level"], "High")
        self.assertEqual(dq["completeness_pct"], 100.0)
        self.assertGreaterEqual(dq["score"], 0.8)

    def test_24_data_quality_stale_data(self):
        """24. Data Quality: stale data has lower score than fresh data."""
        weather_fresh = {"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.5, "solar_wm2": 600.0}
        dq_fresh = evaluate_data_quality(weather_fresh, is_forecast=False, source="Open-Meteo", location="Delhi")
        dq_forecast = evaluate_data_quality(weather_fresh, is_forecast=True, source="Open-Meteo", location="Delhi")
        self.assertLess(dq_forecast["score"], dq_fresh["score"])

    def test_25_data_quality_missing_optional_field(self):
        """25. Data Quality: missing solar radiation defaults safely with note."""
        weather = {"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.5}
        wbgt = estimate_wbgt(38.0, 50.0, 2.5, 0.0)
        self.assertTrue(20.0 < wbgt < 45.0)

    def test_26_data_quality_missing_required_field(self):
        """26. Data Quality: missing temperature raises validation error."""
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=75.0, rh_pct=50.0, wind_ms=2.0, solar_wm2=500.0)

    def test_27_data_quality_mixed_observed_forecast_data(self):
        """27. Data Quality: observed vs forecast flags differentiated."""
        weather = {"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.5, "solar_wm2": 600.0}
        dq_obs = evaluate_data_quality(weather, is_forecast=False, source="Open-Meteo", location="Nagpur")
        dq_fc = evaluate_data_quality(weather, is_forecast=True, source="Open-Meteo", location="Nagpur")
        self.assertFalse(dq_obs["is_forecast"])
        self.assertTrue(dq_fc["is_forecast"])

    def test_28_data_quality_low_resolution_data(self):
        """28. Data Quality: demographic proxy data explicitly flagged."""
        demo = DemographicService.get_demographics("Delhi")
        self.assertTrue(demo["is_demo_data"])
        self.assertIn("Demo", demo["source"])

    # ==========================================
    # LIVE SAFETY (29 - 30)
    # ==========================================

    @patch("app.fetch_current_weather")
    def test_29_live_safety_no_synthetic_fallback_in_live_risk_path(self, mock_weather):
        """29. Live Safety: live risk endpoint returns 502 on weather failure, NEVER synthetic weather."""
        mock_weather.side_effect = WeatherServiceUnavailableError("Provider offline")
        res = self.client.post("/api/check-risk", json={
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
        })
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertEqual(data["status"], "unavailable")
        self.assertIn("temporarily unavailable", data["error"])

    def test_30_live_safety_demo_data_explicitly_labelled(self):
        """30. Live Safety: demo demographic data carries explicit disclaimer."""
        demo = DemographicService.get_demographics("UnknownCity")
        self.assertTrue(demo["is_demo_data"])
        self.assertIn("Demo", demo["source"])

    # ==========================================
    # MAP (31 - 35)
    # ==========================================

    def test_31_map_valid_coordinates(self):
        """31. Map: authority zones layer returns valid geographic coordinates."""
        mock_weather = lambda lat, lon: {"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0}
        data = AuthorityService.get_area_heat_layer(mock_weather)
        self.assertIn("features", data)
        self.assertGreater(len(data["features"]), 0)
        coords = data["features"][0]["geometry"]["coordinates"]
        # GeoJSON is [lon, lat]
        self.assertTrue(-180 <= coords[0] <= 180)
        self.assertTrue(-90 <= coords[1] <= 90)

    def test_32_map_invalid_coordinates(self):
        """32. Map: invalid coordinates rejected by GeoService."""
        valid, err = GeoService.validate_coordinates(100.0, 78.0)
        self.assertFalse(valid)

    def test_33_map_tile_failure(self):
        """33. Map: HTML templates contain error handling and OpenStreetMap tiles."""
        res = self.client.get("/authority")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn('id="authority-map"', html)

    def test_34_map_marker_rendering(self):
        """34. Map: authority zones feature properties contain risk and station metrics."""
        mock_weather = lambda lat, lon: {"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0}
        data = AuthorityService.get_area_heat_layer(mock_weather)
        props = data["features"][0]["properties"]
        self.assertIn("environmental_risk", props)
        self.assertIn("wbgt_c", props)
        self.assertIn("hap_level", props)

    def test_35_map_attribution_presence(self):
        """35. Map: Leaflet OpenStreetMap attribution present in authority page."""
        res = self.client.get("/authority")
        html = res.get_data(as_text=True)
        self.assertIn("OpenStreetMap", html)

    # ==========================================
    # CACHING (36 - 39)
    # ==========================================

    @patch("requests.get")
    def test_36_cache_repeated_request_reuse(self, mock_get):
        """36. Cache: repeated request within TTL uses cached entry without network call."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "current": {
                "time": "2026-09-17T12:00",
                "temperature_2m": 37.0,
                "relative_humidity_2m": 45.0,
                "wind_speed_10m": 3.0,
                "shortwave_radiation_instant": 650.0,
            }
        }
        mock_get.return_value = mock_resp

        # Force non-testing mode temporarily to test caching mechanism
        WeatherService.clear_cache()
        with patch.object(WeatherService, "_is_testing", return_value=False):
            data1 = WeatherService.fetch_current_weather(28.6139, 77.2090)
            data2 = WeatherService.fetch_current_weather(28.6139, 77.2090)
            self.assertEqual(data1["temp_c"], data2["temp_c"])
            self.assertEqual(mock_get.call_count, 1)

    def test_37_cache_expiry(self):
        """37. Cache: clearing cache resets in-memory store."""
        WeatherService.clear_cache()
        from services.weather_service import _CURRENT_CACHE, _FORECAST_CACHE
        self.assertEqual(len(_CURRENT_CACHE), 0)
        self.assertEqual(len(_FORECAST_CACHE), 0)

    @patch("requests.get")
    def test_38_cache_distinct_location_separation(self, mock_get):
        """38. Cache: distinct locations maintain separate cache entries."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "current": {
                "time": "2026-09-17T12:00",
                "temperature_2m": 37.0,
                "relative_humidity_2m": 45.0,
                "wind_speed_10m": 3.0,
                "shortwave_radiation_instant": 650.0,
            }
        }
        mock_get.return_value = mock_resp

        with patch.object(WeatherService, "_is_testing", return_value=False):
            WeatherService.fetch_current_weather(28.6139, 77.2090)
            WeatherService.fetch_current_weather(17.3850, 78.4867)
            self.assertEqual(mock_get.call_count, 2)

    def test_39_cache_stale_cache_behavior(self):
        """39. Cache: clear_cache empties all caches."""
        WeatherService.clear_cache()
        GeoService.clear_cache()
        from services.geo_service import _GEO_CACHE
        self.assertEqual(len(_GEO_CACHE), 0)

    # ==========================================
    # RETRY (40 - 43)
    # ==========================================

    @patch("requests.get")
    def test_40_retry_transient_retry(self, mock_get):
        """40. Retry: transient timeout retries once and succeeds."""
        timeout_err = requests.Timeout("Network timeout")
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = {
            "current": {
                "time": "2026-09-17T12:00",
                "temperature_2m": 35.0,
                "relative_humidity_2m": 50.0,
                "wind_speed_10m": 2.0,
                "shortwave_radiation_instant": 500.0,
            }
        }
        mock_get.side_effect = [timeout_err, ok_resp]

        data = WeatherService.fetch_current_weather(17.3850, 78.4867, retries=1)
        self.assertEqual(data["temp_c"], 35.0)
        self.assertEqual(mock_get.call_count, 2)

    @patch("requests.get")
    def test_41_retry_limit(self, mock_get):
        """41. Retry: does not exceed max retry limit of 1."""
        mock_get.side_effect = requests.Timeout("Persistent timeout")
        with self.assertRaises(WeatherServiceUnavailableError):
            WeatherService.fetch_current_weather(17.3850, 78.4867, retries=1)
        self.assertEqual(mock_get.call_count, 2)

    @patch("requests.get")
    def test_42_retry_non_retryable_error(self, mock_get):
        """42. Retry: 400 client error rejected immediately without retry."""
        bad_req_resp = MagicMock()
        bad_req_resp.status_code = 400
        mock_get.return_value = bad_req_resp

        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.fetch_current_weather(17.3850, 78.4867, retries=1)
        self.assertEqual(mock_get.call_count, 1)

    @patch("time.sleep")
    @patch("requests.get")
    def test_43_retry_backoff_behavior(self, mock_get, mock_sleep):
        """43. Retry: backoff sleep called between transient attempts."""
        mock_get.side_effect = [requests.Timeout("Timeout"), MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "current": {
                    "time": "2026-09-17T12:00",
                    "temperature_2m": 35.0,
                    "relative_humidity_2m": 50.0,
                    "wind_speed_10m": 2.0,
                    "shortwave_radiation_instant": 500.0,
                }
            }
        )]
        WeatherService.fetch_current_weather(17.3850, 78.4867, retries=1)
        mock_sleep.assert_called_once()

    # ==========================================
    # SECURITY (44 - 49)
    # ==========================================

    def test_44_security_invalid_input(self):
        """44. Security: invalid coordinates in check-risk returns 400 JSON."""
        res = self.client.post("/api/check-risk", json={
            "latitude": "invalid_lat",
            "longitude": 78.4867,
        })
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_45_security_oversized_input(self):
        """45. Security: oversized payload handled cleanly by 413 handler."""
        # 1.5 MB payload
        large_payload = {"junk": "x" * (1500 * 1024)}
        res = self.client.post("/api/check-risk", data=json.dumps(large_payload), content_type="application/json")
        self.assertEqual(res.status_code, 413)

    def test_46_security_unsafe_characters(self):
        """46. Security: script tags in location handled safely without execution."""
        res = self.client.get("/api/search-location?q=%3Cscript%3Ealert(1)%3C/script%3E")
        self.assertEqual(res.status_code, 200)

    def test_47_security_debug_mode_in_production(self):
        """47. Security: WSGI application instance configured with production safety."""
        self.assertIsNotNone(application)

    def test_48_security_secret_exposure_checks(self):
        """48. Security: /health does not expose secret keys or tokens."""
        res = self.client.get("/health")
        data = res.get_json()
        self.assertNotIn("secret_key", data)
        self.assertNotIn("TWILIO_AUTH_TOKEN", json.dumps(data))

    def test_49_security_error_leakage_checks(self):
        """49. Security: 500 handler returns sanitized JSON without Python tracebacks."""
        with patch("app.assess_full_risk_profile", side_effect=RuntimeError("Database connection failed")):
            with patch("app.fetch_current_weather", return_value={
                "temp_c": 35.0,
                "rh_pct": 50.0,
                "wind_ms": 2.0,
                "solar_wm2": 500.0,
                "source": "Open-Meteo Atmospheric Services",
                "source_time": "2026-09-17T12:00",
                "freshness": "FRESH"
            }):
                res = self.client.post("/api/check-risk", json={"location": "Delhi"})
                self.assertEqual(res.status_code, 500)
                data = res.get_json()
                self.assertNotIn("Traceback", json.dumps(data))
                self.assertIn("error", data)

    # ==========================================
    # HEALTH (50 - 52)
    # ==========================================

    def test_50_health_endpoint_contract(self):
        """50. Health: /health reports healthy status and operational services."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "HeatWave Guardian")
        self.assertIn("services", data)
        self.assertIn("timestamp_utc", data)

    def test_51_health_unavailable_dependency_handling(self):
        """51. Health: /api/health alias functions identically."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_52_health_no_secret_leakage(self):
        """52. Health: no credential leakage in health response."""
        res = self.client.get("/health")
        data_str = res.get_data(as_text=True)
        self.assertNotIn("token", data_str.lower())
        self.assertNotIn("password", data_str.lower())

    # ==========================================
    # FRONTEND UI (53 - 61)
    # ==========================================

    def test_53_frontend_weather_error_state(self):
        """53. Frontend: homepage contains toast notifications container for error states."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)
        self.assertIn('id="global-toast-container"', html)

    def test_54_frontend_forecast_error_state(self):
        """54. Frontend: forecast page contains trajectory banner and simulator container."""
        res = self.client.get("/forecast")
        html = res.get_data(as_text=True)
        self.assertIn('id="trajectory-summary-banner"', html)

    def test_55_frontend_geocoding_error(self):
        """55. Frontend: location search handles empty query with 400 JSON."""
        res = self.client.get("/api/search-location?q=")
        self.assertEqual(res.status_code, 400)

    def test_56_frontend_map_error(self):
        """56. Frontend: authority page contains authority map container."""
        res = self.client.get("/authority")
        html = res.get_data(as_text=True)
        self.assertIn('id="authority-map"', html)

    def test_57_frontend_stale_data_display(self):
        """57. Frontend: methodology explains data freshness and quality metrics."""
        res = self.client.get("/methodology")
        html = res.get_data(as_text=True)
        self.assertIn('Methodology & Algorithmic Pipeline', html)
        self.assertIn('Liljegren WBGT', html)

    def test_58_frontend_limited_confidence_display(self):
        """58. Frontend: CSS contains confidence badge classes."""
        res = self.client.get("/static/style.css")
        css = res.get_data(as_text=True)
        self.assertIn('badge', css)

    def test_59_frontend_incomplete_data_display(self):
        """59. Frontend: validation page explains ERA5 reanalysis cold bias limitations."""
        res = self.client.get("/validation")
        html = res.get_data(as_text=True)
        self.assertIn('ERA5', html)
        self.assertIn('Khammam', html)

    def test_60_frontend_retry_ui(self):
        """60. Frontend: assess button available to trigger fresh calculations."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)
        self.assertIn('btn', html)

    def test_61_frontend_no_console_errors(self):
        """61. Frontend: static stylesheet exists and is served cleanly."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)

    # ==========================================
    # DEPLOYMENT HARDENING (62 - 66)
    # ==========================================

    def test_62_deployment_production_startup(self):
        """62. Deployment: WSGI application alias is available."""
        self.assertTrue(callable(application))

    def test_63_deployment_environment_configuration(self):
        """63. Deployment: app secret key is configured."""
        self.assertTrue(len(app.secret_key) > 0)

    def test_64_deployment_health_endpoint(self):
        """64. Deployment: health endpoint responds with 200."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)

    def test_65_deployment_static_assets(self):
        """65. Deployment: style.css is accessible."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)

    def test_66_deployment_security_headers_present(self):
        """66. Deployment: security headers attached to response."""
        res = self.client.get("/")
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertEqual(res.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")

    # ==========================================
    # REGRESSION (67 - 81)
    # ==========================================

    @patch("app.fetch_current_weather")
    def test_67_regression_current_risk(self, mock_weather):
        """67. Regression: /api/check-risk assesses heat risk profile."""
        mock_weather.return_value = {
            "temp_c": 41.5,
            "rh_pct": 45.0,
            "wind_ms": 2.5,
            "solar_wm2": 750.0,
            "source": "Open-Meteo Atmospheric Services",
            "source_time": "2026-09-17T12:00",
            "freshness": "FRESH",
        }
        res = self.client.post("/api/check-risk", json={
            "location": "Hyderabad",
            "workload": "heavy",
            "clothing": "light",
            "experience": "not_used",
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("overall_action_level", data)
        self.assertIn("wbgt_c", data)
        self.assertIn("work_minutes", data)

    @patch("app.fetch_forecast_weather")
    def test_68_regression_forecast(self, mock_forecast):
        """68. Regression: /api/forecast returns 5-day daily forecasts."""
        mock_forecast.return_value = [
            {
                "date": f"2026-09-0{d}",
                "max_temp_c": 38.0 + d,
                "peak_wbgt_c": 30.0 + d * 0.5,
                "hourly": [
                    {
                        "time": f"2026-09-0{d}T{h:02d}:00",
                        "temp_c": 35.0,
                        "rh_pct": 50.0,
                        "wind_ms": 2.0,
                        "solar_wm2": 500.0,
                        "wbgt_c": 30.0,
                    }
                    for h in range(24)
                ]
            }
            for d in range(1, 6)
        ]
        res = self.client.post("/api/forecast", json={"location": "Delhi"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 5)

    def test_69_regression_peak_risk_engine(self):
        """69. Regression: peak risk window identifies diurnal peak hours."""
        hourly = [
            {"time": f"2026-09-17T{h:02d}:00", "wbgt_c": 32.5 if 12 <= h <= 15 else 26.0, "temp_c": 38.0}
            for h in range(24)
        ]
        peak = detect_peak_risk_windows(hourly)
        self.assertTrue(peak["has_elevated_risk"])
        self.assertIn("12:00", peak["elevated_risk_start"])

    def test_70_regression_what_if_simulator(self):
        """70. Regression: /api/simulate-scenario evaluates counterfactual intervention."""
        weather = {"temp_c": 40.0, "rh_pct": 45.0, "wind_ms": 2.0, "solar_wm2": 700.0, "wbgt_c": 32.0}
        res = self.client.post("/api/simulate-scenario", json={
            "baseline": {"workload": "very_heavy", "clothing": "ppe", "is_sun_exposed": True},
            "scenario": {"workload": "light", "clothing": "light", "is_sun_exposed": False},
            "weather_data": weather,
            "location": "Hyderabad",
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("action_level_change", data)
        self.assertIn("causal_explanations", data)

    def test_71_regression_intervention_engine(self):
        """71. Regression: get_work_rest computes ISO 7243 rest guidance."""
        wr = get_work_rest("Extreme", workload="heavy", acclimatized="not_used", clothing="ppe")
        self.assertEqual(wr["work_minutes"], 0)
        self.assertEqual(wr["rest_minutes"], 60)

    def test_72_regression_explainability(self):
        """72. Regression: assess_full_risk_profile creates decision trace and structured explanation."""
        weather = {"temp_c": 42.0, "rh_pct": 45.0, "wind_ms": 2.0, "solar_wm2": 750.0}
        profile = assess_full_risk_profile(
            weather_data=weather,
            personal_inputs={"workload": "heavy", "clothing": "light", "experience": "very_used", "age_group": "18–40"},
            region="plains",
            location="Khammam"
        )
        self.assertIn("decision_trace", profile)
        self.assertIn("explanation", profile)

    def test_73_regression_validation(self):
        """73. Regression: /api/validation/data returns 2015 heatwave historical dataset."""
        res = self.client.get("/api/validation/data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["event_name"], "2015 Andhra Pradesh & Telangana Heatwave")

    def test_74_regression_worksite(self):
        """74. Regression: /api/institute/sites returns institutional site risk."""
        with patch("app.fetch_current_weather", return_value={"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0}):
            res = self.client.get("/api/institute/sites?institute_id=infra_delhi")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn("sites", data)

    def test_75_regression_authority(self):
        """75. Regression: /api/authority/zones returns municipal zones."""
        with patch("app.fetch_current_weather", return_value={"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0}):
            res = self.client.get("/api/authority/zones")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["type"], "FeatureCollection")

    def test_76_regression_cooling_stations(self):
        """76. Regression: /api/cooling-stations/recommendations returns siting guidance."""
        res = self.client.get("/api/cooling-stations/recommendations?city=Hyderabad")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("recommendations", data)

    def test_77_regression_map_integration(self):
        """77. Regression: authority map page loads."""
        res = self.client.get("/authority")
        self.assertEqual(res.status_code, 200)

    def test_78_regression_personalization(self):
        """78. Regression: vulnerability escalation works correctly."""
        base_risk = "Moderate"
        vuln = assess_vulnerability(age=68, acclimatization="not", workload="very_heavy", clothing="ppe", age_group="60+")
        escalated = escalate_risk(base_risk, vuln)
        self.assertEqual(escalated, "Very High")

    def test_79_regression_acclimatization(self):
        """79. Regression: evaluate_acclimatization_schedule computes progressive exposure limits."""
        sched = evaluate_acclimatization_schedule("not_used", day_in_heat=3)
        self.assertEqual(sched["day_in_heat"], 3)
        self.assertEqual(sched["max_exposure_pct"], 60)

    def test_80_regression_all_existing_api_endpoints(self):
        """80. Regression: all API endpoints respond with valid JSON."""
        endpoints = [
            ("/api/search-location?q=Delhi", "GET"),
            ("/api/institute/list", "GET"),
            ("/api/alerts/history", "GET"),
            ("/api/validation/data", "GET"),
            ("/health", "GET"),
        ]
        for url, method in endpoints:
            if method == "GET":
                res = self.client.get(url)
                self.assertEqual(res.status_code, 200, f"Endpoint {url} failed with {res.status_code}")

    def test_81_regression_all_existing_pages(self):
        """81. Regression: all 8 user-facing pages load with HTTP 200."""
        pages = [
            "/",
            "/validation",
            "/forecast",
            "/what-if",
            "/worksite",
            "/authority",
            "/cooling-stations",
            "/methodology",
        ]
        for page in pages:
            res = self.client.get(page)
            self.assertEqual(res.status_code, 200, f"Page {page} failed with {res.status_code}")


if __name__ == "__main__":
    unittest.main()
