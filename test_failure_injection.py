"""
HeatWave Guardian — Failure Injection & Safe Degradation Test Suite
===================================================================
Phase 28 Verification:
Deliberately injects simulated real-world external provider failures
(timeouts, 500s, 400s, malformed responses, partial series, stale timestamps,
out-of-bounds readings) to verify that HeatWave Guardian fails safely,
transparently, and predictably without synthetic data fallbacks.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
import json

from app import app
from engine import (
    evaluate_data_quality,
    estimate_wbgt,
    classify_thermal_stress,
)
from services.weather_service import (
    WeatherService,
    WeatherServiceError,
    WeatherServiceUnavailableError,
    WeatherServiceValidationError,
    CURRENT_WEATHER_MAX_AGE_SECONDS,
)
from services.geo_service import GeoService


class TestFailureInjectionSuite(unittest.TestCase):
    """Rigorous failure-injection testing suite for Phase 28."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        WeatherService.clear_cache()
        GeoService.clear_cache()

    # 1. Timeout Injection — Current Weather
    @patch("requests.get")
    def test_inject_current_weather_timeout(self, mock_get):
        """Inject persistent socket timeout into current weather fetch."""
        mock_get.side_effect = requests.Timeout("Gateway connection timed out after 8s")
        with self.assertRaises(WeatherServiceUnavailableError) as ctx:
            WeatherService.fetch_current_weather(28.6139, 77.2090, retries=1)
        self.assertIn("temporarily unavailable", str(ctx.exception))

    # 2. Timeout Injection — Forecast Weather
    @patch("requests.get")
    def test_inject_forecast_weather_timeout(self, mock_get):
        """Inject persistent socket timeout into forecast weather fetch."""
        mock_get.side_effect = requests.Timeout("Gateway connection timed out after 10s")
        with self.assertRaises(WeatherServiceUnavailableError) as ctx:
            WeatherService.fetch_forecast_weather(17.3850, 78.4867, retries=1)
        self.assertIn("temporarily unavailable", str(ctx.exception))

    # 3. HTTP 500 Server Error with Recovery on Retry
    @patch("requests.get")
    def test_inject_transient_500_with_retry_recovery(self, mock_get):
        """Inject HTTP 500 on first call, followed by valid 200 on retry."""
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = {
            "current": {
                "time": "2026-09-21T12:00",
                "temperature_2m": 37.5,
                "relative_humidity_2m": 42.0,
                "wind_speed_10m": 3.1,
                "shortwave_radiation_instant": 700.0,
            }
        }
        mock_get.side_effect = [fail_resp, ok_resp]

        result = WeatherService.fetch_current_weather(17.3850, 78.4867, retries=1)
        self.assertEqual(result["temp_c"], 37.5)
        self.assertEqual(mock_get.call_count, 2)

    # 4. HTTP 400 Bad Request — Verify NO retries attempted
    @patch("requests.get")
    def test_inject_400_bad_request_no_retries(self, mock_get):
        """Inject 400 Bad Request; verify validator raises immediately without retry loops."""
        bad_resp = MagicMock()
        bad_resp.status_code = 400
        bad_resp.raise_for_status.side_effect = requests.HTTPError("400 Client Error: Bad Request")
        mock_get.return_value = bad_resp

        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.fetch_current_weather(17.3850, 78.4867, retries=3)
        # Verify request was attempted exactly once (no retry storm on 4xx)
        self.assertEqual(mock_get.call_count, 1)

    # 5. Malformed Non-Dict JSON Injection
    @patch("requests.get")
    def test_inject_malformed_json_payload(self, mock_get):
        """Inject non-dict JSON string from upstream API."""
        bad_resp = MagicMock()
        bad_resp.status_code = 200
        bad_resp.raise_for_status.return_value = None
        bad_resp.json.return_value = ["not", "a", "valid", "dictionary"]
        mock_get.return_value = bad_resp

        with self.assertRaises(WeatherServiceValidationError) as ctx:
            WeatherService.fetch_current_weather(17.3850, 78.4867)
        self.assertIn("Malformed weather response", str(ctx.exception))

    # 6. Missing Required Field Injection
    @patch("requests.get")
    def test_inject_missing_required_variable(self, mock_get):
        """Inject payload missing relative_humidity_2m."""
        bad_resp = MagicMock()
        bad_resp.status_code = 200
        bad_resp.raise_for_status.return_value = None
        bad_resp.json.return_value = {
            "current": {
                "time": "2026-09-21T12:00",
                "temperature_2m": 40.0,
                # relative_humidity_2m missing
                "wind_speed_10m": 2.0,
            }
        }
        mock_get.return_value = bad_resp

        with self.assertRaises(WeatherServiceValidationError) as ctx:
            WeatherService.fetch_current_weather(17.3850, 78.4867)
        self.assertIn("Missing required 'relative_humidity_2m'", str(ctx.exception))

    # 7. Out-of-Bounds Physical Value Injection
    def test_inject_out_of_bounds_parameters(self):
        """Inject absurd, physically impossible parameters."""
        # Absurd temperature
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=75.0, rh_pct=40.0, wind_ms=2.0, solar_wm2=500.0)
        # Negative relative humidity
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=35.0, rh_pct=-5.0, wind_ms=2.0, solar_wm2=500.0)
        # Negative wind speed
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=35.0, rh_pct=40.0, wind_ms=-3.0, solar_wm2=500.0)
        # Impossible solar radiation
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_weather_parameters(temp_c=35.0, rh_pct=40.0, wind_ms=2.0, solar_wm2=2000.0)

    # 8. Stale Observation Timestamp Injection with Confidence Downgrade
    def test_inject_stale_timestamp_confidence_penalty(self):
        """Inject observation older than freshness threshold; confirm confidence downgrade."""
        fresh_weather = {
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "wind_ms": 2.5,
            "solar_wm2": 600.0,
            "freshness": "FRESH",
        }
        stale_weather = {
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "wind_ms": 2.5,
            "solar_wm2": 600.0,
            "freshness": "STALE",
        }

        dq_fresh = evaluate_data_quality(fresh_weather, is_forecast=False)
        dq_stale = evaluate_data_quality(stale_weather, is_forecast=False)

        self.assertEqual(dq_fresh["level"], "High")
        self.assertEqual(dq_stale["level"], "Medium")
        self.assertLess(dq_stale["score"], dq_fresh["score"])
        self.assertTrue(any("stale" in f.lower() for f in dq_stale["flags"]))

    # 9. Invalid Geographic Coordinates Injection
    def test_inject_invalid_coordinates(self):
        """Inject out-of-range coordinates."""
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_coordinates(95.0, 78.0)
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_coordinates(17.0, -195.0)
        with self.assertRaises(WeatherServiceValidationError):
            WeatherService.validate_coordinates("invalid_non_numeric", 78.0)

    # 10. Incomplete Forecast Series (< 5 Days) Injection
    @patch("requests.get")
    def test_inject_incomplete_forecast_days(self, mock_get):
        """Inject only 3 days into forecast pipeline."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        times = [f"2026-09-0{d}T{h:02d}:00" for d in range(1, 4) for h in range(24)]
        mock_resp.json.return_value = {
            "hourly": {
                "time": times,
                "temperature_2m": [36.0] * len(times),
                "relative_humidity_2m": [45.0] * len(times),
                "wind_speed_10m": [2.0] * len(times),
                "shortwave_radiation": [500.0] * len(times),
            }
        }
        mock_get.return_value = mock_resp

        with self.assertRaises(WeatherServiceValidationError) as ctx:
            WeatherService.fetch_forecast_weather(17.3850, 78.4867)
        self.assertIn("expected at least 5", str(ctx.exception))

    # 11. Duplicate Timestamp Injection in Forecast
    @patch("requests.get")
    def test_inject_duplicate_forecast_timestamps(self, mock_get):
        """Inject duplicated timestamp entries into hourly forecast array."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        times, temps, rhs, winds = [], [], [], []
        for d in range(1, 6):
            for h in range(24):
                t_str = f"2026-09-{d:02d}T{h:02d}:00"
                times.append(t_str)
                temps.append(36.0)
                rhs.append(50.0)
                winds.append(2.0)
                if h == 10:  # Inject duplicate
                    times.append(t_str)
                    temps.append(36.0)
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
        # Deduplication must ensure exactly 24 hourly points per day
        self.assertEqual(len(forecast[0]["hourly"]), 24)

    # 12. Live Safety Check: Injected Failure Returns 502, NEVER Synthetic Fallback
    @patch("app.fetch_current_weather")
    def test_inject_live_endpoint_failure_never_returns_synthetic(self, mock_weather):
        """Verify /api/check-risk returns 502 with error message and zero synthetic risk assessment."""
        mock_weather.side_effect = WeatherServiceUnavailableError("Simulated upstream provider outage")

        res = self.client.post("/api/check-risk", json={
            "location": "Delhi",
            "workload": "heavy",
        })
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertEqual(data["status"], "unavailable")
        self.assertIn("error", data)
        # Verify no synthetic risk metrics were returned
        self.assertNotIn("wbgt_c", data)
        self.assertNotIn("overall_action_level", data)

    # 13. Live Safety Check: Injected Forecast Failure Returns 502
    @patch("app.fetch_forecast_weather")
    def test_inject_forecast_endpoint_failure_never_returns_synthetic(self, mock_forecast):
        """Verify /api/forecast returns 502 with error message and zero synthetic forecast."""
        mock_forecast.side_effect = WeatherServiceUnavailableError("Simulated forecast outage")

        res = self.client.post("/api/forecast", json={"location": "Delhi"})
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertEqual(data["status"], "unavailable")
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
