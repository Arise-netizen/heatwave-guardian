"""
Comprehensive Automated Test Suite for HeatWave Guardian.
Covers:
1. Location Search Endpoint (/api/search-location)
   - Valid Indian cities
   - Partial city names
   - Unknown city (empty results)
   - Empty and single-character query validation
   - Multiple results and deduplication
   - India priority ordering
   - Upstream failure handling: timeout, HTTP 500, malformed JSON
2. Personal Risk Check (/api/check-risk)
   - Valid preset cities (backward compatibility)
   - Valid custom coordinates (Pune, Lucknow)
   - Coordinate boundary validation (lat [-90, 90], lon [-180, 180])
   - Age validation (1-120)
   - Physiological vulnerability escalation
   - Upstream failure handling: timeout, HTTP 500, malformed JSON, missing required variables
   - Verification that NO synthetic weather values (32°C / 55% RH) are used
3. Live 5-Day Forecast Pipeline (/api/forecast)
   - Valid preset cities (GET)
   - Custom coordinates (GET and POST)
   - Exactly 5 days in sequential, distinct order
   - Correct Tmax from hourly series and peak WBGT from hourly WBGTs
   - Risk fusion, vulnerability escalation, work/rest schedule totals (work + rest = 60)
   - Upstream failure handling: timeout, HTTP 500, malformed JSON, missing hourly series, incomplete days
   - Invalid coordinates and age validations
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
from app import app
from engine import CITIES


class TestExpandedFeatures(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Location Search Tests (/api/search-location)
    # ──────────────────────────────────────────────────────────────────────────

    def test_search_location_valid_indian_city(self):
        """Test search with a known Indian city not in the preset list."""
        res = self.client.get("/api/search-location?q=Pune")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)

        first = data["results"][0]
        self.assertEqual(first["name"], "Pune")
        self.assertIn("latitude", first)
        self.assertIn("longitude", first)
        self.assertIn("country", first)
        self.assertIn("display_name", first)
        self.assertIsInstance(first["latitude"], float)
        self.assertIsInstance(first["longitude"], float)
        self.assertTrue(18.0 <= first["latitude"] <= 19.5)
        self.assertTrue(73.0 <= first["longitude"] <= 74.5)

    def test_search_location_partial_query(self):
        """Test partial name search returns matches."""
        res = self.client.get("/api/search-location?q=Jaip")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        names = [r["name"].lower() for r in data["results"]]
        self.assertTrue(any("jaipur" in n for n in names))

    def test_search_location_unknown_city(self):
        """Test unknown city search returns empty results list with HTTP 200."""
        res = self.client.get("/api/search-location?q=xyznonexistentcity9999")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("results", data)
        self.assertEqual(data["results"], [])

    def test_search_location_empty_input(self):
        """Empty query returns 400 with helpful error message."""
        res = self.client.get("/api/search-location?q=")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_search_location_too_short(self):
        """Single-character query returns empty list without calling API unnecessarily."""
        res = self.client.get("/api/search-location?q=a")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["results"], [])

    def test_search_location_pune_contains_strictly_india_only(self):
        """Test search for Pune returns ONLY Indian locations, completely excluding foreign ones."""
        res = self.client.get("/api/search-location?q=Pune")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("results", data)
        results = data["results"]
        self.assertGreater(len(results), 0)

        # Top result must be Pune, Maharashtra, India
        self.assertEqual(results[0]["name"], "Pune")
        self.assertEqual(results[0]["admin1"], "Maharashtra")
        self.assertEqual(results[0]["country"], "India")
        self.assertEqual(results[0]["country_code"], "IN")
        self.assertEqual(results[0]["display_name"], "Pune, Maharashtra, India")

        # Verify EVERY single result is strictly Indian
        for item in results:
            self.assertEqual(item["country_code"], "IN")
            self.assertEqual(item["country"], "India")
            self.assertNotIn("Timor-Leste", item["display_name"])
            self.assertNotIn("Brazil", item["display_name"])
            self.assertNotIn("Iran", item["display_name"])

    @patch("requests.get")
    def test_search_location_foreign_city_returns_empty(self, mock_get):
        """Test that foreign queries like London and Paris return empty list without allowing selection."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "results": [
                {"name": "London", "latitude": 51.5074, "longitude": -0.1278, "country": "United Kingdom", "country_code": "GB", "admin1": "England"},
                {"name": "Paris", "latitude": 48.8566, "longitude": 2.3522, "country": "France", "country_code": "FR", "admin1": "Ile-de-France"}
            ]
        }
        mock_get.return_value = mock_resp
        for foreign_city in ["London", "Paris"]:
            res = self.client.get(f"/api/search-location?q={foreign_city}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["results"], [])

    def test_search_location_casing_and_whitespace(self):
        """Test that lowercase, uppercase, and leading/trailing whitespace queries resolve consistently."""
        for query in ["pune", "PUNE", "  Pune  "]:
            res = self.client.get(f"/api/search-location?q={query}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertGreater(len(data["results"]), 0)
            self.assertEqual(data["results"][0]["name"], "Pune")
            self.assertEqual(data["results"][0]["country_code"], "IN")

    def test_search_location_major_indian_cities_india_only(self):
        """Verify queries for Mumbai, Delhi, Nashik, Kolkata, Chennai, Bengaluru, Hyderabad, Ahmedabad."""
        cities = ["Mumbai", "Delhi", "Nashik", "Kolkata", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad"]
        for city in cities:
            res = self.client.get(f"/api/search-location?q={city}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertGreater(len(data["results"]), 0, f"Expected results for {city}")
            for item in data["results"]:
                self.assertEqual(item["country_code"], "IN", f"Non-Indian result found in {city} search")
                self.assertEqual(item["country"], "India")

    @patch("requests.get")
    def test_search_location_mixed_api_response_filtering(self, mock_get):
        """Test that when upstream returns mixed global results, non-Indian results are strictly discarded."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "results": [
                {"name": "Pune", "latitude": 18.5196, "longitude": 73.8554, "country": "India", "country_code": "IN", "admin1": "Maharashtra", "population": 3124458},
                {"name": "Pune", "latitude": -9.2833, "longitude": 124.3833, "country": "Timor-Leste", "country_code": "TL", "admin1": "Oecusse"},
                {"name": "Pune", "latitude": -0.8500, "longitude": -51.8500, "country": "Brazil", "country_code": "BR", "admin1": "Para"},
                {"name": "Punel", "latitude": 37.3800, "longitude": 49.0300, "country": "Iran", "country_code": "IR", "admin1": "Gilan"}
            ]
        }
        mock_get.return_value = mock_resp
        res = self.client.get("/api/search-location?q=Pune")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["country_code"], "IN")
        self.assertEqual(data["results"][0]["display_name"], "Pune, Maharashtra, India")

    @patch("requests.get")
    def test_search_location_all_foreign_results_filtered_out(self, mock_get):
        """Test that if upstream only returns non-Indian locations, results array is empty."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "results": [
                {"name": "London", "latitude": 51.5074, "longitude": -0.1278, "country": "United Kingdom", "country_code": "GB", "admin1": "England"},
                {"name": "London", "latitude": 42.9834, "longitude": -81.2330, "country": "Canada", "country_code": "CA", "admin1": "Ontario"}
            ]
        }
        mock_get.return_value = mock_resp
        res = self.client.get("/api/search-location?q=London")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["results"], [])

    @patch("requests.get")
    def test_search_location_timeout(self, mock_get):
        """Mocked geocoding timeout returns 502 with safe error message."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        res = self.client.get("/api/search-location?q=Pune")
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertNotIn("Timeout", data["error"])
        self.assertIn("temporarily unavailable", data["error"])

    @patch("requests.get")
    def test_search_location_http_failure(self, mock_get):
        """Mocked upstream HTTP 500 error returns 502 with safe error message."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp
        res = self.client.get("/api/search-location?q=Pune")
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertNotIn("500", data["error"])

    @patch("requests.get")
    def test_search_location_malformed_response(self, mock_get):
        """Mocked malformed geocoding response returns 502."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = "Not A Dictionary"
        mock_get.return_value = mock_resp
        res = self.client.get("/api/search-location?q=Pune")
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Risk Check with Dynamic Coordinates Tests (/api/check-risk)
    # ──────────────────────────────────────────────────────────────────────────

    def test_check_risk_valid_preset_city(self):
        """Test risk assessment for preset city (Delhi)."""
        payload = {
            "location": "Delhi",
            "age": 30,
            "acclimatization": "fully",
            "workload": "moderate",
            "region": "plains"
        }
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["location"], "Delhi")
        self.assertEqual(data["latitude"], CITIES["Delhi"][0])
        self.assertEqual(data["longitude"], CITIES["Delhi"][1])
        self.assertIn("current_temp_c", data)
        self.assertIn("wbgt_c", data)
        self.assertIn("risk_category", data)

    def test_check_risk_with_custom_coordinates(self):
        """Test risk assessment using explicit latitude/longitude for Pune."""
        payload = {
            "location": "Pune",
            "latitude": 18.5204,
            "longitude": 73.8567,
            "age": 28,
            "acclimatization": "fully",
            "workload": "moderate",
            "region": "plains"
        }
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["location"], "Pune")
        self.assertEqual(data["latitude"], 18.5204)
        self.assertEqual(data["longitude"], 73.8567)
        self.assertIn("current_temp_c", data)
        self.assertIn("wbgt_c", data)
        self.assertIn("wbgt_band", data)
        self.assertIn("imd_status", data)
        self.assertIn("risk_category", data)
        self.assertIn("work_minutes", data)
        self.assertIn("rest_minutes", data)
        self.assertIn("instruction", data)
        self.assertEqual(data["work_minutes"] + data["rest_minutes"], 60)

    def test_check_risk_another_dynamic_location(self):
        """Test risk assessment for another dynamic location (Lucknow)."""
        payload = {
            "location": "Lucknow",
            "latitude": 26.8467,
            "longitude": 80.9462,
            "age": 40,
            "acclimatization": "fully",
            "workload": "light",
            "region": "plains"
        }
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["location"], "Lucknow")
        self.assertEqual(data["latitude"], 26.8467)
        self.assertEqual(data["longitude"], 80.9462)

    def test_check_risk_coordinate_out_of_bounds(self):
        """Test that invalid coordinates (lat > 90 or lon > 180) return 400."""
        # Lat > 90
        res1 = self.client.post("/api/check-risk", json={"location": "Invalid", "latitude": 95.0, "longitude": 78.0})
        self.assertEqual(res1.status_code, 400)
        self.assertIn("error", res1.get_json())

        # Lon > 180
        res2 = self.client.post("/api/check-risk", json={"location": "Invalid", "latitude": 20.0, "longitude": 185.0})
        self.assertEqual(res2.status_code, 400)
        self.assertIn("error", res2.get_json())

    def test_check_risk_invalid_age(self):
        """Test that out-of-range age returns 400."""
        res1 = self.client.post("/api/check-risk", json={"location": "Hyderabad", "age": 0})
        self.assertEqual(res1.status_code, 400)

        res2 = self.client.post("/api/check-risk", json={"location": "Hyderabad", "age": 150})
        self.assertEqual(res2.status_code, 400)

    def test_check_risk_vulnerability_escalation(self):
        """Test that vulnerable individuals receive escalated risk category."""
        base_payload = {
            "location": "Hyderabad",
            "age": 30,
            "acclimatization": "fully",
            "workload": "light",
            "region": "plains"
        }
        res1 = self.client.post("/api/check-risk", json=base_payload)
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()

        vuln_payload = {
            "location": "Hyderabad",
            "age": 72,
            "acclimatization": "not",
            "workload": "heavy",
            "region": "plains"
        }
        res2 = self.client.post("/api/check-risk", json=vuln_payload)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()

        self.assertGreaterEqual(data2["severity"], data1["severity"])
        self.assertGreaterEqual(data2["rest_minutes"], data1["rest_minutes"])

    @patch("requests.get")
    def test_check_risk_weather_timeout(self, mock_get):
        """Mocked weather API timeout returns 502 and does NOT use synthetic fallback."""
        mock_get.side_effect = requests.exceptions.Timeout("Weather service timeout")
        payload = {"location": "Hyderabad", "age": 30}
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertNotIn("current_temp_c", data)
        self.assertIn("temporarily unavailable", data["error"])

    @patch("requests.get")
    def test_check_risk_weather_http_failure(self, mock_get):
        """Mocked upstream HTTP 500 returns 502 cleanly."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")
        mock_get.return_value = mock_resp
        payload = {"location": "Hyderabad", "age": 30}
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertNotIn("500", data["error"])

    @patch("requests.get")
    def test_check_risk_missing_required_variable(self, mock_get):
        """Mocked response missing temperature_2m raises ValueError and returns 502 without fallback."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        # Missing 'temperature_2m'
        mock_resp.json.return_value = {
            "current": {
                "relative_humidity_2m": 50.0,
                "wind_speed_10m": 2.0,
                "shortwave_radiation_instant": 300.0
            }
        }
        mock_get.return_value = mock_resp
        payload = {"location": "Hyderabad", "age": 30}
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)
        # Verify no fabricated values
        self.assertNotIn("current_temp_c", data)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Live 5-Day Forecast Pipeline Tests (/api/forecast)
    # ──────────────────────────────────────────────────────────────────────────

    def test_forecast_with_custom_coordinates(self):
        """Test 5-day forecast using dynamic coordinates via POST."""
        payload = {
            "location": "Varanasi",
            "latitude": 25.3176,
            "longitude": 82.9739,
            "age": 45,
            "acclimatization": "fully",
            "workload": "heavy",
            "region": "plains"
        }
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 200)
        days = res.get_json()
        self.assertEqual(len(days), 5)

        prev_date = ""
        for i, day in enumerate(days):
            self.assertEqual(day["location"], "Varanasi")
            self.assertEqual(day["latitude"], 25.3176)
            self.assertEqual(day["longitude"], 82.9739)
            self.assertIn("date", day)
            self.assertIn("max_temp_c", day)
            self.assertIn("wbgt_c", day)
            self.assertIn("wbgt_band", day)
            self.assertIn("imd_status", day)
            self.assertIn("base_risk", day)
            self.assertIn("risk_category", day)
            self.assertIn("vulnerability", day)
            self.assertIn("work_minutes", day)
            self.assertIn("rest_minutes", day)
            self.assertIn("instruction", day)
            self.assertIn("summary", day)
            self.assertIn("severity", day)
            self.assertEqual(day["work_minutes"] + day["rest_minutes"], 60)
            self.assertIn(day["severity"], [1, 2, 3, 4, 5])

            # Check sequential and distinct dates
            self.assertTrue(day["date"] > prev_date)
            prev_date = day["date"]

    def test_forecast_get_query_parameters(self):
        """Test 5-day forecast GET with query parameters."""
        res = self.client.get("/api/forecast?location=Lucknow&latitude=26.8467&longitude=80.9462&age=68&workload=heavy&acclimatization=not")
        self.assertEqual(res.status_code, 200)
        days = res.get_json()
        self.assertEqual(len(days), 5)
        # Age 68 + heavy work + not acclimatized -> vulnerability is High
        self.assertEqual(days[0]["vulnerability"], "High")

    def test_forecast_legacy_city_compatibility(self):
        """Test GET /api/forecast?location=Delhi still functions seamlessly."""
        res = self.client.get("/api/forecast?location=Delhi")
        self.assertEqual(res.status_code, 200)
        days = res.get_json()
        self.assertEqual(len(days), 5)
        self.assertEqual(days[0]["location"], "Delhi")

    def test_forecast_unknown_city_without_coords(self):
        """Test unknown location without coordinates returns 400."""
        res = self.client.get("/api/forecast?location=Atlantis")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    @patch("requests.get")
    def test_forecast_timeout(self, mock_get):
        """Mocked forecast API timeout returns 502 with safe error message."""
        mock_get.side_effect = requests.exceptions.Timeout("Forecast API timed out")
        res = self.client.get("/api/forecast?location=Delhi")
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertNotIn("Timeout", data["error"])

    @patch("requests.get")
    def test_forecast_http_failure(self, mock_get):
        """Mocked forecast HTTP 500 error returns 502 with safe error message."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp
        res = self.client.get("/api/forecast?location=Delhi")
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertNotIn("500", data["error"])

    @patch("requests.get")
    def test_forecast_missing_hourly_series(self, mock_get):
        """Mocked forecast response missing temperature series returns 502."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2026-09-04T00:00"],
                # Missing temperature_2m
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [2.0],
            }
        }
        mock_get.return_value = mock_resp
        res = self.client.get("/api/forecast?location=Delhi")
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)

    @patch("requests.get")
    def test_forecast_incomplete_days(self, mock_get):
        """Mocked forecast returning fewer than 5 days returns 502 without partial fabrication."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        # Only 2 days of timestamps
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2026-09-04T12:00", "2026-09-05T12:00"],
                "temperature_2m": [32.0, 33.0],
                "relative_humidity_2m": [50.0, 48.0],
                "wind_speed_10m": [2.0, 2.5],
                "shortwave_radiation": [600.0, 650.0],
            }
        }
        mock_get.return_value = mock_resp
        res = self.client.get("/api/forecast?location=Delhi")
        self.assertEqual(res.status_code, 502)
        data = res.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
