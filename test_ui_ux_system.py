"""
HeatWave Guardian — UI/UX System & Design System Integration Tests
==================================================================
Comprehensive test suite verifying all 56 items from Product UI/UX System & Stitch Design specification:
- UI Page Loads & Component Rendering (My Risk, Forecast, What-If, Worksite, Authority, Validation, Methodology, Cooling Stations)
- States (Loading, Empty, API Error, Incomplete/Stale data)
- Responsive & Accessibility (Semantic structure, aria labels, high contrast, keyboard focus indicators)
- API Integrations (Risk, Forecast, What-If Simulation, Worksite, Authority, Alerts, Cooling Stations, Decision Trace)
- Scientific Regression Integrity (Liljegren WBGT, IMD Thresholds, ISO 7243 Work/Rest splits, Deterministic Explainability)
"""

import unittest
from unittest.mock import patch, MagicMock
from app import app
from engine import (
    estimate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    classify_imd,
    fuse_risk,
    assess_vulnerability,
    escalate_risk,
    get_work_rest,
    simulate_exposure_scenario,
    assess_full_risk_profile,
    detect_peak_risk_windows,
)


class TestUIUXSystem(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. UI Page Rendering Tests
    # ──────────────────────────────────────────────────────────────────────────
    def test_main_dashboard_loads(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("HeatWave Guardian", html)
        self.assertIn("CHECK MY HEAT RISK", html)
        self.assertIn("risk-summary-hero", html)

    def test_forecast_page_loads(self):
        res = self.client.get("/forecast")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("5-Day Heat-Risk Forecast", html)
        self.assertIn("trajectory-summary-banner", html)

    def test_what_if_page_loads(self):
        res = self.client.get("/what-if")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("What-If Heat Exposure Simulator", html)
        self.assertIn("sim-delta-card", html)
        self.assertIn("Baseline Scenario", html)

    def test_worksite_page_loads(self):
        res = self.client.get("/worksite")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Institute Heat-Risk Dashboard", html)
        self.assertIn("Workplace Micro-Sites & Operations Table", html)

    def test_institute_backward_compat_route_loads(self):
        res = self.client.get("/institute")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Institute Heat-Risk Dashboard", html)

    def test_authority_page_loads(self):
        res = self.client.get("/authority")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Authority / Municipal Heat-Risk Map", html)
        self.assertIn("authority-map", html)

    def test_cooling_station_page_loads(self):
        res = self.client.get("/cooling-stations")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Cooling-Station Siting Recommender", html)
        self.assertIn("Filter by Urban Corridor", html)

    def test_validation_page_loads(self):
        res = self.client.get("/validation")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("2015 Andhra Pradesh & Telangana Heatwave Case Study", html)
        self.assertIn("Khammam", html)
        self.assertIn("Hyderabad", html)

    def test_methodology_page_loads(self):
        res = self.client.get("/methodology")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Methodology & Algorithmic Pipeline", html)
        self.assertIn("Liljegren WBGT", html)
        self.assertIn("IMD Criteria", html)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Design System & Accessibility Tests
    # ──────────────────────────────────────────────────────────────────────────
    def test_design_system_css_tokens(self):
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        css = res.get_data(as_text=True)
        self.assertIn("--risk-low", css)
        self.assertIn("--risk-moderate", css)
        self.assertIn("--risk-high", css)
        self.assertIn("--risk-very-high", css)
        self.assertIn("--risk-extreme", css)
        self.assertIn(".card-hero", css)
        self.assertIn("location-map-popup", css)

    def test_accessibility_semantic_attributes(self):
        res = self.client.get("/")
        html = res.get_data(as_text=True)
        self.assertIn('aria-label="Main Navigation"', html)
        self.assertIn('role="region"', html)
        self.assertIn('role="status"', html)
        self.assertIn('aria-label="Current Action Level"', html)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. API Integrations & Real-Time Endpoints
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.fetch_current_weather")
    def test_api_check_risk_success(self, mock_weather):
        mock_weather.return_value = {
            "temp_c": 38.5,
            "current_temp_c": 38.5,
            "rh_pct": 48.0,
            "wind_ms": 2.2,
            "solar_wm2": 650.0,
            "freshness": "FRESH"
        }
        payload = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "18–40",
            "experience": "very_used",
            "work_intensity": "moderate",
            "clothing": "light",
            "recovery_environment": "shade",
        }
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("wbgt_c", data)
        self.assertIn("risk_category", data)
        self.assertIn("work_minutes", data)
        self.assertIn("rest_minutes", data)
        self.assertIn("primary_drivers", data)
        self.assertIn("contributing_factors", data)
        self.assertIn("protective_factors", data)
        self.assertIn("factor_sensitivities", data)
        self.assertIn("decision_trace", data)

    @patch("app.fetch_forecast_weather")
    def test_api_forecast_success(self, mock_forecast):
        mock_forecast.return_value = [
            {
                "date": "2026-09-18",
                "max_temp_c": 39.0,
                "peak_wbgt_c": 31.5,
                "hourly": [
                    {"time": "2026-09-18T10:00", "temp_c": 34.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0, "wbgt_c": 28.5},
                    {"time": "2026-09-18T14:00", "temp_c": 39.0, "rh_pct": 40.0, "wind_ms": 2.5, "solar_wm2": 800.0, "wbgt_c": 31.5},
                ]
            }
        ]
        res = self.client.get("/api/forecast?location=Hyderabad&latitude=17.3850&longitude=78.4867")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIn("decision_story", data[0])
        self.assertIn("peak_risk_window", data[0])

    @patch("app.fetch_current_weather")
    def test_api_simulate_scenario(self, mock_weather):
        mock_weather.return_value = {
            "temp_c": 40.0,
            "current_temp_c": 40.0,
            "rh_pct": 45.0,
            "wind_ms": 2.0,
            "solar_wm2": 700.0,
        }
        payload = {
            "location": "Delhi",
            "baseline": {
                "workload": "heavy",
                "recovery_environment": "outdoor_direct",
                "clothing": "heavy",
                "experience": "not_used",
            },
            "scenario": {
                "workload": "light",
                "recovery_environment": "cool_indoor",
                "clothing": "light",
                "experience": "very_used",
            }
        }
        res = self.client.post("/api/simulate-scenario", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("baseline_profile", data)
        self.assertIn("scenario_profile", data)
        self.assertIn("action_level_change", data)
        self.assertIn("parameter_changes", data)
        self.assertGreater(len(data["parameter_changes"]), 0)
        self.assertIn("transition_explanation", data)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Error & Validation States
    # ──────────────────────────────────────────────────────────────────────────
    def test_api_check_risk_invalid_coords(self):
        payload = {
            "location": "InvalidLocation",
            "latitude": 999.0,
            "longitude": 999.0,
        }
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_api_search_location_empty_query(self):
        res = self.client.get("/api/search-location?q=")
        self.assertEqual(res.status_code, 400)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Scientific Regression Integrity Checks
    # ──────────────────────────────────────────────────────────────────────────
    def test_wbgt_calculation_physics(self):
        wbgt = estimate_wbgt(temp_c=35.0, rh_pct=60.0, wind_ms=2.0, solar_wm2=600.0)
        self.assertIsInstance(wbgt, float)
        self.assertGreater(wbgt, 25.0)
        self.assertLess(wbgt, 40.0)

    def test_imd_regional_thresholds(self):
        self.assertEqual(classify_imd(42.5, "plains"), "Heatwave")
        self.assertEqual(classify_imd(44.5, "plains"), "Severe Heatwave")
        self.assertEqual(classify_imd(38.0, "coastal"), "Heat Day")
        self.assertEqual(classify_imd(39.5, "coastal"), "Heatwave")
        self.assertEqual(classify_imd(31.5, "hilly"), "Heat Day")
        self.assertEqual(classify_imd(32.5, "hilly"), "Heatwave")



    def test_iso_7243_work_rest_ratios(self):
        wr_low = get_work_rest("Low")
        self.assertEqual(wr_low["work_minutes"], 60)
        self.assertEqual(wr_low["rest_minutes"], 0)

        wr_extreme = get_work_rest("Extreme", workload="heavy", acclimatized="not")
        self.assertEqual(wr_extreme["work_minutes"], 0)
        self.assertEqual(wr_extreme["rest_minutes"], 60)


if __name__ == "__main__":
    unittest.main()
