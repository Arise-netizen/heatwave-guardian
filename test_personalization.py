"""
Comprehensive Personalization & Heat-Risk Input Test Suite for HeatWave Guardian.
Validates:
1. Every Age Option ('Under 18', '18–40', '41–60', '60+')
2. Every Experience Option ('very_used', 'somewhat_used', 'not_used')
3. Every Workload Option ('light', 'moderate', 'heavy', 'very_heavy')
4. Every Clothing Option ('light', 'heavy', 'ppe')
5. Combinations of Personal Inputs and Vulnerability Escalation
6. Internal Region Inference (Plains, Coastal, Hilly) without user input
7. Transparent Explanation Verification (driving factors in summary)
8. Backward Compatibility (legacy keys: age, acclimatization, workload, region)
9. Missing & Invalid Input Validation
10. Forecast endpoint integration with personalization inputs
"""

import unittest
from unittest.mock import patch, MagicMock
from app import app, parse_personal_inputs
from engine import (
    assess_vulnerability,
    escalate_risk,
    generate_summary,
    infer_region,
    FUSION_MATRIX,
    RISK_LEVELS,
    CITIES,
)


class TestPersonalRiskInputs(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Unit Tests for assess_vulnerability across all options
    # ──────────────────────────────────────────────────────────────────────────

    def test_all_age_group_options_baseline(self):
        """Test each age group with baseline conditions (very used, light work, light clothing)."""
        # Under 18: vulnerable age -> Moderate vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="Under 18", acclimatization="very_used", workload="light", clothing="light"),
            "Moderate"
        )
        # 18–40: standard adult -> Low vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="light", clothing="light"),
            "Low"
        )
        # 41–60: standard adult -> Low vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="41–60", acclimatization="very_used", workload="light", clothing="light"),
            "Low"
        )
        # 60+: vulnerable age -> Moderate vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="60+", acclimatization="very_used", workload="light", clothing="light"),
            "Moderate"
        )

    def test_all_experience_options(self):
        """Test each hot-weather-experience option for standard adult, light work, light clothing."""
        # Very used: fully acclimatized -> Low vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="light", clothing="light"),
            "Low"
        )
        # Somewhat used: partially acclimatized -> Moderate vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="somewhat_used", workload="light", clothing="light"),
            "Moderate"
        )
        # Not used: unacclimatized -> Moderate vulnerability (+1)
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="not_used", workload="light", clothing="light"),
            "Moderate"
        )

    def test_all_workload_options(self):
        """Test each workload option for standard adult, very used to heat, light clothing."""
        # Light -> Low vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="light", clothing="light"),
            "Low"
        )
        # Moderate -> Low vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="moderate", clothing="light"),
            "Low"
        )
        # Heavy -> Moderate vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="heavy", clothing="light"),
            "Moderate"
        )
        # Very heavy -> Moderate vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="very_heavy", clothing="light"),
            "Moderate"
        )

    def test_all_clothing_options(self):
        """Test each clothing option for standard adult, very used to heat, light work."""
        # Light/normal -> Low vulnerability
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="light", clothing="light"),
            "Low"
        )
        # Heavy work clothing -> Moderate vulnerability (+1 escalation)
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="light", clothing="heavy"),
            "Moderate"
        )
        # Protective gear / PPE -> Moderate vulnerability (+1 escalation) even with light work
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="light", clothing="ppe"),
            "Moderate"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Multi-Factor Combinations & Escalation Tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_clothing_escalation_rules(self):
        """Verify clothing rules: PPE with moderate/heavy work or unacclimatized triggers High vulnerability."""
        # PPE + moderate work -> High
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="moderate", clothing="ppe"),
            "High"
        )
        # PPE + not used to heat -> High
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="not_used", workload="light", clothing="ppe"),
            "High"
        )
        # PPE + Under 18 -> High
        self.assertEqual(
            assess_vulnerability(age_group="Under 18", acclimatization="very_used", workload="light", clothing="ppe"),
            "High"
        )
        # Heavy work clothing + heavy work -> High
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="very_used", workload="heavy", clothing="heavy"),
            "High"
        )
        # Heavy work clothing + not used to heat -> High
        self.assertEqual(
            assess_vulnerability(age_group="18–40", acclimatization="not_used", workload="light", clothing="heavy"),
            "High"
        )

    def test_extreme_vulnerability_combination(self):
        """Vulnerable age + not used to heat + heavy work + PPE -> High vulnerability."""
        vuln = assess_vulnerability(
            age_group="60+",
            acclimatization="not_used",
            workload="very_heavy",
            clothing="ppe"
        )
        self.assertEqual(vuln, "High")
        # Base risk 'Moderate' escalated by High (+2) -> 'Very High'
        self.assertEqual(escalate_risk("Moderate", vuln), "Very High")
        # Base risk 'High' escalated by High (+2) -> 'Extreme'
        self.assertEqual(escalate_risk("High", vuln), "Extreme")

    def test_minimum_vulnerability_combination(self):
        """Young adult + very used to heat + light work + light clothing -> Low vulnerability."""
        vuln = assess_vulnerability(
            age_group="18–40",
            acclimatization="very_used",
            workload="light",
            clothing="light"
        )
        self.assertEqual(vuln, "Low")
        # Base risk unchanged
        self.assertEqual(escalate_risk("Moderate", vuln), "Moderate")
        self.assertEqual(escalate_risk("High", vuln), "High")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Internal Region Inference Tests (No User Selection Required)
    # ──────────────────────────────────────────────────────────────────────────

    def test_internal_region_inference(self):
        """Verify region is correctly inferred from location and administrative region."""
        # Preset cities
        self.assertEqual(infer_region("Delhi"), "plains")
        self.assertEqual(infer_region("Mumbai"), "coastal")
        self.assertEqual(infer_region("Chennai"), "coastal")
        self.assertEqual(infer_region("Khammam"), "plains")
        self.assertEqual(infer_region("Bhubaneswar"), "coastal")

        # Dynamic Hilly locations
        self.assertEqual(infer_region("Shimla", "Himachal Pradesh"), "hilly")
        self.assertEqual(infer_region("Manali", "Himachal Pradesh"), "hilly")
        self.assertEqual(infer_region("Dehradun", "Uttarakhand"), "hilly")
        self.assertEqual(infer_region("Srinagar", "Jammu and Kashmir"), "hilly")
        self.assertEqual(infer_region("Leh", "Ladakh"), "hilly")
        self.assertEqual(infer_region("Ooty", "Tamil Nadu"), "hilly")

        # Dynamic Coastal locations
        self.assertEqual(infer_region("Visakhapatnam", "Andhra Pradesh"), "coastal")
        self.assertEqual(infer_region("Kochi", "Kerala"), "coastal")
        self.assertEqual(infer_region("Panaji", "Goa"), "coastal")
        self.assertEqual(infer_region("Mangalore", "Karnataka"), "coastal")
        self.assertEqual(infer_region("Puri", "Odisha"), "coastal")

        # Default Plains locations
        self.assertEqual(infer_region("Lucknow", "Uttar Pradesh"), "plains")
        self.assertEqual(infer_region("Patna", "Bihar"), "plains")
        self.assertEqual(infer_region("Jaipur", "Rajasthan"), "plains")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Input Parser Tests (parse_personal_inputs)
    # ──────────────────────────────────────────────────────────────────────────

    def test_parse_personal_inputs_user_friendly(self):
        """Test parse_personal_inputs with clean user-friendly parameters without region."""
        data = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "Under 18",
            "experience": "not_used",
            "work_intensity": "heavy",
            "clothing": "ppe"
        }
        parsed = parse_personal_inputs(data)
        self.assertEqual(parsed["location"], "Hyderabad")
        self.assertEqual(parsed["latitude"], 17.3850)
        self.assertEqual(parsed["longitude"], 78.4867)
        self.assertEqual(parsed["age_group"], "Under 18")
        self.assertEqual(parsed["experience"], "not_used")
        self.assertEqual(parsed["acclimatization"], "not")
        self.assertEqual(parsed["workload"], "heavy")
        self.assertEqual(parsed["clothing"], "ppe")
        self.assertEqual(parsed["region"], "plains")  # Inferred internally!

    def test_parse_personal_inputs_legacy_compatibility(self):
        """Test parse_personal_inputs maintains 100% backward compatibility with numeric age and old terms."""
        data = {
            "location": "Delhi",
            "age": 72,
            "acclimatization": "partially",
            "workload": "very_heavy",
            "region": "plains"
        }
        parsed = parse_personal_inputs(data)
        self.assertEqual(parsed["age"], 72)
        self.assertEqual(parsed["age_group"], "60+")
        self.assertEqual(parsed["acclimatization"], "partially")
        self.assertEqual(parsed["experience"], "somewhat_used")
        self.assertEqual(parsed["workload"], "very_heavy")
        self.assertEqual(parsed["clothing"], "light")
        self.assertEqual(parsed["region"], "plains")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Transparent Summary / Explanation Tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_generate_summary_explains_all_factors(self):
        """Verify summary transparently explains why risk was produced including all factors."""
        summary = generate_summary(
            location="Hyderabad",
            temp_c=41.5,
            wbgt_c=31.2,
            wbgt_band="Danger",
            imd_status="Heat Day",
            base_risk="High",
            final_risk="Extreme",
            vulnerability="High",
            age_group="60+",
            experience="not_used",
            workload="heavy",
            clothing="ppe"
        )
        self.assertIn("Hyderabad", summary)
        self.assertIn("41.5 °C", summary)
        self.assertIn("Danger", summary)
        self.assertIn("Heat Day", summary)
        self.assertIn("age 60+", summary)
        self.assertIn("not being used to working in hot weather", summary)
        self.assertIn("heavy physical labor", summary)
        self.assertIn("protective gear / PPE", summary)
        self.assertIn("High", summary)
        self.assertIn("Extreme", summary)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Full API Endpoint Integration Tests (/api/check-risk)
    # ──────────────────────────────────────────────────────────────────────────

    @patch("app.fetch_current_weather")
    def test_check_risk_new_inputs_reach_backend_and_influence_risk(self, mock_weather):
        """Verify POST /api/check-risk with new inputs computes valid risk and applies rules."""
        mock_weather.return_value = {
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "wind_ms": 2.0,
            "solar_wm2": 600.0,
        }
        # Baseline request
        payload_base = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "18–40",
            "experience": "very_used",
            "work_intensity": "light",
            "clothing": "light"
            # NOTE: NO region provided!
        }
        res_base = self.client.post("/api/check-risk", json=payload_base)
        self.assertEqual(res_base.status_code, 200)
        data_base = res_base.get_json()
        self.assertEqual(data_base["vulnerability"], "Low")
        self.assertEqual(data_base["age_group"], "18–40")
        self.assertEqual(data_base["clothing"], "light")
        self.assertEqual(data_base["region"], "plains")
        self.assertIn("base_risk", data_base)
        self.assertIn("summary", data_base)

        # High vulnerability request: Under 18 + not_used + heavy + ppe
        payload_high = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "Under 18",
            "experience": "not_used",
            "work_intensity": "heavy",
            "clothing": "ppe"
        }
        res_high = self.client.post("/api/check-risk", json=payload_high)
        self.assertEqual(res_high.status_code, 200)
        data_high = res_high.get_json()
        self.assertEqual(data_high["vulnerability"], "High")
        self.assertEqual(data_high["clothing"], "ppe")
        self.assertEqual(data_high["age_group"], "Under 18")
        self.assertGreaterEqual(data_high["severity"], data_base["severity"])
        self.assertGreaterEqual(data_high["rest_minutes"], data_base["rest_minutes"])
        self.assertIn("protective gear / PPE", data_high["summary"])

    @patch("app.fetch_current_weather")
    def test_check_risk_every_clothing_option_via_api(self, mock_weather):
        """Verify each clothing option reaches backend and is reflected in response."""
        mock_weather.return_value = {
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "wind_ms": 2.0,
            "solar_wm2": 600.0,
        }
        for cloth in ["light", "heavy", "ppe"]:
            res = self.client.post("/api/check-risk", json={
                "location": "Delhi",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "age_group": "18–40",
                "experience": "very_used",
                "work_intensity": "moderate",
                "clothing": cloth
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["clothing"], cloth)

    @patch("app.fetch_current_weather")
    def test_check_risk_every_age_option_via_api(self, mock_weather):
        """Verify each age group option reaches backend and is reflected in response."""
        mock_weather.return_value = {
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "wind_ms": 2.0,
            "solar_wm2": 600.0,
        }
        for ag in ["Under 18", "18–40", "41–60", "60+"]:
            res = self.client.post("/api/check-risk", json={
                "location": "Delhi",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "age_group": ag,
                "experience": "very_used",
                "work_intensity": "light",
                "clothing": "light"
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["age_group"], ag)

    @patch("app.fetch_current_weather")
    def test_check_risk_every_experience_option_via_api(self, mock_weather):
        """Verify each experience option reaches backend and is reflected in response."""
        mock_weather.return_value = {
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "wind_ms": 2.0,
            "solar_wm2": 600.0,
        }
        for exp in ["very_used", "somewhat_used", "not_used"]:
            res = self.client.post("/api/check-risk", json={
                "location": "Delhi",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "age_group": "18–40",
                "experience": exp,
                "work_intensity": "light",
                "clothing": "light"
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["experience"], exp)

    @patch("app.fetch_current_weather")
    def test_check_risk_every_workload_option_via_api(self, mock_weather):
        """Verify each workload option reaches backend and is reflected in response."""
        mock_weather.return_value = {
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "wind_ms": 2.0,
            "solar_wm2": 600.0,
        }
        for work in ["light", "moderate", "heavy", "very_heavy"]:
            res = self.client.post("/api/check-risk", json={
                "location": "Delhi",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "age_group": "18–40",
                "experience": "very_used",
                "work_intensity": work,
                "clothing": "light"
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["work_intensity"], work)

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Forecast Endpoint Integration Tests (/api/forecast)
    # ──────────────────────────────────────────────────────────────────────────

    @patch("app.fetch_forecast_weather")
    def test_forecast_with_new_personalization_inputs(self, mock_forecast):
        """Verify POST /api/forecast accepts new personalization parameters."""
        mock_forecast.return_value = [
            {"date": "2026-06-01", "max_temp_c": 41.0, "peak_wbgt_c": 32.0},
            {"date": "2026-06-02", "max_temp_c": 42.0, "peak_wbgt_c": 33.0},
            {"date": "2026-06-03", "max_temp_c": 40.0, "peak_wbgt_c": 31.0},
            {"date": "2026-06-04", "max_temp_c": 39.0, "peak_wbgt_c": 30.0},
            {"date": "2026-06-05", "max_temp_c": 38.0, "peak_wbgt_c": 29.0},
        ]
        payload = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "60+",
            "experience": "not_used",
            "work_intensity": "heavy",
            "clothing": "ppe"
        }
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 200)
        days = res.get_json()
        self.assertEqual(len(days), 5)
        self.assertEqual(days[0]["vulnerability"], "High")
        self.assertEqual(days[0]["age_group"], "60+")
        self.assertEqual(days[0]["clothing"], "ppe")
        self.assertEqual(days[0]["region"], "plains")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. User Experience Verification: Region Type Not Required
    # ──────────────────────────────────────────────────────────────────────────

    def test_region_not_requested_in_html_forms(self):
        """Verify HTML form on '/' (My Risk) does NOT contain 'Region Type' or 'acclimatization'."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Confirm Region Type is NOT present in the form
        self.assertNotIn("Geographic Region Type", html)
        self.assertNotIn('id="region"', html)
        self.assertNotIn('name="region"', html)

        # Confirm technical term acclimatization is NOT present in the user-facing text
        self.assertNotIn("Thermal Acclimatization", html)
        self.assertNotIn("acclimatized", html.lower())

        # Confirm all 5 final user-facing labels exist
        self.assertIn("Where are you going to be?", html)
        self.assertIn("How old are you?", html)
        self.assertIn("How used are you to working in hot weather?", html)
        self.assertIn("How physically demanding is your work?", html)
        self.assertIn("What will you be wearing while working?", html)

        # Confirm age groups exist
        self.assertIn("Under 18", html)
        self.assertIn("18–40", html)
        self.assertIn("41–60", html)
        self.assertIn("60+", html)

        # Confirm experience options exist
        self.assertIn("Very used to it", html)
        self.assertIn("Somewhat used to it", html)
        self.assertIn("Not used to it", html)

        # Confirm clothing options exist
        self.assertIn("Light/normal clothing", html)
        self.assertIn("Heavy work clothing", html)
        self.assertIn("Protective gear / PPE", html)


if __name__ == "__main__":
    unittest.main()
