"""
Integration Tests for HeatWave Guardian New Pages and API Endpoints
===================================================================
Tests:
- GET /institute (200 OK)
- GET /authority (200 OK)
- GET /cooling-stations (200 OK)
- GET /methodology (200 OK)
- GET /api/institute/list (200 OK)
- GET /api/institute/sites (200 OK)
- GET /api/authority/zones (200 OK)
- GET /api/cooling-stations/recommendations (200 OK)
- POST /api/alerts/send (200 OK / simulation mode)
- GET /api/alerts/history (200 OK)
- GET /api/mortality-risk (200 OK)
- GET /api/hap-guidelines (200 OK)
- GET /api/validation/data (200 OK)

"""

import unittest
from unittest.mock import patch
from app import app


class TestNewEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # 1. Page Routes Tests
    def test_page_institute(self):
        res = self.client.get("/institute")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Institute Heat-Risk Dashboard", html)
        self.assertIn("Workplace Micro-Sites & Operations Table", html)

    def test_page_authority(self):
        res = self.client.get("/authority")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Authority / Municipal Heat-Risk Map", html)
        self.assertIn("authority-map", html)

    def test_page_cooling_stations(self):
        res = self.client.get("/cooling-stations")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Cooling-Station Siting Recommender", html)
        self.assertIn("Filter by Urban Corridor", html)

    def test_page_methodology(self):
        res = self.client.get("/methodology")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Methodology & Algorithmic Pipeline", html)
        self.assertIn("Liljegren WBGT", html)
        self.assertIn("IMD Criteria", html)

    # 2. REST Endpoints Tests
    def test_api_institute_list(self):
        res = self.client.get("/api/institute/list")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("institutes", data)
        self.assertGreaterEqual(len(data["institutes"]), 1)

    @patch("app.fetch_current_weather")
    def test_api_institute_sites(self, mock_weather):
        mock_weather.return_value = {
            "temp_c": 36.0,
            "rh_pct": 50.0,
            "wind_ms": 2.0,
            "solar_wm2": 500.0,
        }
        res = self.client.get("/api/institute/sites?institute_id=infra_delhi")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("counts", data)
        self.assertIn("sites", data)
        self.assertEqual(data["institute"]["id"], "infra_delhi")

    @patch("app.fetch_current_weather")
    def test_api_authority_zones(self, mock_weather):
        mock_weather.return_value = {
            "temp_c": 37.0,
            "rh_pct": 45.0,
            "wind_ms": 2.5,
            "solar_wm2": 600.0,
        }
        res = self.client.get("/api/authority/zones")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertIn("features", data)
        self.assertIn("summary", data)

    def test_api_cooling_recommendations(self):
        res = self.client.get("/api/cooling-stations/recommendations")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("recommendations", data)
        self.assertIn("disclaimer", data)
        self.assertTrue(data.get("is_decision_support"))

    def test_api_alerts_send_and_history(self):
        # Simulation send
        payload = {
            "phone": "+919999988888",
            "location": "Pune",
            "risk_category": "Extreme",
            "wbgt_c": 33.5,
            "temp_c": 43.0,
            "work_minutes": 0,
            "rest_minutes": 60,
            "instruction": "STOP continuous work",
            "channel": "sms",
            "force": True
        }
        res_send = self.client.post("/api/alerts/send", json=payload)
        self.assertEqual(res_send.status_code, 200)
        data_send = res_send.get_json()
        self.assertTrue(data_send["success"])
        self.assertTrue(data_send.get("simulated", False))

        # Check history
        res_hist = self.client.get("/api/alerts/history")
        self.assertEqual(res_hist.status_code, 200)
        data_hist = res_hist.get_json()
        self.assertIn("history", data_hist)
        self.assertGreater(len(data_hist["history"]), 0)

    def test_api_mortality_risk(self):
        res = self.client.get("/api/mortality-risk?wbgt_c=32.5&temp_c=42.0&imd_status=Heatwave&location=Delhi")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("proxy_score", data)
        self.assertIn("disclaimer", data)
        self.assertTrue(data["is_experimental"])

    def test_api_hap_guidelines(self):
        res = self.client.get("/api/hap-guidelines?risk_category=Extreme&imd_status=Severe+Heatwave&wbgt_c=33.0")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["alert_level"], "RED")
        self.assertIn("public_actions", data)

    def test_api_validation_data(self):
        res = self.client.get("/api/validation/data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("event_name", data)
        self.assertIn("limitation_note", data)


if __name__ == "__main__":
    unittest.main()
