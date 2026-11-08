"""
Unit and Integration Test Suite for HeatWave Guardian Services
==============================================================
Validates:
1. GeoService (validation, presets, caching)
2. DemographicService (schema, demo data labeling, vulnerability score)
3. HAPService (Red/Orange/Yellow/Green triggers, response protocols)
4. AlertService (formatting, cooldown, simulation mode audit trail)
5. MortalityRiskService (proxy score range 0-10, disclaimer, factors)
6. CoolingRecommenderService (scoring model, distance gap, explainability)
7. InstituteService (multi-site aggregation, core pipeline reuse)
8. AuthorityService (GeoJSON layer, summary counts, HAP mapping)
"""

import unittest
from unittest.mock import patch, MagicMock

from services.geo_service import GeoService
from services.demographic_service import DemographicService
from services.hap_service import HAPService
from services.alert_service import AlertService
from services.mortality_service import MortalityRiskService
from services.cooling_recommender import CoolingRecommenderService
from services.institute_service import InstituteService
from services.authority_service import AuthorityService


class TestServices(unittest.TestCase):

    # 1. GeoService Tests
    def test_geo_service_coordinates_validation(self):
        valid, coords = GeoService.validate_coordinates(17.3850, 78.4867)
        self.assertTrue(valid)
        self.assertEqual(coords, (17.3850, 78.4867))

        invalid_lat, _ = GeoService.validate_coordinates(95.0, 78.4867)
        self.assertFalse(invalid_lat)

        invalid_lon, _ = GeoService.validate_coordinates(17.3850, 190.0)
        self.assertFalse(invalid_lon)

        invalid_str, _ = GeoService.validate_coordinates("abc", "def")
        self.assertFalse(invalid_str)

    def test_geo_service_presets(self):
        presets = GeoService.get_preset_locations()
        self.assertGreaterEqual(len(presets), 10)
        names = [p["name"] for p in presets]
        self.assertIn("Delhi", names)
        self.assertIn("Hyderabad", names)
        self.assertIn("Mumbai", names)
        self.assertIn("Khammam", names)

    def test_geo_service_india_only_live_and_filtering(self):
        # 1. Foreign query 'London' must return 0 results
        london_res = GeoService.search_locations("London")
        self.assertEqual(len(london_res), 0)

        # 2. Unknown gibberish location must return 0 results
        unknown_res = GeoService.search_locations("XyZqwk987nonexistent")
        self.assertEqual(len(unknown_res), 0)

        # 3. Empty or short queries must return empty list
        self.assertEqual(GeoService.search_locations(""), [])
        self.assertEqual(GeoService.search_locations("a"), [])

        # 4. Indian city 'Pune' must return 1 best exact match
        pune_res = GeoService.search_locations("Pune")
        self.assertGreaterEqual(len(pune_res), 1)
        self.assertEqual(pune_res[0]["name"], "Pune")
        self.assertEqual(pune_res[0]["country_code"], "IN")
        self.assertIn("Maharashtra", pune_res[0]["display_name"])

        # 5. Indian cities Mumbai, Nashik, Delhi
        for city in ["Mumbai", "Nashik", "Delhi"]:
            res = GeoService.search_locations(city)
            self.assertGreater(len(res), 0)
            self.assertEqual(res[0]["country_code"], "IN")
            self.assertIn("India", res[0]["display_name"])
            self.assertIn("latitude", res[0])
            self.assertIn("longitude", res[0])
            self.assertIn("inferred_region", res[0])

    @patch("services.geo_service.requests.get")
    def test_geo_service_independent_backend_validation_rejects_foreign(self, mock_get):
        # Mock upstream response containing a mix of Indian and foreign items
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "name": "London",
                    "latitude": 51.5074,
                    "longitude": -0.1278,
                    "country": "United Kingdom",
                    "country_code": "GB",
                    "admin1": "England"
                },
                {
                    "name": "Paris",
                    "latitude": 48.8566,
                    "longitude": 2.3522,
                    "country": "France",
                    "country_code": "FR",
                    "admin1": "Ile-de-France"
                },
                {
                    "name": "Delhi",
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                    "country": "India",
                    "country_code": "IN",
                    "admin1": "Delhi",
                    "population": 16787941
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Clear cache for this query
        res = GeoService.search_locations("TestMultiCityQuery")
        # Must strictly reject non-Indian items
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "Delhi")
        self.assertEqual(res[0]["country_code"], "IN")

    # 2. DemographicService Tests
    def test_demographic_service_known_and_fallback(self):
        delhi_demo = DemographicService.get_demographics("Delhi")
        self.assertEqual(delhi_demo["state"], "Delhi")
        self.assertTrue(delhi_demo["is_demo_data"])
        self.assertIn("population", delhi_demo)

        fallback_demo = DemographicService.get_demographics("UnknownPlace123", "SomeState")
        self.assertTrue(fallback_demo["is_demo_data"])
        self.assertEqual(fallback_demo["state"], "SomeState")

    def test_demographic_vulnerability_score(self):
        demo = {
            "elderly_pct": 10.0,
            "children_pct": 15.0,
            "outdoor_worker_pct": 40.0,
            "informal_housing_pct": 30.0,
        }
        score = DemographicService.get_vulnerability_score(demo)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    # 3. HAPService Tests
    def test_hap_service_triggers(self):
        # Red alert
        red = HAPService.evaluate_hap_status("Extreme", "Severe Heatwave", 33.5)
        self.assertEqual(red["alert_level"], "RED")
        self.assertIn("Red Alert", red["alert_title"])
        self.assertGreater(len(red["public_actions"]), 0)
        self.assertGreater(len(red["institutional_actions"]), 0)

        # Orange alert
        orange = HAPService.evaluate_hap_status("High", "Heatwave", 31.5)
        self.assertEqual(orange["alert_level"], "ORANGE")

        # Yellow alert
        yellow = HAPService.evaluate_hap_status("Moderate", "Heat Day", 29.5)
        self.assertEqual(yellow["alert_level"], "YELLOW")

        # Green normal
        green = HAPService.evaluate_hap_status("Low", "Normal", 24.0)
        self.assertEqual(green["alert_level"], "GREEN")

    # 4. AlertService Tests
    def test_alert_formatting_and_simulation(self):
        msg = AlertService.format_alert_message(
            location="Hyderabad",
            risk_category="High",
            wbgt_c=31.2,
            temp_c=41.5,
            work_minutes=30,
            rest_minutes=30,
            instruction="Take 30 min rest in shaded areas",
            channel="sms"
        )
        self.assertIn("HEATWAVE GUARDIAN ALERT", msg)
        self.assertIn("Hyderabad", msg)
        self.assertIn("HIGH", msg)
        self.assertIn("31.2°C", msg)
        self.assertIn("30m work / 30m rest", msg)

    def test_alert_invalid_phone(self):
        res = AlertService.dispatch_alert(
            recipient_phone="123",  # invalid
            location="Delhi",
            risk_category="High",
            wbgt_c=31.0,
            temp_c=40.0,
            work_minutes=30,
            rest_minutes=30,
            instruction="Rest in shade"
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "INVALID_PHONE")

    def test_alert_simulation_mode_and_cooldown(self):
        phone = "+919876543210"
        # First dispatch
        res1 = AlertService.dispatch_alert(
            recipient_phone=phone,
            location="Delhi",
            risk_category="High",
            wbgt_c=31.0,
            temp_c=40.0,
            work_minutes=30,
            rest_minutes=30,
            instruction="Rest in shade",
            force=True
        )
        self.assertTrue(res1["success"])
        self.assertTrue(res1.get("simulated", False))
        self.assertEqual(res1["status"], "SIMULATED")

        # Second dispatch without force triggers cooldown
        res2 = AlertService.dispatch_alert(
            recipient_phone=phone,
            location="Delhi",
            risk_category="High",
            wbgt_c=31.0,
            temp_c=40.0,
            work_minutes=30,
            rest_minutes=30,
            instruction="Rest in shade",
            force=False
        )
        self.assertTrue(res2["success"])
        self.assertEqual(res2["status"], "COOLDOWN_ACTIVE")

    # 5. MortalityRiskService Tests
    def test_mortality_proxy_index(self):
        res = MortalityRiskService.calculate_proxy_index(
            wbgt_c=33.2,
            imd_status="Severe Heatwave",
            temp_c=45.0,
            location="Delhi",
            consecutive_heat_days=3
        )
        self.assertIn("proxy_score", res)
        self.assertGreaterEqual(res["proxy_score"], 0.0)
        self.assertLessEqual(res["proxy_score"], 10.0)
        self.assertTrue(res["is_experimental"])
        self.assertFalse(res["is_medical_prediction"])
        self.assertIn("NOT a clinically or epidemiologically validated mortality prediction", res["disclaimer"])
        self.assertGreater(len(res["contributing_factors"]), 0)

    # 6. CoolingRecommenderService Tests
    def test_cooling_recommender(self):
        recs = CoolingRecommenderService.get_recommendations()
        self.assertGreaterEqual(len(recs), 5)
        # Check sorting by siting_score descending
        for i in range(len(recs) - 1):
            self.assertGreaterEqual(recs[i]["siting_score"], recs[i+1]["siting_score"])

        first = recs[0]
        self.assertIn("zone_name", first)
        self.assertIn("priority", first)
        self.assertIn("suggested_facility", first)
        self.assertIn("rationale", first)
        self.assertTrue(first["is_decision_support"])

        # City filter
        pune_recs = CoolingRecommenderService.get_recommendations(target_city="Pune")
        self.assertGreater(len(pune_recs), 0)
        for r in pune_recs:
            self.assertEqual(r["city"], "Pune")

    # 7. InstituteService Tests
    def test_institute_service(self):
        inst_list = InstituteService.get_institute_list()
        self.assertGreater(len(inst_list), 0)
        inst_id = inst_list[0]["id"]

        def mock_weather(lat, lon):
            return {"temp_c": 38.5, "rh_pct": 55.0, "wind_ms": 1.5, "solar_wm2": 650.0}

        eval_res = InstituteService.evaluate_institute(inst_id, mock_weather)
        self.assertIn("counts", eval_res)
        self.assertIn("sites", eval_res)
        self.assertEqual(eval_res["counts"]["total"], len(eval_res["sites"]))
        for s in eval_res["sites"]:
            self.assertIn(s["risk_category"], ["Low", "Moderate", "High", "Very High", "Extreme"])
            self.assertEqual(s["work_minutes"] + s["rest_minutes"], 60)

    # 8. AuthorityService Tests
    def test_authority_service(self):
        def mock_weather(lat, lon):
            return {"temp_c": 36.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0}

        layer = AuthorityService.get_area_heat_layer(mock_weather)
        self.assertEqual(layer["type"], "FeatureCollection")
        self.assertIn("summary", layer)
        self.assertIn("features", layer)
        self.assertGreater(len(layer["features"]), 5)

        first_feat = layer["features"][0]
        self.assertEqual(first_feat["geometry"]["type"], "Point")
        self.assertEqual(len(first_feat["geometry"]["coordinates"]), 2)
        p = first_feat["properties"]
        self.assertIn("hap_level", p)
        self.assertIn("wbgt_c", p)
        self.assertIn("base_risk", p)


if __name__ == "__main__":
    unittest.main()
