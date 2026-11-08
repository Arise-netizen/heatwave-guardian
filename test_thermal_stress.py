import pytest
from unittest.mock import patch
from engine import (
    classify_thermal_stress,
    calculate_thermal_stress_index,
    estimate_wbgt,
    classify_wbgt_band,
    classify_imd,
    fuse_risk,
    assess_vulnerability,
    escalate_risk,
    WBGT_TO_THERMAL_STRESS_LEVEL,
    THERMAL_STRESS_DESCRIPTIONS,
)
from app import app


# ---------------------------------------------------------------------------
# UNIT TESTS: Thermal Stress Classification & WBGT Bands
# ---------------------------------------------------------------------------

class TestThermalStressClassification:
    def test_low_thermal_stress(self):
        result = classify_thermal_stress(25.0)
        assert result["thermal_stress_level"] == "Low"
        assert result["wbgt_band"] == "Safe"
        assert result["severity"] == 1
        assert result["risk_color"] == "#2f9e44"
        assert "Minimal environmental heat stress" in result["thermal_stress_description"]

    def test_moderate_thermal_stress(self):
        result = classify_thermal_stress(28.0)
        assert result["thermal_stress_level"] == "Moderate"
        assert result["wbgt_band"] == "Caution"
        assert result["severity"] == 2
        assert result["risk_color"] == "#f2a900"
        assert "Moderate environmental heat stress" in result["thermal_stress_description"]

    def test_high_thermal_stress(self):
        result = classify_thermal_stress(30.0)
        assert result["thermal_stress_level"] == "High"
        assert result["wbgt_band"] == "Extreme Caution"
        assert result["severity"] == 3
        assert result["risk_color"] == "#e8590c"
        assert "High environmental heat stress" in result["thermal_stress_description"]

    def test_very_high_thermal_stress(self):
        result = classify_thermal_stress(31.5)
        assert result["thermal_stress_level"] == "Very High"
        assert result["wbgt_band"] == "Danger"
        assert result["severity"] == 4
        assert result["risk_color"] == "#c1121f"
        assert "Very high environmental heat stress" in result["thermal_stress_description"]

    def test_extreme_thermal_stress(self):
        result = classify_thermal_stress(33.5)
        assert result["thermal_stress_level"] == "Extreme"
        assert result["wbgt_band"] == "Extreme Danger"
        assert result["severity"] == 5
        assert result["risk_color"] == "#6a040f"
        assert "Extreme environmental heat stress" in result["thermal_stress_description"]


class TestThermalStressBoundaryThresholds:
    def test_threshold_26_9_safe(self):
        assert classify_thermal_stress(26.9)["thermal_stress_level"] == "Low"

    def test_threshold_27_0_moderate(self):
        assert classify_thermal_stress(27.0)["thermal_stress_level"] == "Moderate"

    def test_threshold_29_3_moderate(self):
        assert classify_thermal_stress(29.3)["thermal_stress_level"] == "Moderate"

    def test_threshold_29_4_high(self):
        assert classify_thermal_stress(29.4)["thermal_stress_level"] == "High"

    def test_threshold_31_0_high(self):
        assert classify_thermal_stress(31.0)["thermal_stress_level"] == "High"

    def test_threshold_31_1_very_high(self):
        assert classify_thermal_stress(31.1)["thermal_stress_level"] == "Very High"

    def test_threshold_32_1_very_high(self):
        assert classify_thermal_stress(32.1)["thermal_stress_level"] == "Very High"

    def test_threshold_32_2_extreme(self):
        assert classify_thermal_stress(32.2)["thermal_stress_level"] == "Extreme"


class TestThermalStressIndexCalculation:
    def test_calculate_thermal_stress_index_mild(self):
        res = calculate_thermal_stress_index(temp_c=25.0, rh_pct=40.0, wind_ms=2.5, solar_wm2=300.0)
        assert "wbgt_c" in res
        assert "thermal_stress_level" in res
        assert "wbgt_band" in res
        assert "environmental_conditions" in res
        assert res["environmental_conditions"]["temp_c"] == 25.0
        assert res["thermal_stress_level"] == "Low"

    def test_calculate_thermal_stress_index_hot_humid(self):
        res = calculate_thermal_stress_index(temp_c=42.0, rh_pct=65.0, wind_ms=1.0, solar_wm2=850.0)
        assert res["wbgt_c"] > 32.0
        assert res["thermal_stress_level"] in ["Very High", "Extreme"]
        assert res["wbgt_band"] in ["Danger", "Extreme Danger"]

    def test_calculate_thermal_stress_index_edge_inputs(self):
        # 0% humidity, 0 solar, calm wind
        res = calculate_thermal_stress_index(temp_c=30.0, rh_pct=0.0, wind_ms=0.0, solar_wm2=0.0)
        assert isinstance(res["wbgt_c"], float)
        assert res["thermal_stress_level"] in ["Low", "Moderate"]


# ---------------------------------------------------------------------------
# CONCEPT SEPARATION TESTS: Thermal Stress vs Base Risk vs Personal Risk
# ---------------------------------------------------------------------------

class TestThreeConceptSeparation:
    def test_personal_risk_escalation_above_thermal_stress(self):
        """A vulnerable worker in moderate thermal stress should have escalated Personal Risk."""
        wbgt = estimate_wbgt(temp_c=30.0, rh_pct=50.0, wind_ms=2.0, solar_wm2=400.0)
        wbgt_band = classify_wbgt_band(wbgt)
        ts_info = classify_thermal_stress(wbgt)
        
        imd_status = classify_imd(temp_c=30.0, region="plains")
        base_risk = fuse_risk(wbgt_band, imd_status)
        
        # High vulnerability profile
        vuln_level = assess_vulnerability(
            age_group="60+",
            acclimatization="not",
            workload="very_heavy",
            clothing="heavy"
        )
        personal_risk = escalate_risk(base_risk, vuln_level)

        severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4, "Extreme": 5}
        assert severity_map[personal_risk] >= severity_map[base_risk]
        assert ts_info["thermal_stress_level"] in ["Low", "Moderate"]


# ---------------------------------------------------------------------------
# INTEGRATION TESTS: Flask Endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestThermalStressEndpoints:
    def test_check_risk_endpoint_thermal_stress(self, client):
        mock_weather = {
            "temp_c": 38.0,
            "rh_pct": 55.0,
            "wind_ms": 2.0,
            "solar_wm2": 700.0,
        }
        with patch("app.fetch_current_weather", return_value=mock_weather):
            resp = client.post("/api/check-risk", json={
                "location": "Hyderabad",
                "latitude": 17.385,
                "longitude": 78.486,
                "age_group": "18_40",
                "experience": "somewhat_used",
                "work_intensity": "moderate",
                "clothing": "light",
                "admin1": "Telangana"
            })
            assert resp.status_code == 200
            data = resp.get_json()

            # Check core thermal stress fields
            assert "thermal_stress" in data
            assert "thermal_stress_level" in data
            assert "thermal_stress_description" in data
            assert "environmental_risk" in data
            assert "personal_risk" in data
            assert data["thermal_stress_level"] in ["Low", "Moderate", "High", "Very High", "Extreme"]
            assert "wbgt_c" in data["thermal_stress"]
            assert "environmental_conditions" in data["thermal_stress"]
            assert data["thermal_stress"]["environmental_conditions"]["temp_c"] == 38.0

    def test_forecast_endpoint_thermal_stress(self, client):
        mock_forecast = [
            {"date": "2026-06-01", "max_temp_c": 40.0, "peak_wbgt_c": 31.5},
            {"date": "2026-06-02", "max_temp_c": 41.0, "peak_wbgt_c": 32.5},
            {"date": "2026-06-03", "max_temp_c": 38.0, "peak_wbgt_c": 29.5},
            {"date": "2026-06-04", "max_temp_c": 37.0, "peak_wbgt_c": 28.0},
            {"date": "2026-06-05", "max_temp_c": 36.0, "peak_wbgt_c": 26.5},
        ]
        with patch("app.fetch_forecast_weather", return_value=mock_forecast):
            resp = client.post("/api/forecast", json={
                "location": "Delhi",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "age_group": "18_40",
                "work_intensity": "moderate",
                "experience": "somewhat_used",
                "clothing": "light",
            })
            assert resp.status_code == 200
            days = resp.get_json()
            assert len(days) == 5
            for day in days:
                assert "thermal_stress_level" in day
                assert "thermal_stress_description" in day
                assert "thermal_stress" in day
                assert "environmental_risk" in day
                assert "personal_risk" in day
                assert day["thermal_stress_level"] in ["Low", "Moderate", "High", "Very High", "Extreme"]

    def test_institute_sites_endpoint_thermal_stress(self, client):
        mock_weather = {
            "temp_c": 36.0,
            "rh_pct": 60.0,
            "wind_ms": 2.0,
            "solar_wm2": 650.0,
        }
        with patch("app.fetch_current_weather", return_value=mock_weather):
            resp = client.get("/api/institute/sites?institute_id=infra_delhi")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "sites" in data
            assert len(data["sites"]) > 0
            for site in data["sites"]:
                assert "thermal_stress_level" in site
                assert "thermal_stress_description" in site
                assert "thermal_stress" in site
                assert site["thermal_stress_level"] in ["Low", "Moderate", "High", "Very High", "Extreme"]

    def test_authority_zones_endpoint_thermal_stress(self, client):
        mock_weather = {
            "temp_c": 39.0,
            "rh_pct": 50.0,
            "wind_ms": 2.5,
            "solar_wm2": 750.0,
        }
        with patch("app.fetch_current_weather", return_value=mock_weather):
            resp = client.get("/api/authority/zones")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "features" in data
            assert len(data["features"]) > 0
            for feat in data["features"]:
                props = feat["properties"]
                assert "thermal_stress_level" in props
                assert "thermal_stress_description" in props
                assert props["thermal_stress_level"] in ["Low", "Moderate", "High", "Very High", "Extreme"]


# ---------------------------------------------------------------------------
# UI EXPLAINABILITY & SCIENTIFIC TEXT ACCURACY TESTS
# ---------------------------------------------------------------------------

class TestThermalStressExplainabilityAndHierarchy:
    def test_ui_thermal_stress_explanation_text(self, client):
        """Verify the exact Liljegren explanation text and equation are present in My Risk HTML."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert "How is Thermal Stress calculated?" in html
        assert "Thermal Stress is derived from outdoor Wet-Bulb Globe Temperature (WBGT) using the" in html
        assert "Liljegren et al. (2008)" in html
        assert "Meteorological conditions such as air temperature, humidity, wind and solar radiation are used to estimate the natural wet-bulb and globe temperatures, which are then combined as:" in html
        assert "WBGT = 0.7 × T" in html

    def test_ui_thermal_stress_vs_personal_risk_distinction_text(self, client):
        """Verify the exact distinction text separating Thermal Stress from Personal Risk."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert "describes the environmental heat load based on WBGT. <strong class=\"text-slate-900\">Personal Risk</strong> additionally considers individual vulnerability and workload." in html

    def test_pipeline_hierarchy_on_my_risk_page(self, client):
        """Verify the 8-step evaluation pipeline hierarchy is clearly represented."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert "8-Step Hierarchy" in html
        assert "Weather" in html
        assert "WBGT" in html
        assert "Thermal Stress" in html
        assert "IMD" in html
        assert "Personal Factors" in html
        assert "Personal Risk" in html
        assert "Action" in html
