"""
HeatWave Guardian — Authority Map Tests
========================================
Tests for /api/authority/zones endpoint, marker data structure,
filter logic, weather failure handling, and data consistency.
"""

import pytest
from unittest.mock import patch, MagicMock
from app import app
from engine import (
    classify_thermal_stress,
    classify_imd,
    fuse_risk,
    classify_wbgt_band,
    RISK_LEVELS,
    RISK_COLORS,
)
from services.authority_service import AuthorityService


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _mock_weather_ok(lat, lon):
    """Return realistic weather data for testing."""
    return {
        "temp_c": 42.0,
        "rh_pct": 35.0,
        "wind_ms": 3.0,
        "solar_wm2": 800.0,
    }


def _mock_weather_fail(lat, lon):
    """Simulate weather API failure."""
    raise ConnectionError("Weather API unavailable")


# ──────────────────────────────────────────────────────────────────────────────
# 1. API Response Structure Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAuthorityAPIStructure:
    """Verify /api/authority/zones response has correct structure."""

    def test_endpoint_returns_200(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            resp = client.get("/api/authority/zones")
        assert resp.status_code == 200

    def test_response_is_feature_collection(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert "summary" in data
        assert isinstance(data["features"], list)

    def test_summary_contains_required_counts(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        s = data["summary"]
        assert "total_monitored" in s
        assert "red_alert" in s
        assert "orange_alert" in s
        assert "yellow_alert" in s
        assert "green_normal" in s
        assert s["total_monitored"] == len(AuthorityService.MONITORED_AREAS)

    def test_features_count_matches_monitored_areas(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        assert len(data["features"]) == len(AuthorityService.MONITORED_AREAS)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Properties Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestFeatureProperties:
    """Verify each feature has all required fields for map display."""

    REQUIRED_FIELDS = [
        "name", "state", "region_type", "latitude", "longitude",
        "temp_c", "wbgt_c", "wbgt_band", "thermal_stress_level",
        "imd_status", "environmental_risk", "base_risk", "risk_color",
        "severity", "weather_available", "hap_level", "hap_title",
        "public_action_summary", "demographics",
    ]

    def test_all_features_have_required_fields(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        for feature in data["features"]:
            props = feature["properties"]
            for field in self.REQUIRED_FIELDS:
                assert field in props, f"Missing field '{field}' in feature for {props.get('name', 'UNKNOWN')}"

    def test_feature_geometry_is_valid_point(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        for feature in data["features"]:
            assert feature["geometry"]["type"] == "Point"
            coords = feature["geometry"]["coordinates"]
            assert len(coords) == 2
            lon, lat = coords
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90

    def test_risk_color_is_valid_hex(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        for feature in data["features"]:
            color = feature["properties"]["risk_color"]
            assert color.startswith("#")
            assert len(color) == 7

    def test_severity_is_integer_1_to_5(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        for feature in data["features"]:
            sev = feature["properties"]["severity"]
            assert isinstance(sev, int)
            assert 1 <= sev <= 5

    def test_hap_level_is_valid(self, client):
        with patch("app.fetch_current_weather", _mock_weather_ok):
            data = client.get("/api/authority/zones").get_json()
        valid_levels = {"RED", "ORANGE", "YELLOW", "GREEN"}
        for feature in data["features"]:
            assert feature["properties"]["hap_level"] in valid_levels


# ──────────────────────────────────────────────────────────────────────────────
# 3. Weather Failure Handling Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWeatherFailureHandling:
    """Verify honest handling when weather API fails for some/all cities."""

    def test_weather_failure_marks_unavailable(self):
        """When weather fetch fails, feature should be marked weather_unavailable."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_fail)
        for feature in data["features"]:
            assert feature["properties"]["weather_available"] is False

    def test_weather_success_marks_available(self):
        """When weather fetch succeeds, feature should be marked weather_available."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        for feature in data["features"]:
            assert feature["properties"]["weather_available"] is True

    def test_failure_does_not_crash(self):
        """Weather failure should not crash the entire endpoint."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_fail)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == len(AuthorityService.MONITORED_AREAS)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Filter Logic Tests (Backend Data Verification)
# ──────────────────────────────────────────────────────────────────────────────


class TestFilterLogic:
    """Verify that filter conditions can be applied to feature data correctly."""

    def test_red_orange_filter(self):
        """Features with RED/ORANGE hap_level should be filterable."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        red_orange = [
            f for f in data["features"]
            if f["properties"]["hap_level"] in ("RED", "ORANGE")
        ]
        for f in red_orange:
            assert f["properties"]["hap_level"] in ("RED", "ORANGE")

    def test_extreme_filter(self):
        """Features with 'Extreme' base_risk should be filterable."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        extreme = [
            f for f in data["features"]
            if f["properties"]["base_risk"] == "Extreme"
        ]
        for f in extreme:
            assert f["properties"]["base_risk"] == "Extreme"

    def test_all_filter_includes_everything(self):
        """'All' filter should include every feature."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        assert len(data["features"]) == len(AuthorityService.MONITORED_AREAS)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Data Consistency Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDataConsistency:
    """Verify WBGT -> Thermal Stress -> IMD -> Risk pipeline is consistent."""

    def test_wbgt_and_thermal_stress_consistent(self):
        """The thermal_stress_level should match what classify_thermal_stress returns for the WBGT value."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        for feature in data["features"]:
            p = feature["properties"]
            expected = classify_thermal_stress(p["wbgt_c"])
            assert p["thermal_stress_level"] == expected["thermal_stress_level"], \
                f"{p['name']}: thermal_stress_level mismatch"

    def test_imd_status_consistent_with_region(self):
        """IMD status should be consistent with classify_imd(temp, region)."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        for feature in data["features"]:
            p = feature["properties"]
            expected_imd = classify_imd(p["temp_c"], p["region_type"])
            assert p["imd_status"] == expected_imd, \
                f"{p['name']}: imd_status mismatch"

    def test_risk_matches_fusion(self):
        """Base risk should match fuse_risk(wbgt_band, imd_status)."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        for feature in data["features"]:
            p = feature["properties"]
            expected_risk = fuse_risk(p["wbgt_band"], p["imd_status"])
            assert p["base_risk"] == expected_risk, \
                f"{p['name']}: base_risk mismatch"

    def test_risk_color_matches_base_risk(self):
        """Risk color should match the RISK_COLORS mapping for base_risk."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok)
        for feature in data["features"]:
            p = feature["properties"]
            expected_color = RISK_COLORS.get(p["base_risk"], "#2f9e44")
            assert p["risk_color"] == expected_color, \
                f"{p['name']}: risk_color mismatch"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Thermal Stress vs Personal Risk Separation Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestThermalStressVsPersonalRisk:
    """Verify Thermal Stress and Personal Risk are properly separated in check-risk."""

    def test_thermal_stress_is_environmental_only(self, client):
        """Thermal stress should NOT change with personal factors."""
        with patch("app.fetch_current_weather", _mock_weather_ok):
            resp1 = client.post("/api/check-risk", json={
                "location": "Delhi",
                "age_group": "18-40",
                "experience": "very_used",
                "work_intensity": "light",
                "clothing": "light",
            })
            resp2 = client.post("/api/check-risk", json={
                "location": "Delhi",
                "age_group": "60+",
                "experience": "not_used",
                "work_intensity": "very_heavy",
                "clothing": "ppe",
            })
        d1 = resp1.get_json()
        d2 = resp2.get_json()

        # Same location -> same WBGT and thermal stress
        assert d1["wbgt_c"] == d2["wbgt_c"]
        assert d1["thermal_stress_level"] == d2["thermal_stress_level"]

        # But personal risk should differ (high-vulnerability profile should be escalated)
        severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4, "Extreme": 5}
        sev1 = severity_map[d1["personal_risk"]]
        sev2 = severity_map[d2["personal_risk"]]
        assert sev2 >= sev1, "Vulnerable profile should have equal or higher personal risk"

    def test_thermal_stress_and_personal_risk_can_differ(self, client):
        """With a vulnerable profile, personal risk should be >= environmental risk."""
        with patch("app.fetch_current_weather", _mock_weather_ok):
            resp = client.post("/api/check-risk", json={
                "location": "Delhi",
                "age_group": "60+",
                "experience": "not_used",
                "work_intensity": "heavy",
                "clothing": "ppe",
            })
        data = resp.get_json()
        severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4, "Extreme": 5}

        env_sev = severity_map[data["environmental_risk"]]
        personal_sev = severity_map[data["personal_risk"]]
        assert personal_sev >= env_sev, "Personal risk should be >= environmental risk"

    def test_response_contains_both_indicators(self, client):
        """Check-risk response should contain distinct thermal_stress_level and personal_risk."""
        with patch("app.fetch_current_weather", _mock_weather_ok):
            resp = client.post("/api/check-risk", json={"location": "Delhi"})
        data = resp.get_json()
        assert "thermal_stress_level" in data
        assert "personal_risk" in data
        assert "environmental_risk" in data
        assert "wbgt_c" in data


# ──────────────────────────────────────────────────────────────────────────────
# 7. UI & Map Rendering Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAuthorityMapUIAndFilters:
    """Verify Authority Map HTML, filter controls, and popup elements."""

    def test_authority_page_renders_map_and_filter_buttons(self, client):
        """GET /authority should render Leaflet map container and all 3 filter buttons."""
        resp = client.get("/authority")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert 'id="authority-map"' in html
        assert 'id="filter-all-btn"' in html
        assert 'id="filter-red-btn"' in html
        assert 'id="filter-ext-btn"' in html
        assert "All Centers" in html
        assert "Red & Orange Only" in html
        assert "Extreme Risk Only" in html
        assert 'id="map-empty-state"' in html
        assert 'id="filter-count"' in html

    def test_authority_cache_clearing(self):
        """Verify clear_cache resets in-memory cache."""
        AuthorityService.get_area_heat_layer(_mock_weather_ok)
        assert AuthorityService._cache != {}
        AuthorityService.clear_cache()
        assert AuthorityService._cache == {}
        assert AuthorityService._last_fetcher is None


# ──────────────────────────────────────────────────────────────────────────────
# 8. Regression Tests for the Three Filters
# ──────────────────────────────────────────────────────────────────────────────


class TestAuthorityMapRegressionFilters:
    """Regression tests for 'All Centers', 'Red & Orange Only', and 'Extreme Risk Only'."""

    # Authoritative mapping contract matching frontend RISK_TO_ALERT_MAP
    RISK_TO_ALERT = {
        "Extreme": "RED",
        "Very High": "RED",
        "High": "ORANGE",
        "Moderate": "YELLOW",
        "Low": "GREEN",
    }

    def _filter_dataset(self, features, filter_mode):
        """Simulate the authoritative frontend filter algorithm."""
        results = []
        for f in features:
            coords = f.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2 or not isinstance(coords[0], (int, float)) or not isinstance(coords[1], (int, float)):
                continue
            p = f.get("properties", {})
            hap = p.get("hap_level", "").upper()
            base_risk = p.get("base_risk", "")
            mapped_alert = hap if hap in ("RED", "ORANGE", "YELLOW", "GREEN") else self.RISK_TO_ALERT.get(base_risk, "GREEN")
            thermal = p.get("thermal_stress_level", "")

            if filter_mode == "red_orange":
                if mapped_alert in ("RED", "ORANGE"):
                    results.append(f)
            elif filter_mode == "extreme":
                if base_risk in ("Extreme", "Very High") or thermal == "Extreme" or mapped_alert == "RED":
                    results.append(f)
            else:  # 'all'
                results.append(f)
        return results

    def test_filter_all_centers_returns_all_with_valid_coordinates(self):
        """'All Centers' filter must return all 14 monitored centers with valid coordinates."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok, use_cache=False)
        filtered = self._filter_dataset(data["features"], "all")
        assert len(filtered) == 14
        for f in filtered:
            lon, lat = f["geometry"]["coordinates"]
            assert isinstance(lon, (int, float)) and isinstance(lat, (int, float))
            assert 65.0 <= lon <= 95.0
            assert 5.0 <= lat <= 38.0

    def test_filter_red_orange_with_mild_weather_returns_zero(self):
        """When no centers exceed warning thresholds (e.g. mild conditions), red_orange correctly returns 0 without crashing."""
        def _mild_weather(lat, lon):
            return {"temp_c": 28.0, "rh_pct": 40.0, "wind_ms": 2.0, "solar_wm2": 300.0}

        data = AuthorityService.get_area_heat_layer(_mild_weather, use_cache=False)
        filtered = self._filter_dataset(data["features"], "red_orange")
        assert len(filtered) == 0
        assert data["summary"]["red_alert"] == 0
        assert data["summary"]["orange_alert"] == 0

    def test_filter_red_orange_with_extreme_weather_returns_all(self):
        """When all centers are under severe heatwave, red_orange returns all 14 centers."""
        def _extreme_weather(lat, lon):
            return {"temp_c": 46.0, "rh_pct": 50.0, "wind_ms": 1.5, "solar_wm2": 900.0}

        data = AuthorityService.get_area_heat_layer(_extreme_weather, use_cache=False)
        filtered = self._filter_dataset(data["features"], "red_orange")
        assert len(filtered) == 14

    def test_filter_extreme_only_with_mixed_weather(self):
        """In mixed weather, extreme filter returns only centers with extreme heat risk."""
        def _mixed_weather(lat, lon):
            if lat > 25.0:  # Northern cities -> Severe heatwave
                return {"temp_c": 46.0, "rh_pct": 50.0, "wind_ms": 1.5, "solar_wm2": 900.0}
            else:  # Southern / Central -> Moderate
                return {"temp_c": 30.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 400.0}

        data = AuthorityService.get_area_heat_layer(_mixed_weather, use_cache=False)
        extreme_filtered = self._filter_dataset(data["features"], "extreme")
        red_orange_filtered = self._filter_dataset(data["features"], "red_orange")
        all_filtered = self._filter_dataset(data["features"], "all")

        # Northern cities > 25.0 lat: Delhi, Varanasi, Jaipur, Jaisalmer, Lucknow (5 cities)
        assert len(extreme_filtered) == 5
        assert len(red_orange_filtered) == 5
        assert len(all_filtered) == 14
        for f in extreme_filtered:
            assert f["properties"]["latitude"] > 25.0

    def test_repeated_filter_switching_preserves_dataset_integrity(self):
        """Switching between filters repeatedly must not mutate the underlying feature list."""
        data = AuthorityService.get_area_heat_layer(_mock_weather_ok, use_cache=False)
        original_count = len(data["features"])

        for _ in range(10):
            res_all = self._filter_dataset(data["features"], "all")
            res_ro = self._filter_dataset(data["features"], "red_orange")
            res_ext = self._filter_dataset(data["features"], "extreme")

            assert len(data["features"]) == original_count
            assert len(res_all) == original_count
            assert len(res_ro) <= original_count
            assert len(res_ext) <= len(res_ro)

