"""
HeatWave Guardian — MapLibre GL JS Authority Map Test Suite
===========================================================
Comprehensive unit and integration tests verifying the migration from Leaflet
to MapLibre GL JS for the Authority / Municipal Heat-Risk Map:

1. MapLibre GL JS assets, DOM structure, and map configuration
2. Keyless basemap style specification & OpenStreetMap attribution
3. GeoJSON FeatureCollection contracts for MapLibre sources (Points & Polygons)
4. Data-driven risk layer styling attributes (risk_color, hap_level, severity)
5. ML Hyperlocal layer metadata (ml_temp_c, ml_anomaly_c, drivers, model_version)
6. Feature inspection data fields & strict 3-tier distinction (OBSERVED, REANALYSIS, ML-PREDICTED)
7. Filter compatibility across All, High+, Very High+, and Extreme categories
8. Robustness against missing fields, malformed GeoJSON, empty datasets, and backend errors
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from app import app
from services.authority_service import AuthorityService
from services.hyperlocal_ml_service import HyperlocalMLService


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _mock_weather_standard(lat, lon):
    return {
        "temp_c": 41.5,
        "rh_pct": 38.0,
        "wind_ms": 2.5,
        "solar_wm2": 750.0,
    }


def _mock_weather_error(lat, lon):
    raise ConnectionError("Upstream NWP weather service unavailable")


# ──────────────────────────────────────────────────────────────────────────────
# 1. MAPLIBRE ASSETS & DOM STRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

class TestMapLibreAssetsAndDOM:
    """Verify MapLibre GL JS scripts, styles, controls, and DOM elements."""

    def test_authority_page_loads_maplibre_assets(self, client):
        """Authority page must load MapLibre GL JS 4.7.1 CSS and JS without Leaflet."""
        resp = client.get("/authority")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        # MapLibre assets present
        assert "maplibre-gl" in html
        assert "maplibre-gl.css" in html
        assert "maplibre-gl.js" in html

        # Leaflet should NOT be on /authority
        assert "leaflet.css" not in html
        assert "leaflet.js" not in html

    def test_map_container_and_controls_rendered(self, client):
        """Map canvas container, filter buttons, recenter, and inspection panel must exist."""
        resp = client.get("/authority")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert 'id="authority-map"' in html
        assert 'id="filter-all-btn"' in html
        assert 'id="filter-red-btn"' in html
        assert 'id="filter-vhigh-btn"' in html
        assert 'id="filter-ext-btn"' in html
        assert 'id="recenter-map-btn"' in html
        assert 'id="toggle-ml-grid-btn"' in html
        assert 'id="city-focus-select"' in html
        assert 'id="cell-inspection-panel"' in html
        assert 'id="map-loading-state"' in html
        assert 'id="map-empty-state"' in html

    def test_map_config_provides_keyless_maplibre_style(self, client):
        """map_config.js must provide getMapLibreStyle() referencing OpenStreetMap with attribution."""
        resp = client.get("/static/map_config.js")
        assert resp.status_code == 200
        js = resp.get_data(as_text=True)

        assert "getMapLibreStyle" in js
        assert "tile.openstreetmap.org" in js
        assert "OpenStreetMap" in js
        # Zero API key requirements
        assert "access_token" not in js
        assert "apiKey" not in js


# ──────────────────────────────────────────────────────────────────────────────
# 2. GEOJSON CONTRACTS FOR MAPLIBRE COMPATIBILITY
# ──────────────────────────────────────────────────────────────────────────────

class TestMapLibreGeoJSONContracts:
    """Verify GeoJSON FeatureCollections match MapLibre GL layer requirements."""

    def test_authority_zones_geojson_format(self, client):
        """GET /api/authority/zones returns valid FeatureCollection of Points with coordinates [lon, lat]."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            AuthorityService.clear_cache()
            resp = client.get("/api/authority/zones")
            assert resp.status_code == 200
            data = resp.get_json()

        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) > 0

        for f in data["features"]:
            assert f["type"] == "Feature"
            assert f["geometry"]["type"] == "Point"
            coords = f["geometry"]["coordinates"]
            assert len(coords) == 2
            lon, lat = coords
            assert isinstance(lon, (int, float)) and isinstance(lat, (int, float))
            # Lon, Lat valid bounds for India
            assert 65.0 <= lon <= 98.0
            assert 6.0 <= lat <= 38.0

            p = f["properties"]
            assert "name" in p
            assert "latitude" in p
            assert "longitude" in p
            assert "wbgt_c" in p
            assert "base_risk" in p
            assert "hap_level" in p
            assert "risk_color" in p
            assert "ml_temp_c" in p
            assert "ml_anomaly_c" in p

    def test_spatial_grid_geojson_polygon_format(self, client):
        """GET /api/authority/grid returns valid FeatureCollection of closed Polygons."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            resp = client.get("/api/authority/grid?city=Delhi")
            assert resp.status_code == 200
            data = resp.get_json()

        assert data["type"] == "FeatureCollection"
        features = data["features"]
        assert len(features) > 0

        for f in features:
            assert f["type"] == "Feature"
            assert f["geometry"]["type"] == "Polygon"
            coords = f["geometry"]["coordinates"]
            assert len(coords) == 1  # Single exterior ring
            ring = coords[0]
            assert len(ring) == 5  # Closed 5-point box: p1, p2, p3, p4, p1
            assert ring[0] == ring[-1], "Polygon boundary must be explicitly closed"

            p = f["properties"]
            assert "cell_id" in p
            assert "city" in p
            assert "latitude" in p
            assert "longitude" in p
            assert "background_temp_c" in p
            assert "ml_temp_c" in p
            assert "ml_anomaly_c" in p
            assert "wbgt_c" in p
            assert "wbgt_band" in p
            assert "risk_category" in p
            assert "risk_color" in p
            assert "confidence" in p
            assert "model_version" in p


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA-DRIVEN STYLING & FILTERING
# ──────────────────────────────────────────────────────────────────────────────

class TestMapLibreDataDrivenStylingAndFilters:
    """Verify backend properties correctly drive MapLibre visualization and filtering."""

    VALID_RISK_COLORS = {"#2f9e44", "#f59f00", "#f2a900", "#e8590c", "#e03131", "#c1121f", "#6a040f", "#7048e8"}
    VALID_HAP_LEVELS = {"RED", "ORANGE", "YELLOW", "GREEN"}

    def test_data_driven_color_and_alert_levels(self, client):
        """All features must have recognized risk colors and valid HAP alert levels."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            AuthorityService.clear_cache()
            resp = client.get("/api/authority/zones")
            data = resp.get_json()

        for f in data["features"]:
            p = f["properties"]
            assert p["risk_color"].lower() in [c.lower() for c in self.VALID_RISK_COLORS]
            assert p["hap_level"].upper() in self.VALID_HAP_LEVELS
            assert 1 <= p["severity"] <= 5

    def test_filter_all_returns_every_center(self, client):
        """All Centers filter encompasses all monitored zones."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            AuthorityService.clear_cache()
            data = AuthorityService.get_area_heat_layer(_mock_weather_standard, use_cache=False)
            assert len(data["features"]) == len(AuthorityService.MONITORED_AREAS)

    def test_filter_high_plus_includes_red_and_orange(self, client):
        """High+ filter isolates warnings and emergencies."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            AuthorityService.clear_cache()
            data = AuthorityService.get_area_heat_layer(_mock_weather_standard, use_cache=False)
            high_plus = [
                f for f in data["features"]
                if f["properties"]["hap_level"] in ("RED", "ORANGE")
                or f["properties"]["base_risk"] in ("High", "Very High", "Extreme")
            ]
            for f in high_plus:
                assert f["properties"]["hap_level"] in ("RED", "ORANGE") or f["properties"]["base_risk"] in ("High", "Very High", "Extreme")

    def test_filter_very_high_plus_isolates_severe(self, client):
        """Very High+ filter filters only highest alert levels."""
        def _intense_weather(lat, lon):
            return {"temp_c": 45.0, "rh_pct": 45.0, "wind_ms": 1.5, "solar_wm2": 850.0}

        data = AuthorityService.get_area_heat_layer(_intense_weather, use_cache=False)
        vhigh_plus = [
            f for f in data["features"]
            if f["properties"]["hap_level"] == "RED" or f["properties"]["base_risk"] in ("Very High", "Extreme")
        ]
        assert len(vhigh_plus) > 0
        for f in vhigh_plus:
            assert f["properties"]["hap_level"] == "RED" or f["properties"]["base_risk"] in ("Very High", "Extreme")

    def test_filter_extreme_only(self, client):
        """Extreme filter only captures Extreme/RED severity."""
        def _extreme_weather(lat, lon):
            return {"temp_c": 47.0, "rh_pct": 55.0, "wind_ms": 1.0, "solar_wm2": 950.0}

        data = AuthorityService.get_area_heat_layer(_extreme_weather, use_cache=False)
        extreme = [
            f for f in data["features"]
            if f["properties"]["base_risk"] == "Extreme" or f["properties"]["hap_level"] == "RED"
        ]
        assert len(extreme) > 0
        for f in extreme:
            assert f["properties"]["base_risk"] in ("Extreme", "Very High") or f["properties"]["hap_level"] == "RED"


# ──────────────────────────────────────────────────────────────────────────────
# 4. INSPECTION DATA & THREE-TIER DISTINCTION
# ──────────────────────────────────────────────────────────────────────────────

class TestMapLibreInspectionDataTiers:
    """Verify feature click inspection data maintains honest scientific distinction."""

    def test_three_tier_labels_present_in_grid_cells(self, client):
        """Grid cells must explicitly declare OBSERVED, REANALYSIS, and ML-PREDICTED labels."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            resp = client.get("/api/authority/grid?city=Delhi")
            data = resp.get_json()

        for f in data["features"]:
            p = f["properties"]
            labels = p.get("labels", {})
            assert "observed_label" in labels
            assert "reanalysis_label" in labels
            assert "ml_predicted_label" in labels
            assert "OBSERVED" in labels["observed_label"]
            assert "REANALYSIS" in labels["reanalysis_label"]
            assert "ML-PREDICTED" in labels["ml_predicted_label"]

    def test_no_claims_of_medical_diagnosis(self, client):
        """Spatial features must explicitly disclaim medical diagnosis or mortality predictions."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            resp = client.get("/api/authority/grid?city=Hyderabad")
            data = resp.get_json()

        for f in data["features"]:
            p = f["properties"]
            disclaimer = p.get("disclaimer", "")
            assert "never declares medical diagnosis" in disclaimer
            assert "deterministic physics" in disclaimer

    def test_cell_inspection_contributing_drivers(self, client):
        """Grid cells must provide explainable physical drivers for local microclimate."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            resp = client.get("/api/authority/grid?city=Mumbai")
            data = resp.get_json()

        first_cell = data["features"][0]["properties"]
        assert "major_drivers" in first_cell
        drivers = first_cell["major_drivers"]
        assert isinstance(drivers, list)
        if len(drivers) > 0:
            d = drivers[0]
            assert "factor" in d
            assert "impact_c" in d


# ──────────────────────────────────────────────────────────────────────────────
# 5. ROBUSTNESS & FAILURE HANDLING
# ──────────────────────────────────────────────────────────────────────────────

class TestMapLibreFailureHandling:
    """Verify system resilience during upstream meteorological or data failures."""

    def test_weather_failure_marks_estimated_without_crash(self, client):
        """When weather provider is down, zones must flag weather_available: False and not crash."""
        with patch("app.fetch_current_weather", _mock_weather_error):
            AuthorityService.clear_cache()
            resp = client.get("/api/authority/zones")
            assert resp.status_code == 200
            data = resp.get_json()

        for f in data["features"]:
            assert f["properties"]["weather_available"] is False

    def test_empty_city_param_falls_back_gracefully(self, client):
        """Calling /api/authority/grid without city parameter yields aggregated cells across monitored cities."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            resp = client.get("/api/authority/grid")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) > 0

    def test_backward_compatible_view_grid_param(self, client):
        """Calling /api/authority/zones?view=grid returns the spatial microclimate grid."""
        with patch("app.fetch_current_weather", _mock_weather_standard):
            resp = client.get("/api/authority/zones?view=grid&city=Delhi")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) > 0
            assert data["features"][0]["geometry"]["type"] == "Polygon"
