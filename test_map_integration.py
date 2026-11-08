"""
Automated tests verifying the interactive location map integration in HeatWave Guardian.
Tests:
1. GET / verifies map HTML elements and Leaflet CDN scripts/links.
2. GET /api/search-location verifies coordinates and structure for map pinning.
3. POST /api/check-risk returns all required payload attributes for marker popups and metrics strip.
4. Testing multiple Indian locations (Delhi, Mumbai, Kolkata, Hyderabad, Chennai).
5. Testing resilience against invalid location searches or edge inputs.
"""

import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_map_html_elements_present(client):
    """Verify that the landing page renders Leaflet assets and map DOM elements."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # Leaflet CDN assets and map config
    assert "leaflet@1.9.4/dist/leaflet.css" in html
    assert "leaflet@1.9.4/dist/leaflet.js" in html
    assert "map_config.js" in html

    # Map Card and Canvas elements
    assert 'id="location-map"' in html
    assert 'id="map-section-card"' in html
    assert 'id="recenter-map-btn"' in html
    assert 'id="map-stat-wbgt"' in html
    assert 'id="map-stat-imd"' in html
    assert 'id="map-stat-risk"' in html
    assert 'id="map-stat-action"' in html
    assert 'id="map-tile-error"' in html
    assert "Heat Risk:" in html


def test_map_config_asset_and_authority_map(client):
    """Verify that map_config.js is served correctly and authority page includes MapLibre map assets."""
    # Verify map_config.js is served
    config_resp = client.get("/static/map_config.js")
    assert config_resp.status_code == 200
    config_js = config_resp.get_data(as_text=True)
    assert "tile.openstreetmap.org" in config_js
    assert "MAP_CONFIG" in config_js
    assert "createTileLayer" in config_js
    assert "getMapLibreStyle" in config_js

    # Verify authority page renders MapLibre map elements and config
    auth_resp = client.get("/authority")
    assert auth_resp.status_code == 200
    auth_html = auth_resp.get_data(as_text=True)
    assert "maplibre-gl@4.7.1/dist/maplibre-gl.css" in auth_html
    assert "maplibre-gl@4.7.1/dist/maplibre-gl.js" in auth_html
    assert "map_config.js" in auth_html
    assert 'id="authority-map"' in auth_html


def test_geocoding_coordinates_for_map(client):
    """Verify that location search returns real latitude and longitude numbers for map centering."""
    response = client.get("/api/search-location?q=Delhi")
    assert response.status_code == 200
    data = response.get_json()

    assert "results" in data
    assert len(data["results"]) > 0

    first_res = data["results"][0]
    assert "latitude" in first_res
    assert "longitude" in first_res
    assert "name" in first_res
    assert isinstance(first_res["latitude"], (int, float))
    assert isinstance(first_res["longitude"], (int, float))

    # Bounded to India
    assert 8.0 <= first_res["latitude"] <= 38.0
    assert 68.0 <= first_res["longitude"] <= 98.0


def test_check_risk_provides_all_marker_popup_data(client):
    """Verify that check-risk API returns all necessary fields for the map popup and metrics strip."""
    payload = {
        "location": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "admin1": "Delhi",
        "age_group": "18–40",
        "experience": "very_used",
        "work_intensity": "heavy",
        "clothing": "heavy"
    }
    response = client.post("/api/check-risk", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    required_keys = [
        "location", "latitude", "longitude",
        "wbgt_c", "wbgt_band",
        "current_temp_c", "imd_status",
        "risk_category", "work_minutes", "rest_minutes",
        "instruction", "vuln_level"
    ]
    for k in required_keys:
        assert k in data, f"Missing key '{k}' required by map popup"

    assert data["risk_category"] in ["Low", "Moderate", "High", "Very High", "Extreme"]
    assert isinstance(data["wbgt_c"], (int, float))
    assert isinstance(data["work_minutes"], int)
    assert isinstance(data["rest_minutes"], int)
    assert data["work_minutes"] + data["rest_minutes"] == 60


def test_multiple_indian_locations_for_map(client):
    """Verify risk check works seamlessly across diverse Indian geographies for map plotting."""
    locations = [
        ("Mumbai", 19.0760, 72.8777, "Maharashtra"),
        ("Kolkata", 22.5726, 88.3639, "West Bengal"),
        ("Chennai", 13.0827, 80.2707, "Tamil Nadu"),
        ("Hyderabad", 17.3850, 78.4867, "Telangana"),
    ]

    for city, lat, lon, admin1 in locations:
        payload = {
            "location": city,
            "latitude": lat,
            "longitude": lon,
            "admin1": admin1,
            "age_group": "41–60",
            "experience": "somewhat_used",
            "work_intensity": "moderate",
            "clothing": "light"
        }
        res = client.post("/api/check-risk", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["location"] == city
        assert data["latitude"] == lat
        assert data["longitude"] == lon
        assert "risk_category" in data


def test_invalid_search_graceful_handling(client):
    """Verify that invalid search inputs return clean empty results without application failure."""
    res = client.get("/api/search-location?q=ZZZZZZZZZZNOTAPLACE999")
    assert res.status_code == 200
    data = res.get_json()
    assert "results" in data
    assert len(data["results"]) == 0
