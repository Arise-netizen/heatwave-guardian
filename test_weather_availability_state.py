"""
HeatWave Guardian — Weather Availability State Regression Suite
================================================================
Comprehensive verification of weather state transitions and zero-synthetic-fallback guarantees:
1. SUCCESS -> Live (freshness=FRESH, is_cached=False, confidence=High, weather_status=live)
2. STALE CACHE -> Cached/Stale (freshness=STALE, is_cached=True, confidence=Medium/Limited, weather_status=stale_cache, timestamp shown)
3. FAILURE -> Unavailable (weather_status=unavailable, status=unavailable, 502, zero fabricated/default numbers)
4. Scientific integrity: no default temperatures (30°C / 35°C) when weather API fails.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import pytest
import requests

from app import app
from engine import assess_full_risk_profile
from services.weather_service import (
    WeatherService,
    WeatherServiceError,
    WeatherServiceUnavailableError,
    WeatherServiceValidationError,
)
from services.authority_service import AuthorityService
from services.institute_service import InstituteService


@pytest.fixture(autouse=True)
def clean_cache():
    WeatherService.clear_cache()
    AuthorityService.clear_cache()
    yield
    WeatherService.clear_cache()
    AuthorityService.clear_cache()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# =============================================================================
# 1. SUCCESSFUL LIVE WEATHER TESTS
# =============================================================================

def test_successful_live_weather_service():
    """Verify live weather returns live status, FRESH freshness, and High confidence."""
    # Open-Meteo current.time is in local timezone (IST when utc_offset_seconds=19800)
    ist_tz = timezone(timedelta(seconds=19800))
    recent_time = (datetime.now(ist_tz) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")
    mock_payload = {
        "current": {
            "time": recent_time,
            "temperature_2m": 38.5,
            "relative_humidity_2m": 42.0,
            "wind_speed_10m": 3.2,
            "shortwave_radiation_instant": 750.0,
        },
        "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        res = WeatherService.fetch_current_weather(17.3850, 78.4867)

    assert res["weather_status"] == "live"
    assert res["freshness"] == "FRESH"
    assert res["is_cached"] is False
    assert res["temp_c"] == 38.5
    assert res["rh_pct"] == 42.0
    assert res["source_transparency"]["confidence"] == "High"
    assert res["source_transparency"]["weather_status"] == "live"


def test_successful_live_weather_api_endpoint(client):
    """Verify /api/check-risk returns weather_status=live and High confidence on live fetch."""
    ist_tz = timezone(timedelta(seconds=19800))
    recent_time = (datetime.now(ist_tz) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")
    mock_payload = {
        "current": {
            "time": recent_time,
            "temperature_2m": 38.5,
            "relative_humidity_2m": 42.0,
            "wind_speed_10m": 3.2,
            "shortwave_radiation_instant": 750.0,
        },
        "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        resp = client.post("/api/check-risk", json={
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "18-40",
            "workload": "moderate",
            "clothing": "light",
            "experience": "very_used",
        })

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["weather_status"] == "live"
    assert data["freshness"] == "FRESH"
    assert data["is_cached"] is False
    assert "wbgt_c" in data
    assert data["wbgt_c"] > 0
    assert data["source_transparency"]["confidence"] == "High"
    assert data["confidence"]["level"] == "High"


# =============================================================================
# 2. TIMEOUT TESTS
# =============================================================================

def test_weather_timeout_service():
    """Verify persistent timeout raises WeatherServiceUnavailableError without synthetic fallback."""
    with patch("requests.get", side_effect=requests.Timeout("Connection timed out after 8s")):
        with pytest.raises(WeatherServiceUnavailableError) as exc_info:
            WeatherService.fetch_current_weather(28.6139, 77.2090, retries=0)
    assert "temporarily unavailable" in str(exc_info.value)


def test_weather_timeout_api_endpoint(client):
    """Verify timeout in /api/check-risk returns 502 with weather_status=unavailable and no metrics."""
    with patch("requests.get", side_effect=requests.Timeout("Connection timed out")):
        resp = client.post("/api/check-risk", json={
            "location": "Delhi",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "age_group": "18-40",
            "workload": "moderate",
            "clothing": "light",
        })

    assert resp.status_code == 502
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["weather_status"] == "unavailable"
    assert "Weather service is temporarily unavailable" in data["error"]
    # Ensure NO default scientific values are populated
    assert "wbgt_c" not in data
    assert "temp_c" not in data
    assert "risk_level" not in data
    assert "thermal_stress" not in data


# =============================================================================
# 3. HTTP FAILURE TESTS
# =============================================================================

@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_weather_http_server_error_service(status_code):
    """Verify HTTP 5xx errors raise WeatherServiceUnavailableError."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code} Server Error")

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(WeatherServiceUnavailableError):
            WeatherService.fetch_current_weather(19.0760, 72.8777, retries=0)


def test_weather_http_failure_api_endpoint(client):
    """Verify HTTP 500 upstream in /api/check-risk returns 502 unavailable without data."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")

    with patch("requests.get", return_value=mock_resp):
        resp = client.post("/api/check-risk", json={
            "location": "Mumbai",
            "latitude": 19.0760,
            "longitude": 72.8777,
        })

    assert resp.status_code == 502
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["weather_status"] == "unavailable"
    assert "wbgt_c" not in data


# =============================================================================
# 4. MALFORMED RESPONSE TESTS
# =============================================================================

@pytest.mark.parametrize("malformed_data", [
    {},
    {"current": {}},
    {"current": {"temperature_2m": None, "relative_humidity_2m": 50, "wind_speed_10m": 2}},
    {"current": {"temperature_2m": "non_numeric", "relative_humidity_2m": 50, "wind_speed_10m": 2}},
    {"current": {"temperature_2m": 85.0, "relative_humidity_2m": 50, "wind_speed_10m": 2}},  # Exceeds physical bound
])
def test_weather_malformed_response_service(malformed_data):
    """Verify provider structural or physical anomalies raise WeatherServiceValidationError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = malformed_data

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(WeatherServiceValidationError):
            WeatherService.fetch_current_weather(13.0827, 80.2707, retries=0)


def test_weather_malformed_response_api_endpoint(client):
    """Verify malformed upstream response in /api/check-risk returns 502 unavailable."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"current": {"broken": True}}

    with patch("requests.get", return_value=mock_resp):
        resp = client.post("/api/check-risk", json={
            "location": "Chennai",
            "latitude": 13.0827,
            "longitude": 80.2707,
        })

    assert resp.status_code == 502
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["weather_status"] == "unavailable"


# =============================================================================
# 5. STALE CACHE TESTS
# =============================================================================

def test_stale_cache_by_timestamp_observation():
    """Verify weather observation > 1 hour old is labelled stale_cache with downgraded confidence."""
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
    mock_payload = {
        "current": {
            "time": stale_time,
            "temperature_2m": 39.0,
            "relative_humidity_2m": 45.0,
            "wind_speed_10m": 2.5,
            "shortwave_radiation_instant": 600.0,
        },
        "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        res = WeatherService.fetch_current_weather(26.9124, 75.7873)

    assert res["weather_status"] == "stale_cache"
    assert res["freshness"] == "STALE"
    assert res["source_time"] == stale_time
    assert res["source_transparency"]["confidence"] in ("Medium", "Limited")
    assert res["source_transparency"]["confidence"] != "High"


def test_stale_cache_api_endpoint_downgrades_confidence(client):
    """Verify /api/check-risk returns weather_status=stale_cache and downgraded confidence."""
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
    mock_payload = {
        "current": {
            "time": stale_time,
            "temperature_2m": 40.0,
            "relative_humidity_2m": 35.0,
            "wind_speed_10m": 2.0,
            "shortwave_radiation_instant": 700.0,
        },
        "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        resp = client.post("/api/check-risk", json={
            "location": "Jaipur",
            "latitude": 26.9124,
            "longitude": 75.7873,
        })

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["weather_status"] == "stale_cache"
    assert data["freshness"] == "STALE"
    assert data["source_time"] == stale_time
    # Confidence MUST be downgraded from High
    assert data["confidence"]["level"] in ("Medium", "Limited")
    assert data["confidence"]["level"] != "High"
    assert data["source_transparency"]["confidence"] in ("Medium", "Limited")
    assert data["source_transparency"]["confidence"] != "High"


def test_explicit_get_cached_weather():
    """Verify get_cached_weather returns stale_cache with timestamp and downgraded confidence."""
    lat, lon = 21.1458, 79.0882
    cache_key = (round(lat, 3), round(lon, 3))
    fake_entry = {
        "temp_c": 37.0,
        "rh_pct": 40.0,
        "wind_ms": 2.0,
        "solar_wm2": 500.0,
        "source_time": "2026-09-17T12:00",
        "retrieved_at": "2026-09-17T12:05:00Z",
        "freshness": "FRESH",
        "weather_status": "live",
        "source_transparency": {
            "confidence": "High",
            "freshness": "FRESH",
            "updated": "2026-09-17T12:00",
        }
    }
    from services.weather_service import _CURRENT_CACHE
    _CURRENT_CACHE[cache_key] = (time.time() - 3600, fake_entry)

    cached = WeatherService.get_cached_weather(lat, lon)
    assert cached is not None
    assert cached["is_cached"] is True
    assert cached["freshness"] == "STALE"
    assert cached["weather_status"] == "stale_cache"
    assert cached["source_transparency"]["confidence"] == "Medium"


# =============================================================================
# 6. NO WEATHER DATA & ZERO-SYNTHETIC-FALLBACK TESTS
# =============================================================================

def test_no_weather_data_raises_value_error():
    """Verify assess_full_risk_profile refuses to evaluate when weather data is missing or empty."""
    with pytest.raises(ValueError) as exc1:
        assess_full_risk_profile(weather_data={}, personal_inputs={}, region="plains")
    assert "required" in str(exc1.value).lower()

    with pytest.raises(ValueError) as exc2:
        assess_full_risk_profile(weather_data={"rh_pct": 50.0}, personal_inputs={}, region="plains")
    assert "temperature" in str(exc2.value).lower()

    with pytest.raises(ValueError) as exc3:
        assess_full_risk_profile(weather_data={"temp_c": 38.0}, personal_inputs={}, region="plains")
    assert "humidity" in str(exc3.value).lower()


def test_authority_service_offline_returns_unavailable():
    """Verify AuthorityService marks offline sites as unavailable without fabricating 32.5°C."""
    def offline_fetcher(lat, lon):
        raise WeatherServiceUnavailableError("Provider offline")

    layer = AuthorityService.get_area_heat_layer(offline_fetcher, use_cache=False)
    assert "features" in layer
    f0 = layer["features"][0]["properties"]
    assert f0["weather_available"] is False
    assert f0["temp_c"] is None
    assert f0["wbgt_c"] is None
    assert f0["environmental_risk"] == "Unavailable"
    assert f0["base_risk"] == "Unavailable"
    assert f0["hap_level"] == "Unavailable"


def test_institute_service_offline_raises_unavailable():
    """Verify InstituteService raises on provider outage without fabricating 33.0°C."""
    def offline_fetcher(lat, lon):
        raise WeatherServiceUnavailableError("Provider offline")

    with pytest.raises(WeatherServiceUnavailableError):
        InstituteService.evaluate_institute("infra_delhi", offline_fetcher)


def test_api_explain_offline_returns_502(client):
    """Verify /api/explain returns 502 unavailable rather than fabricating 35°C on outage."""
    with patch("app.fetch_current_weather", side_effect=WeatherServiceUnavailableError("Provider offline")):
        resp = client.post("/api/decision-trace", json={
            "location": "Delhi",
            "latitude": 28.6139,
            "longitude": 77.2090,
        })
    assert resp.status_code == 502
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["weather_status"] == "unavailable"


# =============================================================================
# 7. FRONTEND CONTRACT & UI ELEMENTS
# =============================================================================

def test_frontend_my_risk_contains_state_containers(client):
    """Verify my_risk.html contains the necessary elements for dynamic state transitions."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Header badge
    assert 'id="form-weather-badge"' in html
    # Hero status indicators
    assert 'id="risk-summary-hero"' in html
    assert 'id="risk-summary-sync-text"' in html
    assert 'id="risk-summary-freshness"' in html
    assert 'id="risk-summary-confidence"' in html
    # Explicit Unavailable and Stale containers
    assert 'id="weather-unavailable-card"' in html
    assert 'id="weather-stale-banner"' in html
    assert 'id="weather-stale-timestamp"' in html
    # Initial state should NOT be hardcoded "Live Weather" / "Confidence: High"
    assert 'Freshness: Checking...' in html
    assert 'Confidence: Pending' in html
    assert 'AWAITING WEATHER' in html
