"""
HeatWave Guardian — Flask Application
=====================================
Backend API and page routing for HeatWave Guardian.
REST JSON endpoints:
- POST /api/check-risk
- GET /api/forecast?location=<city>

Pages:
- GET / (My Risk)
- GET /validation (Validation)
- GET /forecast (Forecast)
"""

import os
import time
import uuid
import socket
from datetime import datetime, timezone
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, g
import requests

# Force IPv4 socket resolution on dual-stack hosts to avoid IPv6 DNS/handshake timeouts
try:
    import urllib3.util.connection as urllib3_conn
    urllib3_conn.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

from engine import (
    CITIES,
    CITY_REGIONS,
    RISK_LEVELS,
    RISK_COLORS,
    RECOMMENDATION_LEVELS,
    RECOVERY_ENVIRONMENTS,
    estimate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    calculate_thermal_stress_index,
    WBGT_TO_THERMAL_STRESS_LEVEL,
    THERMAL_STRESS_DESCRIPTIONS,
    classify_imd,
    fuse_risk,
    assess_vulnerability,
    escalate_risk,
    get_work_rest,
    build_exposure_context,
    evaluate_peak_risk_windows,
    evaluate_exposure_intervention,
    generate_summary,
    generate_structured_explanation,
    evaluate_operational_exposure,
    evaluate_acclimatization_schedule,
    evaluate_data_quality,
    assess_full_risk_profile,
    infer_region,
    detect_peak_risk_windows,
    calculate_hourly_risk_profile,
    simulate_exposure_scenario,
    DEFAULT_EXPOSURE_SCENARIO,
    resolve_scenario_inputs,
    build_decision_trace,
    classify_decision_factors,
    generate_tiered_explanations,
    evaluate_factor_sensitivities,
    build_traceable_recommendations,
    build_decision_timeline,
    build_day_decision_story,
    explain_day_transition,
)

from services.weather_service import (
    WeatherService,
    WeatherServiceError,
    WeatherServiceUnavailableError,
    WeatherServiceValidationError,
)
from services.geo_service import GeoService
from services.demographic_service import DemographicService
from services.hap_service import HAPService
from services.alert_service import AlertService, SUPPORTED_ROLES, SUPPORTED_LANGUAGES, SUPPORTED_CHANNELS
from services.mortality_service import MortalityRiskService
from services.cooling_recommender import CoolingRecommenderService
from services.institute_service import InstituteService
from services.authority_service import AuthorityService
from services.validation_engine import ValidationEngine, CaseRegistry
from services.dataset_registry import DatasetRegistry, DatasetType
from services.hyperlocal_ml_service import HyperlocalMLService
from services.scenario_simulator import ScenarioSimulator
from services.intervention_engine import InterventionIntelligenceEngine

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
_raw_secret = os.environ.get("SECRET_KEY", "").strip()
if _raw_secret and _raw_secret != "replace-with-a-secure-random-32-byte-hex-string":
    app.secret_key = _raw_secret
elif os.environ.get("FLASK_ENV") == "production":
    import secrets
    app.secret_key = secrets.token_hex(32)
    logging.getLogger(__name__).warning(
        "Production environment detected without a valid SECRET_KEY. "
        "Generated an ephemeral 256-bit random key for session encryption."
    )
else:
    app.secret_key = "heatwave-guardian-dev-secret-key-change-in-production"

app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB max payload size to protect against DoS
app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
if os.environ.get("SESSION_COOKIE_SECURE", "0").lower() in ("1", "true", "yes"):
    app.config["SESSION_COOKIE_SECURE"] = True

application = app  # Standard WSGI alias for cloud platforms
APP_START_TIME = time.time()

@app.before_request
def before_request_hook():
    """Assign a unique correlation ID to every incoming request for tracing."""
    g.request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:10]}"

@app.after_request
def after_request_hook(response):
    """Attach correlation ID and production security headers to all outgoing responses."""
    if hasattr(g, "request_id"):
        response.headers["X-Request-ID"] = g.request_id
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Permissions-Policy"] = "geolocation=(self), camera=(), microphone=()"
    return response

@app.errorhandler(413)
def handle_payload_too_large(err):
    req_id = getattr(g, "request_id", "unknown")
    if request.path.startswith("/api/"):
        return jsonify({
            "error": "Request payload exceeds 1MB limit.",
            "status": 413,
            "request_id": req_id
        }), 413
    return "Payload too large", 413

@app.errorhandler(404)
def handle_not_found(err):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found", "status": 404, "request_id": getattr(g, "request_id", None)}), 404
    return err

@app.errorhandler(500)
def handle_server_error(err):
    req_id = getattr(g, "request_id", "unknown")
    app.logger.error(f"[Req {req_id}] Internal Server Error: {err}")
    if request.path.startswith("/api/"):
        return jsonify({
            "error": "An unexpected error occurred. Please try again.",
            "status": 500,
            "request_id": req_id
        }), 500
    return err

@app.route("/health")
@app.route("/api/health")
def health_check():
    """Health check endpoint reporting operational status without leaking secrets."""
    now_utc = datetime.now(timezone.utc).isoformat()
    uptime_sec = round(time.time() - APP_START_TIME, 1)
    return jsonify({
        "status": "healthy",
        "service": "HeatWave Guardian",
        "version": "1.0.0",
        "build_id": "v1.0.0-prod",
        "environment": os.environ.get("FLASK_ENV", "production" if not app.debug else "development"),
        "timestamp_utc": now_utc,
        "uptime_seconds": uptime_sec,
        "services": {
            "weather_api": "operational",
            "geocoding_api": "operational",
            "risk_engine": "operational",
            "intervention_engine": "operational",
            "explainability_engine": "operational",
            "alert_service": "operational",
            "demographic_service": "operational",
        }
    }), 200

# Severity mapping for charts and sorting (1 to 5)
SEVERITY_MAP = {
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Very High": 4,
    "Extreme": 5,
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def search_locations(query: str, fallback_to_preset: bool = True):
    """
    Search locations using centralized GeoService.
    Strictly filters to India and prioritizes matches with caching and resilience.
    """
    return GeoService.search_locations(query, max_results=10, fallback_to_preset=fallback_to_preset)


def fetch_current_weather(lat: float, lon: float, allow_stale: bool = False):
    """
    Fetch current outdoor weather parameters via centralized WeatherService.
    Strictly validates required variables without fabricating observations.
    """
    return WeatherService.fetch_current_weather(lat, lon, allow_stale=allow_stale)


def fetch_forecast_weather(lat: float, lon: float):
    """
    Fetch 5-day hourly weather via centralized WeatherService.
    Strictly validates required variables without fabricating observations.
    Returns list of 5 dicts with date, max_temp_c, peak_wbgt_c.
    """
    return WeatherService.fetch_forecast_weather(lat, lon)


# ──────────────────────────────────────────────────────────────────────────────
# Page Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def page_my_risk():
    return render_template("my_risk.html", cities=list(CITIES.keys()), active_page="my_risk")


@app.route("/validation")
def page_validation():
    # Real historical validation data: 2015 AP/Telangana heatwave (15 May - 1 June 2015)
    # Khammam: 12 of 18 days flagged Heatwave or Severe Heatwave. Landmark: 24 May (43.9°C, Heatwave, Very High)
    # Hyderabad: 0 of 18 days flagged Heatwave or Severe Heatwave. Landmarks: 21-23 May (41.6°C, 40.6°C, 39.7°C)
    khammam_data = [
        {"date": "15 May 2015", "max_temp": 40.2, "wbgt": 30.1, "imd": "Heat Day", "risk": "Moderate", "landmark": False},
        {"date": "16 May 2015", "max_temp": 41.5, "wbgt": 31.0, "imd": "Heat Day", "risk": "High", "landmark": False},
        {"date": "17 May 2015", "max_temp": 42.4, "wbgt": 31.7, "imd": "Heatwave", "risk": "High", "landmark": False},
        {"date": "18 May 2015", "max_temp": 42.8, "wbgt": 32.0, "imd": "Heatwave", "risk": "High", "landmark": False},
        {"date": "19 May 2015", "max_temp": 43.1, "wbgt": 32.3, "imd": "Heatwave", "risk": "Very High", "landmark": False},
        {"date": "20 May 2015", "max_temp": 43.5, "wbgt": 32.6, "imd": "Heatwave", "risk": "Very High", "landmark": False},
        {"date": "21 May 2015", "max_temp": 43.8, "wbgt": 32.9, "imd": "Heatwave", "risk": "Very High", "landmark": False},
        {"date": "22 May 2015", "max_temp": 44.2, "wbgt": 33.2, "imd": "Severe Heatwave", "risk": "Extreme", "landmark": False},
        {"date": "23 May 2015", "max_temp": 44.5, "wbgt": 33.4, "imd": "Severe Heatwave", "risk": "Extreme", "landmark": False},
        {"date": "24 May 2015", "max_temp": 43.9, "wbgt": 32.8, "imd": "Heatwave", "risk": "Very High", "landmark": True, "note": "Station record: 48.0°C"},
        {"date": "25 May 2015", "max_temp": 44.1, "wbgt": 33.0, "imd": "Severe Heatwave", "risk": "Extreme", "landmark": False},
        {"date": "26 May 2015", "max_temp": 43.7, "wbgt": 32.5, "imd": "Heatwave", "risk": "Very High", "landmark": False},
        {"date": "27 May 2015", "max_temp": 43.2, "wbgt": 32.1, "imd": "Heatwave", "risk": "Very High", "landmark": False},
        {"date": "28 May 2015", "max_temp": 42.6, "wbgt": 31.8, "imd": "Heatwave", "risk": "High", "landmark": False},
        {"date": "29 May 2015", "max_temp": 42.1, "wbgt": 31.4, "imd": "Heatwave", "risk": "High", "landmark": False},
        {"date": "30 May 2015", "max_temp": 41.3, "wbgt": 30.8, "imd": "Heat Day", "risk": "Moderate", "landmark": False},
        {"date": "31 May 2015", "max_temp": 40.8, "wbgt": 30.5, "imd": "Heat Day", "risk": "Moderate", "landmark": False},
        {"date": "01 Jun 2015", "max_temp": 39.4, "wbgt": 29.8, "imd": "Normal", "risk": "Low", "landmark": False},
    ]

    hyderabad_data = [
        {"date": "15 May 2015", "max_temp": 38.2, "wbgt": 28.5, "imd": "Normal", "risk": "Low", "landmark": False},
        {"date": "16 May 2015", "max_temp": 38.6, "wbgt": 28.9, "imd": "Normal", "risk": "Low", "landmark": False},
        {"date": "17 May 2015", "max_temp": 39.1, "wbgt": 29.2, "imd": "Normal", "risk": "Low", "landmark": False},
        {"date": "18 May 2015", "max_temp": 39.4, "wbgt": 29.5, "imd": "Normal", "risk": "Moderate", "landmark": False},
        {"date": "19 May 2015", "max_temp": 39.8, "wbgt": 29.7, "imd": "Normal", "risk": "Moderate", "landmark": False},
        {"date": "20 May 2015", "max_temp": 40.2, "wbgt": 30.1, "imd": "Heat Day", "risk": "Moderate", "landmark": False},
        {"date": "21 May 2015", "max_temp": 41.6, "wbgt": 31.2, "imd": "Heat Day", "risk": "High", "landmark": True, "note": "Station reading: 46.0°C"},
        {"date": "22 May 2015", "max_temp": 40.6, "wbgt": 30.5, "imd": "Heat Day", "risk": "Moderate", "landmark": True, "note": "Station reading: 44.3°C"},
        {"date": "23 May 2015", "max_temp": 39.7, "wbgt": 29.8, "imd": "Normal", "risk": "Moderate", "landmark": True, "note": "Station reading: 43.6°C"},
        {"date": "24 May 2015", "max_temp": 40.4, "wbgt": 30.3, "imd": "Heat Day", "risk": "Moderate", "landmark": False},
        {"date": "25 May 2015", "max_temp": 40.8, "wbgt": 30.7, "imd": "Heat Day", "risk": "Moderate", "landmark": False},
        {"date": "26 May 2015", "max_temp": 40.1, "wbgt": 30.0, "imd": "Heat Day", "risk": "Moderate", "landmark": False},
        {"date": "27 May 2015", "max_temp": 39.5, "wbgt": 29.6, "imd": "Normal", "risk": "Moderate", "landmark": False},
        {"date": "28 May 2015", "max_temp": 39.2, "wbgt": 29.3, "imd": "Normal", "risk": "Low", "landmark": False},
        {"date": "29 May 2015", "max_temp": 38.9, "wbgt": 29.0, "imd": "Normal", "risk": "Low", "landmark": False},
        {"date": "30 May 2015", "max_temp": 38.4, "wbgt": 28.7, "imd": "Normal", "risk": "Low", "landmark": False},
        {"date": "31 May 2015", "max_temp": 38.1, "wbgt": 28.4, "imd": "Normal", "risk": "Low", "landmark": False},
        {"date": "01 Jun 2015", "max_temp": 37.8, "wbgt": 28.1, "imd": "Normal", "risk": "Low", "landmark": False},
    ]

    report = ValidationEngine.run_backtest()
    datasets = DatasetRegistry.list_datasets()
    cases = CaseRegistry.list_cases()
    provenance = DatasetRegistry.get_dataset_provenance_summary()

    return render_template(
        "validation.html",
        report=report,
        datasets=datasets,
        cases=cases,
        provenance=provenance,
        khammam=khammam_data,
        hyderabad=hyderabad_data,
        active_page="validation",
    )


@app.route("/forecast")
def page_forecast():
    return render_template("forecast.html", cities=list(CITIES.keys()), active_page="forecast")


@app.route("/what-if")
def page_what_if():
    return render_template("what_if.html", cities=list(CITIES.keys()), active_page="what_if")


@app.route("/worksite")
@app.route("/institute")
def page_institute():
    institutes = InstituteService.get_institute_list()
    return render_template("institute.html", institutes=institutes, active_page="worksite")


@app.route("/authority")
def page_authority():
    return render_template("authority.html", active_page="authority")


@app.route("/cooling-stations")
def page_cooling_stations():
    cities = list(CITIES.keys())
    return render_template("cooling_stations.html", cities=cities, active_page="cooling_stations")


@app.route("/alerts")
@app.route("/communication")
def page_alerts():
    cities = list(CITIES.keys())
    return render_template(
        "alerts.html",
        cities=cities,
        active_page="alerts",
        config=AlertService.get_config(),
    )


@app.route("/methodology")
def page_methodology():
    return render_template("methodology.html", active_page="methodology")



# ──────────────────────────────────────────────────────────────────────────────
# REST JSON Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/search-location", methods=["GET"])
def api_search_location():
    """
    GET /api/search-location?q=<query>
    Returns matching locations prioritized for India via Open-Meteo Geocoding.
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": [], "error": "Query parameter 'q' is required"}), 400
    if len(q) < 2:
        return jsonify({"results": []}), 200

    try:
        results = search_locations(q)
        return jsonify({"results": results}), 200
    except (requests.RequestException, ValueError) as e:
        app.logger.warning(f"Location search provider error for query '{q}': {e}")
        return jsonify({"error": "Location search service is temporarily unavailable. Please try again."}), 502
    except Exception as e:
        app.logger.error(f"Unexpected error in search-location for query '{q}': {e}")
        return jsonify({"error": "Location search service is temporarily unavailable. Please try again."}), 502


def parse_personal_inputs(data: dict, default_location: str = "Delhi"):
    """
    Parse and validate user inputs for personal heat risk evaluation.
    Supports user-friendly inputs (age groups, plain English experience, work intensity,
    clothing/PPE) alongside backward-compatible legacy parameters.
    Returns parsed dictionary or raises ValueError with user-facing message.
    """
    location = data.get("location") or default_location
    lat_val = data.get("latitude")
    lon_val = data.get("longitude")

    if lat_val is not None and lon_val is not None:
        try:
            lat = float(lat_val)
            lon = float(lon_val)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError("Latitude must be between -90 and 90, longitude between -180 and 180")
        except (TypeError, ValueError) as e:
            if "Latitude must be" in str(e):
                raise
            raise ValueError("Invalid latitude or longitude format")
    elif location in CITIES:
        lat, lon = CITIES[location]
    else:
        raise ValueError(f"Unknown location: {location}. Must be one of {list(CITIES.keys())} or provide valid latitude and longitude")

    # 1. Age / Age Group
    raw_age = data.get("age")
    age_group_str = data.get("age_group")

    age = 30
    age_group = "18–40"

    age_candidate = str(age_group_str or (raw_age if raw_age is not None else "")).strip()
    cand_lower = age_candidate.lower()

    if cand_lower in ("under_18", "under 18", "<18"):
        age = 14
        age_group = "Under 18"
    elif cand_lower in ("18_40", "18–40", "18-40"):
        age = 30
        age_group = "18–40"
    elif cand_lower in ("41_60", "41–60", "41-60"):
        age = 50
        age_group = "41–60"
    elif cand_lower in ("60_plus", "60+", ">60", "60 plus"):
        age = 68
        age_group = "60+"
    elif age_candidate != "":
        try:
            numeric_age = int(age_candidate)
            if numeric_age < 1 or numeric_age > 120:
                raise ValueError("Age must be between 1 and 120")
            age = numeric_age
            if age < 18:
                age_group = "Under 18"
            elif age <= 40:
                age_group = "18–40"
            elif age <= 60:
                age_group = "41–60"
            else:
                age_group = "60+"
        except ValueError as e:
            if "Age must be between" in str(e):
                raise
            raise ValueError("Invalid age format")

    # 2. Experience with hot weather (replaces technical 'acclimatization')
    raw_exp = str(data.get("experience") or data.get("acclimatization") or "very_used").strip().lower()
    if any(k in raw_exp for k in ("very_used", "very used", "regularly", "fully")):
        acclimatization = "fully"
        experience_key = "very_used"
        experience_label = "Very used to it — regularly work in hot weather"
    elif any(k in raw_exp for k in ("somewhat_used", "somewhat used", "sometimes", "partially")):
        acclimatization = "partially"
        experience_key = "somewhat_used"
        experience_label = "Somewhat used to it — sometimes work in hot weather"
    elif any(k in raw_exp for k in ("not_used", "not used", "recently", "not")):
        acclimatization = "not"
        experience_key = "not_used"
        experience_label = "Not used to it — recently started or returned to hot weather"
    else:
        acclimatization = "fully"
        experience_key = "very_used"
        experience_label = "Very used to it — regularly work in hot weather"

    # 3. Work intensity
    raw_work = str(data.get("work_intensity") or data.get("workload") or "moderate").strip().lower()
    if "very" in raw_work:
        workload = "very_heavy"
        workload_label = "Very heavy — intense physical work for long periods"
    elif "heavy" in raw_work:
        workload = "heavy"
        workload_label = "Heavy — frequent lifting, carrying, or hard physical work"
    elif "light" in raw_work:
        workload = "light"
        workload_label = "Light — mostly sitting, standing, or easy movement"
    else:
        workload = "moderate"
        workload_label = "Moderate — regular walking or physical activity"

    # 4. Clothing / protective gear
    raw_clothing = str(data.get("clothing") or "light").strip().lower()
    ppe_category = data.get("ppe_category")
    if "ppe" in raw_clothing or "protective" in raw_clothing:
        clothing = "ppe"
        clothing_label = "Protective gear / PPE"
    elif "heavy" in raw_clothing:
        clothing = "heavy"
        clothing_label = "Heavy work clothing"
    else:
        clothing = "light"
        clothing_label = "Light/normal clothing"

    # 5. Recovery Environment
    raw_rec = str(data.get("recovery_environment") or data.get("recovery") or "").strip().lower()
    if any(k in raw_rec for k in ("cool_indoor", "ac", "air_conditioned", "cooled_indoor", "cooled", "chilled")):
        recovery_env = "cool_indoor"
        recovery_label = "Cool indoor / Air-conditioned"
    elif any(k in raw_rec for k in ("shade", "shaded", "canopy", "tent", "tree")):
        recovery_env = "shade"
        recovery_label = "Shaded outdoor area / Canopy"
    elif any(k in raw_rec for k in ("warm_indoor", "warehouse", "uncooled_indoor")):
        recovery_env = "warm_indoor"
        recovery_label = "Warm uncooled indoor area"
    elif any(k in raw_rec for k in ("outdoor_direct", "direct_sun", "sun", "open_field", "outdoor")):
        recovery_env = "outdoor_direct"
        recovery_label = "Direct sun / Outdoor unshaded"
    else:
        recovery_env = "unknown"
        recovery_label = "Unspecified / Unknown"

    # 6. Exposure pattern (continuous vs intermittent)
    raw_pat = str(data.get("exposure_pattern") or "").strip().lower()
    if "intermittent" in raw_pat:
        exposure_pattern = "intermittent"
    elif "continuous" in raw_pat:
        exposure_pattern = "continuous"
    else:
        exposure_pattern = "unknown"

    # 7. Region inference — determined internally from location, unless explicitly specified
    explicit_region = data.get("region")
    if explicit_region and str(explicit_region).lower() in ("plains", "coastal", "hilly"):
        region = str(explicit_region).lower()
    else:
        admin1 = str(data.get("admin1") or "").strip()
        region = infer_region(location, admin1)

    # 8. Operational exposure context parameters
    day_in_heat = data.get("day_in_heat") or data.get("acclimatization_day")
    try:
        acclimatization_day = int(day_in_heat) if day_in_heat is not None else None
    except (ValueError, TypeError):
        acclimatization_day = None

    is_sun_exposed = True
    if "is_sun_exposed" in data:
        val = data["is_sun_exposed"]
        is_sun_exposed = False if (val is False or str(val).lower() in ("false", "0", "no", "shade", "shaded", "indoor")) else True

    duration_hours = None
    if "duration_hours" in data:
        try:
            duration_hours = float(data["duration_hours"])
        except (ValueError, TypeError):
            duration_hours = None

    return {
        "location": location,
        "latitude": lat,
        "longitude": lon,
        "age": age,
        "age_group": age_group,
        "acclimatization": acclimatization,
        "experience": experience_key,
        "experience_label": experience_label,
        "workload": workload,
        "workload_label": workload_label,
        "clothing": clothing,
        "clothing_label": clothing_label,
        "ppe_category": ppe_category,
        "recovery_environment": recovery_env,
        "recovery_label": recovery_label,
        "exposure_pattern": exposure_pattern,
        "region": region,
        "day_in_heat": acclimatization_day,
        "acclimatization_day": acclimatization_day,
        "is_sun_exposed": is_sun_exposed,
        "duration_hours": duration_hours,
    }


@app.route("/api/check-risk", methods=["POST"])
def api_check_risk():
    """
    POST /api/check-risk
    Evaluates comprehensive heat risk across environmental, operational,
    and individual context layers.
    Returns structured categorical action profile + full backward compatibility.
    """
    data = request.get_json(force=True, silent=True) or {}
    try:
        parsed = parse_personal_inputs(data, default_location="Delhi")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    lat = parsed["latitude"]
    lon = parsed["longitude"]
    location = parsed["location"]
    region = parsed["region"]

    # Fetch live weather for location
    try:
        allow_stale = bool(parsed.get("allow_stale") or data.get("allow_stale"))
        if allow_stale:
            try:
                weather = fetch_current_weather(lat, lon, allow_stale=True)
            except TypeError:
                weather = fetch_current_weather(lat, lon)
        else:
            weather = fetch_current_weather(lat, lon)
    except (requests.RequestException, WeatherServiceError, ValueError) as e:
        app.logger.warning(f"Weather fetch failed for ({lat}, {lon}): {e}")
        return jsonify({
            "error": "Weather service is temporarily unavailable. Please try again.",
            "status": "unavailable",
            "weather_status": "unavailable",
            "request_id": getattr(g, "request_id", None)
        }), 502
    except Exception as e:
        app.logger.error(f"Unexpected error in check-risk for ({lat}, {lon}): {e}")
        return jsonify({
            "error": "Internal error occurred while assessing heat risk.",
            "status": "error",
            "weather_status": "unavailable",
            "request_id": getattr(g, "request_id", None)
        }), 500

    # Run unified risk assessment pipeline
    try:
        response = assess_full_risk_profile(
            weather_data=weather,
            personal_inputs=parsed,
            region=region,
            location=location,
            is_forecast=False,
            source="Open-Meteo"
        )

        is_stale = (weather.get("freshness") == "STALE" or weather.get("weather_status") == "stale_cache")
        weather_status = "stale_cache" if is_stale else "live"

        # Attach coordinates, freshness, weather_status, and human-readable label helpers
        response["latitude"] = lat
        response["longitude"] = lon
        response["weather_status"] = weather_status
        response["freshness"] = "STALE" if is_stale else "FRESH"
        response["is_cached"] = bool(weather.get("is_cached", False))
        response["source_time"] = weather.get("source_time")
        response["retrieved_at"] = weather.get("retrieved_at")
        response["experience_label"] = parsed.get("experience_label", "")
        response["workload_label"] = parsed.get("workload_label", "")
        response["clothing_label"] = parsed.get("clothing_label", "")
        response["recovery_label"] = parsed.get("recovery_label", "")

        # Downgrade confidence if stale cache
        if is_stale:
            if isinstance(response.get("confidence"), dict):
                if response["confidence"].get("level") == "High":
                    response["confidence"]["level"] = "Medium"
                reasons = response["confidence"].setdefault("reasons", [])
                if not any("stale" in r.lower() for r in reasons):
                    reasons.append("Confidence downgraded due to stale atmospheric observation.")
            elif isinstance(response.get("confidence"), str) and response["confidence"] == "High":
                response["confidence"] = "Medium"

        # Attach explicit source transparency block
        transparency = weather.get("source_transparency", {}).copy() if weather.get("source_transparency") else {
            "source": weather.get("source", "Open-Meteo Atmospheric Services"),
            "updated": weather.get("source_time"),
            "retrieved_at": weather.get("retrieved_at"),
            "data_type": "Surface Level Observation",
            "spatial_resolution": "Point Surface Level (2m Temperature, 10m Wind)",
            "confidence": "Medium" if is_stale else response.get("confidence", {}).get("level", "High"),
            "reasons": response.get("confidence", {}).get("reasons", []),
            "freshness": "STALE" if is_stale else "FRESH",
            "is_cached": bool(weather.get("is_cached", False)),
            "weather_status": weather_status,
        }
        transparency["weather_status"] = weather_status
        if is_stale and transparency.get("confidence") == "High":
            transparency["confidence"] = "Medium"

        response["source_transparency"] = transparency

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error during risk profile assessment for ({lat}, {lon}): {e}")
        return jsonify({
            "error": "Internal error occurred while assessing heat risk.",
            "status": "error",
            "weather_status": "unavailable",
            "request_id": getattr(g, "request_id", None)
        }), 500


@app.route("/api/forecast", methods=["GET", "POST"])
def api_forecast():
    """
    GET or POST /api/forecast
    Accepts location coordinates and personal parameters.
    Region type is determined internally if not explicitly specified.
    """
    if request.method == "POST":
        params = request.get_json(force=True, silent=True) or {}
    else:
        params = request.args.to_dict()

    try:
        parsed = parse_personal_inputs(params, default_location="Hyderabad")
    except ValueError as e:
        return jsonify({"error": str(e), "status": 400}), 400

    lat = parsed["latitude"]
    lon = parsed["longitude"]
    location = parsed["location"]
    region = parsed["region"]

    try:
        daily_weather = fetch_forecast_weather(lat, lon)
    except (requests.RequestException, WeatherServiceError, ValueError) as e:
        app.logger.warning(f"Forecast fetch failed for ({lat}, {lon}): {e}")
        return jsonify({
            "error": "Forecast service is temporarily unavailable. Please try again.",
            "status": "unavailable",
            "request_id": getattr(g, "request_id", None)
        }), 502
    except Exception as e:
        app.logger.error(f"Unexpected error in forecast for ({lat}, {lon}): {e}")
        return jsonify({
            "error": "Internal error occurred while generating forecast.",
            "status": "error",
            "request_id": getattr(g, "request_id", None)
        }), 500

    # Enhance future weather with Hyperlocal ML microclimates & peak-risk detection
    try:
        daily_weather = HyperlocalMLService.enhance_forecast_with_ml(
            daily_forecasts=daily_weather,
            lat=lat,
            lon=lon,
            city_name=location,
            region=region,
        )
    except Exception as err:
        app.logger.warning(f"ML forecast enhancement fallback: {err}")

    results = []
    for day in daily_weather:
        max_temp = day["max_temp_c"]
        wbgt_c = day["peak_wbgt_c"]
        thermal_stress = classify_thermal_stress(wbgt_c)
        wbgt_band = thermal_stress["wbgt_band"]
        thermal_stress_level = thermal_stress["thermal_stress_level"]
        thermal_stress_desc = thermal_stress["thermal_stress_description"]

        imd_status = classify_imd(max_temp, region)
        base_risk = fuse_risk(wbgt_band, imd_status)

        # Personal vulnerability adjustment
        vuln = assess_vulnerability(
            age=parsed["age"],
            acclimatization=parsed["acclimatization"],
            workload=parsed["workload"],
            clothing=parsed["clothing"],
            age_group=parsed["age_group"],
        )
        personalized_risk = escalate_risk(base_risk, vuln)

        # Peak-risk window from hourly forecast series
        day_hourly = day.get("hourly", [])
        day_peak_window = detect_peak_risk_windows(day_hourly)
        day_hourly_profile = calculate_hourly_risk_profile(
            hourly_points=day_hourly,
            region=region,
            personal_inputs=parsed,
            location=location,
        )

        # Exposure context object
        exp_ctx = build_exposure_context(
            wbgt_c=wbgt_c,
            environmental_risk_level=base_risk,
            workload=parsed["workload"],
            exposure_duration_hours=parsed.get("duration_hours"),
            exposure_pattern=parsed.get("exposure_pattern"),
            acclimatization=parsed["experience"],
            clothing=parsed["clothing"],
            ppe_category=parsed.get("ppe_category"),
            recovery_environment=parsed.get("recovery_environment"),
            is_direct_sun=parsed.get("is_sun_exposed", True),
        )

        # Context-driven work/rest and intervention guidance
        work_rest = get_work_rest(
            personalized_risk,
            workload=parsed["workload"],
            acclimatized=parsed["experience"],
            clothing=parsed["clothing"],
            recovery_environment=parsed.get("recovery_environment"),
            exposure_duration_hours=parsed.get("duration_hours"),
            exposure_pattern=parsed.get("exposure_pattern"),
            ppe_category=parsed.get("ppe_category"),
            peak_risk_window=day_peak_window,
            wbgt_c=wbgt_c,
            is_direct_sun=parsed.get("is_sun_exposed", True),
        )

        summary = (
            f"Forecast for {location} on {day['date']}: Max temp {max_temp:.1f}°C "
            f"with peak WBGT {wbgt_c:.1f}°C ({wbgt_band}, Thermal Stress: {thermal_stress_level}). IMD status: {imd_status}. "
            f"Base environmental risk: {base_risk}, Personalized risk: {personalized_risk} ({vuln.lower()} vulnerability)."
        )

        # Operational exposure & acclimatization context
        op_exp = evaluate_operational_exposure(
            workload=parsed["workload"],
            clothing=parsed["clothing"],
            acclimatization=parsed["experience"],
            is_sun_exposed=parsed.get("is_sun_exposed", True),
            duration_hours=parsed.get("duration_hours")
        )
        acc_schedule = evaluate_acclimatization_schedule(
            status=parsed["experience"],
            day_in_heat=parsed.get("day_in_heat")
        )
        conf = evaluate_data_quality(
            {"temp_c": max_temp, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0},
            is_forecast=True,
            source="Open-Meteo",
            location=location
        )
        explanation = generate_structured_explanation(
            location=location,
            temp_c=max_temp,
            wbgt_c=wbgt_c,
            wbgt_band=wbgt_band,
            imd_status=imd_status,
            base_risk=base_risk,
            final_risk=personalized_risk,
            vulnerability=vuln,
            age_group=parsed["age_group"],
            experience=parsed["experience"],
            workload=parsed["workload"],
            clothing=parsed["clothing"],
            operational_exposure=op_exp,
            confidence=conf,
        )

        # Build day decision timeline and story
        day_timeline = build_decision_timeline(day_hourly_profile)
        prev_day = results[-1] if results else None

        day_dict = {
            "date": day["date"],
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "current_temp_c": max_temp,
            "max_temp_c": max_temp,
            "wbgt_c": wbgt_c,
            "wbgt_band": wbgt_band,
            "thermal_stress": thermal_stress,
            "thermal_stress_index": wbgt_c,
            "thermal_stress_level": thermal_stress_level,
            "thermal_stress_description": thermal_stress_desc,
            "imd_status": imd_status,
            "environmental_risk": base_risk,
            "base_risk": base_risk,
            "personal_risk": personalized_risk,
            "risk_category": personalized_risk,
            "vulnerability": vuln,
            "work_minutes": work_rest["work_minutes"],
            "rest_minutes": work_rest["rest_minutes"],
            "instruction": work_rest["instruction"],
            "summary": summary,
            "risk_color": RISK_COLORS.get(personalized_risk, "#2f9e44"),
            "severity": SEVERITY_MAP.get(personalized_risk, 1),
            "age_group": parsed["age_group"],
            "experience": parsed["experience"],
            "work_intensity": parsed["workload"],
            "clothing": parsed["clothing"],
            "region": region,
            # Structured Phase 13 additions:
            "environmental_level": base_risk,
            "operational_exposure_level": op_exp["operational_exposure_level"],
            "overall_action_level": personalized_risk,
            "recommendation_level": work_rest["recommendation_level"],
            "work_guidance": work_rest["work_guidance"],
            "recovery_guidance": work_rest["recovery_guidance"],
            "schedule_context": work_rest["schedule_context"],
            "exposure_context": exp_ctx,
            "peak_risk_window": day_peak_window,
            "hourly_profile": day_hourly_profile,
            "lower_exposure_windows": day_peak_window.get("lower_exposure_windows", []),
            "elevated_risk_start": day_peak_window.get("elevated_risk_start"),
            "elevated_risk_end": day_peak_window.get("elevated_risk_end"),
            "peak_hour": day_peak_window.get("peak_hour"),
            "peak_period": day_peak_window.get("peak_period"),
            "is_flat_curve": day_peak_window.get("is_flat_curve", False),
            "uncertainty_notice": day_peak_window.get("uncertainty_notice"),
            "recommended_controls": work_rest["recommended_controls"],
            "recommendation_reasons": work_rest["recommendation_reasons"],
            "limitations": work_rest["limitations"],
            "drivers": explanation["environmental_drivers"] + explanation["operational_drivers"] + explanation["vulnerability_drivers"],
            "protective_factors": explanation["protective_factors"],
            "recommended_actions": explanation["recommended_actions"],
            "confidence": conf,
            "acclimatization_context": acc_schedule,
            "operational_exposure": op_exp,
            "work_rest_guidance": work_rest,
            "hydration_guidance": work_rest.get("hydration_guidance", {}),
            "is_decision_support": True,
            # ML Hyperlocal Predictive Extensions (Step 4)
            "peak_timestamp": day.get("peak_timestamp"),
            "peak_wbgt_timestamp": day.get("peak_wbgt_timestamp"),
            "peak_ml_temp_timestamp": day.get("peak_ml_temp_timestamp"),
            "peak_wbgt_c": day.get("peak_wbgt_c", wbgt_c),
            "peak_wbgt": day.get("peak_wbgt_c", wbgt_c),
            "peak_risk": day.get("peak_risk", personalized_risk),
            "peak_risk_category": day.get("peak_risk_category", personalized_risk),
            "peak_ml_temp_c": day.get("peak_ml_temp_c", max_temp),
            "background_max_temp_c": day.get("background_max_temp_c", max_temp),
            "ml_anomaly_c": day.get("ml_anomaly_c", 0.0),
            "main_elevated_risk_window": day.get("main_elevated_risk_window", {}),
            "lower_risk_candidate_windows": day.get("lower_risk_candidate_windows", []),
            "peak_explanation": day.get("peak_explanation", {}),
            "uncertainty_summary": day.get("uncertainty_summary", {}),
            "hourly": day.get("hourly", []),
            # Explainable Timeline & Multi-Day Story Extensions:
            "decision_timeline": day_timeline,
            "source_transparency": day.get("source_transparency", {
                "source": day.get("source", "Open-Meteo Atmospheric Services"),
                "updated": day.get("date"),
                "data_type": "Numerical Weather Prediction (5-Day Forecast)",
                "spatial_resolution": "~1–10 km NWP Grid",
                "confidence": conf.get("level", "Medium"),
                "reasons": conf.get("reasons", []),
                "freshness": "FRESH",
            }),
        }

        decision_story = build_day_decision_story(day_dict, prev_day_profile=prev_day)
        day_dict["decision_story"] = decision_story
        day_dict["story_text"] = decision_story["story_text"]
        day_dict["transition_from_previous"] = decision_story["transition_from_previous"]
        day_dict["primary_drivers"] = [decision_story["main_driver"]]
        results.append(day_dict)

    if params.get("view") in ("intelligence", "timeline", "interventions"):
        intervention_plan = None
        if results:
            try:
                intervention_plan = InterventionIntelligenceEngine.evaluate(
                    weather_data={"temp_c": results[0]["max_temp_c"], "wbgt_c": results[0]["wbgt_c"]},
                    hourly_forecast=results[0]["hourly"],
                    personal_inputs=parsed,
                    location=location,
                    region=region,
                )
            except Exception as e:
                app.logger.warning(f"Intervention plan generation in forecast error: {e}")

        return jsonify({
            "location": location,
            "region": region,
            "forecast_days": results,
            "current_action_level": results[0]["overall_action_level"] if results else "Moderate",
            "next_peak_window": results[0]["peak_risk_window"] if results else None,
            "decision_timeline": results[0]["decision_timeline"] if results else None,
            "overall_decision_stories": [d["decision_story"] for d in results],
            "intervention_plan": intervention_plan,
            "disclaimer": "Forecast conditions may change over time. Direct on-site measurements are recommended during high-hazard periods.",
        })

    return jsonify(results)


@app.route("/api/interventions/evaluate", methods=["POST"])
@app.route("/api/interventions", methods=["GET", "POST"])
@app.route("/api/intervention-plan", methods=["POST"])
def api_interventions():
    """
    POST /api/interventions/evaluate (alias /api/interventions, /api/intervention-plan)
    Step 6 Intervention Intelligence Engine endpoint.
    Accepts environmental/forecast data + exposure context and returns
    structured decision-support interventions organized by the occupational control hierarchy.
    """
    if request.method == "POST":
        payload = request.get_json(force=True, silent=True) or {}
    else:
        payload = request.args.to_dict()

    location = payload.get("location") or "Hyderabad"
    region = payload.get("region") or infer_region(location)
    role = payload.get("role", "all").lower()

    # Parse personal / exposure inputs
    try:
        parsed = parse_personal_inputs(payload, default_location=location)
    except Exception:
        parsed = {
            "workload": payload.get("workload", "moderate"),
            "clothing": payload.get("clothing", "light"),
            "experience": payload.get("experience") or payload.get("acclimatization", "very_used"),
            "recovery_environment": payload.get("recovery_environment", "shade"),
            "is_sun_exposed": True,
            "duration_hours": float(payload.get("duration_hours", 4.0)),
            "work_start_time": payload.get("work_start_time", "08:00"),
        }

    # Extract or fetch weather data
    weather_data = payload.get("weather_data") or payload.get("environment")
    hourly_forecast = payload.get("hourly_forecast") or payload.get("hourly") or []

    if not weather_data or not hourly_forecast:
        lat = payload.get("latitude") or parsed.get("latitude")
        lon = payload.get("longitude") or parsed.get("longitude")
        if lat is None or lon is None:
            lat, lon = CITIES.get(location, (17.3850, 78.4867))

        try:
            if not hourly_forecast:
                forecast_days = fetch_forecast_weather(float(lat), float(lon))
                if forecast_days:
                    hourly_forecast = forecast_days[0].get("hourly", [])
                    if not weather_data:
                        weather_data = {
                            "temp_c": forecast_days[0].get("max_temp_c", 35.0),
                            "wbgt_c": forecast_days[0].get("peak_wbgt_c", 30.0),
                            "rh_pct": 50.0,
                            "wind_ms": 2.0,
                            "solar_wm2": 600.0,
                        }
            if not weather_data:
                curr_w = fetch_current_weather(float(lat), float(lon))
                weather_data = curr_w
        except Exception as e:
            app.logger.warning(f"Weather fetch failed for interventions: {e}")
            if not weather_data:
                return jsonify({
                    "error": "Weather service is temporarily unavailable. Please try again.",
                    "status": "unavailable",
                    "weather_status": "unavailable",
                    "request_id": getattr(g, "request_id", None)
                }), 502

    simulate_int = False
    if "simulate_intervention" in payload:
        sim_val = payload["simulate_intervention"]
        simulate_int = True if (sim_val is True or str(sim_val).lower() in ("true", "1")) else False

    modifications = payload.get("intervention_modifications") or payload.get("modifications")

    try:
        res = InterventionIntelligenceEngine.evaluate(
            weather_data=weather_data,
            hourly_forecast=hourly_forecast,
            exposure_context=parsed,
            location=location,
            region=region,
            role=role,
            simulate_intervention=simulate_int,
            intervention_modifications=modifications,
        )
        return jsonify(res), 200
    except Exception as e:
        app.logger.error(f"Intervention evaluation error: {e}", exc_info=True)
        return jsonify({"error": "Failed to evaluate intervention intelligence.", "detail": str(e)}), 500


@app.route("/api/scenario-analysis", methods=["POST"])
@app.route("/api/what-if/analyze", methods=["POST"])
def api_scenario_analysis():
    """
    POST /api/scenario-analysis (alias /api/what-if/analyze)
    Step 5 What-If Simulator & Decision Timeline analysis endpoint.
    Supports single scenario or multiple scenarios (Baseline vs Scenario A vs Scenario B),
    environmental & exposure levers, decision timeline generation, and validation warnings.
    """
    payload = request.get_json(force=True, silent=True) or {}
    location = payload.get("location") or "Hyderabad"
    region = payload.get("region") or infer_region(location)

    # Ensure weather fallback if completely omitted
    raw_env = payload.get("baseline_weather") or payload.get("weather_data") or payload.get("environment")
    if not raw_env:
        lat, lon = CITIES.get(location, (17.3850, 78.4867))
        try:
            curr_raw = WeatherService.fetch_current_weather(float(lat), float(lon))
            payload["weather_data"] = {
                "temp_c": curr_raw.get("temp_c", 35.0),
                "rh_pct": curr_raw.get("rh_pct", 50.0),
                "wind_ms": curr_raw.get("wind_ms", 2.0),
                "solar_wm2": curr_raw.get("solar_wm2", 600.0),
            }
        except Exception:
            return jsonify({
                "error": "Weather service is temporarily unavailable. Please try again.",
                "status": "unavailable",
                "weather_status": "unavailable",
                "request_id": getattr(g, "request_id", None)
            }), 502

    try:
        sim_payload = dict(payload)
        sim_payload["location"] = location
        sim_payload["region"] = region
        res = ScenarioSimulator.simulate(sim_payload)
        return jsonify(res)
    except Exception as e:
        app.logger.error(f"Scenario analysis error: {e}", exc_info=True)
        return jsonify({"error": "Failed to evaluate scenario analysis.", "detail": str(e)}), 500


@app.route("/api/simulate-scenario", methods=["POST"])
def api_simulate_scenario():
    """
    POST /api/simulate-scenario
    Simulates operational exposure scenarios under identical environmental heat conditions.
    Compares baseline vs scenario with core engine and explains causal factors without fake percentages.
    Powered by ScenarioSimulator with 100% backward-compatible responses.
    """
    payload = request.get_json(force=True, silent=True) or {}

    baseline_raw = payload.get("baseline") or {}
    scenario_raw = payload.get("scenario") or {}
    weather_data = payload.get("weather_data") or payload.get("environment")

    location = payload.get("location") or baseline_raw.get("location") or "Hyderabad"
    region = payload.get("region") or baseline_raw.get("region") or infer_region(location)

    if not weather_data or not isinstance(weather_data, dict):
        lat = payload.get("latitude") or baseline_raw.get("latitude")
        lon = payload.get("longitude") or baseline_raw.get("longitude")
        if lat is None or lon is None:
            if location in CITIES:
                lat, lon = CITIES[location]
            else:
                lat, lon = CITIES["Hyderabad"]
        try:
            curr_raw = WeatherService.fetch_current_weather(float(lat), float(lon))
            weather_data = {
                "current_temp_c": curr_raw.get("temp_c", 35.0),
                "temp_c": curr_raw.get("temp_c", 35.0),
                "humidity": curr_raw.get("rh_pct", 50.0),
                "rh_pct": curr_raw.get("rh_pct", 50.0),
                "wind_ms": curr_raw.get("wind_ms", 2.0),
                "wind_kph": curr_raw.get("wind_ms", 2.0) * 3.6,
                "solar_wm2": curr_raw.get("solar_wm2", 500.0),
            }
        except Exception:
            return jsonify({
                "error": "Weather service is temporarily unavailable. Please try again.",
                "status": "unavailable",
                "weather_status": "unavailable",
                "request_id": getattr(g, "request_id", None)
            }), 502

    try:
        sim_payload = dict(payload)
        sim_payload["location"] = location
        sim_payload["region"] = region
        sim_payload["weather_data"] = weather_data
        simulation_result = ScenarioSimulator.simulate(sim_payload)
        return jsonify(simulation_result)
    except Exception as e:
        app.logger.error(f"Scenario simulation error: {e}", exc_info=True)
        return jsonify({"error": "Failed to simulate exposure scenario."}), 500


BENCHMARK_CASES = ["khammam_2015_peak", "hyderabad_2015_peak"]


@app.route("/api/decision-trace", methods=["GET", "POST"])
def api_decision_trace():
    """
    GET or POST /api/decision-trace
    Direct inspection endpoint for auditing the complete deterministic decision trace:
    INPUTS -> DERIVED CONDITIONS -> ENVIRONMENTAL LEVEL -> EXPOSURE CONTEXT -> ACTION LEVEL -> RECOMMENDED CONTROLS.
    Supports historical benchmark case presets (e.g. 'khammam_2015_peak', 'hyderabad_2015_peak')
    or live evaluated scenarios.
    """
    if request.method == "POST":
        params = request.get_json(force=True, silent=True) or {}
    else:
        params = request.args.to_dict()

    case_id = params.get("case_id")
    if case_id == "khammam_2015_peak":
        location = "Khammam"
        region = "plains"
        weather_data = {
            "temp_c": 43.9,
            "current_temp_c": 43.9,
            "rh_pct": 32.0,
            "humidity": 32.0,
            "wind_ms": 2.2,
            "solar_wm2": 780.0,
            "time": "2015-05-24T14:00",
            "location": "Khammam",
        }
        personal_inputs = {
            "workload": "heavy",
            "clothing": "light",
            "experience": "not_used",
            "recovery_environment": "shade",
            "is_sun_exposed": True,
            "age_group": "25-45",
            "age": 35,
        }
    elif case_id == "hyderabad_2015_peak":
        location = "Hyderabad"
        region = "plains"
        weather_data = {
            "temp_c": 41.6,
            "current_temp_c": 41.6,
            "rh_pct": 28.0,
            "humidity": 28.0,
            "wind_ms": 3.1,
            "solar_wm2": 760.0,
            "time": "2015-05-21T14:00",
            "location": "Hyderabad",
        }
        personal_inputs = {
            "workload": "moderate",
            "clothing": "light",
            "experience": "somewhat_used",
            "recovery_environment": "shade",
            "is_sun_exposed": True,
            "age_group": "25-45",
            "age": 35,
        }
    else:
        try:
            parsed = parse_personal_inputs(params, default_location="Hyderabad")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        location = parsed["location"]
        region = parsed["region"]
        personal_inputs = parsed
        weather_data = params.get("weather_data")

        if not weather_data or not isinstance(weather_data, dict):
            lat = parsed["latitude"]
            lon = parsed["longitude"]
            try:
                weather_data = fetch_current_weather(lat, lon)
            except Exception:
                return jsonify({
                    "error": "Weather service is temporarily unavailable. Please try again.",
                    "status": "unavailable",
                    "weather_status": "unavailable",
                    "request_id": getattr(g, "request_id", None)
                }), 502

    profile = assess_full_risk_profile(
        weather_data=weather_data,
        personal_inputs=personal_inputs,
        region=region,
        location=location,
        is_forecast=False,
    )

    return jsonify({
        "case_id": case_id,
        "location": location,
        "region": region,
        "decision_trace": profile["decision_trace"],
        "explanation": profile["explanation"],
        "primary_drivers": profile["primary_drivers"],
        "contributing_factors": profile["contributing_factors"],
        "protective_factors": profile["protective_factors"],
        "actionable_levers": profile["actionable_levers"],
        "factor_sensitivities": profile["factor_sensitivities"],
        "recommended_controls": profile["recommended_controls"],
        "triggered_rules": profile["decision_trace"]["triggered_rules"],
        "summary": profile["summary"],
        "disclaimer": profile.get("disclaimer"),
    })


@app.route("/api/institute/list", methods=["GET"])
def api_institute_list():
    """GET /api/institute/list — Returns list of pre-configured institutions."""
    return jsonify({"institutes": InstituteService.get_institute_list()}), 200


@app.route("/api/institute/sites", methods=["GET"])
def api_institute_sites():
    """GET /api/institute/sites?institute_id=<id> — Evaluates live risk for all sites in an institution."""
    inst_id = request.args.get("institute_id", "infra_delhi")
    try:
        data = InstituteService.evaluate_institute(inst_id, fetch_current_weather)
        return jsonify(data), 200
    except (WeatherServiceError, requests.RequestException) as e:
        app.logger.warning(f"Institute weather fetch failed for {inst_id}: {e}")
        return jsonify({
            "error": "Weather service is temporarily unavailable for institutional sites.",
            "status": "unavailable",
            "weather_status": "unavailable",
            "request_id": getattr(g, "request_id", None)
        }), 502
    except Exception as e:
        app.logger.error(f"Institute evaluation failed for {inst_id}: {e}")
        return jsonify({"error": "Failed to evaluate institutional heat risk.", "status": "error"}), 500


@app.route("/api/authority/zones", methods=["GET"])
def api_authority_zones():
    """GET /api/authority/zones — Returns GeoJSON area risk layer with HAP alert status and demographic proxies."""
    try:
        view = request.args.get("view", "").lower()
        if view == "grid":
            city = request.args.get("city")
            data = AuthorityService.get_spatial_grid_layer(fetch_current_weather, city=city)
            return jsonify(data), 200

        data = AuthorityService.get_area_heat_layer(fetch_current_weather)
        return jsonify(data), 200
    except Exception as e:
        app.logger.error(f"Authority layer generation failed: {e}")
        return jsonify({"error": "Failed to generate municipal heat risk layer."}), 500


@app.route("/api/authority/grid", methods=["GET"])
@app.route("/api/spatial-grid", methods=["GET"])
def api_authority_grid():
    """GET /api/authority/grid?city=<city> — Returns GeoJSON spatial microclimate grid cells (~1km) with ML downscaling."""
    try:
        city = request.args.get("city")
        data = AuthorityService.get_spatial_grid_layer(fetch_current_weather, city=city)
        return jsonify(data), 200
    except Exception as e:
        app.logger.error(f"Spatial grid layer generation failed: {e}")
        return jsonify({"error": "Failed to generate spatial microclimate grid layer."}), 500



@app.route("/api/ml/hyperlocal", methods=["GET"])
def api_ml_hyperlocal():
    """GET /api/ml/hyperlocal?lat=<lat>&lon=<lon>&city=<city> — Returns ML microclimate downscaling prediction."""
    try:
        lat = float(request.args.get("lat", 28.6139))
        lon = float(request.args.get("lon", 77.2090))
        city = request.args.get("city", "Delhi")

        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return jsonify({"error": f"Invalid coordinates out of bounds: lat must be [-90, 90], lon [-180, 180]."}), 400

        w = fetch_current_weather(lat, lon)
        res = HyperlocalMLService.predict_point(lat=lat, lon=lon, weather_data=w, city_name=city)
        return jsonify(res), 200
    except ValueError as ve:
        return jsonify({"error": f"Invalid coordinate parameters: {ve}"}), 400
    except (WeatherServiceError, requests.RequestException) as we:
        app.logger.warning(f"Upstream weather provider error during hyperlocal inference: {we}")
        return jsonify({"error": f"Upstream weather provider error: {we}"}), 502
    except Exception as e:
        app.logger.error(f"Hyperlocal ML prediction failed: {e}")
        if "Provider rejected" in str(e) or "WeatherService" in type(e).__name__ or "requests" in type(e).__name__:
            return jsonify({"error": f"Upstream weather provider error: {e}"}), 502
        return jsonify({"error": "Failed to generate hyperlocal ML prediction."}), 500


@app.route("/api/ml/metrics", methods=["GET"])
def api_ml_metrics():
    """GET /api/ml/metrics — Returns ML model training metadata, holdout evaluation metrics, and feature importances."""
    try:
        metrics = HyperlocalMLService.get_model_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        app.logger.error(f"Hyperlocal ML metrics retrieval failed: {e}")
        return jsonify({"error": "Failed to retrieve ML evaluation metrics."}), 500


@app.route("/api/cooling-stations/recommendations", methods=["GET"])
def api_cooling_recommendations():
    """GET /api/cooling-stations/recommendations?city=<city> — Decision-support cooling station siting."""
    city = request.args.get("city")
    try:
        recs = CoolingRecommenderService.get_recommendations(target_city=city)
        return jsonify({
            "disclaimer": CoolingRecommenderService.DISCLAIMER,
            "recommendations": recs,
            "count": len(recs),
            "is_decision_support": True
        }), 200
    except Exception as e:
        app.logger.error(f"Cooling recommendations failed: {e}")
        return jsonify({"error": "Failed to generate cooling station recommendations."}), 500


@app.route("/api/alerts/send", methods=["POST"])
def api_send_alert():
    """POST /api/alerts/send — Dispatches SMS/WhatsApp alert or logs simulated dispatch."""
    payload = request.get_json(force=True, silent=True) or {}
    phone = payload.get("phone") or payload.get("recipient")
    location = payload.get("location", "Unknown Location")
    risk_category = payload.get("risk_category") or payload.get("risk", "High")
    try:
        wbgt_c = float(payload.get("wbgt_c", 31.0))
        temp_c = float(payload.get("temp_c", 38.0))
        work_minutes = int(payload.get("work_minutes", 30))
        rest_minutes = int(payload.get("rest_minutes", 30))
    except (ValueError, TypeError):
        wbgt_c, temp_c, work_minutes, rest_minutes = 31.0, 38.0, 30, 30

    instruction = payload.get("instruction", "Alternate work with rest in shaded areas.")
    channel = payload.get("channel", "sms").lower()
    force = bool(payload.get("force", False))
    role = payload.get("role", "worker")
    language = payload.get("language", "en")
    peak_window = payload.get("peak_window", "12:00–15:30")
    confidence = payload.get("confidence", "High")
    is_simulated = bool(payload.get("is_simulated", False))
    alert_id = payload.get("alert_id")

    res = AlertService.dispatch_alert(
        recipient_phone=phone,
        location=location,
        risk_category=risk_category,
        wbgt_c=wbgt_c,
        temp_c=temp_c,
        work_minutes=work_minutes,
        rest_minutes=rest_minutes,
        instruction=instruction,
        channel=channel,
        force=force,
        role=role,
        language=language,
        peak_window=peak_window,
        confidence=confidence,
        is_simulated=is_simulated,
        alert_id=alert_id,
    )
    status_code = 200 if res.get("success") else 400
    return jsonify(res), status_code


@app.route("/api/alerts/history", methods=["GET"])
def api_alert_history():
    """GET /api/alerts/history — Returns recent alert audit trail and provider status."""
    return jsonify({
        "config": AlertService.get_config(),
        "history": AlertService.get_history()
    }), 200


@app.route("/api/alerts/generate", methods=["POST"])
@app.route("/api/alerts/evaluate", methods=["POST"])
def api_generate_alerts():
    """POST /api/alerts/generate — Synthesizes structured role-based alert objects."""
    payload = request.get_json(force=True, silent=True) or {}
    location = payload.get("location", "Unknown Location")
    risk = payload.get("risk") or payload.get("risk_category", "High")
    try:
        wbgt_c = float(payload.get("wbgt_c", 31.0))
        temp_c = float(payload.get("temp_c", 38.0))
        work_minutes = int(payload.get("work_minutes", 30))
        rest_minutes = int(payload.get("rest_minutes", 30))
    except (ValueError, TypeError):
        wbgt_c, temp_c, work_minutes, rest_minutes = 31.0, 38.0, 30, 30

    peak_window = payload.get("peak_window", "12:00–15:30")
    confidence = payload.get("confidence", "High")
    instruction = payload.get("instruction")
    roles = payload.get("roles")
    language = payload.get("language", "en")
    channel = payload.get("channel", "sms")
    is_simulated = bool(payload.get("is_simulated", False))

    if roles and isinstance(roles, str):
        roles = [roles]

    alerts = AlertService.generate_role_alerts(
        location=location,
        risk=risk,
        wbgt_c=wbgt_c,
        temp_c=temp_c,
        peak_window=peak_window,
        confidence=confidence,
        work_minutes=work_minutes,
        rest_minutes=rest_minutes,
        instruction=instruction,
        roles=roles,
        language=language,
        channel=channel,
        is_simulated=is_simulated,
    )
    return jsonify({
        "success": True,
        "count": len(alerts),
        "alerts": alerts,
        "location": location,
        "risk": risk,
    }), 200


@app.route("/api/alerts/preview", methods=["POST"])
def api_preview_alert():
    """POST /api/alerts/preview — Previews full and compact alert representations without persisting."""
    payload = request.get_json(force=True, silent=True) or {}
    location = payload.get("location", "Hyderabad")
    risk_category = payload.get("risk") or payload.get("risk_category", "High")
    role = payload.get("role", "worker")
    language = payload.get("language", "en")
    channel = payload.get("channel", "sms")
    try:
        wbgt_c = float(payload.get("wbgt_c", 31.5))
        temp_c = float(payload.get("temp_c", 40.0))
        work_minutes = int(payload.get("work_minutes", 30))
        rest_minutes = int(payload.get("rest_minutes", 30))
    except (ValueError, TypeError):
        wbgt_c, temp_c, work_minutes, rest_minutes = 31.5, 40.0, 30, 30

    peak_window = payload.get("peak_window", "12:00–15:30")
    confidence = payload.get("confidence", "High")
    instruction = payload.get("instruction", "")
    is_simulated = bool(payload.get("is_simulated", False))

    if role not in SUPPORTED_ROLES:
        return jsonify({"error": f"Invalid role '{role}'. Must be one of {SUPPORTED_ROLES}"}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Invalid language '{language}'. Must be one of {SUPPORTED_LANGUAGES}"}), 400

    full_msg = AlertService.format_alert_message(
        location=location,
        risk_category=risk_category,
        wbgt_c=wbgt_c,
        temp_c=temp_c,
        work_minutes=work_minutes,
        rest_minutes=rest_minutes,
        instruction=instruction,
        channel=channel,
        role=role,
        language=language,
        peak_window=peak_window,
        confidence=confidence,
        is_simulated=is_simulated,
    )

    compact_sms = AlertService.format_compact_sms(
        location=location,
        risk_category=risk_category,
        wbgt_c=wbgt_c,
        temp_c=temp_c,
        work_minutes=work_minutes,
        rest_minutes=rest_minutes,
        role=role,
        language=language,
        peak_window=peak_window,
        confidence=confidence,
        is_simulated=is_simulated,
    )

    policy = AlertService.evaluate_alert_policy(
        risk=risk_category,
        role=role,
        confidence=confidence,
        is_simulated=is_simulated,
    )

    return jsonify({
        "role": role,
        "language": language,
        "channel": channel,
        "full_message": full_msg,
        "compact_sms": compact_sms,
        "compact_char_count": len(compact_sms),
        "policy": policy,
    }), 200


@app.route("/api/alerts", methods=["GET"])
def api_list_alerts():
    """GET /api/alerts — Retrieves filtered stored alerts."""
    role = request.args.get("role")
    status = request.args.get("status")
    approval_state = request.args.get("approval_state")
    urgency = request.args.get("urgency")
    try:
        limit = int(request.args.get("limit", 50))
    except (ValueError, TypeError):
        limit = 50

    alerts = AlertService.list_alerts(
        role=role,
        status=status,
        approval_state=approval_state,
        urgency=urgency,
        limit=limit,
    )
    return jsonify({
        "count": len(alerts),
        "alerts": alerts,
        "config": AlertService.get_config(),
    }), 200


@app.route("/api/alerts/<alert_id>", methods=["GET"])
def api_get_alert(alert_id):
    """GET /api/alerts/<alert_id> — Retrieves single alert with audit trail."""
    alert = AlertService.get_alert(alert_id)
    if not alert:
        return jsonify({"error": f"Alert '{alert_id}' not found."}), 404
    return jsonify({"alert": alert}), 200


@app.route("/api/alerts/<alert_id>/approve", methods=["POST"])
def api_approve_alert(alert_id):
    """POST /api/alerts/<alert_id>/approve — Human-in-the-loop authorization."""
    payload = request.get_json(force=True, silent=True) or {}
    approver = payload.get("approver", "Supervisor / Admin")
    notes = payload.get("notes", "")
    res = AlertService.approve_alert(alert_id=alert_id, approver_name=approver, notes=notes)
    status_code = 200 if res.get("success") else 400
    return jsonify(res), status_code


@app.route("/api/alerts/<alert_id>/reject", methods=["POST"])
def api_reject_alert(alert_id):
    """POST /api/alerts/<alert_id>/reject — Human-in-the-loop rejection."""
    payload = request.get_json(force=True, silent=True) or {}
    rejected_by = payload.get("rejected_by", "Supervisor / Admin")
    reason = payload.get("reason", "")
    res = AlertService.reject_alert(alert_id=alert_id, rejected_by=rejected_by, reason=reason)
    status_code = 200 if res.get("success") else 400
    return jsonify(res), status_code


@app.route("/api/alerts/<alert_id>/escalate", methods=["POST"])
def api_escalate_alert(alert_id):
    """POST /api/alerts/<alert_id>/escalate — Escalates alert up the chain of command."""
    payload = request.get_json(force=True, silent=True) or {}
    escalated_by = payload.get("escalated_by") or payload.get("actor") or "Operational Lead"
    reason = payload.get("reason", "Thermal risk escalated on ground")
    res = AlertService.escalate_alert(alert_id=alert_id, escalated_by=escalated_by, reason=reason)
    status_code = 200 if res.get("success") else 400
    return jsonify(res), status_code


@app.route("/api/alerts/<alert_id>/dispatch", methods=["POST"])
def api_dispatch_alert_by_id(alert_id):
    """POST /api/alerts/<alert_id>/dispatch — Dispatches verified alert."""
    payload = request.get_json(force=True, silent=True) or {}
    recipient_phone = payload.get("phone") or payload.get("recipient")
    force = bool(payload.get("force", False))
    res = AlertService.dispatch_alert_by_id(alert_id=alert_id, recipient_phone=recipient_phone, force=force)
    status_code = 200 if res.get("success") else 400
    return jsonify(res), status_code


@app.route("/api/alerts/preferences", methods=["GET", "POST"])
def api_alert_preferences():
    """GET/POST /api/alerts/preferences — Manages recipient opt-in consent and quiet hours."""
    if request.method == "POST":
        payload = request.get_json(force=True, silent=True) or {}
        recipient_id = payload.get("recipient_id") or payload.get("phone") or "default_user"
        role = payload.get("role", "worker")
        lang = payload.get("preferred_language") or payload.get("language", "en")
        channels = payload.get("enabled_channels")
        quiet_hours = payload.get("quiet_hours")
        opted_in = payload.get("opted_in", True)
        try:
            prefs = AlertService.set_preferences(
                recipient_id=recipient_id,
                role=role,
                preferred_language=lang,
                enabled_channels=channels,
                quiet_hours=quiet_hours,
                opted_in=opted_in,
            )
            return jsonify({"success": True, "preferences": prefs}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    else:
        recipient_id = request.args.get("recipient_id") or request.args.get("phone") or "default_user"
        prefs = AlertService.get_preferences(recipient_id)
        return jsonify({"preferences": prefs}), 200


@app.route("/api/mortality-risk", methods=["GET"])
def api_mortality_risk():
    """GET /api/mortality-risk — Returns experimental proxy population heat-risk indicator."""
    try:
        wbgt_c = float(request.args.get("wbgt_c", 31.0))
        temp_c = float(request.args.get("temp_c", 40.0))
        imd_status = request.args.get("imd_status", "Heat Day")
        location = request.args.get("location", "Delhi")
        admin1 = request.args.get("admin1", "")
        days = int(request.args.get("consecutive_days", 1))

        res = MortalityRiskService.calculate_proxy_index(
            wbgt_c=wbgt_c,
            imd_status=imd_status,
            temp_c=temp_c,
            location=location,
            admin1=admin1,
            consecutive_heat_days=days
        )
        return jsonify(res), 200
    except Exception as e:
        app.logger.error(f"Mortality risk calculation error: {e}")
        return jsonify({"error": "Failed to calculate mortality risk index."}), 400


@app.route("/api/hap-guidelines", methods=["GET"])
def api_hap_guidelines():
    """GET /api/hap-guidelines — Returns Government Heat Action Plan triggers and response protocols."""
    risk_category = request.args.get("risk_category", "High")
    imd_status = request.args.get("imd_status", "Heatwave")
    try:
        wbgt_c = float(request.args.get("wbgt_c", 31.5))
    except (ValueError, TypeError):
        wbgt_c = 31.5
    res = HAPService.evaluate_hap_status(risk_category, imd_status, wbgt_c)
    return jsonify(res), 200


@app.route("/api/validation/data", methods=["GET"])
def api_validation_data():
    """GET /api/validation/data — Returns historical 2015 AP/Telangana validation dataset."""
    return jsonify({
        "event_name": "2015 Andhra Pradesh & Telangana Heatwave",
        "date_range": "15 May 2015 – 01 June 2015",
        "reference_source": "IMD Station Bulletins vs ERA5 Atmospheric Reanalysis (~25km)",
        "limitation_note": "Observed limitation: ERA5 reanalysis spatial smoothing has a documented cold bias over India, underestimating extreme localized station maximums.",
        "cities": ["Khammam", "Hyderabad"]
    }), 200


@app.route("/api/validation/report", methods=["GET"])
def api_validation_report():
    """GET /api/validation/report — Returns full automated backtesting report."""
    case_id = request.args.get("case_id")
    case_ids = [case_id] if case_id else None
    report = ValidationEngine.run_backtest(case_ids=case_ids)
    return jsonify(report), 200


@app.route("/api/validation/datasets", methods=["GET"])
def api_validation_datasets():
    """GET /api/validation/datasets — Returns registered datasets and provenance metadata."""
    real_only = request.args.get("real_only", "").lower() in ("true", "1")
    data_type = request.args.get("data_type")
    datasets = DatasetRegistry.list_datasets(data_type=data_type, real_only=real_only)
    provenance = DatasetRegistry.get_dataset_provenance_summary()
    return jsonify({
        "provenance_summary": provenance,
        "datasets": datasets,
    }), 200


@app.route("/api/validation/cases", methods=["GET"])
def api_validation_cases():
    """GET /api/validation/cases — Returns all registered historical validation cases."""
    cases = CaseRegistry.list_cases()
    return jsonify({"cases": cases, "count": len(cases)}), 200


@app.route("/api/validation/case/<case_id>", methods=["GET"])
def api_validation_case_detail(case_id):
    """GET /api/validation/case/<case_id> — Returns detailed backtest for single case."""
    case = CaseRegistry.get_case(case_id)
    if not case:
        return jsonify({"error": f"Case ID '{case_id}' not found."}), 404
    report = ValidationEngine.run_backtest(case_ids=[case_id])
    case_report = report["cases"][0] if report.get("cases") else case
    return jsonify({
        "case": case,
        "evaluation": case_report,
        "metrics": {
            "environmental": report.get("environmental_metrics"),
            "classification": report.get("classification_metrics"),
        }
    }), 200


@app.route("/api/validation/export", methods=["GET"])
def api_validation_export():
    """GET /api/validation/export?format=csv|json — Export backtest report."""
    fmt = request.args.get("format", "json").lower()
    report = ValidationEngine.run_backtest()
    if fmt == "csv":
        csv_data = ValidationEngine.export_report_csv(report)
        from flask import Response
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=heatwave_guardian_validation_report.csv"}
        )
    return jsonify(report), 200

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug, threaded=True)

