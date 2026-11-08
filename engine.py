"""
HeatWave Guardian — Core Risk Intelligence Engine
=================================================
Scientific and engineering biometeorological decision-support engine.

Architecture:
    OBSERVE -> MODEL -> PERSONALIZE -> EXPLAIN -> PREDICT -> ACT -> VALIDATE

Core Concepts:
1. Environmental Heat Stress:
   - Atmospheric inputs: Dry-bulb temperature (Ta), Relative humidity (RH),
     10m Wind speed (v), Downward solar shortwave radiation (S).
   - Outdoor Wet-Bulb Globe Temperature (WBGT) via Liljegren et al. (2008)
     iterative black-globe energy balance and Stull (2011) wet-bulb regression.
   - India Meteorological Department (IMD) regional threshold criteria:
     Plains (40°C), Coastal (37°C), Hilly (30°C).
   - Fused Environmental Base Risk via categorical lookup table (no additive double counting).

2. Operational Exposure:
   - Metabolic workload categories (ISO 7243:2017 / ACGIH).
   - Clothing adjustment values / evaporative resistance (ISO 9920 / ACGIH).
   - Direct sunlight vs. shaded/indoor microclimate exposure.

3. Physiological Context & Acclimatization:
   - Acclimatization status (Acclimatized, Partially Acclimatized, Unacclimatized).
   - Recognized occupational ramp-up schedule (NIOSH 2016 / OSHA protocols).
   - Age-related thermoregulatory vulnerability context (children <18, older adults >=60).

4. Decision Support & Interventions:
   - Categorical action tiers: Low, Moderate, High, Very High, Extreme.
   - Hourly work/rest screening guidance.
   - Non-prescriptive, safety-focused occupational hydration protocols.
   - Transparent, traceable drivers and protective factors.
   - Qualitative data quality and confidence rating (High, Medium, Limited).

DISCLAIMER:
This system is an occupational decision-support and environmental heat-exposure
screening platform. It is NOT an individualized medical diagnostic tool, does NOT
predict clinical probability of illness, and does NOT prescribe universal fluid volumes.
"""

import math
import time
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Categorical Tiers
# ──────────────────────────────────────────────────────────────────────────────

RISK_LEVELS: List[str] = ["Low", "Moderate", "High", "Very High", "Extreme"]

RISK_COLORS: Dict[str, str] = {
    "Low":       "#2f9e44",
    "Moderate":  "#f2a900",
    "High":      "#e8590c",
    "Very High": "#c1121f",
    "Extreme":   "#6a040f",
}

CITIES: Dict[str, Tuple[float, float]] = {
    "Delhi":       (28.6139, 77.2090),
    "Hyderabad":   (17.3850, 78.4867),
    "Khammam":     (17.2473, 80.1514),
    "Chennai":     (13.0827, 80.2707),
    "Mumbai":      (19.0760, 72.8777),
    "Kolkata":     (22.5726, 88.3639),
    "Jaisalmer":   (26.9157, 70.9083),
    "Nagpur":      (21.1458, 79.0882),
    "Ahmedabad":   (23.0225, 72.5714),
    "Bhubaneswar": (20.2961, 85.8245),
    "Bengaluru":   (12.9716, 77.5946),
    "Jaipur":      (26.9124, 75.7873),
    "Lucknow":     (26.8467, 80.9462),
}

# Default region type for major preset cities (used in IMD classification)
CITY_REGIONS: Dict[str, str] = {
    "Delhi":       "plains",
    "Hyderabad":   "plains",
    "Khammam":     "plains",
    "Chennai":     "coastal",
    "Mumbai":      "coastal",
    "Kolkata":     "plains",
    "Jaisalmer":   "plains",
    "Nagpur":      "plains",
    "Ahmedabad":   "plains",
    "Bhubaneswar": "coastal",
    "Bengaluru":   "plains",
    "Jaipur":      "plains",
    "Lucknow":     "plains",
}

# IMD Regional qualifying maximum temperature thresholds
# Source: India Meteorological Department Heatwave Criteria
REGION_THRESHOLDS: Dict[str, float] = {
    "plains":  40.0,
    "coastal": 37.0,
    "hilly":   30.0,
}

# 20-Cell Environmental Risk Fusion Matrix: (WBGT band, IMD status) -> Risk Category
# Lookup table avoids double-counting since both inputs correlate with temperature.
FUSION_MATRIX: Dict[Tuple[str, str], str] = {
    ("Safe",            "Normal"):          "Low",
    ("Safe",            "Heat Day"):        "Low",
    ("Safe",            "Heatwave"):        "Moderate",
    ("Safe",            "Severe Heatwave"): "Moderate",
    ("Caution",         "Normal"):          "Low",
    ("Caution",         "Heat Day"):        "Moderate",
    ("Caution",         "Heatwave"):        "Moderate",
    ("Caution",         "Severe Heatwave"): "High",
    ("Extreme Caution", "Normal"):          "Moderate",
    ("Extreme Caution", "Heat Day"):        "Moderate",
    ("Extreme Caution", "Heatwave"):        "High",
    ("Extreme Caution", "Severe Heatwave"): "High",
    ("Danger",          "Normal"):          "Moderate",
    ("Danger",          "Heat Day"):        "High",
    ("Danger",          "Heatwave"):        "High",
    ("Danger",          "Severe Heatwave"): "Very High",
    ("Extreme Danger",  "Normal"):          "High",
    ("Extreme Danger",  "Heat Day"):        "Very High",
    ("Extreme Danger",  "Heatwave"):        "Very High",
    ("Extreme Danger",  "Severe Heatwave"): "Extreme",
}

# Baseline hourly work/rest splits (Screening guidance derived from ISO 7243:2017 principles)
# Note: These values serve as initial screening guidance and must be adapted to specific
# workplace metabolic demands and site conditions.
WORK_REST: Dict[str, Dict[str, Any]] = {
    "Low": {
        "work_minutes": 60,
        "rest_minutes": 0,
        "instruction": "Standard work schedule — environmental heat load within normal seasonal tolerance.",
    },
    "Moderate": {
        "work_minutes": 45,
        "rest_minutes": 15,
        "instruction": "Provide 15 minutes of rest per hour in shaded or cooled recovery zones.",
    },
    "High": {
        "work_minutes": 30,
        "rest_minutes": 30,
        "instruction": "Alternate 30 minutes of work with 30 minutes of shaded rest. Ensure continuous potable water access.",
    },
    "Very High": {
        "work_minutes": 15,
        "rest_minutes": 45,
        "instruction": "Curtail outdoor exertion to 15 minutes per hour. Rotate workers and prioritize shaded recovery.",
    },
    "Extreme": {
        "work_minutes": 0,
        "rest_minutes": 60,
        "instruction": "STOP: Continuous manual outdoor work not recommended. Suspend non-essential outdoor operations.",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# WBGT Estimation — Liljegren et al. (2008) & Stull (2011)
# ──────────────────────────────────────────────────────────────────────────────

def _stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """
    Psychrometric wet-bulb temperature via Stull (2011) empirical regression.
    Accurate to ±0.3 °C for RH 5–99 % and Ta -20 to 50 °C.

    THERMODYNAMIC BOUNDARY ENFORCEMENT:
    Wet-bulb temperature cannot exceed dry-bulb temperature (Tw <= Ta) for RH <= 100%.
    At complete saturation (RH >= 100%), wet bulb equals dry bulb (Tw = Ta).

    Reference:
        Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity and Air
        Temperature. Journal of Applied Meteorology and Climatology, 50(11), 2267–2269.
    """
    T = float(temp_c)
    rh_val = float(rh_pct)
    if math.isnan(T) or math.isinf(T):
        raise ValueError(f"Wet-bulb calculation received non-finite temperature: {temp_c}")
    if math.isnan(rh_val) or math.isinf(rh_val):
        raise ValueError(f"Wet-bulb calculation received non-finite humidity: {rh_pct}")

    # Physically clamp RH to empirical validity domain (5% to 99%)
    RH = max(min(rh_val, 99.0), 5.0)

    tw = (T * math.atan(0.151977 * (RH + 8.313659) ** 0.5)
          + math.atan(T + RH)
          - math.atan(RH - 1.676331)
          + 0.00391838 * (RH ** 1.5) * math.atan(0.023101 * RH)
          - 4.686035)

    # Physical bounding: wet-bulb cannot exceed dry-bulb under atmospheric evaporation
    tw = min(tw, T)
    # Physical lower bound: wet-bulb depression never exceeds 40°C in terrestrial atmosphere
    tw = max(tw, T - 40.0)
    return tw


def _globe_temperature(temp_c: float, wind_ms: float, solar_wm2: float) -> float:
    """
    Black globe temperature via iterative energy balance (Liljegren et al. 2008).
    Models a standard 150 mm matte-black globe using Newton–Raphson iteration
    to solve radiation absorption + thermal radiation emission + forced convection equilibrium.

    Energy balance:
        α · S / 4  =  ε · σ · (Tg⁴ − Ta⁴)  +  h_c · (Tg − Ta)

    where:
        α = 0.95 (absorptivity of matte black globe)
        ε = 0.95 (surface emissivity)
        σ = 5.67e-8 W/(m²·K⁴) (Stefan-Boltzmann constant)
        D = 0.15 m (standard globe diameter)
        h_c = forced convection heat transfer coefficient (Nu · k_air / D)

    Reference:
        Liljegren, J. C., et al. (2008). Modeling the Wet Bulb Globe Temperature
        Using Standard Meteorological Measurements. J. Occup. Environ. Hyg., 5(10), 645–655.
    """
    t_val = float(temp_c)
    w_val = float(wind_ms)
    s_val = float(solar_wm2)
    if any(math.isnan(v) or math.isinf(v) for v in (t_val, w_val, s_val)):
        raise ValueError(f"Black globe temperature received non-finite input: temp={t_val}, wind={w_val}, solar={s_val}")

    if s_val <= 0:
        return t_val  # In absence of solar radiation, Tg ≈ Ta

    STEFAN_BOLTZMANN = 5.67e-8
    EMISSIVITY = 0.95
    ABSORPTIVITY = 0.95
    DIAMETER = 0.15  # metres

    wind = max(w_val, 0.1)  # floor at 0.1 m/s to prevent division by zero

    # Air properties at standard warm boundary layer (~35 °C)
    nu_air = 1.655e-5   # kinematic viscosity (m²/s)
    k_air  = 0.0271     # thermal conductivity (W/(m·K))
    Pr     = 0.710      # Prandtl number for dry air

    # Forced convection around a sphere: Nu = 2 + 0.6 · Re^0.5 · Pr^0.33
    reynolds = wind * DIAMETER / nu_air
    nusselt = 2.0 + 0.6 * (reynolds ** 0.5) * (Pr ** 0.33)
    h_c = nusselt * k_air / DIAMETER

    ta_k = t_val + 273.15
    tg = t_val + 10.0  # Initial iteration guess

    # Newton-Raphson convergence with divergence dampening
    for _ in range(100):
        tg_k = max(tg + 273.15, 100.0)  # Absolute temperature guard
        f = (EMISSIVITY * STEFAN_BOLTZMANN * (tg_k ** 4 - ta_k ** 4)
             + h_c * (tg - t_val)
             - (ABSORPTIVITY * s_val / 4.0))
        f_prime = 4.0 * EMISSIVITY * STEFAN_BOLTZMANN * (tg_k ** 3) + h_c
        if abs(f_prime) < 1e-12:
            break
        delta = f / f_prime
        tg -= delta
        if abs(delta) < 0.001:
            break

    # Physical bounding: with positive solar radiation, black globe cannot be cooler
    # than ambient air, nor exceed theoretical extreme terrestrial equilibrium (Ta + 45°C)
    tg = max(t_val, min(t_val + 45.0, tg))
    return tg


@lru_cache(maxsize=32768)
def _estimate_wbgt_cached(t_round: float, rh_round: float, w_round: float, s_round: float) -> float:
    t_nwb = _stull_wet_bulb(t_round, rh_round)
    t_g = _globe_temperature(t_round, w_round, s_round)
    wbgt = 0.7 * t_nwb + 0.2 * t_g + 0.1 * t_round
    return round(wbgt, 1)


def clear_wbgt_cache() -> None:
    """Clear in-memory LRU cache for Liljegren WBGT calculations."""
    _estimate_wbgt_cached.cache_clear()


def estimate_wbgt(temp_c: float, rh_pct: float, wind_ms: float, solar_wm2: float) -> float:
    """
    Estimate outdoor Wet-Bulb Globe Temperature (°C).
    Liljegren et al. (2008) formulation:
        WBGT = 0.7 × T_nwb  +  0.2 × T_g  +  0.1 × T_a

    Inputs:
        temp_c: Dry-bulb ambient air temperature (°C)
        rh_pct: Relative humidity (%)
        wind_ms: 10-metre wind speed (m/s)
        solar_wm2: Downward shortwave solar radiation (W/m²)
    """
    try:
        t_f = float(temp_c)
        rh_f = float(rh_pct)
        w_f = float(wind_ms)
        s_f = float(solar_wm2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid non-numeric input to estimate_wbgt: {e}")

    if any(math.isnan(v) or math.isinf(v) for v in (t_f, rh_f, w_f, s_f)):
        raise ValueError(f"Non-finite input encountered in estimate_wbgt: temp={t_f}, rh={rh_f}, wind={w_f}, solar={s_f}")

    t_clamped = max(-40.0, min(65.0, t_f))
    rh_clamped = max(1.0, min(100.0, rh_f))
    w_clamped = max(0.0, min(50.0, w_f))
    s_clamped = max(0.0, min(1500.0, s_f))

    t_r = round(t_clamped, 2)
    rh_r = round(rh_clamped, 1)
    w_r = round(w_clamped, 2)
    s_r = round(s_clamped, 1)
    return _estimate_wbgt_cached(t_r, rh_r, w_r, s_r)


def calculate_wbgt(
    temp_c: float,
    humidity: float,
    solar_wm2: float = 0.0,
    wind_ms: float = 2.0,
    is_sun_exposed: bool = True,
    **kwargs: Any,
) -> float:
    """
    Calculate outdoor Wet-Bulb Globe Temperature (°C) with solar shielding context.
    Wraps estimate_wbgt for flexible naming and parameter conventions.
    """
    eff_solar = max(float(solar_wm2), 0.0) if is_sun_exposed else 0.0
    return estimate_wbgt(temp_c=float(temp_c), rh_pct=float(humidity), wind_ms=float(wind_ms), solar_wm2=eff_solar)


# ──────────────────────────────────────────────────────────────────────────────
# Thermal Stress Classification
# ──────────────────────────────────────────────────────────────────────────────

WBGT_TO_THERMAL_STRESS_LEVEL: Dict[str, str] = {
    "Safe":            "Low",
    "Caution":         "Moderate",
    "Extreme Caution": "High",
    "Danger":          "Very High",
    "Extreme Danger":  "Extreme",
}

THERMAL_STRESS_DESCRIPTIONS: Dict[str, str] = {
    "Safe":            "Minimal environmental heat stress; safe for standard outdoor activities.",
    "Caution":         "Moderate environmental heat stress; fatigue possible with prolonged physical exertion.",
    "Extreme Caution": "High environmental heat stress; increased risk of heat cramps and exhaustion.",
    "Danger":          "Very high environmental heat stress; heat exhaustion likely, protective action required.",
    "Extreme Danger":  "Extreme environmental heat stress; critical heat stroke danger, immediate mitigation required.",
}


def classify_wbgt_band(wbgt_c: float) -> str:
    """
    Classify outdoor WBGT into 5 standard physiological stress bands.
    Thresholds reflect established environmental heat-stress screening criteria
    (US Navy / ACSM / OSHA screening bands).
    """
    val = float(wbgt_c)
    if val < 27.0:
        return "Safe"
    elif val <= 29.3:
        return "Caution"
    elif val <= 31.0:
        return "Extreme Caution"
    elif val <= 32.1:
        return "Danger"
    else:
        return "Extreme Danger"


def classify_thermal_stress(wbgt_c: float) -> Dict[str, Any]:
    """
    Thermal Stress Index classification derived directly from Liljegren WBGT.
    Returns level, band, description, severity score (1-5), and risk color.
    """
    wbgt_val = round(float(wbgt_c), 1)
    band = classify_wbgt_band(wbgt_val)
    level = WBGT_TO_THERMAL_STRESS_LEVEL[band]
    description = THERMAL_STRESS_DESCRIPTIONS[band]
    severity = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4, "Extreme": 5}[level]
    color = RISK_COLORS[level]

    return {
        "wbgt_c": wbgt_val,
        "thermal_stress_level": level,
        "wbgt_band": band,
        "thermal_stress_description": description,
        "severity": severity,
        "risk_color": color,
    }


def calculate_thermal_stress_index(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    solar_wm2: float
) -> Dict[str, Any]:
    """
    Authoritative calculation for Thermal Stress Index.
    Estimates outdoor WBGT (Liljegren et al. 2008) and attaches environmental conditions.
    """
    wbgt = estimate_wbgt(temp_c, rh_pct, wind_ms, solar_wm2)
    res = classify_thermal_stress(wbgt)
    res["environmental_conditions"] = {
        "temp_c": round(float(temp_c), 1),
        "rh_pct": round(float(rh_pct), 1),
        "wind_ms": round(float(wind_ms), 1),
        "solar_wm2": round(max(float(solar_wm2), 0.0), 1),
    }
    return res


# ──────────────────────────────────────────────────────────────────────────────
# IMD Regional Heatwave Classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_imd(temp_c: float, region: Optional[str] = None, region_type: Optional[str] = None) -> str:
    """
    Evaluates temperature against regional qualifying threshold departures (plains: 40°C, coastal: 37°C, hilly: 30°C).

    SCIENTIFIC METHODOLOGY & REGULATORY CONTEXT:
    - Official IMD Heatwave Warning Criteria (India Meteorological Department / MoES Standard Operating Procedure):
        1. Applies exclusively to daily maximum temperature (T_max) observed at an official meteorological observatory.
        2. Qualifying threshold: Plains >= 40.0°C, Coastal >= 37.0°C, Hilly >= 30.0°C.
        3. Based on departure from climatological normal:
           - Heat Wave: Departure is +4.5°C to +6.4°C
           - Severe Heat Wave: Departure is > +6.4°C
        4. Based on absolute maximum (Plains):
           - Heat Wave: T_max >= 45.0°C
           - Severe Heat Wave: T_max >= 47.0°C
        5. Persistence: Declared only when qualifying conditions persist at >= 2 stations in a subdivision for >= 2 consecutive days.

    OPERATIONAL SCREENING DEFINITION:
    When applied to instantaneous or model-downscaled temperatures, this function acts as an immediate
    regional thermal departure screening index. It does NOT constitute an official IMD daily subdivision declaration.
    """
    effective_region = region_type or region or "plains"
    reg_clean = str(effective_region).lower().strip()
    threshold = REGION_THRESHOLDS.get(reg_clean, 40.0)
    departure = float(temp_c) - threshold

    if departure < 0.0:
        return "Normal"
    elif departure < 2.0:
        return "Heat Day"
    elif departure < 4.0:
        return "Heatwave"
    else:
        return "Severe Heatwave"


def classify_official_imd(
    daily_max_temp_c: float,
    normal_temp_c: Optional[float] = None,
    region: Optional[str] = "plains",
    consecutive_days: int = 2,
    stations_meeting_criteria: int = 2,
    is_daily_max: bool = True,
) -> Dict[str, Any]:
    """
    Authoritative India Meteorological Department (IMD) Heat Wave Classification.
    Standard Operating Procedure (SOP) criteria (MoES / IMD Heat Wave Guidelines).

    RULES:
    1. Scope: Applies strictly to verified daily maximum temperature (T_max) at synoptic observatories.
       Instantaneous or model downscaled temperatures CANNOT declare an official IMD Heat Wave.
    2. Qualifying Thresholds:
       - Plains: T_max >= 40.0 °C
       - Coastal: T_max >= 37.0 °C
       - Hilly: T_max >= 30.0 °C
       If T_max does not meet the qualifying threshold, the condition is categorized as 'Normal'.
    3. Departure from Normal Criteria (when normal_temp_c is provided):
       - Heat Wave: Departure is +4.5 °C to +6.4 °C
       - Severe Heat Wave: Departure is >= +6.5 °C
    4. Absolute Maximum Temperature Criteria (Plains only, regardless of normal):
       - Heat Wave: T_max >= 45.0 °C
       - Severe Heat Wave: T_max >= 47.0 °C
    5. Persistence & Spatial Extent (for official declaration):
       - Declared only when qualifying conditions persist for >= 2 consecutive days,
         across >= 2 stations in a meteorological sub-division.
    """
    reg_clean = str(region or "plains").lower().strip()
    threshold = REGION_THRESHOLDS.get(reg_clean, 40.0)
    t_max = float(daily_max_temp_c)

    qualifying_met = t_max >= threshold
    departure = (t_max - float(normal_temp_c)) if normal_temp_c is not None else (t_max - threshold)
    has_normal = normal_temp_c is not None

    met_heatwave = False
    met_severe = False

    if qualifying_met:
        # Check departure criteria if climatological normal is provided
        if has_normal:
            if departure >= 6.5:
                met_severe = True
            elif departure >= 4.5:
                met_heatwave = True

        # Check absolute criteria for plains
        if reg_clean == "plains":
            if t_max >= 47.0:
                met_severe = True
            elif t_max >= 45.0:
                met_heatwave = True

        # For coastal and hilly without explicit normal: use departure from qualifying threshold
        if not has_normal and not met_heatwave and not met_severe:
            if departure >= 6.5:
                met_severe = True
            elif departure >= 4.5:
                met_heatwave = True

    persistence_met = int(consecutive_days) >= 2
    spatial_met = int(stations_meeting_criteria) >= 2
    temporal_valid = bool(is_daily_max)

    if not qualifying_met:
        status = "Normal"
        declaration_valid = False
    elif met_severe:
        if temporal_valid and persistence_met and spatial_met:
            status = "Severe Heatwave"
            declaration_valid = True
        elif not temporal_valid:
            status = "Modelled Condition: Severe Thermal Departure"
            declaration_valid = False
        else:
            status = "Pending Persistence Verification (Severe Threshold Met)"
            declaration_valid = False
    elif met_heatwave:
        if temporal_valid and persistence_met and spatial_met:
            status = "Heatwave"
            declaration_valid = True
        elif not temporal_valid:
            status = "Modelled Condition: Moderate Thermal Departure"
            declaration_valid = False
        else:
            status = "Pending Persistence Verification (Heatwave Threshold Met)"
            declaration_valid = False
    else:
        status = "Normal"
        declaration_valid = False

    return {
        "official_imd_status": status,
        "is_official_declaration_valid": declaration_valid,
        "qualifying_threshold_met": qualifying_met,
        "qualifying_threshold_c": threshold,
        "daily_max_temp_c": t_max,
        "normal_temp_c": normal_temp_c,
        "departure_c": round(departure, 1),
        "met_heatwave_threshold": met_heatwave,
        "met_severe_heatwave_threshold": met_severe,
        "persistence_criterion_met": persistence_met,
        "spatial_criterion_met": spatial_met,
        "is_daily_max": temporal_valid,
        "region": reg_clean,
        "classification_type": "OFFICIAL_IMD_DECLARATION" if declaration_valid else "MODELLED_ENVIRONMENTAL_HEAT_CONDITION",
        "regulatory_provenance": "IMD MoES Standard Operating Procedure (Heat Wave SOP)",
    }


evaluate_official_imd_criteria = classify_official_imd


def fuse_risk(wbgt_band: str, imd_status: str) -> str:
    """
    Fuse WBGT band and IMD regional status into categorical environmental risk.
    Uses a non-additive 20-cell lookup matrix to avoid double-counting.
    """
    key = (wbgt_band, imd_status)
    if key not in FUSION_MATRIX:
        return "Moderate"
    return FUSION_MATRIX[key]


# ──────────────────────────────────────────────────────────────────────────────
# Geographic Region Inference
# ──────────────────────────────────────────────────────────────────────────────

HILLY_KEYWORDS = {
    "himachal", "uttarakhand", "jammu", "kashmir", "ladakh", "sikkim",
    "arunachal", "meghalaya", "nagaland", "manipur", "mizoram", "tripura",
    "shimla", "manali", "dehradun", "mussoorie", "nainital", "darjeeling",
    "ooty", "munnar", "leh", "srinagar", "kullu", "dharamshala", "shillong",
    "gangtok", "almora", "rishikesh", "mount abu", "coonoor", "kodaikanal",
    "solan", "mandi", "chamba", "tehri", "pauri", "chamoli", "rudraprayag"
}

COASTAL_KEYWORDS = {
    "goa", "kerala", "puducherry", "lakshadweep", "andaman", "mumbai",
    "chennai", "kochi", "cochin", "thiruvananthapuram", "trivandrum",
    "mangalore", "mangaluru", "visakhapatnam", "vizag", "puri", "bhubaneswar",
    "surat", "panaji", "kozhikode", "alappuzha", "kollam", "tuticorin",
    "thoothukudi", "daman", "diu", "karwar", "ratnagiri", "kakinada",
    "machilipatnam", "nellore", "cuddalore", "nagapattinam", "kanyakumari",
    "udupi", "porbandar", "bhavnagar", "dwarka", "jamnagar", "paradip",
    "balasore", "digha", "haldia"
}


def infer_region(location: str, admin1: str = "") -> str:
    """
    Infer IMD geographic region classification ('plains', 'coastal', 'hilly')
    from location name and administrative subdivision.
    """
    loc_clean = (location or "").strip()
    for city, reg in CITY_REGIONS.items():
        if city.lower() in loc_clean.lower():
            return reg

    text = f"{loc_clean} {admin1}".lower()
    for kw in HILLY_KEYWORDS:
        if kw in text:
            return "hilly"
    for kw in COASTAL_KEYWORDS:
        if kw in text:
            return "coastal"
    return "plains"


# ──────────────────────────────────────────────────────────────────────────────
# Acclimatization as a First-Class Factor (NIOSH / OSHA Guidance)
# ──────────────────────────────────────────────────────────────────────────────

ACCLIMATIZATION_STATUS_ACCLIMATIZED = "acclimatized"
ACCLIMATIZATION_STATUS_PARTIAL = "partially_acclimatized"
ACCLIMATIZATION_STATUS_UNACCLIMATIZED = "unacclimatized"

ACCLIMATIZATION_STATUSES: Dict[str, Dict[str, Any]] = {
    "acclimatized": {
        "key": "acclimatized",
        "label": "Acclimatized (Regular heat exposure)",
        "description": "Worker has regular hot weather exposure (7–14 days). Thermoregulatory mechanisms (earlier sweating, lower sodium loss, cardiac stability) are active.",
        "risk_modifier": "Standard baseline exposure tolerance",
        "protocol_note": "Standard work/rest cycles apply. Maintain monitoring during sudden microclimate temperature spikes.",
    },
    "partially_acclimatized": {
        "key": "partially_acclimatized",
        "label": "Partially Acclimatized / Returning Worker",
        "description": "Worker occasionally works in heat or is returning after 3–9 days absence. Physiological heat adaptations have partially decayed.",
        "risk_modifier": "Caution: +1 escalation under heavy exertion or high environmental load",
        "protocol_note": "Follow returning worker exposure ramp-up: Day 1: 50%, Day 2: 60%, Day 3: 80%, Day 4: 100% full shift heat exposure.",
    },
    "unacclimatized": {
        "key": "unacclimatized",
        "label": "Unacclimatized / New Worker",
        "description": "New to hot environment or returning after extended absence (>9 days). Delayed sweating response and unexpanded plasma volume pose peak heat illness risk.",
        "risk_modifier": "High caution: +1 to +2 escalation depending on workload and protective clothing",
        "protocol_note": "Enforce NIOSH progressive exposure schedule: Day 1: 20%, Day 2: 40%, Day 3: 60%, Day 4: 80%, Day 5: 100% full shift heat exposure.",
    },
}

ACCLIMATIZATION_ALIASES: Dict[str, str] = {
    "fully": "acclimatized",
    "acclimatized": "acclimatized",
    "very_used": "acclimatized",
    "regularly": "acclimatized",
    "yes": "acclimatized",
    "partially": "partially_acclimatized",
    "somewhat_used": "partially_acclimatized",
    "returning": "partially_acclimatized",
    "occasional": "partially_acclimatized",
    "sometimes": "partially_acclimatized",
    "not": "unacclimatized",
    "not_used": "unacclimatized",
    "unacclimatized": "unacclimatized",
    "new": "unacclimatized",
    "recently": "unacclimatized",
    "no": "unacclimatized",
}

# Recognized progressive exposure schedules (NIOSH 2016 / OSHA)
ACCLIMATIZATION_SCHEDULES: Dict[str, Dict[int, Dict[str, Any]]] = {
    "new_worker": {
        1: {"max_heat_exposure_pct": 20, "recommendation": "Day 1: Maximum 20% of normal shift duration in heat. Remaining hours in shaded/cooled areas."},
        2: {"max_heat_exposure_pct": 40, "recommendation": "Day 2: Maximum 40% of normal shift duration in heat. Monitor buddy worker for early fatigue."},
        3: {"max_heat_exposure_pct": 60, "recommendation": "Day 3: Maximum 60% of normal shift duration in heat with enforced hourly hydration."},
        4: {"max_heat_exposure_pct": 80, "recommendation": "Day 4: Maximum 80% of normal shift duration in heat."},
        5: {"max_heat_exposure_pct": 100, "recommendation": "Day 5: Full work duration permitted subject to environmental work/rest cycles."},
    },
    "returning_worker": {
        1: {"max_heat_exposure_pct": 50, "recommendation": "Day 1: Maximum 50% shift heat exposure for workers returning after 3–9 days absence."},
        2: {"max_heat_exposure_pct": 60, "recommendation": "Day 2: Maximum 60% shift heat exposure."},
        3: {"max_heat_exposure_pct": 80, "recommendation": "Day 3: Maximum 80% shift heat exposure."},
        4: {"max_heat_exposure_pct": 100, "recommendation": "Day 4: Full operational shift heat exposure permitted."},
    }
}


def evaluate_acclimatization_schedule(
    status: str,
    day_in_heat: Optional[int] = None,
    is_returning_worker: bool = False
) -> Dict[str, Any]:
    """
    Occupational decision support for worker heat acclimatization (NIOSH 2016).
    Provides structured exposure pacing for new or returning outdoor workers.

    Disclaimer:
        Occupational decision-support guideline. Does NOT clinically measure
        physiological adaptation.
    """
    clean_status = str(status or "").strip().lower()
    canonical = ACCLIMATIZATION_ALIASES.get(clean_status, "acclimatized")
    info = ACCLIMATIZATION_STATUSES[canonical]

    schedule_type = "returning_worker" if (is_returning_worker or canonical == "partially_acclimatized") else "new_worker"
    schedule_table = ACCLIMATIZATION_SCHEDULES[schedule_type]

    schedule_detail = None
    if day_in_heat is not None:
        try:
            day_num = int(day_in_heat)
            clamped_day = max(1, min(day_num, max(schedule_table.keys())))
            schedule_detail = schedule_table.get(clamped_day)
        except (ValueError, TypeError):
            schedule_detail = None

    if schedule_detail:
        rec = schedule_detail["recommendation"]
        max_pct = schedule_detail["max_heat_exposure_pct"]
    elif canonical == "acclimatized":
        rec = info["protocol_note"]
        max_pct = 100
    elif canonical == "partially_acclimatized":
        rec = schedule_table[1]["recommendation"]
        max_pct = 50
    else:  # unacclimatized
        rec = schedule_table[1]["recommendation"]
        max_pct = 20

    return {
        "status": canonical,
        "label": info["label"],
        "description": info["description"],
        "risk_modifier": info["risk_modifier"],
        "protocol_note": info["protocol_note"],
        "day_in_heat": day_in_heat,
        "schedule_recommendation": rec,
        "max_exposure_pct": max_pct,
        "is_decision_support": True,
        "disclaimer": "Occupational decision support based on NIOSH (2016) / OSHA guidance. Not individualized medical advice."
    }


# ──────────────────────────────────────────────────────────────────────────────
# Operational Exposure & Physiological Vulnerability Assessment
# ──────────────────────────────────────────────────────────────────────────────

# ISO 7243:2017 & ACGIH Metabolic rate estimates (Watts)
METABOLIC_RATE_WATTS: Dict[str, int] = {
    "light": 150,        # Sitting, light arm work, driving, inspection
    "moderate": 240,     # Sustained walking, lifting, machinery operation
    "heavy": 360,        # Shoveling, masonry, manual cargo loading
    "very_heavy": 480,   # Maximum physical exertion, intense digging, rapid climb
}

# ISO 9920 / ACGIH Clothing Adjustment Values (CAVs in °C WBGT)
CLOTHING_ADJUSTMENT_VALUES: Dict[str, float] = {
    "light": 0.0,        # Light cotton work clothing / standard shorts and shirt
    "heavy": 2.0,        # Heavy fabric / double-layer overalls / denim
    "ppe": 6.0,          # Vapor-barrier impermeable chemical/protective gear
}


def evaluate_operational_exposure(
    workload: str = "moderate",
    clothing: str = "light",
    acclimatization: str = "fully",
    is_sun_exposed: bool = True,
    duration_hours: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluate operational exposure context (how work activity and protective gear modify heat load).
    Categorizes operational exposure into 'Low', 'Moderate', 'High', or 'Very High'.
    """
    wl_clean = str(workload or "moderate").strip().lower()
    cl_clean = str(clothing or "light").strip().lower()
    acc_clean = str(acclimatization or "fully").strip().lower()
    canonical_acc = ACCLIMATIZATION_ALIASES.get(acc_clean, "acclimatized")

    # Resolve metabolic load
    if any(k in wl_clean for k in ("very_heavy", "very heavy")):
        work_key = "very_heavy"
    elif "heavy" in wl_clean:
        work_key = "heavy"
    elif "light" in wl_clean:
        work_key = "light"
    else:
        work_key = "moderate"

    # Resolve clothing
    if any(k in cl_clean for k in ("ppe", "protective")):
        cloth_key = "ppe"
    elif "heavy" in cl_clean:
        cloth_key = "heavy"
    else:
        cloth_key = "light"

    metabolic_w = METABOLIC_RATE_WATTS[work_key]
    cav_c = CLOTHING_ADJUSTMENT_VALUES[cloth_key]

    drivers: List[str] = []
    protective: List[str] = []

    if work_key in ("heavy", "very_heavy"):
        drivers.append(f"{work_key.replace('_', ' ').capitalize()} physical labor generates substantial internal metabolic heat (~{metabolic_w} W).")
    elif work_key == "light":
        protective.append(f"Light physical workload generates low internal metabolic heat (~{metabolic_w} W).")

    if cloth_key == "ppe":
        drivers.append(f"Protective gear / PPE adds high evaporative resistance (effective clothing adjustment +{cav_c:.1f} °C).")
    elif cloth_key == "heavy":
        drivers.append(f"Heavy work clothing traps convective heat (effective clothing adjustment +{cav_c:.1f} °C).")
    else:
        protective.append("Light breathable clothing facilitates unhindered sweat evaporation.")

    if canonical_acc == "unacclimatized":
        drivers.append("Unacclimatized status: delayed onset of sweating and higher cardiovascular heart-rate strain.")
    elif canonical_acc == "partially_acclimatized":
        drivers.append("Partially acclimatized status: incomplete physiological heat adaptation.")
    else:
        protective.append("Acclimatized worker: optimized sweating efficiency and preserved plasma volume.")

    if not is_sun_exposed:
        protective.append("Shaded or indoor work eliminates direct solar radiation heat burden.")

    # Determine operational exposure tier
    if cloth_key == "ppe" and work_key in ("moderate", "heavy", "very_heavy"):
        op_tier = "Very High"
    elif (cloth_key == "ppe") or (cloth_key == "heavy" and work_key in ("heavy", "very_heavy")) or (work_key == "very_heavy" and canonical_acc == "unacclimatized"):
        op_tier = "High"
    elif (cloth_key == "heavy") or (work_key in ("heavy", "very_heavy")) or (canonical_acc != "acclimatized" and work_key == "moderate"):
        op_tier = "Moderate"
    elif work_key == "moderate":
        op_tier = "Moderate"
    else:
        op_tier = "Low"

    return {
        "operational_exposure_level": op_tier,
        "metabolic_rate_watts": metabolic_w,
        "clothing_adjustment_c": cav_c,
        "workload_category": work_key,
        "clothing_category": cloth_key,
        "acclimatization_status": canonical_acc,
        "is_sun_exposed": is_sun_exposed,
        "duration_hours": duration_hours,
        "operational_drivers": drivers,
        "protective_factors": protective,
        "is_decision_support": True,
    }


def assess_vulnerability(
    age: int = 30,
    acclimatization: str = "fully",
    workload: str = "moderate",
    clothing: str = "light",
    age_group: Optional[str] = None
) -> str:
    """
    Determine physiological vulnerability level ('Low', 'Moderate', 'High').
    Preserved for strict backward compatibility with existing tests and services.

    Scientific Basis:
        - Age extremes (<18, >=60): Children have higher body surface-area-to-mass ratio
          and lower sweat rates; older adults experience reduced cardiac output reserve,
          delayed cutaneous vasodilation, and blunted thirst sensation.
        - High metabolic workload (heavy, very heavy): Multiplies internal thermal burden.
        - Protective clothing / PPE: Prevents sweat evaporation, trapping moisture and heat.
        - Lack of acclimatization: Limits sweat efficiency and cardiovascular endurance.
    """
    if age_group is not None:
        ag_clean = str(age_group).strip().lower()
        vulnerable_age = ag_clean in ("under 18", "under_18", "<18", "60+", "60_plus", ">60", "60 plus")
    else:
        vulnerable_age = (int(age) < 18 or int(age) >= 60)

    workload_clean = str(workload or "moderate").strip().lower()
    heavy_work = workload_clean in ("heavy", "very_heavy", "very heavy")

    acc_clean = str(acclimatization or "fully").strip().lower()
    not_acclimatized = acc_clean in ("not", "not_used", "not used", "recently", "unacclimatized", "no")
    not_fully = acc_clean not in ("fully", "very_used", "very used", "regularly", "yes", "acclimatized")

    clothing_clean = str(clothing or "light").strip().lower()
    is_ppe = clothing_clean in ("ppe", "protective", "protective gear / ppe")
    is_heavy_clothing = clothing_clean in ("heavy", "heavy work clothing", "heavy_work")

    # High vulnerability (+2 escalation steps):
    # 1. PPE traps metabolic heat & prevents sweat evaporation:
    #    triggers High vulnerability with ANY workload > light, or unacclimatized, or vulnerable age
    if is_ppe and (workload_clean != "light" or not_fully or vulnerable_age):
        return "High"

    # 2. Heavy work clothing adds thermal insulation:
    #    triggers High vulnerability when combined with heavy work, unacclimatized, or vulnerable age with partial experience
    if is_heavy_clothing and (heavy_work or not_acclimatized or (vulnerable_age and not_fully)):
        return "High"

    # 3. Standard high vulnerability criteria:
    #    (not used to heat AND heavy work) OR (vulnerable age AND not fully used to heat)
    if (not_acclimatized and heavy_work) or (vulnerable_age and not_fully):
        return "High"

    # Moderate vulnerability (+1 escalation step):
    # Any one of: vulnerable age, heavy work, not fully acclimatized, or restrictive clothing
    if vulnerable_age or heavy_work or not_fully or is_ppe or is_heavy_clothing:
        return "Moderate"

    return "Low"


def escalate_risk(base_risk: str, vulnerability: str) -> str:
    """
    Escalate fused environmental risk by individual exposure vulnerability:
        Low +0 steps
        Moderate +1 step
        High +2 steps
    Capped at 'Extreme'.
    """
    bump = {"Low": 0, "Moderate": 1, "High": 2}.get(vulnerability, 0)
    idx = RISK_LEVELS.index(base_risk) if base_risk in RISK_LEVELS else 0
    new_idx = min(idx + bump, len(RISK_LEVELS) - 1)
    return RISK_LEVELS[new_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Work / Rest & Exposure Intervention Engine
# (ISO 7243:2017 & Occupational Health Decision Support)
# ──────────────────────────────────────────────────────────────────────────────

RECOMMENDATION_LEVELS: List[str] = [
    "Normal precautions",
    "Enhanced precautions",
    "Frequent recovery needed",
    "High-exposure controls required",
    "Suspend/defer strenuous exposure pending site assessment",
    "Site-specific assessment required",
]

RECOVERY_ENVIRONMENTS: List[str] = [
    "cool_indoor",
    "shade",
    "warm_indoor",
    "outdoor_direct",
    "unknown"
]

WORKLOAD_METABOLIC_WATTS: Dict[str, int] = {
    "light": 150,
    "moderate": 250,
    "heavy": 400,
    "very_heavy": 550,
    "unknown": 250
}

DEFAULT_EXPOSURE_SCENARIO: Dict[str, Any] = {
    "workload": "moderate",
    "work_intensity": "moderate",
    "workload_label": "Moderate Work (~250W)",
    "acclimatization": "very_used",
    "experience": "very_used",
    "experience_label": "Acclimatized",
    "clothing": "standard",
    "clothing_label": "Standard Work Clothing",
    "recovery_environment": "shade",
    "recovery_label": "Shaded Recovery Area",
    "is_sun_exposed": True,
    "duration_hours": 4.0,
    "exposure_pattern": "continuous",
    "age_group": "18–40",
    "age": 30,
    "is_default_scenario": True,
    "scenario_label": "Default Exposure Scenario (Moderate work, acclimatized worker, standard clothing, shaded recovery)",
}


def resolve_scenario_inputs(
    inputs: Optional[Dict[str, Any]] = None,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve and populate exposure inputs for scenario evaluation.
    If fields are omitted or None, uses fallback or DEFAULT_EXPOSURE_SCENARIO.
    Explicitly tags whether the resulting scenario is based on defaults.
    """
    base = dict(fallback or DEFAULT_EXPOSURE_SCENARIO)
    if not inputs:
        resolved = dict(base)
        resolved["is_default_scenario"] = True
        return resolved

    resolved = dict(base)
    has_custom = False
    for k, v in inputs.items():
        if v is not None and v != "":
            resolved[k] = v
            has_custom = True

    resolved["is_default_scenario"] = not has_custom

    # Harmonize workload and work_intensity
    if "workload" in inputs and inputs["workload"]:
        resolved["work_intensity"] = resolved["workload"]
    elif "work_intensity" in inputs and inputs["work_intensity"]:
        resolved["workload"] = resolved["work_intensity"]

    # Harmonize experience and acclimatization
    if "experience" in inputs and inputs["experience"]:
        resolved["acclimatization"] = resolved["experience"]
    elif "acclimatization" in inputs and inputs["acclimatization"]:
        resolved["experience"] = resolved["acclimatization"]

    return resolved


def normalize_workload(workload: Optional[str]) -> str:
    """Normalize user workload string to canonical category without fabricating."""
    if not workload:
        return "unknown"
    w = str(workload).strip().lower().replace("-", "_").replace(" ", "_")
    if "very_heavy" in w or "intense" in w:
        return "very_heavy"
    if "heavy" in w:
        return "heavy"
    if "light" in w or "sedentary" in w:
        return "light"
    if "moderate" in w or "medium" in w:
        return "moderate"
    return "unknown"


def normalize_acclimatization(acc: Any) -> str:
    """Normalize acclimatization input to canonical states without fabricating."""
    if acc is True:
        return "acclimatized"
    if acc is False:
        return "unacclimatized"
    if acc is None or acc == "":
        return "unknown"
    s = str(acc).strip().lower().replace("-", "_").replace(" ", "_")
    if any(k in s for k in ("very_used", "fully", "acclimatized", "adapted", "regular")):
        return "acclimatized"
    if any(k in s for k in ("somewhat_used", "partially", "returning", "partial")):
        return "partially_acclimatized"
    if any(k in s for k in ("not_used", "unacclimatized", "not", "new", "unadapted")):
        return "unacclimatized"
    return "unknown"


def normalize_clothing(clothing: Optional[str], ppe_category: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Normalize clothing and PPE inputs explicitly."""
    if not clothing:
        return ("unknown", ppe_category)
    c = str(clothing).strip().lower().replace("-", "_").replace(" ", "_")
    if "ppe" in c or "protective" in c or "impermeable" in c:
        ppe_cat = ppe_category or "generic_protective_gear"
        return ("ppe", ppe_cat)
    if "heavy" in c or "overalls" in c or "thick" in c:
        return ("heavy", ppe_category)
    if "standard" in c or "cotton" in c or "uniform" in c or "normal" in c:
        return ("standard", ppe_category)
    if "light" in c or "breathable" in c:
        return ("light", ppe_category)
    return ("unknown", ppe_category)


def normalize_recovery_environment(rec: Optional[str]) -> str:
    """Normalize recovery setting explicitly without fabricating unknown values."""
    if not rec:
        return "unknown"
    r = str(rec).strip().lower().replace("-", "_").replace(" ", "_")
    if any(k in r for k in ("cool_indoor", "ac", "air_conditioned", "cooled_indoor", "cooled", "chilled", "indoor_cool")):
        return "cool_indoor"
    if any(k in r for k in ("shade", "shaded", "canopy", "tent", "tree", "covered")):
        return "shade"
    if any(k in r for k in ("warm_indoor", "warehouse", "uncooled_indoor", "indoor_warm", "uncooled")):
        return "warm_indoor"
    if any(k in r for k in ("outdoor_direct", "direct_sun", "sun", "open_field", "outdoor")):
        return "outdoor_direct"
    return "unknown"


def build_exposure_context(
    wbgt_c: Optional[float] = None,
    environmental_risk_level: Optional[str] = None,
    wind_ms: Optional[float] = None,
    solar_wm2: Optional[float] = None,
    is_direct_sun: Optional[bool] = None,
    workload: Optional[str] = None,
    exposure_duration_hours: Optional[float] = None,
    exposure_pattern: Optional[str] = None,
    acclimatization: Any = None,
    clothing: Optional[str] = None,
    ppe_category: Optional[str] = None,
    recovery_environment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build explicit exposure-context model across environment, work, worker context,
    clothing/PPE, and recovery setting. Never fabricates missing information.
    """
    norm_workload = normalize_workload(workload)
    norm_acc = normalize_acclimatization(acclimatization)
    norm_cloth, norm_ppe = normalize_clothing(clothing, ppe_category)
    norm_rec = normalize_recovery_environment(recovery_environment)

    pat = "unknown"
    if exposure_pattern:
        p_clean = str(exposure_pattern).strip().lower()
        if "continuous" in p_clean:
            pat = "continuous"
        elif "intermittent" in p_clean:
            pat = "intermittent"

    # Clothing thermal adjustment and vapor-permeability characteristics
    cloth_adj = 0.0
    is_vapor = False
    cloth_uncertain = (norm_cloth == "unknown")
    if norm_cloth == "heavy":
        cloth_adj = 2.0
    elif norm_cloth == "ppe":
        if norm_ppe and any(k in str(norm_ppe).lower() for k in ("chemical", "encapsulated", "barrier", "impermeable", "suit")):
            cloth_adj = 11.0
            is_vapor = True
        else:
            cloth_adj = 3.0

    clothing_dict = {
        "clothing_type": norm_cloth,
        "ppe_category": norm_ppe if norm_cloth == "ppe" else None,
        "clothing_adjustment_c": cloth_adj,
        "is_vapor_barrier": is_vapor,
        "has_uncertainty": cloth_uncertain,
    }

    recovery_dict = {
        "recovery_environment": norm_rec,
        "environment_type": norm_rec,
        "has_uncertainty": (norm_rec == "unknown"),
    }

    return {
        "environment": {
            "wbgt_c": round(wbgt_c, 1) if wbgt_c is not None else "unknown",
            "environmental_risk_level": environmental_risk_level if environmental_risk_level in RISK_LEVELS else "unknown",
            "wind_ms": round(wind_ms, 1) if wind_ms is not None else "unknown",
            "solar_wm2": round(solar_wm2, 1) if solar_wm2 is not None else "unknown",
            "is_direct_sun": is_direct_sun if is_direct_sun is not None else "unknown",
        },
        "work": {
            "workload": norm_workload,
            "workload_label": norm_workload.replace("_", " ").title(),
            "exposure_duration_hours": round(float(exposure_duration_hours), 1) if exposure_duration_hours is not None else "unknown",
            "exposure_pattern": pat,
            "metabolic_watts": WORKLOAD_METABOLIC_WATTS.get(norm_workload, 250),
        },
        "worker_context": {
            "acclimatization_status": norm_acc,
        },
        "clothing": clothing_dict,
        "clothing_and_ppe": clothing_dict,
        "recovery": recovery_dict,
        "recovery_environment": recovery_dict,
    }


def detect_peak_risk_windows(
    hourly_forecast: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Identify lower, moderate, and peak exposure windows from forecast environmental heat data.
    Does NOT use simplistic time-of-day assumptions; relies strictly on actual forecast metrics.
    Detects diurnal peak, elevated risk period boundaries, daytime lower-exposure windows,
    underlying drivers, flat thermal curves, and uncertainty.
    """
    unavailable_fallback = {
        "status": "unavailable",
        "message": "Peak window cannot be estimated reliably from available data.",
        "windows": [],
        "peak_period": None,
        "peak_wbgt_c": None,
        "peak_temp_c": None,
        "peak_hour": None,
        "elevated_risk_start": None,
        "elevated_risk_end": None,
        "has_elevated_risk": False,
        "is_flat_curve": False,
        "lower_exposure_windows": [],
        "drivers": [],
        "environmental_drivers": [],
        "confidence": "Limited",
        "uncertainty_notice": "Forecast conditions may change over time. Direct on-site measurements are recommended during high-hazard periods.",
        "is_reliable": False,
    }

    if not hourly_forecast or not isinstance(hourly_forecast, list) or len(hourly_forecast) < 6:
        return unavailable_fallback

    valid_points = []
    for item in hourly_forecast:
        if not isinstance(item, dict):
            continue
        time_val = item.get("time") or item.get("hour") or item.get("time_str")
        if not time_val:
            continue
        wbgt = item.get("wbgt_c")
        if wbgt is None:
            wbgt = item.get("peak_wbgt_c")
        temp = item.get("temp_c")
        if temp is None:
            temp = item.get("current_temp_c")
        solar = item.get("solar_wm2")
        wind = item.get("wind_kph")
        if wind is None and item.get("wind_ms") is not None:
            try:
                wind = float(item["wind_ms"]) * 3.6
            except (ValueError, TypeError):
                wind = None
        humidity = item.get("humidity")

        if wbgt is None and temp is None:
            continue

        try:
            metric_wbgt = float(wbgt) if wbgt is not None else (float(temp) * 0.72)
            metric_temp = float(temp) if temp is not None else None
            metric_solar = float(solar) if solar is not None else None
            metric_wind = float(wind) if wind is not None else None
            metric_hum = float(humidity) if humidity is not None else None
        except (ValueError, TypeError):
            continue

        valid_points.append({
            "time_str": str(time_val),
            "wbgt_c": metric_wbgt,
            "temp_c": metric_temp,
            "solar_wm2": metric_solar,
            "wind_kph": metric_wind,
            "humidity": metric_hum,
        })

    if len(valid_points) < 6:
        return unavailable_fallback

    # Helper to parse hour integer from ISO or HH:MM string
    def _parse_hour(t_str: str) -> Optional[int]:
        try:
            part = t_str.split("T")[-1][:2]
            return int(part)
        except Exception:
            return None

    def _format_time_label(t_str: str) -> str:
        if "T" in t_str:
            return t_str.split("T")[-1][:5]
        return str(t_str)[:5]

    # Diurnal range analysis
    min_wbgt = min(p["wbgt_c"] for p in valid_points)
    max_wbgt = max(p["wbgt_c"] for p in valid_points)
    diff_wbgt = max_wbgt - min_wbgt
    is_flat_curve = (diff_wbgt < 1.0 and max_wbgt < 28.0)

    # Peak point identification
    peak_pt = max(valid_points, key=lambda x: x["wbgt_c"])
    peak_hour = _format_time_label(peak_pt["time_str"])
    peak_wbgt_c = round(peak_pt["wbgt_c"], 1)
    peak_temp_c = round(peak_pt["temp_c"], 1) if peak_pt["temp_c"] is not None else None

    # Elevated risk boundaries (WBGT >= 30.0 or within 1.5C of peak when peak >= 28.0)
    elevated_points = [
        p for p in valid_points
        if p["wbgt_c"] >= 30.0 or (max_wbgt >= 28.0 and p["wbgt_c"] >= max(28.0, max_wbgt - 1.5))
    ]
    has_elevated_risk = len(elevated_points) > 0
    if has_elevated_risk:
        elevated_risk_start = _format_time_label(elevated_points[0]["time_str"])
        elevated_risk_end = _format_time_label(elevated_points[-1]["time_str"])
    else:
        elevated_risk_start = None
        elevated_risk_end = None

    # Tiered classification for backward compatibility & multi-window grouping
    classified = []
    for p in valid_points:
        w = p["wbgt_c"]
        t = p["temp_c"]
        if w >= 30.0 or (t is not None and t >= 38.0):
            tier = "PEAK EXPOSURE WINDOW"
            tier_code = "peak"
        elif w >= 26.0 or (t is not None and t >= 32.0):
            tier = "MODERATE EXPOSURE WINDOW"
            tier_code = "moderate"
        else:
            tier = "LOWER EXPOSURE WINDOW"
            tier_code = "lower"
        classified.append({
            "time_str": p["time_str"],
            "tier": tier,
            "tier_code": tier_code,
            "wbgt_c": w,
            "temp_c": t,
        })

    windows = []
    current_block = [classified[0]]
    for pt in classified[1:]:
        if pt["tier_code"] == current_block[-1]["tier_code"]:
            current_block.append(pt)
        else:
            t_start = _format_time_label(current_block[0]["time_str"])
            t_end = _format_time_label(current_block[-1]["time_str"])
            tier_name = current_block[0]["tier"]
            w_min = min(x["wbgt_c"] for x in current_block)
            w_max = max(x["wbgt_c"] for x in current_block)
            desc = (
                "Lower environmental heat burden — preferred window for strenuous tasks."
                if tier_name == "LOWER EXPOSURE WINDOW" else
                "Elevated heat burden — curtail continuous heavy exertion and schedule active recovery."
                if tier_name == "PEAK EXPOSURE WINDOW" else
                "Moderate environmental heat burden — maintain paced exertion and standard hydration."
            )
            windows.append({
                "window": f"{t_start}–{t_end}",
                "tier": tier_name,
                "tier_code": current_block[0]["tier_code"],
                "wbgt_range_c": (round(w_min, 1), round(w_max, 1)),
                "description": desc,
            })
            current_block = [pt]

    if current_block:
        t_start = _format_time_label(current_block[0]["time_str"])
        t_end = _format_time_label(current_block[-1]["time_str"])
        tier_name = current_block[0]["tier"]
        w_min = min(x["wbgt_c"] for x in current_block)
        w_max = max(x["wbgt_c"] for x in current_block)
        desc = (
            "Lower environmental heat burden — preferred window for strenuous tasks."
            if tier_name == "LOWER EXPOSURE WINDOW" else
            "Elevated heat burden — curtail continuous heavy exertion and schedule active recovery."
            if tier_name == "PEAK EXPOSURE WINDOW" else
            "Moderate environmental heat burden — maintain paced exertion and standard hydration."
        )
        windows.append({
            "window": f"{t_start}–{t_end}",
            "tier": tier_name,
            "tier_code": current_block[0]["tier_code"],
            "wbgt_range_c": (round(w_min, 1), round(w_max, 1)),
            "description": desc,
        })

    # Group contiguous hours around the peak to define primary peak period
    peak_blocks = [w for w in windows if w["tier_code"] == "peak"]
    if peak_blocks:
        primary_peak_window = peak_blocks[0]["window"]
    elif has_elevated_risk and elevated_risk_start and elevated_risk_end:
        primary_peak_window = f"{elevated_risk_start}–{elevated_risk_end}"
    else:
        primary_peak_window = f"{peak_hour}–{peak_hour}"

    # Daytime lower-exposure window detection (06:00 to 19:00, WBGT < 28.0C)
    daytime_pts = [p for p in valid_points if _parse_hour(p["time_str"]) is not None and 6 <= _parse_hour(p["time_str"]) <= 19]
    lower_daytime_pts = [p for p in daytime_pts if p["wbgt_c"] < 28.0]
    lower_exposure_windows = []

    if lower_daytime_pts:
        curr_low = [lower_daytime_pts[0]]
        for p in lower_daytime_pts[1:]:
            prev_h = _parse_hour(curr_low[-1]["time_str"])
            curr_h = _parse_hour(p["time_str"])
            if prev_h is not None and curr_h is not None and (curr_h - prev_h) <= 2:
                curr_low.append(p)
            else:
                t0 = _format_time_label(curr_low[0]["time_str"])
                t1 = _format_time_label(curr_low[-1]["time_str"])
                w0 = min(x["wbgt_c"] for x in curr_low)
                w1 = max(x["wbgt_c"] for x in curr_low)
                lower_exposure_windows.append({
                    "window": f"{t0}–{t1}",
                    "wbgt_range_c": (round(w0, 1), round(w1, 1)),
                    "description": "Daytime lower-exposure window suitable for scheduled high-exertion tasks."
                })
                curr_low = [p]
        if curr_low:
            t0 = _format_time_label(curr_low[0]["time_str"])
            t1 = _format_time_label(curr_low[-1]["time_str"])
            w0 = min(x["wbgt_c"] for x in curr_low)
            w1 = max(x["wbgt_c"] for x in curr_low)
            lower_exposure_windows.append({
                "window": f"{t0}–{t1}",
                "wbgt_range_c": (round(w0, 1), round(w1, 1)),
                "description": "Daytime lower-exposure window suitable for scheduled high-exertion tasks."
            })

    # Causal environmental drivers of the peak
    drivers = []
    if peak_pt.get("solar_wm2") is not None and peak_pt["solar_wm2"] > 600:
        drivers.append(f"Intense solar irradiance peaking at {int(peak_pt['solar_wm2'])} W/m².")
    if peak_pt.get("temp_c") is not None and peak_pt["temp_c"] >= 38.0:
        drivers.append(f"High ambient dry-bulb temperature reaching {peak_pt['temp_c']:.1f}°C.")
    if peak_pt.get("humidity") is not None and peak_pt["humidity"] >= 65.0:
        drivers.append(f"Elevated atmospheric humidity ({int(peak_pt['humidity'])}%) restricting evaporative sweat dissipation.")
    if peak_pt.get("wind_kph") is not None and peak_pt["wind_kph"] < 10.0:
        drivers.append(f"Stagnant air velocity ({peak_pt['wind_kph']:.1f} km/h) impeding convective heat transfer.")
    if not drivers:
        drivers.append(f"Diurnal thermal accumulation elevating WBGT to {peak_wbgt_c}°C around {peak_hour}.")

    # Summary message
    if is_flat_curve:
        message = "Flat diurnal thermal curve observed — consistently low environmental heat hazard without distinct midday peak."
    elif has_elevated_risk:
        message = f"Elevated peak risk period identified between {elevated_risk_start} and {elevated_risk_end} (peak {peak_wbgt_c}°C at {peak_hour})."
    else:
        message = f"Diurnal thermal peak identified at {peak_hour} ({peak_wbgt_c}°C WBGT)."

    confidence = "High" if len(valid_points) >= 20 else ("Medium" if len(valid_points) >= 12 else "Limited")

    return {
        "status": "available",
        "message": message,
        "windows": windows,
        "peak_period": primary_peak_window,
        "peak_wbgt_c": peak_wbgt_c,
        "peak_temp_c": peak_temp_c,
        "peak_hour": peak_hour,
        "elevated_risk_start": elevated_risk_start,
        "elevated_risk_end": elevated_risk_end,
        "has_elevated_risk": has_elevated_risk,
        "is_flat_curve": is_flat_curve,
        "lower_exposure_windows": lower_exposure_windows,
        "drivers": drivers,
        "environmental_drivers": drivers,
        "confidence": confidence,
        "uncertainty_notice": "Forecast conditions may change over time. Direct on-site measurements are recommended during high-hazard periods.",
        "is_reliable": True,
    }


evaluate_peak_risk_windows = detect_peak_risk_windows


def evaluate_exposure_intervention(
    exposure_context: Dict[str, Any],
    peak_risk_window: Optional[Dict[str, Any]] = None,
    is_peak_window: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Transparent rule engine evaluating environmental heat, work intensity, acclimatization,
    clothing/PPE, recovery environment, exposure duration, and peak forecast window.
    Does NOT invent arbitrary mathematics; produces traceable, explainable precautions.
    """
    env = exposure_context.get("environment", {})
    work = exposure_context.get("work", {})
    worker = exposure_context.get("worker_context", {})
    cloth = exposure_context.get("clothing", {})
    rec = exposure_context.get("recovery", {})

    base_risk = env.get("environmental_risk_level", "unknown")
    wbgt_c = env.get("wbgt_c")
    workload = work.get("workload", "unknown")
    acc_status = worker.get("acclimatization_status", "unknown")
    cloth_type = cloth.get("clothing_type", "unknown")
    ppe_cat = cloth.get("ppe_category")
    rec_env = rec.get("recovery_environment", "unknown")
    duration = work.get("exposure_duration_hours")

    # Determine baseline environmental tier (1 to 5)
    risk_tier_map = {
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Very High": 4,
        "Extreme": 5,
    }
    if base_risk in risk_tier_map:
        base_tier = risk_tier_map[base_risk]
    elif isinstance(wbgt_c, (int, float)):
        if wbgt_c < 26.0:
            base_tier = 1
        elif wbgt_c < 30.0:
            base_tier = 2
        elif wbgt_c < 32.0:
            base_tier = 3
        elif wbgt_c < 34.0:
            base_tier = 4
        else:
            base_tier = 5
    else:
        # Information insufficient for environmental baseline
        return {
            "recommendation_level": "Site-specific assessment required",
            "work_guidance": "Environmental heat data is insufficient for screening guidance. Conduct an on-site heat stress assessment.",
            "recovery_guidance": "Ensure accessible shaded or cooled rest zones until environmental conditions are verified.",
            "recommended_controls": [
                {
                    "category": "MONITORING",
                    "action": "Perform localized wet-bulb globe temperature or heat index screening.",
                    "reason": "Environmental heat data is currently missing or unverified.",
                    "priority": "High"
                }
            ],
            "schedule_context": {
                "work_window": "Site-specific determination",
                "recovery_window": "Continuous shaded access",
                "basis": ["missing environmental parameters"],
                "confidence": "low"
            },
            "recommendation_reasons": ["Atmospheric heat burden could not be computed from available inputs."],
            "limitations": [
                "Environmental heat data is unavailable. Screening models cannot evaluate physiological thermal load without valid meteorological observations.",
                "Site-specific environmental monitoring and occupational assessment must be conducted before establishing operational work-rest schedules."
            ],
            "work_minutes": 30,
            "rest_minutes": 30,
            "instruction": "Site-specific assessment required: verify local heat conditions before assigning continuous strenuous outdoor work.",
            "hydration_guidance": {
                "general_principle": "Maintain continuous access to cool potable water near active work areas.",
                "drinking_pattern": "Encourage regular, small fluid intake throughout the work shift (e.g. one cup of water every 15–20 minutes).",
                "electrolytes": "For prolonged heavy physical exertion (>2 hours) with heavy sweating, provide balanced electrolyte solutions.",
                "medical_caution": "Workers with medical fluid restrictions should follow individual healthcare provider advice.",
                "is_individualized_medical_prescription": False
            },
            "is_decision_support": True
        }

    score = base_tier
    reasons: List[str] = []
    recommended_controls: List[Dict[str, str]] = []
    limitations: List[str] = []
    schedule_basis: List[str] = []

    # 1. Environmental Heat Baseline
    env_basis_desc = f"{base_risk} environmental heat burden" if base_risk != "unknown" else f"WBGT {wbgt_c}°C"
    schedule_basis.append(env_basis_desc)

    # 2. Workload Modifier
    if workload == "light":
        if score >= 3:
            score -= 1
            reasons.append("Light physical intensity generates minimal internal metabolic heat (~150 W), permitting extended work cycles.")
            schedule_basis.append("light physical intensity (~150W)")
        else:
            schedule_basis.append("light workload (~150W)")
    elif workload == "heavy":
        if score < 5:
            score += 1
        reasons.append("Heavy manual labor generates high internal metabolic heat (~400 W), accelerating core temperature elevation.")
        schedule_basis.append("heavy physical workload (~400W)")
        recommended_controls.append({
            "category": "WORKLOAD",
            "action": "Implement task rotation with lighter duties and utilize mechanical lifting aids where practical.",
            "reason": "Heavy metabolic exertion (~400W) accelerates internal body heat accumulation.",
            "priority": "High"
        })
    elif workload == "very_heavy":
        score = min(5, score + (2 if base_tier >= 3 else 1))
        reasons.append("Very heavy physical exertion generates extreme metabolic heat (>500 W), severely compounding environmental thermal burden.")
        schedule_basis.append("very heavy physical exertion (>500W)")
        recommended_controls.append({
            "category": "WORKLOAD",
            "action": "Curtail or rotate high-intensity manual tasks; introduce auxiliary mechanical assistance.",
            "reason": "Extremely high metabolic heat rate (>500W) creates dangerous internal thermal load.",
            "priority": "High"
        })
    elif workload == "unknown":
        limitations.append("Workload intensity is unspecified; screening recommendations assume moderate physical activity.")
        schedule_basis.append("moderate baseline workload (assumed)")
    else:
        schedule_basis.append("moderate workload (~250W)")

    # 3. Acclimatization Modifier
    if acc_status == "acclimatized":
        reasons.append("Worker is fully heat-acclimatized: efficient thermoregulatory sweating and lower cardiovascular strain support standard operational pacing.")
        schedule_basis.append("full heat acclimatization")
    elif acc_status == "partially_acclimatized":
        if score <= 3 and base_tier >= 2:
            score += 1
        reasons.append("Worker is partially acclimatized / returning from absence: incomplete physiological heat tolerance requires conservative pacing.")
        schedule_basis.append("partial acclimatization")
        recommended_controls.append({
            "category": "ACCLIMATIZATION",
            "action": "Apply progressive / gradual re-acclimatization schedule (50% to 70% duration) for returning workers.",
            "reason": "Physiological sweating and cardiovascular heat defense are partially diminished.",
            "priority": "High"
        })
    elif acc_status == "unacclimatized":
        if score < 5:
            score += 1
        reasons.append("Worker is not heat-acclimatized: delayed sweating and higher cardiac strain significantly elevate heat illness vulnerability.")
        schedule_basis.append("unacclimatized worker status")
        recommended_controls.append({
            "category": "ACCLIMATIZATION",
            "action": "Implement progressive exposure ramp-up protocol (e.g. 20% to 50% shift exposure on Day 1) under active supervision.",
            "reason": "Unacclimatized individuals lack physiological heat adaptation mechanisms.",
            "priority": "High"
        })
    else:
        limitations.append("Worker heat acclimatization status is unspecified; recommendations assume an acclimatized worker.")

    # 4. Clothing / PPE Modifier
    if cloth_type == "light":
        reasons.append("Lightweight breathable clothing permits unhindered evaporative and convective cooling.")
        schedule_basis.append("light breathable clothing")
    elif cloth_type == "heavy":
        if base_tier >= 2 and score < 5:
            score += 1
        reasons.append("Heavy work clothing reduces sweat evaporation and traps body heat, increasing thermal burden.")
        schedule_basis.append("heavy work clothing")
        recommended_controls.append({
            "category": "CLOTHING_PPE",
            "action": "Loosen restrictive outerwear during recovery intervals and select high-breathability fabrics.",
            "reason": "Heavy fabrics trap body heat and impede sweat evaporation.",
            "priority": "Medium"
        })
    elif cloth_type == "ppe":
        is_vapor = cloth.get("is_vapor_barrier", False) or (ppe_cat and any(k in str(ppe_cat).lower() for k in ("chemical", "encapsulated", "barrier", "impermeable")))
        if is_vapor:
            score = max(score, 4)
            reasons.append("Impermeable vapor-barrier PPE virtually eliminates evaporative heat loss through sweating.")
            schedule_basis.append("vapor-barrier PPE configuration")
            recommended_controls.append({
                "category": "MONITORING",
                "action": "Implement mandatory physiological core temperature / heart rate monitoring and shortened work periods.",
                "reason": "Vapor-barrier clothing completely blocks sweat evaporation.",
                "priority": "mandatory"
            })
            recommended_controls.append({
                "category": "CLOTHING_PPE",
                "action": "Conduct PPE-specific heat assessment; deploy active microclimate cooling vests or air-fed suits.",
                "reason": "Protective clothing restricts convective and evaporative heat transfer.",
                "priority": "High"
            })
        else:
            if score < 5:
                score += 1
            reasons.append("Protective equipment / PPE creates a significant barrier to evaporative cooling, substantially increasing heat strain.")
            schedule_basis.append("protective equipment / PPE")
            recommended_controls.append({
                "category": "CLOTHING_PPE",
                "action": "Conduct PPE-specific heat assessment; consider active cooling vests or shortened work cycles.",
                "reason": "Protective clothing restricts convective and evaporative heat transfer.",
                "priority": "High"
            })
        limitations.append("PPE-specific heat assessment recommended: impermeable or multi-layered protective equipment severely restricts sweat evaporation. Standard work/rest schedules may be insufficient without physiological monitoring or microclimate cooling.")
    else:
        limitations.append("Clothing/PPE configuration is unspecified; screening recommendations assume standard lightweight work clothing.")

    # 5. Recovery Environment Modifier
    if rec_env == "cool_indoor":
        if score >= 3 and base_tier < 5:
            score = max(score - 1, base_tier)
        reasons.append("Air-conditioned or mechanically cooled recovery area accelerates core body temperature recovery and cardiovascular stabilization.")
        schedule_basis.append("cool indoor recovery available")
        recommended_controls.append({
            "category": "RECOVERY",
            "action": "Ensure rest periods take place within the cooled indoor recovery area to maximize cooling efficacy.",
            "reason": "Mechanically cooled air provides optimal recovery between exposure intervals.",
            "priority": "Low"
        })
    elif rec_env == "shade":
        reasons.append("Shaded recovery area provides baseline protection from solar radiation.")
        schedule_basis.append("shaded recovery available")
        recommended_controls.append({
            "category": "RECOVERY",
            "action": "Ensure rest breaks occur under robust shade canopy or natural tree cover with adequate airflow.",
            "reason": "Shading reduces direct radiative radiant heat load.",
            "priority": "Medium"
        })
    elif rec_env == "warm_indoor":
        if score < 5:
            score += 1
        reasons.append("Unconditioned or warm indoor recovery area limits radiant dissipation and slows physiological cooling.")
        schedule_basis.append("warm unconditioned indoor recovery")
        recommended_controls.append({
            "category": "RECOVERY",
            "action": "Introduce auxiliary spot cooling fans or misting fans in the recovery zone to enhance airflow.",
            "reason": "Warm stagnant recovery environments impede core temperature reduction.",
            "priority": "High"
        })
    elif rec_env == "outdoor_direct":
        score = min(5, score + 2 if base_tier >= 2 else 1)
        reasons.append("Recovery directly in open sunlight offers zero solar protection, severely compromising physiological heat dissipation.")
        schedule_basis.append("direct sun recovery setting (compromised)")
        recommended_controls.append({
            "category": "RECOVERY",
            "action": "Establish portable sun canopies, umbrellas, or shaded shelters immediately; do not rest in direct sunlight.",
            "reason": "Resting under direct sun or unshielded solar radiation prevents physiological core cooling and accelerates heat illness.",
            "priority": "High"
        })
    else:
        limitations.append("Recovery setting is unspecified; recommendations assume baseline shaded recovery.")

    # 6. Peak-Risk Window Context (from real forecast data)
    if is_peak_window is True:
        if score < 5:
            score += 1
        reasons.append("Current time coincides with the peak forecast diurnal heat window: solar radiation and ambient temperature are at diurnal maxima.")
        schedule_basis.append("peak forecast diurnal heat window active")
        recommended_controls.append({
            "category": "SCHEDULING",
            "action": "Reschedule high-intensity physical tasks to identified lower-exposure morning or late afternoon windows.",
            "reason": "Diurnal peak solar and ambient conditions magnify environmental strain.",
            "priority": "High"
        })

    # 6. Non-Prescriptive Hydration Controls (Mandatory)
    recommended_controls.append({
        "category": "HYDRATION",
        "action": "Maintain continuous access to cool potable water near active work areas; encourage regular small drinks (e.g. 1 cup every 15–20 minutes).",
        "reason": "Elevated thermal stress induces continuous fluid loss through sweating.",
        "priority": "High"
    })
    if score >= 3 or workload in ("heavy", "very_heavy"):
        recommended_controls.append({
            "category": "HYDRATION",
            "action": "Provide balanced electrolyte solutions (ORS, buttermilk, lemon water with salt) for heavy physical exertion exceeding 2 hours.",
            "reason": "Electrolyte replenishment prevents hyponatremia and cramping from profuse sweating.",
            "priority": "High"
        })

    # 7. Monitoring Controls
    if score >= 3:
        recommended_controls.append({
            "category": "MONITORING",
            "action": "Institute mandatory buddy monitoring and supervisor checks for early signs of heat strain (dizziness, weakness, cessation of sweating).",
            "reason": "Elevated thermal strain poses acute health risks without rapid identification.",
            "priority": "High"
        })

    # 8. Peak Risk Forecast Scheduling Controls
    if peak_risk_window and peak_risk_window.get("peak_period"):
        pk = peak_risk_window["peak_period"]
        pk_wbgt = peak_risk_window.get("peak_wbgt_c")
        recommended_controls.append({
            "category": "SCHEDULING",
            "action": f"Shift strenuous physical work away from the forecast peak period ({pk}) to cooler morning/evening windows.",
            "reason": f"Forecast environmental heat burden is highest during {pk} (peak WBGT {pk_wbgt}°C).",
            "priority": "High"
        })
        reasons.append(f"Forecast heat burden peaks between {pk}, requiring schedule adjustments for strenuous tasks.")

    if duration and isinstance(duration, (int, float)) and duration >= 4.0:
        reasons.append(f"Prolonged continuous exposure ({duration} hours) compounds cumulative physiological thermal strain.")
        recommended_controls.append({
            "category": "SCHEDULING",
            "action": "Rotate personnel across tasks to prevent continuous prolonged heat exposure.",
            "reason": "Extended exposure duration compounds cumulative physiological fatigue.",
            "priority": "Medium"
        })

    # Final Outcome Tier Clamping
    final_score = max(1, min(5, score))
    rec_level = RECOMMENDATION_LEVELS[final_score - 1]

    # Map Tier to Work/Recovery Guidance & Schedule Context
    tier_configs = {
        1: {
            "work_window": "Standard work duration (up to 50–60 min per hour)",
            "recovery_window": "Routine 10-minute rest breaks per schedule in comfortable environment",
            "work_minutes": 60,
            "rest_minutes": 0,
            "instruction": "Standard work schedule — environmental heat load within normal seasonal tolerance.",
            "work_guidance": "Maintain standard operational work pace. Ensure routine hydration breaks and baseline supervision.",
            "recovery_guidance": "Routine rest breaks in ventilated or shaded areas are sufficient under normal seasonal conditions.",
        },
        2: {
            "work_window": "Approximately 45 min per hour",
            "recovery_window": "At least 15 min per hour in shaded or cooled recovery zones",
            "work_minutes": 45,
            "rest_minutes": 15,
            "instruction": "Provide 15 minutes of rest per hour in shaded or cooled recovery zones.",
            "work_guidance": "Pace physical exertion; avoid continuous heavy work exceeding 45 minutes without recovery.",
            "recovery_guidance": "Mandate recovery in shaded or air-conditioned environments with active air movement.",
        },
        3: {
            "work_window": "Approximately 30 min per hour (alternate work/rest cycles)",
            "recovery_window": "At least 30 min per hour in shaded/cooled recovery zones",
            "work_minutes": 30,
            "rest_minutes": 30,
            "instruction": "Alternate 30 minutes of work with 30 minutes of shaded rest. Ensure continuous potable water access.",
            "work_guidance": "Alternate 30 minutes of work with 30 minutes of recovery. Rotate strenuous tasks among workers.",
            "recovery_guidance": "Mandatory shaded or cooled recovery with active air circulation. Loosen tight protective gear.",
        },
        4: {
            "work_window": "Curtail continuous exertion to approximately 15 min per hour; utilize task rotation",
            "recovery_window": "At least 45 min per hour in shaded/cooled zones with active air movement",
            "work_minutes": 15,
            "rest_minutes": 45,
            "instruction": "Curtail outdoor exertion to 15 minutes per hour. Rotate workers and prioritize shaded recovery.",
            "work_guidance": "Curtail continuous outdoor exertion to approximately 15 minutes per hour. Enforce strict task rotation.",
            "recovery_guidance": "Enforce recovery in air-conditioned or well-ventilated shaded zones. Actively monitor hydration and symptoms.",
        },
        5: {
            "work_window": "Continuous strenuous outdoor work not recommended; suspend or reschedule non-essential operations",
            "recovery_window": "Continuous shaded/cooled recovery; restrict activity to critical low-exertion tasks only",
            "work_minutes": 0,
            "rest_minutes": 60,
            "instruction": "STOP: Continuous manual outdoor work not recommended. Suspend non-essential outdoor operations.",
            "work_guidance": "Suspend or defer strenuous manual exposure pending qualified on-site environmental assessment.",
            "recovery_guidance": "Provide continuous cooling in air-conditioned shelters. Monitor personnel for emergent heat stroke symptoms.",
        },
    }
    cfg = tier_configs[final_score]

    # Confidence evaluation for recommendation
    confidence_level = "High" if len(limitations) == 0 else "Medium" if len(limitations) <= 2 else "Limited"

    limitations.append(
        "Recommendations are occupational screening decision-support guidelines, not individualized medical advice or universal safety guarantees. Must be adapted to site-specific tasks and supervisory protocols."
    )

    return {
        "recommendation_level": rec_level,
        "work_guidance": cfg["work_guidance"],
        "recovery_guidance": cfg["recovery_guidance"],
        "recommended_controls": recommended_controls,
        "schedule_context": {
            "work_window": cfg["work_window"],
            "recovery_window": cfg["recovery_window"],
            "basis": schedule_basis,
            "confidence": confidence_level,
        },
        "recommendation_basis": schedule_basis,
        "recommendation_reasons": reasons,
        "limitations": limitations,
        "work_minutes": cfg["work_minutes"],
        "rest_minutes": cfg["rest_minutes"],
        "instruction": cfg["instruction"],
        "screening_basis": "ISO 7243:2017 & ACGIH Occupational Heat Exposure Screening Guidance",
        "is_decision_support": True,
        "hydration_guidance": {
            "general_principle": "Maintain continuous access to cool potable water near active work areas.",
            "drinking_pattern": "Encourage regular, small fluid intake throughout the work shift (e.g. one cup of water every 15–20 minutes) rather than large volumes infrequently.",
            "electrolytes": "For prolonged heavy physical exertion (>2 hours) with heavy sweating, provide balanced electrolyte solutions (ORS, buttermilk, lemon water with salt).",
            "medical_caution": "Workers with medical fluid restrictions, heart failure, or kidney conditions should follow individual healthcare provider advice.",
            "is_individualized_medical_prescription": False,
        },
        "exposure_context": exposure_context,
    }


def get_work_rest(
    risk_category: str,
    workload: Optional[str] = None,
    acclimatized: Any = True,
    clothing: Optional[str] = None,
    recovery_environment: Optional[str] = None,
    exposure_duration_hours: Optional[float] = None,
    exposure_pattern: Optional[str] = None,
    ppe_category: Optional[str] = None,
    peak_risk_window: Optional[Dict[str, Any]] = None,
    is_peak_window: Optional[bool] = None,
    wbgt_c: Optional[float] = None,
    wind_ms: Optional[float] = None,
    solar_wm2: Optional[float] = None,
    is_direct_sun: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Return comprehensive work/rest decision support and operational intervention guidance.
    Considers environmental heat, work intensity, acclimatization, clothing/PPE, recovery
    environment, exposure duration, and peak forecast periods.
    Preserves 100% backward compatibility for legacy callers and tests.
    """
    exp_ctx = build_exposure_context(
        wbgt_c=wbgt_c,
        environmental_risk_level=risk_category,
        wind_ms=wind_ms,
        solar_wm2=solar_wm2,
        is_direct_sun=is_direct_sun,
        workload=workload,
        exposure_duration_hours=exposure_duration_hours,
        exposure_pattern=exposure_pattern,
        acclimatization=acclimatized,
        clothing=clothing,
        ppe_category=ppe_category,
        recovery_environment=recovery_environment,
    )

    return evaluate_exposure_intervention(
        exposure_context=exp_ctx,
        peak_risk_window=peak_risk_window,
        is_peak_window=is_peak_window,
    )



# ──────────────────────────────────────────────────────────────────────────────
# Data Quality and Confidence Engine
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_data_quality(
    weather_data: Dict[str, Any],
    is_forecast: bool = False,
    source: str = "Open-Meteo",
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate atmospheric data completeness, plausibility, and confidence.
    Does NOT fabricate numeric confidence percentages without empirical justification;
    returns transparent qualitative ratings: 'High', 'Medium', or 'Limited'.
    """
    required_keys = ["temp_c", "rh_pct", "wind_ms", "solar_wm2"]
    missing = [k for k in required_keys if k not in weather_data or weather_data[k] is None]

    flags: List[str] = []
    reasons: List[str] = []

    completeness_pct = round(((len(required_keys) - len(missing)) / len(required_keys)) * 100.0, 1)

    if missing:
        flags.append(f"Missing variables: {', '.join(missing)}")
        reasons.append(f"Input data is missing parameters: {', '.join(missing)}")

    temp = weather_data.get("temp_c")
    rh = weather_data.get("rh_pct")
    wind = weather_data.get("wind_ms")
    solar = weather_data.get("solar_wm2")

    # Plausibility checks
    if temp is not None:
        if temp < -20.0 or temp > 60.0:
            flags.append(f"Extreme air temperature ({temp} °C)")
            reasons.append("Air temperature is outside standard empirical meteorological ranges.")
    if rh is not None:
        if rh < 0.0 or rh > 100.0:
            flags.append(f"Relative humidity out of bounds ({rh} %)")
            reasons.append("Relative humidity observation is physically invalid.")
    if wind is not None:
        if wind < 0.0:
            flags.append(f"Negative wind speed ({wind} m/s)")
            reasons.append("Wind speed cannot be negative.")
    if solar is not None:
        if solar < 0.0:
            flags.append(f"Negative solar radiation ({solar} W/m²)")
            reasons.append("Solar radiation cannot be negative.")

    # Freshness checks
    freshness = weather_data.get("freshness")
    is_stale = (freshness == "STALE" or weather_data.get("is_stale") is True)
    if is_stale:
        flags.append("Observation timestamp is stale (> 1 hour old)")
        reasons.append("Atmospheric observation timestamp exceeds freshness threshold; on-site heat stress conditions may have diverged.")

    # Confidence evaluation
    non_stale_flags = [f for f in flags if not f.startswith("Observation timestamp is stale")]
    if not missing and not non_stale_flags:
        if is_stale:
            level = "Medium" if not is_forecast else "Limited"
            reasons.append("Confidence downgraded due to stale atmospheric observation.")
            score = 0.75 if not is_forecast else 0.60
        elif is_forecast:
            level = "Medium"
            reasons.append("Numerical weather prediction forecast model data (~1–10 km resolution). Peak heat stress may vary locally.")
            score = 0.80
        else:
            level = "High"
            reasons.append("Complete observed atmospheric surface measurements available and within physical bounds.")
            score = 0.95
    elif len(missing) == 1 and ("solar_wm2" in missing or "wind_ms" in missing):
        if is_stale:
            level = "Limited"
            reasons.append("Confidence downgraded to Limited due to concurrent missing parameters and stale timestamp.")
            score = 0.50
        else:
            level = "Medium"
            reasons.append("Primary temperature and humidity available; secondary parameters estimated or default-bounded.")
            score = 0.70
    else:
        level = "Limited"
        reasons.append("Incomplete atmospheric observations or physically anomalous sensor values; interpretation requires caution.")
        score = 0.40

    return {
        "level": level,
        "score": score,
        "source": source,
        "location": location,
        "is_forecast": is_forecast,
        "completeness_pct": completeness_pct,
        "freshness": freshness or "FRESH",
        "missing_variables": missing,
        "flags": flags,
        "reasons": reasons,
        "disclaimer": "Atmospheric data quality assessment for screening reliability. Local urban microclimate effects (such as localized heat islands) may alter on-site exposure."
    }


# ──────────────────────────────────────────────────────────────────────────────
# Structured Explainability & Output Model
# ──────────────────────────────────────────────────────────────────────────────

def generate_structured_explanation(
    location: str,
    temp_c: float,
    wbgt_c: float,
    wbgt_band: str,
    imd_status: str,
    base_risk: str,
    final_risk: str,
    vulnerability: str,
    age_group: Optional[str] = None,
    experience: Optional[str] = None,
    workload: Optional[str] = None,
    clothing: Optional[str] = None,
    operational_exposure: Optional[Dict[str, Any]] = None,
    confidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate structured, traceable explanation answering:
    - Why is the risk at this level?
    - What are the environmental drivers?
    - What are the operational drivers?
    - What are the protective factors?
    - What practical actions are recommended?
    """
    env_drivers: List[str] = []
    op_drivers: List[str] = []
    vuln_drivers: List[str] = []
    protective_factors: List[str] = []
    actions: List[str] = []
    uncertainty_notes: List[str] = []

    # 1. Environmental drivers
    if imd_status == "Severe Heatwave":
        env_drivers.append(f"IMD Severe Heatwave: Current temperature ({temp_c:.1f}°C) significantly exceeds regional threshold.")
    elif imd_status == "Heatwave":
        env_drivers.append(f"IMD Heatwave: Current temperature ({temp_c:.1f}°C) is substantially elevated above regional threshold.")
    elif imd_status == "Heat Day":
        env_drivers.append(f"IMD Heat Day: Temperature ({temp_c:.1f}°C) meets or exceeds regional qualifying baseline.")
    else:
        protective_factors.append(f"Ambient temperature ({temp_c:.1f}°C) is below regional IMD heatwave thresholds.")

    env_drivers.append(
        f"Outdoor WBGT is estimated at {wbgt_c:.1f}°C ({wbgt_band} band), accounting for combined humidity, wind, and solar radiative load."
    )

    # 2. Operational drivers & protective factors
    if operational_exposure:
        op_drivers.extend(operational_exposure.get("operational_drivers", []))
        protective_factors.extend(operational_exposure.get("protective_factors", []))
    else:
        wl_clean = str(workload or "").lower()
        if wl_clean in ("very_heavy", "very heavy"):
            op_drivers.append("Very heavy manual labor generates intense internal metabolic heat.")
        elif wl_clean == "heavy":
            op_drivers.append("Heavy physical exertion increases metabolic heat production.")
        elif wl_clean == "light":
            protective_factors.append("Light physical activity generates minimal internal body heat.")

        cl_clean = str(clothing or "").lower()
        if cl_clean in ("ppe", "protective"):
            op_drivers.append("Protective gear / PPE restricts sweat evaporation and heat dissipation.")
        elif cl_clean in ("heavy", "heavy_work"):
            op_drivers.append("Heavy work clothing adds thermal insulation.")
        else:
            protective_factors.append("Light clothing allows effective sweat evaporation.")

    # 3. Vulnerability context drivers
    ag_clean = str(age_group or "").lower()
    if ag_clean in ("under 18", "under_18", "<18"):
        vuln_drivers.append("Youth age (<18): Developing thermoregulatory sweat control and higher surface-area-to-mass ratio.")
    elif ag_clean in ("60+", "60_plus", ">60", "60 plus"):
        vuln_drivers.append("Older adult age (60+): Blunted thirst perception and reduced cardiac output capacity during thermal strain.")

    exp_clean = str(experience or "").lower()
    canonical_acc = ACCLIMATIZATION_ALIASES.get(exp_clean, "acclimatized")
    if canonical_acc == "unacclimatized":
        vuln_drivers.append("Worker is unacclimatized to hot conditions: higher cardiac strain and delayed sweating.")
    elif canonical_acc == "partially_acclimatized":
        vuln_drivers.append("Partial heat acclimatization: incomplete physiological sweat and cardiovascular defense.")
    else:
        protective_factors.append("Full heat acclimatization supports efficient sweat regulation and lower electrolyte loss.")

    # 4. Recommended actions
    work_rest_info = get_work_rest(
        final_risk,
        workload=workload,
        acclimatized=experience,
        clothing=clothing,
        wbgt_c=wbgt_c,
    )
    actions.append(f"Work/Rest Pacing: {work_rest_info['instruction']}")
    actions.append(work_rest_info["hydration_guidance"]["general_principle"])
    actions.append(work_rest_info["hydration_guidance"]["drinking_pattern"])
    for ctrl in work_rest_info.get("recommended_controls", []):
        act_line = f"[{ctrl['category']}] {ctrl['action']}"
        if act_line not in actions:
            actions.append(act_line)

    # 5. Uncertainty notes
    if confidence:
        uncertainty_notes.extend(confidence.get("reasons", []))
    else:
        uncertainty_notes.append("Estimates reflect standard surface meteorological conditions; localized sun/shade variations affect microclimate exposure.")

    # Backward-compatible plain English summary
    summary = generate_summary(
        location=location,
        temp_c=temp_c,
        wbgt_c=wbgt_c,
        wbgt_band=wbgt_band,
        imd_status=imd_status,
        base_risk=base_risk,
        final_risk=final_risk,
        vulnerability=vulnerability,
        age_group=age_group,
        experience=experience,
        workload=workload,
        clothing=clothing,
    )

    return {
        "summary": summary,
        "environmental_drivers": env_drivers,
        "operational_drivers": op_drivers,
        "vulnerability_drivers": vuln_drivers,
        "protective_factors": protective_factors,
        "recommended_actions": actions,
        "uncertainty_notes": uncertainty_notes,
    }


def generate_summary(
    location: str,
    temp_c: float,
    wbgt_c: float,
    wbgt_band: str,
    imd_status: str,
    base_risk: str,
    final_risk: str,
    vulnerability: str,
    age_group: Optional[str] = None,
    experience: Optional[str] = None,
    workload: Optional[str] = None,
    clothing: Optional[str] = None
) -> str:
    """
    Generate a plain-English, transparent summary explaining why the risk result was produced.
    Preserved for strict backward compatibility with existing tests and UI templates.
    """
    parts = [f"Current temperature in {location} is {temp_c:.1f} °C."]

    # 1. IMD heatwave context
    if imd_status == "Severe Heatwave":
        parts.append("IMD heatwave alert: Severe Heatwave — temperatures significantly exceed regional thresholds.")
    elif imd_status == "Heatwave":
        parts.append("IMD heatwave alert: Heatwave — temperature is well above the regional threshold.")
    elif imd_status == "Heat Day":
        parts.append("IMD status: Heat Day — temperature meets or exceeds the regional threshold.")
    else:
        parts.append("No IMD heatwave threshold exceeded for this region.")

    # 2. WBGT environmental heat stress
    parts.append(
        f"Outdoor Wet-Bulb Globe Temperature (WBGT) is estimated at {wbgt_c:.1f} °C ({wbgt_band} zone), "
        f"factoring in air temperature, relative humidity, wind speed, and direct sunlight."
    )

    # 3. Personal physiological drivers
    factors = []
    if age_group in ("Under 18", "under_18", "<18"):
        factors.append("younger age (developing heat regulation)")
    elif age_group in ("60+", "60_plus", ">60", "60 plus"):
        factors.append("age 60+ (higher cardiovascular heat strain)")

    exp_clean = str(experience or "").lower()
    if exp_clean in ("not_used", "not", "unacclimatized"):
        factors.append("not being used to working in hot weather")
    elif exp_clean in ("somewhat_used", "partially"):
        factors.append("only partial experience working in hot weather")

    wl_clean = str(workload or "").lower()
    if wl_clean in ("very_heavy", "very heavy"):
        factors.append("very heavy physical exertion generating intense body heat")
    elif wl_clean == "heavy":
        factors.append("heavy physical labor increasing metabolic heat")

    cl_clean = str(clothing or "").lower()
    if cl_clean in ("ppe", "protective"):
        factors.append("wearing protective gear / PPE which severely restricts sweat evaporation")
    elif cl_clean in ("heavy", "heavy_work"):
        factors.append("wearing heavy work clothing which traps body heat")

    if vulnerability != "Low":
        reasons_text = f" driven by {', '.join(factors)}" if factors else ""
        if base_risk != final_risk:
            parts.append(
                f"Your personal vulnerability is assessed as {vulnerability.lower()}{reasons_text}, "
                f"escalating overall risk from environmental base level ({base_risk}) to {final_risk}."
            )
        else:
            parts.append(
                f"Your personal vulnerability is {vulnerability.lower()}{reasons_text}, "
                f"reinforcing the elevated {final_risk} risk level."
            )
    else:
        parts.append(
            f"Personal vulnerability is low (standard adult age, regular heat experience, and light clothing), "
            f"so your risk matches the environmental base level ({final_risk})."
        )

    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive Explainable Risk Engine & Decision Trace Architecture
# ──────────────────────────────────────────────────────────────────────────────

def classify_decision_factors(
    temp_c: float,
    wbgt_c: float,
    wbgt_band: str,
    imd_status: str,
    base_risk: str,
    final_risk: str,
    vulnerability: str,
    workload: Optional[str] = None,
    clothing: Optional[str] = None,
    experience: Optional[str] = None,
    recovery_environment: Optional[str] = None,
    is_sun_exposed: bool = True,
    duration_hours: Optional[float] = None,
    confidence: Optional[Dict[str, Any]] = None,
    peak_risk_window: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classifies risk factors into four rigorous biometeorological categories:
    1. PRIMARY DRIVERS: Factors materially and causally responsible for the final action level.
    2. CONTRIBUTING FACTORS: Factors that increase thermal strain or elevate pacing concern,
       without independently dictating the final action category.
    3. PROTECTIVE FACTORS: Actual conditions that reduce physiological heat gain or facilitate recovery.
    4. UNCERTAINTY / LIMITATIONS: Explicit caveats regarding measurement limitations and microclimates.
    5. ACTIONABLE LEVERS: Specific operational modifications relevant to this exact scenario.
    """
    primary_drivers: List[str] = []
    contributing_factors: List[str] = []
    protective_factors: List[str] = []
    uncertainty_limitations: List[str] = []
    actionable_levers: List[Dict[str, Any]] = []

    wl_norm = normalize_workload(workload)
    acc_clean = normalize_acclimatization(experience)
    rec_clean = normalize_recovery_environment(recovery_environment)
    cl_clean = str(clothing or "").lower()

    # 1. PRIMARY DRIVERS
    if final_risk in ("Very High", "Extreme"):
        if wbgt_c >= 30.0 or wbgt_band in ("Danger", "Extreme Danger"):
            primary_drivers.append(
                f"Severe environmental heat stress: outdoor WBGT of {wbgt_c:.1f}°C ({wbgt_band} band) exceeds sustained metabolic safety thresholds."
            )
        if imd_status in ("Severe Heatwave", "Heatwave"):
            primary_drivers.append(
                f"Regional IMD alert: {imd_status} active with dry-bulb temperature ({temp_c:.1f}°C) significantly above regional threshold."
            )
        if wl_norm in ("heavy", "very_heavy"):
            watts = WORKLOAD_METABOLIC_WATTS.get(wl_norm, 350)
            primary_drivers.append(
                f"Strenuous physical exertion ({wl_norm}): high internal metabolic heat production (~{watts}W) severely compounds endogenous heat accumulation."
            )
        if final_risk != base_risk and vulnerability in ("Moderate", "High", "Critical"):
            primary_drivers.append(
                f"Individual vulnerability ({vulnerability}) escalated overall action level above environmental baseline ({base_risk}) to {final_risk}."
            )
    elif final_risk == "High":
        if wbgt_c >= 28.0 or wbgt_band in ("Extreme Caution", "Danger"):
            primary_drivers.append(
                f"Elevated environmental heat stress: outdoor WBGT of {wbgt_c:.1f}°C ({wbgt_band} band) imposes substantial cardiovascular cooling demand."
            )
        if imd_status in ("Heatwave", "Heat Day"):
            primary_drivers.append(
                f"Regional IMD condition: {imd_status} active with ambient temperature at {temp_c:.1f}°C."
            )
        if wl_norm in ("heavy", "very_heavy"):
            watts = WORKLOAD_METABOLIC_WATTS.get(wl_norm, 350)
            primary_drivers.append(
                f"Heavy physical workload generates substantial metabolic heat (~{watts}W), compounding ambient heat gain."
            )
        if final_risk != base_risk and vulnerability in ("Moderate", "High", "Critical"):
            primary_drivers.append(
                f"Personal physiological vulnerability factors escalated action category from environmental baseline ({base_risk}) to High."
            )
    elif final_risk == "Moderate":
        primary_drivers.append(
            f"Moderate thermal burden: WBGT of {wbgt_c:.1f}°C ({wbgt_band} band) requires baseline occupational hydration and monitoring during sustained activity."
        )
        if imd_status == "Heat Day":
            primary_drivers.append(
                f"IMD Heat Day: ambient temperature ({temp_c:.1f}°C) approaching regional heatwave criteria."
            )
    else:  # Low
        primary_drivers.append(
            f"Favorable thermal conditions: WBGT of {wbgt_c:.1f}°C ({wbgt_band} band) remains comfortably within safe occupational thermoregulatory capacity."
        )

    # Consistency guarantee: primary_drivers must never be empty
    if not primary_drivers:
        primary_drivers.append(
            f"Combined environmental conditions (WBGT {wbgt_c:.1f}°C, {temp_c:.1f}°C) and {wl_norm} workload establish the {final_risk} action level."
        )

    # 2. CONTRIBUTING FACTORS
    if wl_norm in ("heavy", "very_heavy") and not any("metabolic heat" in d for d in primary_drivers):
        watts = WORKLOAD_METABOLIC_WATTS.get(wl_norm, 350)
        contributing_factors.append(
            f"Physical exertion ({wl_norm}) adds internal metabolic heat load (~{watts}W)."
        )
    if any(k in cl_clean for k in ("ppe", "protective", "impermeable")):
        contributing_factors.append(
            "Protective gear / PPE restricts sweat evaporation and traps convective body heat."
        )
    elif any(k in cl_clean for k in ("heavy", "heavy_work")):
        contributing_factors.append(
            "Heavy work clothing adds thermal insulation, impeding skin heat dissipation."
        )
    if acc_clean == "unacclimatized":
        contributing_factors.append(
            "Unacclimatized physiological state: delayed sweat onset, higher electrolyte loss, and increased heart rate."
        )
    elif acc_clean == "partially_acclimatized":
        contributing_factors.append(
            "Partial heat acclimatization: cardiovascular and sweating adaptations remain incomplete."
        )
    if rec_clean in ("outdoor_direct", "warm_indoor"):
        contributing_factors.append(
            f"Recovery breaks in uncooled or warm area ({rec_clean.replace('_', ' ')}) prevent effective core body temperature cooling between work cycles."
        )
    if is_sun_exposed:
        contributing_factors.append(
            "Direct sunlight exposure adds radiant solar flux (~600–900 W/m² equivalent surface load)."
        )
    if duration_hours and duration_hours >= 6:
        contributing_factors.append(
            f"Prolonged shift duration ({duration_hours:.1f} hours) compounds cumulative physiological and hydration fatigue."
        )

    # 3. PROTECTIVE FACTORS
    if acc_clean == "acclimatized":
        protective_factors.append(
            "Full heat acclimatization: promotes earlier sweating onset, higher sweat volume, and reduced cardiac strain."
        )
    if wl_norm == "light":
        protective_factors.append(
            "Light physical activity generates minimal internal metabolic heat (~125–150W)."
        )
    if any(k in cl_clean for k in ("light", "normal", "breathable")) and not any(k in cl_clean for k in ("heavy", "ppe", "protective")):
        protective_factors.append(
            "Light, breathable cotton clothing facilitates unobstructed sweat evaporation."
        )
    if rec_clean in ("cool_indoor", "shade"):
        protective_factors.append(
            f"Recovery breaks in {rec_clean.replace('_', ' ')} facilitate active dissipation of accumulated core body temperature."
        )
    if not is_sun_exposed:
        protective_factors.append(
            "Work performed in shade/indoors eliminates direct radiant solar heat flux."
        )
    if imd_status == "Normal":
        protective_factors.append(
            f"Ambient air temperature ({temp_c:.1f}°C) is below regional IMD heatwave alert thresholds."
        )
    if vulnerability == "Low":
        protective_factors.append(
            "Low personal vulnerability profile (standard adult age, experienced worker)."
        )

    # 4. UNCERTAINTY / LIMITATIONS (Plain English for nontechnical users)
    if confidence:
        for r in confidence.get("reasons", []):
            r_lower = r.lower()
            if "solar" in r_lower or "secondary" in r_lower or "missing" in r_lower:
                uncertainty_limitations.append(
                    "Solar-radiation input is modeled from clear-sky equations rather than measured with an on-site pyranometer; cloud cover or shade will alter actual radiant heat load."
                )
            elif "forecast" in r_lower:
                uncertainty_limitations.append(
                    "Calculated from numerical weather prediction forecast models (~1–10 km grid); localized microclimates and urban heat islands can elevate on-site temperatures."
                )
            elif r not in uncertainty_limitations:
                uncertainty_limitations.append(r)
    if not uncertainty_limitations:
        uncertainty_limitations.append(
            "Estimates reflect standard surface meteorological observations. Local microclimate variations (e.g. asphalt radiation, enclosed spaces) can alter on-site heat burden."
        )
    uncertainty_limitations.append(
        "Recommendations serve as occupational decision support (ISO 7243 / ACGIH / NDMA) and do not constitute individualized clinical medical prescriptions."
    )

    # 5. ACTIONABLE LEVERS (Only what is actually modifiable in this scenario)
    if wl_norm in ("heavy", "very_heavy"):
        actionable_levers.append({
            "lever": "workload",
            "title": "Reduce Work Intensity",
            "current": wl_norm.capitalize(),
            "suggested": "Moderate or Light",
            "action": "Mechanize heavy lifting or introduce rotating pairs to lower individual metabolic rate by ~150–250W."
        })
    if acc_clean != "acclimatized":
        actionable_levers.append({
            "lever": "acclimatization",
            "title": "Implement Acclimatization Protocol",
            "current": acc_clean.replace('_', ' ').capitalize(),
            "suggested": "Gradual 7–14 day progressive build-up",
            "action": "Limit initial hot exposure to 20% on Day 1, increasing by 20% daily to build cardiovascular tolerance safely."
        })
    if rec_clean not in ("cool_indoor", "shade"):
        actionable_levers.append({
            "lever": "recovery",
            "title": "Provide Shaded or Cooled Break Area",
            "current": rec_clean.replace('_', ' ').capitalize(),
            "suggested": "Shaded canopy or air-conditioned shelter",
            "action": "Set up a ventilated shaded rest shelter with fans and chilled drinking water (<15°C) to establish an active negative thermal gradient."
        })
    if any(k in cl_clean for k in ("heavy", "ppe")):
        actionable_levers.append({
            "lever": "clothing",
            "title": "Review Work Clothing / PPE",
            "current": cl_clean.capitalize(),
            "suggested": "Breathable cotton or evaporative cooling vest",
            "action": "Where safety permits, switch to loose-fitting, light-colored breathable fabrics to reduce evaporative resistance."
        })
    if is_sun_exposed:
        actionable_levers.append({
            "lever": "timing_solar",
            "title": "Shift Peak Exposure or Shade Worksite",
            "current": "Direct Sunlight Exposure",
            "suggested": "Temporary canopies or morning shift schedule",
            "action": "Erect portable tarpaulins/canopies over stationary workstations, or shift strenuous tasks away from midday solar peak."
        })

    return {
        "primary_drivers": primary_drivers,
        "contributing_factors": contributing_factors,
        "protective_factors": protective_factors,
        "uncertainty_limitations": uncertainty_limitations,
        "actionable_levers": actionable_levers,
    }


def build_decision_trace(
    weather_inputs: Dict[str, Any],
    derived_metrics: Dict[str, Any],
    classifications: Dict[str, Any],
    operational_inputs: Dict[str, Any],
    vulnerability_context: Dict[str, Any],
    triggered_rules: List[Dict[str, Any]],
    confidence_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constructs a deterministic, reproducible decision trace of the full HeatWave Guardian pipeline:
    INPUTS -> DERIVED CONDITIONS -> ENVIRONMENTAL LEVEL -> EXPOSURE CONTEXT -> ACTION LEVEL -> RECOMMENDED CONTROLS.
    """
    return {
        "trace_version": "1.0.0",
        "timestamp_evaluated": time.time(),
        "pipeline_stage": "COMPLETE",
        "inputs": {
            "temperature_c": weather_inputs.get("temp_c"),
            "relative_humidity_pct": weather_inputs.get("rh_pct"),
            "wind_speed_ms": weather_inputs.get("wind_ms"),
            "solar_radiation_wm2": weather_inputs.get("solar_wm2"),
            "location": weather_inputs.get("location"),
            "region": weather_inputs.get("region"),
            "is_forecast": weather_inputs.get("is_forecast", False),
            "workload": operational_inputs.get("workload"),
            "clothing": operational_inputs.get("clothing"),
            "ppe": operational_inputs.get("ppe_category"),
            "acclimatization": operational_inputs.get("experience") or operational_inputs.get("acclimatization"),
            "recovery_environment": operational_inputs.get("recovery_environment"),
            "exposure_duration_hours": operational_inputs.get("duration_hours"),
            "is_sun_exposed": operational_inputs.get("is_sun_exposed", True),
        },
        "derived_conditions": {
            "wbgt_c": derived_metrics.get("wbgt_c"),
            "wbgt_band": derived_metrics.get("wbgt_band"),
            "thermal_stress_level": derived_metrics.get("thermal_stress_level"),
            "imd_status": derived_metrics.get("imd_status"),
            "metabolic_watts": derived_metrics.get("metabolic_watts"),
            "clothing_adjustment_c": derived_metrics.get("clothing_adjustment_c"),
        },
        "classifications": {
            "environmental_base_risk": classifications.get("base_risk"),
            "operational_exposure_level": classifications.get("operational_exposure_level"),
            "vulnerability_level": classifications.get("vulnerability"),
            "overall_action_level": classifications.get("final_risk"),
            "recommendation_level": classifications.get("recommendation_level"),
        },
        "pipeline_stages": {
            "1_inputs": {
                "stage_name": "Inputs",
                "summary": f"{weather_inputs.get('temp_c')}°C, {weather_inputs.get('rh_pct')}% RH",
                "status": "OK"
            },
            "2_derived_conditions": {
                "stage_name": "Derived Conditions",
                "summary": f"WBGT {derived_metrics.get('wbgt_c')}°C ({derived_metrics.get('thermal_stress_level')})",
                "status": "OK"
            },
            "3_environmental_level": {
                "stage_name": "Environmental Level",
                "summary": f"{classifications.get('base_risk')} (IMD: {derived_metrics.get('imd_status')})",
                "status": "OK"
            },
            "4_exposure_context": {
                "stage_name": "Exposure Context",
                "summary": f"Workload: {operational_inputs.get('workload')}, Clothing: {operational_inputs.get('clothing')}",
                "status": "OK"
            },
            "5_action_level": {
                "stage_name": "Action Level",
                "summary": classifications.get('final_risk'),
                "status": "OK"
            },
            "6_recommended_controls": {
                "stage_name": "Recommended Controls",
                "summary": classifications.get('recommendation_level'),
                "status": "OK"
            },
        },
        "triggered_rules": triggered_rules,
        "data_quality_trace": confidence_data,
        "decision_support_statement": (
            "Deterministic decision trace generated from biometeorological modeling and occupational exposure rules "
            "(ISO 7243 / ACGIH / IMD). Not an individualized clinical medical diagnosis."
        )
    }


def build_traceable_recommendations(
    recommended_controls: List[Dict[str, Any]],
    final_risk: str,
    wbgt_c: float,
    workload: Optional[str] = None,
    peak_risk_window: Optional[Dict[str, Any]] = None,
    is_forecast: bool = False,
) -> List[Dict[str, Any]]:
    """
    Enriches each recommended control with traceable metadata:
    - category: "HYDRATION", "SCHEDULING", "RECOVERY", "WORKLOAD", "CLOTHING_PPE", "ACCLIMATIZATION", "MONITORING"
    - title: Human-readable title
    - action: Specific operational intervention
    - priority: "mandatory" | "priority" | "recommended" | "good_practice"
    - trigger: Specific physical condition that activated this recommendation
    - rule: Biometeorological standard or occupational rule reference
    - source: Data provenance (On-site observation vs Numerical forecast)
    - confidence: "High" | "Medium"
    - limitation: Microclimate or physiological qualification
    """
    rule_map = {
        "HYDRATION": ("ACGIH / NIOSH Fluid Replacement Protocol", "Ensure water temperature <15°C; consult medical advice for fluid-restricted workers."),
        "SCHEDULING": ("NDMA Heat Action Plan (2019) / ACGIH Guidelines", "Local microclimate and cloud cover may shift actual peak exposure time."),
        "RECOVERY": ("ISO 7243:2017 Work-Rest Thermal Equilibrium Criteria", "Uncooled rest areas delay core temperature recovery, accelerating physiological fatigue."),
        "WORKLOAD": ("ISO 8996 Metabolic Rate Determination", "Actual metabolic rate varies based on worker pacing and mechanical efficiency."),
        "CLOTHING_PPE": ("ISO 9920 Thermal Properties of Clothing", "Impermeable PPE creates an unventilated boundary layer regardless of ambient wind."),
        "ACCLIMATIZATION": ("NIOSH (2016) Occupational Heat Exposure Standard", "Acclimatization decays within 4–7 days away from hot conditions."),
        "MONITORING": ("OSHA Heat Illness Prevention Guidelines", "Early heat exhaustion signs (dizziness, nausea) must trigger immediate cessation."),
    }

    enriched = []
    for ctrl in recommended_controls:
        cat = ctrl.get("category", "RECOVERY")
        std_rule, std_limit = rule_map.get(cat, ("Occupational Heat Stress Screening Standard", "Screening guidance; on-site monitoring mandatory."))

        trigger_desc = ctrl.get("reason")
        if not trigger_desc:
            if cat == "HYDRATION":
                trigger_desc = f"Thermal stress level '{final_risk}' (WBGT {wbgt_c:.1f}°C) with {normalize_workload(workload)} exertion."
            elif cat == "SCHEDULING":
                peak_str = peak_risk_window.get("peak_period") if peak_risk_window else "midday hours"
                trigger_desc = f"Diurnal peak thermal load ({peak_str}) requiring task rescheduling."
            elif cat == "RECOVERY":
                trigger_desc = f"Work/rest cycle in {final_risk} action tier requiring active thermal dissipation."
            elif cat == "WORKLOAD":
                trigger_desc = f"Exertion level '{normalize_workload(workload)}' elevates internal metabolic heat."
            elif cat == "CLOTHING_PPE":
                trigger_desc = "Protective layers or high evaporative resistance detected."
            elif cat == "ACCLIMATIZATION":
                trigger_desc = "Worker acclimatization context warrants progressive pacing."
            else:
                trigger_desc = f"Active action level '{final_risk}'."

        enriched.append({
            "category": cat,
            "title": ctrl.get("title", f"{cat.capitalize()} Control"),
            "action": ctrl.get("action", ""),
            "priority": ctrl.get("priority", "recommended"),
            "trigger": trigger_desc,
            "rule": std_rule,
            "source": "Diurnal Numerical Forecast" if is_forecast else "Observed Surface Meteorological Profile",
            "confidence": "Medium" if is_forecast else "High",
            "limitation": std_limit,
        })
    return enriched


def generate_tiered_explanations(
    location: str,
    temp_c: float,
    wbgt_c: float,
    wbgt_band: str,
    imd_status: str,
    base_risk: str,
    final_risk: str,
    vulnerability: str,
    factors: Dict[str, Any],
    decision_trace: Dict[str, Any],
    workload: Optional[str] = None,
    recommended_controls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generates three distinct explanation depths:
    LEVEL 1 — QUICK: 1–2 plain-English sentences for workers / fast scanning.
    LEVEL 2 — DETAILED: Structured operational summary with drivers, protective factors, levers, controls.
    LEVEL 3 — TECHNICAL / AUDIT: Reproducible parameters, rule IDs, and data quality metrics for researchers.
    """
    wl_norm = normalize_workload(workload)

    # Level 1 Quick
    if final_risk == "Low":
        l1 = f"Low heat exposure in {location}: comfortable atmospheric conditions (WBGT {wbgt_c:.1f}°C) support standard work schedules."
    elif final_risk == "Moderate":
        l1 = f"Moderate heat exposure in {location} (WBGT {wbgt_c:.1f}°C): normal work pacing with scheduled hydration breaks is advised."
    elif final_risk == "High":
        l1 = f"High heat exposure in {location} because outdoor WBGT is elevated ({wbgt_c:.1f}°C, {wbgt_band} band) and {wl_norm} work generates significant internal body heat."
    elif final_risk == "Very High":
        l1 = f"Very High heat exposure in {location} because severe environmental heat (WBGT {wbgt_c:.1f}°C) and {wl_norm} workload create substantial thermal strain requiring frequent recovery."
    else:  # Extreme
        l1 = f"Extreme heat exposure in {location}: hazardous conditions (WBGT {wbgt_c:.1f}°C, {imd_status}) present severe heat-illness risk; defer non-essential strenuous work."

    # Level 2 Detailed
    l2 = {
        "overall_action_level": final_risk,
        "summary": l1,
        "primary_drivers": factors.get("primary_drivers", []),
        "contributing_factors": factors.get("contributing_factors", []),
        "protective_factors": factors.get("protective_factors", []),
        "actionable_levers": factors.get("actionable_levers", []),
        "recommended_controls": recommended_controls or [],
        "plain_english_uncertainty": factors.get("uncertainty_limitations", []),
    }

    # Level 3 Technical / Audit
    l3 = {
        "engine_version": "2.1.0-explainable",
        "decision_trace": decision_trace,
        "standards_referenced": [
            "ISO 7243:2017 (Hot environments — Estimation of heat stress using the WBGT index)",
            "ACGIH TLVs (2023 Threshold Limit Values for Chemical Substances and Physical Agents)",
            "IMD Operational Heatwave Criteria (Classification by climatological regional thresholds)",
            "NDMA National Heat Action Guidelines (2019 Interventions and Advisory Pacing)",
        ],
        "reproducibility": "Deterministic rule-based execution with zero stochastic generation.",
    }

    return {
        "level_1_quick": l1,
        "level_2_detailed": l2,
        "level_3_technical": l3,
    }


def evaluate_factor_sensitivities(
    weather_data: Dict[str, Any],
    personal_inputs: Dict[str, Any],
    region: str = "north",
    location: str = "Location",
) -> List[Dict[str, Any]]:
    """
    Evaluates factor sensitivity / counterfactual analysis under identical weather:
    Tests changes to Workload, Acclimatization, Recovery Environment, Clothing, and Solar Exposure.
    Classifies impact into:
    - CATEGORY_CHANGE: Modifies the final action category (e.g. Very High -> High).
    - RECOMMENDATION_CHANGE: Action category remains unchanged, but work/rest pacing expands or recovery improves.
    - NO_CHANGE: Modifying this factor alone does not alter the action category or work/rest pacing.
    Strictly derives explanations from biometeorological principles without fake percentages.
    """
    baseline_prof = assess_full_risk_profile(
        weather_data=weather_data,
        personal_inputs=personal_inputs,
        region=region,
        location=location,
        include_sensitivities=False,
    )
    b_level = baseline_prof["overall_action_level"]
    b_work = baseline_prof.get("work_minutes", 45)
    b_rest = baseline_prof.get("rest_minutes", 15)
    wbgt_c = baseline_prof.get("wbgt_c", 30.0)

    sensitivities = []

    # 1. Workload lever
    cur_wl = normalize_workload(personal_inputs.get("workload") or personal_inputs.get("work_intensity"))
    target_wl = "moderate" if cur_wl in ("heavy", "very_heavy") else ("light" if cur_wl == "moderate" else None)
    if target_wl:
        mod_inputs = dict(personal_inputs)
        mod_inputs["workload"] = target_wl
        mod_inputs["work_intensity"] = target_wl
        scen_prof = assess_full_risk_profile(weather_data, mod_inputs, region=region, location=location, include_sensitivities=False)
        s_level = scen_prof["overall_action_level"]
        s_work = scen_prof.get("work_minutes", 45)
        s_rest = scen_prof.get("rest_minutes", 15)

        b_watts = WORKLOAD_METABOLIC_WATTS.get(cur_wl, 350)
        s_watts = WORKLOAD_METABOLIC_WATTS.get(target_wl, 250)
        diff_watts = abs(b_watts - s_watts)

        if s_level != b_level:
            imp = "CATEGORY_CHANGE"
            badge = "Category Reduced"
            effect = f"Action level decreases from {b_level} to {s_level}."
        elif (s_work != b_work) or (s_rest != b_rest):
            imp = "RECOMMENDATION_CHANGE"
            badge = "Pacing Relieved"
            effect = f"Action level remains {b_level}, but permissible work time expands from {b_work}m to {s_work}m/hr ({s_rest}m rest)."
        else:
            imp = "NO_CHANGE"
            badge = "No Category Change"
            effect = f"Under current ambient conditions (WBGT {wbgt_c:.1f}°C), lowering workload alone would NOT change the {b_level} action category."

        sensitivities.append({
            "factor": "workload",
            "label": "Workload Reduction",
            "current_value": cur_wl.capitalize(),
            "counterfactual_value": target_wl.capitalize(),
            "impact_type": imp,
            "badge": badge,
            "effect_summary": effect,
            "physical_mechanism": f"Reduces internal metabolic heat production by ~{diff_watts}W (~{b_watts}W to ~{s_watts}W), easing cardiac and thermal strain.",
        })

    # 2. Acclimatization lever
    cur_acc = normalize_acclimatization(personal_inputs.get("experience") or personal_inputs.get("acclimatization"))
    if cur_acc != "acclimatized":
        mod_inputs = dict(personal_inputs)
        mod_inputs["experience"] = "very_used"
        mod_inputs["acclimatization"] = "acclimatized"
        scen_prof = assess_full_risk_profile(weather_data, mod_inputs, region=region, location=location, include_sensitivities=False)
        s_level = scen_prof["overall_action_level"]
        s_work = scen_prof.get("work_minutes", 45)
        s_rest = scen_prof.get("rest_minutes", 15)

        if s_level != b_level:
            imp = "CATEGORY_CHANGE"
            badge = "Category Reduced"
            effect = f"Action level decreases from {b_level} to {s_level}."
        elif (s_work != b_work) or (s_rest != b_rest):
            imp = "RECOMMENDATION_CHANGE"
            badge = "Pacing Relieved"
            effect = f"Action level remains {b_level}, but permissible work time expands from {b_work}m to {s_work}m/hr ({s_rest}m rest)."
        else:
            imp = "NO_CHANGE"
            badge = "No Category Change"
            effect = f"Under current ambient conditions (WBGT {wbgt_c:.1f}°C), full acclimatization alone would NOT change the {b_level} action category."

        sensitivities.append({
            "factor": "acclimatization",
            "label": "Full Acclimatization",
            "current_value": cur_acc.replace('_', ' ').capitalize(),
            "counterfactual_value": "Fully Acclimatized",
            "impact_type": imp,
            "badge": badge,
            "effect_summary": effect,
            "physical_mechanism": "Promotes earlier sweating onset, higher sweat volume, lower electrolyte loss, and lower heart rate under equivalent heat stress.",
        })

    # 3. Recovery environment lever
    cur_rec = normalize_recovery_environment(personal_inputs.get("recovery_environment") or personal_inputs.get("recovery"))
    target_rec = "cool_indoor" if cur_rec != "cool_indoor" else None
    if target_rec:
        mod_inputs = dict(personal_inputs)
        mod_inputs["recovery_environment"] = target_rec
        mod_inputs["recovery"] = target_rec
        scen_prof = assess_full_risk_profile(weather_data, mod_inputs, region=region, location=location, include_sensitivities=False)
        s_level = scen_prof["overall_action_level"]
        s_work = scen_prof.get("work_minutes", 45)
        s_rest = scen_prof.get("rest_minutes", 15)

        if s_level != b_level:
            imp = "CATEGORY_CHANGE"
            badge = "Category Reduced"
            effect = f"Action level decreases from {b_level} to {s_level}."
        elif (s_work != b_work) or (s_rest != b_rest):
            imp = "RECOMMENDATION_CHANGE"
            badge = "Pacing Relieved"
            effect = f"Action level remains {b_level}, but permissible work time expands from {b_work}m to {s_work}m/hr ({s_rest}m rest)."
        else:
            imp = "NO_CHANGE"
            badge = "No Category Change"
            effect = f"Under current severe ambient conditions (WBGT {wbgt_c:.1f}°C), air-conditioned recovery provides vital physiological cooling but would NOT change the overall {b_level} action category."

        sensitivities.append({
            "factor": "recovery",
            "label": "Cool Indoor / AC Recovery",
            "current_value": cur_rec.replace('_', ' ').capitalize(),
            "counterfactual_value": "Air-Conditioned Break Shelter",
            "impact_type": imp,
            "badge": badge,
            "effect_summary": effect,
            "physical_mechanism": "Establishes an active negative thermal gradient during rest breaks, effectively lowering core body temperature between exposure cycles.",
        })

    # 4. Clothing lever
    cur_cl = str(personal_inputs.get("clothing") or "").lower()
    if any(k in cur_cl for k in ("ppe", "protective", "heavy")):
        mod_inputs = dict(personal_inputs)
        mod_inputs["clothing"] = "light"
        scen_prof = assess_full_risk_profile(weather_data, mod_inputs, region=region, location=location, include_sensitivities=False)
        s_level = scen_prof["overall_action_level"]
        s_work = scen_prof.get("work_minutes", 45)
        s_rest = scen_prof.get("rest_minutes", 15)

        if s_level != b_level:
            imp = "CATEGORY_CHANGE"
            badge = "Category Reduced"
            effect = f"Action level decreases from {b_level} to {s_level}."
        elif (s_work != b_work) or (s_rest != b_rest):
            imp = "RECOMMENDATION_CHANGE"
            badge = "Pacing Relieved"
            effect = f"Action level remains {b_level}, but permissible work time expands from {b_work}m to {s_work}m/hr ({s_rest}m rest)."
        else:
            imp = "NO_CHANGE"
            badge = "No Category Change"
            effect = f"Under current thermal conditions (WBGT {wbgt_c:.1f}°C), lighter clothing alone would NOT change the {b_level} action category."

        sensitivities.append({
            "factor": "clothing",
            "label": "Breathable Light Clothing",
            "current_value": cur_cl.capitalize(),
            "counterfactual_value": "Light Cotton Workwear",
            "impact_type": imp,
            "badge": badge,
            "effect_summary": effect,
            "physical_mechanism": "Minimizes clothing evaporative resistance, restoring natural evaporative sweat heat dissipation from the skin surface.",
        })

    # 5. Solar exposure lever
    cur_sun = personal_inputs.get("is_sun_exposed", True)
    if cur_sun:
        mod_inputs = dict(personal_inputs)
        mod_inputs["is_sun_exposed"] = False
        scen_prof = assess_full_risk_profile(weather_data, mod_inputs, region=region, location=location, include_sensitivities=False)
        s_level = scen_prof["overall_action_level"]
        s_work = scen_prof.get("work_minutes", 45)
        s_rest = scen_prof.get("rest_minutes", 15)

        if s_level != b_level:
            imp = "CATEGORY_CHANGE"
            badge = "Category Reduced"
            effect = f"Action level decreases from {b_level} to {s_level}."
        elif (s_work != b_work) or (s_rest != b_rest):
            imp = "RECOMMENDATION_CHANGE"
            badge = "Pacing Relieved"
            effect = f"Action level remains {b_level}, but permissible work time expands from {b_work}m to {s_work}m/hr ({s_rest}m rest)."
        else:
            imp = "NO_CHANGE"
            badge = "No Category Change"
            effect = f"Under current ambient conditions (WBGT {wbgt_c:.1f}°C), eliminating direct sun exposure alone would NOT change the {b_level} action category."

        sensitivities.append({
            "factor": "solar_exposure",
            "label": "Shade / Indoor Worksite",
            "current_value": "Direct Sunlight",
            "counterfactual_value": "Covered / Shaded Worksite",
            "impact_type": imp,
            "badge": badge,
            "effect_summary": effect,
            "physical_mechanism": "Removes radiant solar flux (~600–900 W/m² equivalent surface load), reducing mean radiant temperature.",
        })

    return sensitivities


def build_decision_timeline(hourly_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synthesizes 24-hour hourly risk profiles into an operational decision timeline:
    1. hourly_timeline: 24 annotated points with time, wbgt, action level, primary driver, recommended control.
    2. operational_periods: Grouped into 5 distinct diurnal shift periods.
    """
    if not hourly_profiles:
        return {"hourly_timeline": [], "operational_periods": []}

    hourly_timeline = []
    risk_rank = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Extreme": 4}

    for p in hourly_profiles:
        hour_str = p.get("hour_str", "00:00")
        try:
            h_int = int(hour_str[:2])
            next_h = f"{h_int + 1:02d}:00" if h_int < 23 else "00:00"
            time_win = f"{hour_str}–{next_h}"
        except Exception:
            time_win = hour_str

        wbgt_c = p.get("wbgt_c", 25.0)
        temp_c = p.get("temp_c", 30.0)
        action_level = p.get("overall_action_level", "Moderate")
        controls = p.get("recommended_controls", [])
        top_ctrl = controls[0]["action"] if controls else "Standard pacing and scheduled hydration breaks."

        driver = p.get("primary_driver")
        if not driver:
            if wbgt_c >= 30.0:
                driver = f"Elevated solar-radiant load and ambient heat (WBGT {wbgt_c:.1f}°C)"
            elif wbgt_c >= 28.0:
                driver = f"Rising diurnal thermal stress (WBGT {wbgt_c:.1f}°C)"
            else:
                driver = f"Mild atmospheric conditions (WBGT {wbgt_c:.1f}°C)"

        hourly_timeline.append({
            "hour_str": hour_str,
            "time_window": time_win,
            "temp_c": temp_c,
            "wbgt_c": wbgt_c,
            "action_level": action_level,
            "primary_driver": driver,
            "recommended_control": top_ctrl,
            "is_peak": p.get("is_peak", False),
            "is_elevated_risk": p.get("is_elevated_risk", False),
            "is_lower_exposure": p.get("is_lower_exposure", False),
        })

    # Group into 5 operational diurnal shift periods (ISO / NDMA diurnal curve)
    period_defs = [
        ("early_morning", "Early Morning", 5, 8, "Optimal window for high-exertion tasks; lowest diurnal thermal stress."),
        ("morning_buildup", "Morning Build-up", 8, 11, "Rising ambient and solar load; increase hydration frequency."),
        ("peak_hazard", "Peak Diurnal Hazard", 11, 16, "Maximum solar and ambient load; enforce mandatory shaded recovery."),
        ("late_afternoon", "Late Afternoon", 16, 19, "Declining radiative heat; maintain vigilance for cumulative fatigue."),
        ("night_rest", "Night & Rest", 19, 5, "Nocturnal recovery; minimal occupational thermal stress."),
    ]

    operational_periods = []
    for pid, name, start_h, end_h, default_guidance in period_defs:
        pts = []
        for p in hourly_profiles:
            try:
                h_val = int(p.get("hour_str", "00:00")[:2])
                if pid == "night_rest":
                    if h_val >= 19 or h_val < 5:
                        pts.append(p)
                else:
                    if start_h <= h_val < end_h:
                        pts.append(p)
            except Exception:
                pass

        if not pts:
            continue

        max_pt = max(pts, key=lambda x: x.get("wbgt_c", 0.0))
        highest_level = max(pts, key=lambda x: risk_rank.get(x.get("overall_action_level", "Low"), 0)).get("overall_action_level", "Low")

        ctrls = max_pt.get("recommended_controls", [])
        top_act = ctrls[0]["action"] if (ctrls and isinstance(ctrls[0], dict) and "action" in ctrls[0]) else (ctrls[0] if ctrls and isinstance(ctrls[0], str) else default_guidance)

        operational_periods.append({
            "period_id": pid,
            "period_name": name,
            "name": name,
            "window": f"{start_h:02d}:00 - {end_h:02d}:00" if pid != "night_rest" else "19:00 - 05:00",
            "time_window": f"{start_h:02d}:00 - {end_h:02d}:00" if pid != "night_rest" else "19:00 - 05:00",
            "max_wbgt_c": max_pt.get("wbgt_c", 25.0),
            "dominant_action_level": highest_level,
            "dominant_category": highest_level,
            "primary_driver": max_pt.get("primary_driver", f"Peak WBGT {max_pt.get('wbgt_c', 25.0):.1f}°C in period"),
            "recommended_control": top_act,
            "is_peak_period": any(p.get("is_peak", False) for p in pts),
            "is_elevated_risk": any(p.get("is_elevated_risk", False) for p in pts),
        })

    return {
        "hourly_timeline": hourly_timeline,
        "operational_periods": operational_periods,
        "diurnal_periods": operational_periods,
    }


def build_day_decision_story(
    day_profile: Dict[str, Any],
    prev_day_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs a concise operational daily narrative and explains transitions from previous forecast day.
    """
    date_str = day_profile.get("date", "Today")
    peak_wbgt = day_profile.get("wbgt_c") or day_profile.get("max_wbgt_c", 28.0)
    action_level = day_profile.get("overall_action_level") or day_profile.get("risk_category", "Moderate")
    peak_win = day_profile.get("peak_risk_window") or {}
    peak_period = peak_win.get("peak_period") or peak_win.get("peak_hour") or "13:00 - 15:00"
    low_wins = day_profile.get("lower_exposure_windows") or []
    best_window = (low_wins[0].get("time_window") or low_wins[0].get("window") or "06:00 - 09:00") if (low_wins and isinstance(low_wins[0], dict)) else "06:00 - 09:00"

    drivers = day_profile.get("drivers") or []
    main_driver = drivers[0] if drivers else f"Diurnal peak WBGT of {peak_wbgt:.1f}°C"

    # Transition narrative
    if not prev_day_profile:
        trans_type = "BASELINE"
        delta_wbgt = 0.0
        cat_changed = False
        prev_level = action_level
        transition_msg = "Initial baseline day of the forecast period."
    else:
        prev_level = prev_day_profile.get("risk_category") or prev_day_profile.get("overall_action_level", "Moderate")
        prev_wbgt = prev_day_profile.get("wbgt_c") or prev_day_profile.get("max_wbgt_c", 28.0)
        delta_wbgt = round(peak_wbgt - prev_wbgt, 1)

        risk_rank = {
            "Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Extreme": 4,
            "Normal precautions": 0,
            "Enhanced precautions": 1,
            "Frequent recovery needed": 2,
            "High-exposure controls required": 3,
            "Suspend/defer strenuous exposure pending site assessment": 4,
        }
        p_idx = risk_rank.get(prev_level, 1)
        c_idx = risk_rank.get(action_level, 1)
        cat_changed = (c_idx != p_idx)

        if c_idx > p_idx or (c_idx == p_idx and delta_wbgt >= 1.0):
            trans_type = "ESCALATING"
            transition_msg = f"Action level escalated from {prev_level} to {action_level} because forecast peak WBGT increased by +{delta_wbgt}°C ({prev_wbgt:.1f}°C -> {peak_wbgt:.1f}°C) while workload remained unchanged."
        elif c_idx < p_idx or (c_idx == p_idx and delta_wbgt <= -1.0):
            trans_type = "IMPROVING"
            transition_msg = f"Action level improved from {prev_level} to {action_level} as atmospheric conditions cool (-{abs(delta_wbgt):.1f}°C peak WBGT)."
        else:
            trans_type = "STABLE"
            transition_msg = f"Action level remains stable at {action_level} with consistent thermal conditions (peak WBGT within ±{abs(delta_wbgt):.1f}°C)."

    transition_obj = {
        "type": trans_type,
        "message": transition_msg,
        "wbgt_delta_c": delta_wbgt,
        "category_changed": cat_changed,
        "previous_category": prev_level,
        "current_category": action_level,
    }

    story_text = (
        f"{date_str}: Peak thermal hazard is {action_level} (WBGT {peak_wbgt:.1f}°C) expected around {peak_period}. "
        f"Main driver: {main_driver}. Best operational window for strenuous outdoor work is {best_window}."
    )

    return {
        "date": date_str,
        "action_level": action_level,
        "peak_wbgt_c": peak_wbgt,
        "peak_period": peak_period,
        "best_window": best_window,
        "main_driver": main_driver,
        "story_text": story_text,
        "transition_from_previous": transition_obj,
        "transition_message": transition_msg,
    }


def explain_day_transition(prev_day: Dict[str, Any], next_day: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience alias to explain risk transition between two forecast days."""
    story = build_day_decision_story(next_day, prev_day_profile=prev_day)
    return story["transition_from_previous"]


# ──────────────────────────────────────────────────────────────────────────────
# Unified Comprehensive Risk Assessment (Full Pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def assess_full_risk_profile(
    weather_data: Dict[str, Any],
    personal_inputs: Dict[str, Any],
    region: str,
    location: str = "Delhi",
    is_forecast: bool = False,
    source: str = "Open-Meteo",
    include_sensitivities: bool = True,
) -> Dict[str, Any]:
    """
    Unified entry-point running the full HeatWave Guardian pipeline:
    OBSERVE -> MODEL -> PERSONALIZE -> EXPLAIN -> ACT -> VALIDATE

    Produces both the redesigned primary structured schema AND all legacy keys
    for 100% backward compatibility.
    """
    if not weather_data or not isinstance(weather_data, dict):
        raise ValueError("Valid weather observation dictionary is required to assess heat risk.")

    raw_temp = weather_data.get("temp_c") if weather_data.get("temp_c") is not None else weather_data.get("current_temp_c")
    if raw_temp is None:
        raise ValueError("Missing required weather parameter: temperature (temp_c) is required to assess heat risk.")
    temp_c = float(raw_temp)

    raw_rh = weather_data.get("rh_pct") if weather_data.get("rh_pct") is not None else weather_data.get("humidity")
    if raw_rh is None:
        raise ValueError("Missing required weather parameter: relative humidity (rh_pct) is required to assess heat risk.")
    rh_pct = float(raw_rh)

    wind_raw = weather_data.get("wind_ms") if weather_data.get("wind_ms") is not None else (float(weather_data.get("wind_kph")) / 3.6 if weather_data.get("wind_kph") is not None else 2.0)
    wind_ms = float(wind_raw)
    solar_wm2 = float(weather_data.get("solar_wm2", 0.0))

    # 1. Data quality & confidence evaluation
    confidence = evaluate_data_quality(weather_data, is_forecast=is_forecast, source=source, location=location)

    # 2. Environmental Heat Stress (Liljegren WBGT)
    thermal_stress = calculate_thermal_stress_index(temp_c, rh_pct, wind_ms, solar_wm2)
    wbgt_c = thermal_stress["wbgt_c"]
    wbgt_band = thermal_stress["wbgt_band"]
    thermal_stress_level = thermal_stress["thermal_stress_level"]
    thermal_stress_description = thermal_stress["thermal_stress_description"]

    # 3. IMD regional status
    imd_status = classify_imd(temp_c, region)

    # 4. Environmental Base Risk (Fusion)
    base_risk = fuse_risk(wbgt_band, imd_status)

    # 5. Operational Exposure & Context Evaluation
    workload = personal_inputs.get("workload") or personal_inputs.get("work_intensity") or "moderate"
    clothing = personal_inputs.get("clothing") or "light"
    ppe_category = personal_inputs.get("ppe_category")
    experience = personal_inputs.get("experience") or personal_inputs.get("acclimatization") or "very_used"
    is_sun_exposed = personal_inputs.get("is_sun_exposed", True)
    duration_hours = personal_inputs.get("duration_hours")
    exposure_pattern = personal_inputs.get("exposure_pattern")
    recovery_environment = personal_inputs.get("recovery_environment") or personal_inputs.get("recovery")

    operational_exp = evaluate_operational_exposure(
        workload=workload,
        clothing=clothing,
        acclimatization=experience,
        is_sun_exposed=is_sun_exposed,
        duration_hours=duration_hours
    )

    # 6. Physiological Vulnerability & Risk Escalation
    age = personal_inputs.get("age", 30)
    age_group = personal_inputs.get("age_group")
    acclimatization = personal_inputs.get("acclimatization") or experience

    vulnerability = assess_vulnerability(
        age=age,
        acclimatization=acclimatization,
        workload=workload,
        clothing=clothing,
        age_group=age_group,
    )
    final_risk = escalate_risk(base_risk, vulnerability)

    # 7. Acclimatization Pacing Schedule
    day_in_heat = personal_inputs.get("day_in_heat") or personal_inputs.get("acclimatization_day")
    acclimatization_schedule = evaluate_acclimatization_schedule(
        status=experience,
        day_in_heat=day_in_heat
    )

    # 8. Peak Risk Forecast Windows (from hourly forecast if available)
    hourly_series = weather_data.get("hourly") or weather_data.get("forecast_hourly")
    peak_risk_win = evaluate_peak_risk_windows(hourly_series)

    # 9. Explicit Exposure Context Model
    exposure_ctx = build_exposure_context(
        wbgt_c=wbgt_c,
        environmental_risk_level=base_risk,
        wind_ms=wind_ms,
        solar_wm2=solar_wm2,
        is_direct_sun=is_sun_exposed,
        workload=workload,
        exposure_duration_hours=duration_hours,
        exposure_pattern=exposure_pattern,
        acclimatization=experience,
        clothing=clothing,
        ppe_category=ppe_category,
        recovery_environment=recovery_environment,
    )

    # 10. Work/Rest Decision Support & Comprehensive Interventions
    work_rest = get_work_rest(
        final_risk,
        workload=workload,
        acclimatized=experience,
        clothing=clothing,
        recovery_environment=recovery_environment,
        exposure_duration_hours=duration_hours,
        exposure_pattern=exposure_pattern,
        ppe_category=ppe_category,
        peak_risk_window=peak_risk_win,
        wbgt_c=wbgt_c,
        wind_ms=wind_ms,
        solar_wm2=solar_wm2,
        is_direct_sun=is_sun_exposed,
    )

    # 11. Traceable Structured Explanation
    explanation = generate_structured_explanation(
        location=location,
        temp_c=temp_c,
        wbgt_c=wbgt_c,
        wbgt_band=wbgt_band,
        imd_status=imd_status,
        base_risk=base_risk,
        final_risk=final_risk,
        vulnerability=vulnerability,
        age_group=age_group,
        experience=experience,
        workload=workload,
        clothing=clothing,
        operational_exposure=operational_exp,
        confidence=confidence,
    )

    # 12. Explicit Decision Rule Trace Construction
    wl_norm = normalize_workload(workload)
    metabolic_watts = WORKLOAD_METABOLIC_WATTS.get(wl_norm, 250)
    clothing_adj = CLOTHING_ADJUSTMENT_VALUES.get(str(clothing or "").lower(), 0.0)

    triggered_rules = [
        {
            "rule_id": "RULE_THERMAL_STRESS_BAND",
            "standard": "ISO 7243:2017",
            "description": f"Liljegren WBGT evaluated at {wbgt_c:.1f}°C, classifying thermal stress into the '{wbgt_band}' band.",
            "inputs_evaluated": {"temp_c": round(temp_c, 1), "rh_pct": round(rh_pct, 1), "wind_ms": round(wind_ms, 1), "solar_wm2": round(solar_wm2, 1)},
            "outcome": wbgt_band,
        },
        {
            "rule_id": "RULE_IMD_REGIONAL",
            "standard": "IMD Operational Heatwave Criteria",
            "description": f"Air temperature ({temp_c:.1f}°C) evaluated against {region.capitalize()} threshold; assigned '{imd_status}'.",
            "inputs_evaluated": {"temp_c": round(temp_c, 1), "region": region},
            "outcome": imd_status,
        },
        {
            "rule_id": "RULE_RISK_FUSION",
            "standard": "HeatWave Guardian Dual-Sensor Fusion Matrix",
            "description": f"Fused WBGT band '{wbgt_band}' with IMD status '{imd_status}' to establish environmental base risk '{base_risk}'.",
            "inputs_evaluated": {"wbgt_band": wbgt_band, "imd_status": imd_status},
            "outcome": base_risk,
        },
        {
            "rule_id": "RULE_VULNERABILITY_ESCALATION",
            "standard": "Occupational Physiological Vulnerability Escalation Model",
            "description": f"Personal vulnerability evaluated as '{vulnerability}'; base risk '{base_risk}' escalated to final action level '{final_risk}'.",
            "inputs_evaluated": {"base_risk": base_risk, "vulnerability": vulnerability},
            "outcome": final_risk,
        },
        {
            "rule_id": "RULE_WORK_REST_REGIME",
            "standard": "ISO 7243 / ACGIH Heat Stress TLV Table",
            "description": f"Final action level '{final_risk}' and workload '{workload}' mandate {work_rest['work_minutes']}m work / {work_rest['rest_minutes']}m rest per hour.",
            "inputs_evaluated": {"action_level": final_risk, "workload": workload, "wbgt_c": wbgt_c},
            "outcome": f"{work_rest['work_minutes']}m work / {work_rest['rest_minutes']}m rest",
        },
    ]

    # 13. Build Structured Decision Trace
    decision_trace = build_decision_trace(
        weather_inputs={
            "temp_c": temp_c, "rh_pct": rh_pct, "wind_ms": wind_ms, "solar_wm2": solar_wm2,
            "location": location, "region": region, "is_forecast": is_forecast
        },
        derived_metrics={
            "wbgt_c": wbgt_c, "wbgt_band": wbgt_band, "thermal_stress_level": thermal_stress_level,
            "imd_status": imd_status, "metabolic_watts": metabolic_watts,
            "clothing_adjustment_c": clothing_adj,
        },
        classifications={
            "base_risk": base_risk,
            "operational_exposure_level": operational_exp["operational_exposure_level"],
            "vulnerability": vulnerability,
            "final_risk": final_risk,
            "recommendation_level": work_rest["recommendation_level"],
        },
        operational_inputs=personal_inputs,
        vulnerability_context={"vulnerability": vulnerability, "age": age, "age_group": age_group},
        triggered_rules=triggered_rules,
        confidence_data=confidence,
    )

    # 14. Classify Factors & Actionable Levers
    decision_factors = classify_decision_factors(
        temp_c=temp_c,
        wbgt_c=wbgt_c,
        wbgt_band=wbgt_band,
        imd_status=imd_status,
        base_risk=base_risk,
        final_risk=final_risk,
        vulnerability=vulnerability,
        workload=workload,
        clothing=clothing,
        experience=experience,
        recovery_environment=recovery_environment,
        is_sun_exposed=is_sun_exposed,
        duration_hours=duration_hours,
        confidence=confidence,
        peak_risk_window=peak_risk_win,
    )

    # 15. Traceable Recommended Controls
    traceable_controls = build_traceable_recommendations(
        recommended_controls=work_rest.get("recommended_controls", []),
        final_risk=final_risk,
        wbgt_c=wbgt_c,
        workload=workload,
        peak_risk_window=peak_risk_win,
        is_forecast=is_forecast,
    )

    # 16. Tiered Explanations (Level 1, Level 2, Level 3)
    tiered_explanations = generate_tiered_explanations(
        location=location,
        temp_c=temp_c,
        wbgt_c=wbgt_c,
        wbgt_band=wbgt_band,
        imd_status=imd_status,
        base_risk=base_risk,
        final_risk=final_risk,
        vulnerability=vulnerability,
        factors=decision_factors,
        decision_trace=decision_trace,
        workload=workload,
        recommended_controls=traceable_controls,
    )

    # 17. Counterfactual Factor Sensitivities (prevent recursive loop)
    factor_sensitivities = []
    if include_sensitivities:
        factor_sensitivities = evaluate_factor_sensitivities(
            weather_data=weather_data,
            personal_inputs=personal_inputs,
            region=region,
            location=location,
        )

    # Consolidated output: Primary Redesigned Schema + Legacy Compatible Fields
    return {
        # --- Redesigned Primary Categorical & Explainable Schema ---
        "environmental_level": base_risk,
        "operational_exposure_level": operational_exp["operational_exposure_level"],
        "overall_action_level": final_risk,
        "recommendation_level": work_rest["recommendation_level"],
        "work_guidance": work_rest["work_guidance"],
        "recovery_guidance": work_rest["recovery_guidance"],
        "schedule_context": work_rest["schedule_context"],
        "drivers": explanation["environmental_drivers"] + explanation["operational_drivers"] + explanation["vulnerability_drivers"],
        "protective_factors": decision_factors["protective_factors"] if decision_factors["protective_factors"] else explanation["protective_factors"],
        "recommended_actions": explanation["recommended_actions"],
        "recommended_controls": traceable_controls,
        "recommendation_reasons": work_rest["recommendation_reasons"],
        "recommendation_confidence": work_rest["schedule_context"]["confidence"],
        "limitations": work_rest["limitations"],
        "peak_risk_window": peak_risk_win,
        "exposure_context": exposure_ctx,
        "confidence": confidence,
        "acclimatization_context": acclimatization_schedule,
        "operational_exposure": operational_exp,
        "work_rest_guidance": work_rest,
        "hydration_guidance": work_rest["hydration_guidance"],
        "structured_explanation": explanation,
        "is_decision_support": True,

        # --- Explainable Risk Engine & Decision Trace Extensions ---
        "decision_trace": decision_trace,
        "explanation": {
            "level_1_quick": tiered_explanations["level_1_quick"],
            "level_2_detailed": tiered_explanations["level_2_detailed"],
            "level_3_technical": tiered_explanations["level_3_technical"],
            "primary_drivers": decision_factors["primary_drivers"],
            "contributing_factors": decision_factors["contributing_factors"],
            "protective_factors": decision_factors["protective_factors"],
            "uncertainties": decision_factors["uncertainty_limitations"],
            "what_can_change": decision_factors["actionable_levers"],
            "factor_sensitivities": factor_sensitivities,
        },
        "primary_drivers": decision_factors["primary_drivers"],
        "contributing_factors": decision_factors["contributing_factors"],
        "actionable_levers": decision_factors["actionable_levers"],
        "factor_sensitivities": factor_sensitivities,
        "plain_english_uncertainty": decision_factors["uncertainty_limitations"],
        "level_1_quick": tiered_explanations["level_1_quick"],
        "disclaimer": (
            "Explanations represent the rule-based logic of the HeatWave Guardian decision-support system "
            "based on biometeorological principles (ISO 7243 / ACGIH / IMD). They do not establish individual clinical medical causality."
        ),

        # --- Legacy Backward-Compatible Fields ---
        "location": location,
        "current_temp_c": round(temp_c, 1),
        "environmental_conditions": thermal_stress["environmental_conditions"],
        "thermal_stress": thermal_stress,
        "thermal_stress_index": wbgt_c,
        "thermal_stress_level": thermal_stress_level,
        "thermal_stress_description": thermal_stress_description,
        "wbgt_c": wbgt_c,
        "wbgt_band": wbgt_band,
        "imd_status": imd_status,
        "environmental_risk": base_risk,
        "base_risk": base_risk,
        "personal_risk": final_risk,
        "risk_category": final_risk,
        "summary": explanation["summary"],
        "work_minutes": work_rest["work_minutes"],
        "rest_minutes": work_rest["rest_minutes"],
        "instruction": work_rest["instruction"],
        "risk_color": RISK_COLORS.get(final_risk, "#2f9e44"),
        "severity": RISK_LEVELS.index(final_risk) + 1 if final_risk in RISK_LEVELS else 1,
        "age_group": age_group,
        "experience": experience,
        "work_intensity": workload,
        "clothing": clothing,
        "region": region,
        "vulnerability": vulnerability,
        "vuln_level": vulnerability,
    }


def calculate_hourly_risk_profile(
    hourly_points: List[Dict[str, Any]],
    region: str = "north",
    personal_inputs: Optional[Dict[str, Any]] = None,
    location: str = "Location",
) -> List[Dict[str, Any]]:
    """
    Compute full hourly occupational heat risk profiles across forecast series.
    Normalizes/calculates WBGT per hour and evaluates assess_full_risk_profile.
    Tags peak, elevated, and daytime lower-exposure periods.
    """
    if not hourly_points or not isinstance(hourly_points, list):
        return []

    resolved_personal = resolve_scenario_inputs(personal_inputs)
    is_sun_exposed = resolved_personal.get("is_sun_exposed", True)

    profiles = []
    for pt in hourly_points:
        if not isinstance(pt, dict):
            continue
        time_raw = pt.get("time") or pt.get("hour") or pt.get("time_str") or ""
        hour_str = str(time_raw).split("T")[-1][:5] if "T" in str(time_raw) else str(time_raw)[:5]

        temp_c = pt.get("temp_c")
        if temp_c is None:
            temp_c = pt.get("current_temp_c", 30.0)
        try:
            temp_c = float(temp_c)
        except (ValueError, TypeError):
            temp_c = 30.0

        humidity = pt.get("humidity")
        if humidity is None:
            humidity = pt.get("rh_pct")
        try:
            humidity = float(humidity) if humidity is not None else 50.0
        except (ValueError, TypeError):
            humidity = 50.0

        solar_wm2 = pt.get("solar_wm2")
        try:
            solar_wm2 = float(solar_wm2) if solar_wm2 is not None else 500.0
        except (ValueError, TypeError):
            solar_wm2 = 500.0

        wind_kph = pt.get("wind_kph")
        if wind_kph is None and pt.get("wind_ms") is not None:
            try:
                wind_kph = float(pt["wind_ms"]) * 3.6
            except (ValueError, TypeError):
                wind_kph = 10.0
        try:
            wind_kph = float(wind_kph) if wind_kph is not None else 10.0
        except (ValueError, TypeError):
            wind_kph = 10.0
        wind_ms = wind_kph / 3.6

        # WBGT: preserve or calculate
        wbgt_val = pt.get("wbgt_c")
        if wbgt_val is None:
            wbgt_val = pt.get("peak_wbgt_c")
        if wbgt_val is not None:
            try:
                wbgt_c = round(float(wbgt_val), 1)
            except (ValueError, TypeError):
                wbgt_c = round(calculate_wbgt(temp_c=temp_c, humidity=humidity, solar_wm2=solar_wm2, wind_ms=wind_ms, is_sun_exposed=is_sun_exposed), 1)
        else:
            wbgt_c = round(calculate_wbgt(
                temp_c=temp_c,
                humidity=humidity,
                solar_wm2=solar_wm2,
                wind_ms=wind_ms,
                is_sun_exposed=is_sun_exposed,
            ), 1)

        hour_weather = {
            "current_temp_c": temp_c,
            "temp_c": temp_c,
            "humidity": humidity,
            "rh_pct": humidity,
            "solar_wm2": solar_wm2,
            "wind_kph": wind_kph,
            "wind_ms": wind_ms,
            "wbgt_c": wbgt_c,
            "time": str(time_raw),
            "condition": pt.get("condition", "Clear"),
        }

        prof = assess_full_risk_profile(
            weather_data=hour_weather,
            personal_inputs=resolved_personal,
            region=region,
            location=location,
            include_sensitivities=False,
        )

        p_drivers = prof.get("primary_drivers") or prof.get("drivers", [])
        action_level = prof.get("overall_action_level", "Moderate")
        ml_loc_t = pt.get("ml_localized_temp_c", pt.get("ml_temp_c", temp_c))
        bg_t = pt.get("background_temp_c", temp_c)
        ml_anom = pt.get("ml_anomaly_c", 0.0)
        uncertainty_val = pt.get("uncertainty", {})
        is_peak_val = bool(pt.get("is_peak", False))
        is_lower_val = bool(pt.get("is_lower_heat_stress_window", False) or pt.get("is_lower_exposure", False))
        primary_driver = p_drivers[0] if p_drivers else f"WBGT {wbgt_c:.1f}°C"

        profiles.append({
            "time": str(time_raw),
            "hour_str": hour_str,
            "temp_c": temp_c,
            "background_temp_c": bg_t,
            "ml_localized_temp_c": ml_loc_t,
            "ml_temp_c": ml_loc_t,
            "ml_anomaly_c": ml_anom,
            "humidity": humidity,
            "wind_kph": wind_kph,
            "solar_wm2": solar_wm2,
            "wbgt_c": wbgt_c,
            "wbgt_band": prof.get("wbgt_band", "unknown"),
            "risk_category": action_level,
            "uncertainty": uncertainty_val,
            "environmental_level": prof.get("environmental_level", "Moderate"),
            "operational_exposure_level": prof.get("operational_exposure_level", "Moderate"),
            "overall_action_level": action_level,
            "recommendation_level": prof.get("recommendation_level", "Standard"),
            "work_minutes": prof.get("work_minutes", 45),
            "rest_minutes": prof.get("rest_minutes", 15),
            "hydration_rate_ml_hr": prof.get("hydration_guidance", {}).get("hourly_intake_ml", 500),
            "recommended_controls": prof.get("recommended_controls", []),
            "primary_driver": primary_driver,
            "is_peak": is_peak_val,
            "is_elevated_risk": is_peak_val,
            "is_lower_exposure": is_lower_val,
            "is_lower_heat_stress_window": is_lower_val,
        })

    if not profiles:
        return []

    if not any(p["is_peak"] for p in profiles):
        max_wbgt = max(p["wbgt_c"] for p in profiles)
        for p in profiles:
            if p["wbgt_c"] == max_wbgt:
                p["is_peak"] = True
                break
    else:
        max_wbgt = max(p["wbgt_c"] for p in profiles)

    for p in profiles:
        if p["wbgt_c"] >= 30.0 or (max_wbgt >= 28.0 and p["wbgt_c"] >= max(28.0, max_wbgt - 1.5)):
            p["is_elevated_risk"] = True
        try:
            h_num = int(p["hour_str"][:2])
            if 6 <= h_num <= 19 and p["wbgt_c"] < 28.0 and not p["is_peak"]:
                p["is_lower_exposure"] = True
                p["is_lower_heat_stress_window"] = True
        except Exception:
            pass

    return profiles


def simulate_exposure_scenario(
    baseline_inputs: Optional[Dict[str, Any]],
    scenario_inputs: Optional[Dict[str, Any]],
    weather_data: Dict[str, Any],
    region: str = "north",
    location: str = "Simulation Site",
) -> Dict[str, Any]:
    """
    Comparative What-If Exposure Simulator.
    Strictly evaluates baseline vs scenario using assess_full_risk_profile under
    identical environmental conditions. Derives causal explanations directly from
    underlying physiological/occupational model components without arbitrary percentages.
    """
    resolved_baseline = resolve_scenario_inputs(baseline_inputs)
    is_baseline_default = resolved_baseline.get("is_default_scenario", False)

    # Scenario inherits baseline as fallback
    resolved_scenario = resolve_scenario_inputs(scenario_inputs, fallback=resolved_baseline)

    # 1. Run core risk engine twice under identical weather conditions
    baseline_profile = assess_full_risk_profile(
        weather_data=weather_data,
        personal_inputs=resolved_baseline,
        region=region,
        location=location,
        include_sensitivities=False,
    )
    scenario_profile = assess_full_risk_profile(
        weather_data=weather_data,
        personal_inputs=resolved_scenario,
        region=region,
        location=location,
        include_sensitivities=False,
    )

    # 2. Track parameter changes
    param_keys = [
        ("workload", "Workload Intensity"),
        ("acclimatization", "Acclimatization Status"),
        ("clothing", "Clothing / PPE"),
        ("recovery_environment", "Recovery Environment"),
        ("is_sun_exposed", "Direct Solar Exposure"),
        ("duration_hours", "Shift Duration (Hours)"),
    ]
    parameter_changes = []
    for key, label in param_keys:
        b_val = resolved_baseline.get(key)
        s_val = resolved_scenario.get(key)
        if b_val != s_val:
            parameter_changes.append({
                "parameter": key,
                "label": label,
                "baseline_value": b_val,
                "scenario_value": s_val,
            })

    # 3. Action level comparison
    risk_rank = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Extreme": 4}
    b_level = baseline_profile.get("overall_action_level", "Moderate")
    s_level = scenario_profile.get("overall_action_level", "Moderate")
    b_idx = risk_rank.get(b_level, 1)
    s_idx = risk_rank.get(s_level, 1)
    delta_steps = s_idx - b_idx

    if delta_steps < 0:
        direction = "improved"
    elif delta_steps > 0:
        direction = "worsened"
    else:
        direction = "unchanged"

    action_level_change = {
        "baseline": b_level,
        "scenario": s_level,
        "delta_steps": delta_steps,
        "direction": direction,
    }

    # 4. Work/Rest delta
    b_work = baseline_profile.get("work_minutes", 45)
    s_work = scenario_profile.get("work_minutes", 45)
    b_rest = baseline_profile.get("rest_minutes", 15)
    s_rest = scenario_profile.get("rest_minutes", 15)
    work_rest_change = {
        "baseline_work_minutes": b_work,
        "scenario_work_minutes": s_work,
        "work_delta_minutes": s_work - b_work,
        "baseline_rest_minutes": b_rest,
        "scenario_rest_minutes": s_rest,
        "rest_delta_minutes": s_rest - b_rest,
    }

    # 5. Causal factor explanations (engine-derived, no fake percentages)
    causal_explanations = []

    # Workload causality
    b_workload = normalize_workload(resolved_baseline.get("workload"))
    s_workload = normalize_workload(resolved_scenario.get("workload"))
    if b_workload != s_workload:
        b_watts = WORKLOAD_METABOLIC_WATTS.get(b_workload, 250)
        s_watts = WORKLOAD_METABOLIC_WATTS.get(s_workload, 250)
        diff_watts = s_watts - b_watts
        if diff_watts < 0:
            causal_explanations.append(
                f"Workload reduction ({b_workload} → {s_workload}) decreases internal metabolic heat production by approximately {abs(diff_watts)}W (~{b_watts}W to ~{s_watts}W), easing endogenous cardiac load."
            )
        else:
            causal_explanations.append(
                f"Workload escalation ({b_workload} → {s_workload}) increases metabolic heat output by ~{diff_watts}W (~{b_watts}W to ~{s_watts}W), accelerating core heat accumulation."
            )

    # Acclimatization causality
    b_acc = normalize_acclimatization(resolved_baseline.get("acclimatization"))
    s_acc = normalize_acclimatization(resolved_scenario.get("acclimatization"))
    if b_acc != s_acc:
        if s_acc == "acclimatized":
            causal_explanations.append(
                "Acclimatized physiological state promotes earlier onset of sweating, higher sweat volume, lower electrolyte loss, and reduced cardiovascular strain under equivalent thermal stress."
            )
        elif s_acc == "unacclimatized":
            causal_explanations.append(
                "Unacclimatized status impairs thermoregulatory sweating efficiency and elevates cardiovascular strain, necessitating stricter pacing."
            )

    # Recovery causality
    b_rec = normalize_recovery_environment(resolved_baseline.get("recovery_environment"))
    s_rec = normalize_recovery_environment(resolved_scenario.get("recovery_environment"))
    if b_rec != s_rec:
        if s_rec in ("cool_indoor", "shade") and b_rec in ("outdoor_direct", "warm_indoor"):
            causal_explanations.append(
                f"Upgraded recovery environment ({b_rec} → {s_rec}) establishes an active negative thermal gradient during rest breaks, facilitating core body temperature dissipation."
            )
        elif s_rec in ("outdoor_direct", "warm_indoor") and b_rec in ("cool_indoor", "shade"):
            causal_explanations.append(
                f"Resting in an uncooled/outdoor environment ({s_rec}) impairs core cooling between work cycles, compounding cumulative physiological thermal fatigue."
            )

    # Clothing / PPE causality
    b_cloth = str(resolved_baseline.get("clothing", "")).lower()
    s_cloth = str(resolved_scenario.get("clothing", "")).lower()
    if b_cloth != s_cloth:
        if any(k in s_cloth for k in ("light", "standard")) and any(k in b_cloth for k in ("heavy", "ppe", "impermeable")):
            causal_explanations.append(
                "Switching to breathable, lighter clothing reduces evaporative resistance, maximizing the skin surface's latent heat rejection capacity."
            )
        elif any(k in s_cloth for k in ("heavy", "ppe", "impermeable")):
            causal_explanations.append(
                "Utilizing impermeable PPE or heavy protective layers traps convective and evaporative heat at the microclimate boundary, drastically limiting natural sweat cooling."
            )

    # Solar exposure causality
    b_sun = resolved_baseline.get("is_sun_exposed", True)
    s_sun = resolved_scenario.get("is_sun_exposed", True)
    if b_sun != s_sun:
        if not s_sun:
            causal_explanations.append(
                "Eliminating direct sunlight exposure removes radiant solar heating load (~600–900 W/m² equivalent direct surface flux), reducing mean radiant temperature."
            )
        else:
            causal_explanations.append(
                "Direct exposure to full sun adds radiant solar flux, compounding ambient dry-bulb temperature stress."
            )

    # If no parameter changed:
    if not parameter_changes:
        causal_explanations.append("Baseline and scenario exposure conditions are identical. No operational change observed.")

    # High-heat or low-heat boundary commentary:
    if direction == "unchanged" and parameter_changes:
        wbgt_val = weather_data.get("wbgt_c") or 25.0
        try:
            wbgt_val = float(wbgt_val)
        except (ValueError, TypeError):
            wbgt_val = 25.0
        if wbgt_val >= 32.0 or b_idx >= 3:
            causal_explanations.append(
                f"Note: Environmental heat hazard is severe ({b_level}); while the operational adjustment provides physiological relief, severe atmospheric conditions continue to dictate strict precautions."
            )
        elif wbgt_val < 25.0 or b_idx <= 1:
            causal_explanations.append(
                f"Note: Ambient thermal hazard is mild ({b_level}); the operational adjustment improves ergonomics and reduces cumulative fatigue, but the overall hazard category remains Low/Moderate."
            )

    # 6. Protective factor changes
    b_factors = baseline_profile.get("protective_factors", [])
    s_factors = scenario_profile.get("protective_factors", [])
    protective_factor_changes = {
        "added": [f for f in s_factors if f not in b_factors],
        "removed": [f for f in b_factors if f not in s_factors],
    }

    # 7. Synthesize concise transition explanation
    if direction == "improved":
        transition_explanation = f"Action level improved from {b_level} to {s_level}: " + " ".join(causal_explanations)
    elif direction == "worsened":
        transition_explanation = f"Action level escalated from {b_level} to {s_level}: " + " ".join(causal_explanations)
    else:
        transition_explanation = f"Action level remains {b_level}: " + " ".join(causal_explanations)

    return {
        "baseline_profile": baseline_profile,
        "scenario_profile": scenario_profile,
        "baseline_inputs_used": resolved_baseline,
        "scenario_inputs_used": resolved_scenario,
        "is_baseline_default": is_baseline_default,
        "parameter_changes": parameter_changes,
        "action_level_change": action_level_change,
        "work_rest_change": work_rest_change,
        "causal_explanations": causal_explanations,
        "transition_explanation": transition_explanation,
        "protective_factor_changes": protective_factor_changes,
        "decision_trace_baseline": baseline_profile.get("decision_trace"),
        "decision_trace_scenario": scenario_profile.get("decision_trace"),
        "disclaimer": "Simulations provide operational decision support based on ISO 7243 and ACGIH occupational heat balance principles. They do not constitute individualized clinical diagnoses. Continuous on-site hydration and symptom monitoring remain mandatory.",
    }
