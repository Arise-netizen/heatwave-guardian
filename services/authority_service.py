"""
HeatWave Guardian — Authority & Municipal Disaster Intelligence Service
=======================================================================
Aggregates area-wide geospatial heat stress, IMD heatwave departures,
and demographic vulnerability for city administrators, State Disaster
Management Authorities (SDMAs), and municipal corporations.

PIPELINE INTEGRITY:
All area risks are strictly calculated using the core HeatWave Guardian engine:
estimate_wbgt() -> classify_imd() -> fuse_risk() -> HAP integration.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from engine import (
    CITIES,
    CITY_REGIONS,
    estimate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    calculate_thermal_stress_index,
    classify_imd,
    fuse_risk,
    RISK_COLORS,
    RISK_LEVELS,
)
from services.demographic_service import DemographicService
from services.hap_service import HAPService
from services.hyperlocal_ml_service import HyperlocalMLService


class AuthorityService:
    # Key administrative municipal centers monitored across India
    MONITORED_AREAS = [
        {"name": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "region": "plains"},
        {"name": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "region": "plains"},
        {"name": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "region": "coastal"},
        {"name": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "region": "plains"},
        {"name": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "region": "plains"},
        {"name": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "region": "coastal"},
        {"name": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "region": "plains"},
        {"name": "Nagpur", "state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "region": "plains"},
        {"name": "Jaisalmer", "state": "Rajasthan", "lat": 26.9157, "lon": 70.9083, "region": "plains"},
        {"name": "Bhubaneswar", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "region": "coastal"},
        {"name": "Khammam", "state": "Telangana", "lat": 17.2473, "lon": 80.1514, "region": "plains"},
        {"name": "Varanasi", "state": "Uttar Pradesh", "lat": 25.3176, "lon": 82.9739, "region": "plains"},
        {"name": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "region": "plains"},
        {"name": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "region": "plains"},
    ]

    _cache: Dict[str, Any] = {}
    _cache_time: float = 0.0
    _last_fetcher = None
    _CACHE_TTL_SECONDS: float = 180.0  # 3 minutes TTL cache

    @classmethod
    def clear_cache(cls):
        """Manually clear in-memory cache (e.g. for testing)."""
        cls._cache = {}
        cls._cache_time = 0.0
        cls._last_fetcher = None

    @classmethod
    def get_area_heat_layer(cls, weather_fetcher_func, use_cache: bool = True) -> Dict[str, Any]:
        """
        Evaluate all monitored municipal areas using live weather and the core engine.
        Uses concurrent threads for fast response and short-lived caching.
        Returns GeoJSON-compatible FeatureCollection with full risk metadata, demographic proxies,
        and HAP alert levels.
        """
        now = time.time()
        if (
            use_cache
            and cls._cache
            and cls._last_fetcher is weather_fetcher_func
            and (now - cls._cache_time < cls._CACHE_TTL_SECONDS)
        ):
            return cls._cache

        features = []
        summary_counts = {
            "total_monitored": len(cls.MONITORED_AREAS),
            "red_alert": 0,
            "orange_alert": 0,
            "yellow_alert": 0,
            "green_normal": 0,
            "high_risk_zones": 0,
            "extreme_risk_zones": 0,
        }

        def _evaluate_area(area):
            lat = area["lat"]
            lon = area["lon"]
            name = area["name"]
            region = area["region"]

            try:
                w = weather_fetcher_func(lat, lon)
                if not w or not isinstance(w, dict) or w.get("temp_c") is None:
                    weather_available = False
                else:
                    weather_available = True
            except Exception:
                weather_available = False
                w = None

            demo = DemographicService.get_demographics(name, area["state"])

            if not weather_available or w is None:
                _feat = {
                    "feature": {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                        "properties": {
                            "name": name,
                            "state": area["state"],
                            "region_type": region,
                            "latitude": lat,
                            "longitude": lon,
                            "temp_c": None,
                            "wbgt_c": None,
                            "wbgt_band": "Unavailable",
                            "thermal_stress": None,
                            "thermal_stress_index": None,
                            "thermal_stress_level": "Unavailable",
                            "thermal_stress_description": "Current weather observations offline.",
                            "imd_status": "Unavailable",
                            "environmental_risk": "Unavailable",
                            "base_risk": "Unavailable",
                            "risk_color": "#94a3b8",
                            "severity": 0,
                            "weather_available": False,
                            "hap_level": "Unavailable",
                            "hap_title": "Weather Unavailable",
                            "hap_description": "Atmospheric observation offline. Risk calculations suspended.",
                            "public_action_summary": "Weather service offline. Exercise standard seasonal vigilance.",
                            "institutional_action_summary": "Rely on municipal manual observations.",
                            "demographics": demo,
                        }
                    },
                    "hap_level": "Unavailable",
                    "base_risk": "Unavailable",
                }
                return _feat

            temp_c = float(w["temp_c"])
            rh_pct = float(w["rh_pct"])
            wind_ms = float(w["wind_ms"])
            solar_wm2 = float(w.get("solar_wm2", 0.0))

            # 1. Authoritative Thermal Stress Index (Liljegren WBGT)
            thermal_stress = calculate_thermal_stress_index(temp_c, rh_pct, wind_ms, solar_wm2)
            wbgt_c = thermal_stress["wbgt_c"]
            wbgt_band = thermal_stress["wbgt_band"]
            thermal_stress_level = thermal_stress["thermal_stress_level"]
            thermal_stress_desc = thermal_stress["thermal_stress_description"]

            # 2. IMD status
            imd_status = classify_imd(temp_c, region)

            # 3. Base environmental risk
            base_risk = fuse_risk(wbgt_band, imd_status)

            # 4. HAP protocol evaluation
            hap = HAPService.evaluate_hap_status(base_risk, imd_status, wbgt_c)

            risk_color = RISK_COLORS.get(base_risk, "#2f9e44")

            _feat = {
                "feature": {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],  # GeoJSON standard: [lon, lat]
                    },
                    "properties": {
                        "name": name,
                        "state": area["state"],
                        "region_type": region,
                        "latitude": lat,
                        "longitude": lon,
                        "temp_c": round(temp_c, 1),
                        "wbgt_c": wbgt_c,
                        "wbgt_band": wbgt_band,
                        "thermal_stress": thermal_stress,
                        "thermal_stress_index": wbgt_c,
                        "thermal_stress_level": thermal_stress_level,
                        "thermal_stress_description": thermal_stress_desc,
                        "imd_status": imd_status,
                        "environmental_risk": base_risk,
                        "base_risk": base_risk,
                        "risk_color": risk_color,
                        "severity": RISK_LEVELS.index(base_risk) + 1 if base_risk in RISK_LEVELS else 1,
                        "weather_available": weather_available,
                        "hap_level": hap["alert_level"],
                        "hap_title": hap["alert_title"],
                        "hap_description": hap["description"],
                        "public_action_summary": hap["public_actions"][0] if hap["public_actions"] else "",
                        "institutional_action_summary": hap["institutional_actions"][0] if hap["institutional_actions"] else "",
                        "demographics": demo,
                    }
                },
                "hap_level": hap["alert_level"],
                "base_risk": base_risk,
            }

            # Enrich feature properties with ML Hyperlocal Downscaled Estimates
            _feat["feature"]["properties"] = HyperlocalMLService.enrich_feature_properties(_feat["feature"]["properties"])
            return _feat

        # Fetch concurrently
        with ThreadPoolExecutor(max_workers=min(len(cls.MONITORED_AREAS), 8)) as executor:
            results = list(executor.map(_evaluate_area, cls.MONITORED_AREAS))

        for res in results:
            features.append(res["feature"])
            hap_lvl = res["hap_level"]
            base_r = res["base_risk"]

            if hap_lvl == "RED":
                summary_counts["red_alert"] += 1
            elif hap_lvl == "ORANGE":
                summary_counts["orange_alert"] += 1
            elif hap_lvl == "YELLOW":
                summary_counts["yellow_alert"] += 1
            else:
                summary_counts["green_normal"] += 1

            if base_r in ("High",):
                summary_counts["high_risk_zones"] += 1
            elif base_r in ("Very High", "Extreme"):
                summary_counts["extreme_risk_zones"] += 1

        layer = {
            "type": "FeatureCollection",
            "summary": summary_counts,
            "features": features,
        }

        cls._cache = layer
        cls._cache_time = now
        cls._last_fetcher = weather_fetcher_func
        return layer

    @classmethod
    def get_spatial_grid_layer(cls, weather_fetcher_func, city: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate spatial microclimate grid layer (~1km cells) enriched with ML predictions,
        recalculated Liljegren WBGT, and explainable driver attribution.
        """
        return HyperlocalMLService.get_authority_spatial_grid(weather_fetcher_func, city=city)

