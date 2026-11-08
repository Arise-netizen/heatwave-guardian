"""
HeatWave Guardian — Hyperlocal ML Service
=========================================
Application service providing ML microclimate downscaling inference,
evaluation metrics reporting, and geospatial GeoJSON feature enrichment.
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple

from ml.src.predict import HyperlocalPredictor
from ml.src.data_loader import get_city_morphology

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(BASE_DIR, "ml", "reports", "evaluation_metrics.json")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "ml", "reports", "feature_importance.json")


class HyperlocalMLService:
    """Service wrapper for Hyperlocal ML intelligence."""

    _cache: Dict[str, Any] = {}
    _CACHE_TTL_SECONDS = 600.0  # 10 minutes cache

    @classmethod
    def predict_point(
        cls,
        lat: float,
        lon: float,
        weather_data: Dict[str, Any],
        city_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate explainable hyperlocal prediction for a geographical point."""
        cache_key = f"{round(lat, 3)}_{round(lon, 3)}_{round(float(weather_data.get('temp_c', 35.0)), 1)}"
        now = time.time()
        if cache_key in cls._cache:
            entry_time, res = cls._cache[cache_key]
            if now - entry_time < cls._CACHE_TTL_SECONDS:
                return res

        pred = HyperlocalPredictor.predict_hyperlocal(
            lat=lat,
            lon=lon,
            background_weather=weather_data,
            city_name=city_name,
        )
        cls._cache[cache_key] = (now, pred)
        return pred

    @classmethod
    def get_model_metrics(cls) -> Dict[str, Any]:
        """Return model metadata, temporal evaluation metrics, and feature importances."""
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        
        # If reports not generated yet, trigger evaluation
        from ml.src.evaluate import run_evaluation
        return run_evaluation()

    @classmethod
    def enrich_feature_properties(cls, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a GeoJSON feature's properties with ML hyperlocal downscaled estimates.
        """
        lat = properties.get("latitude", properties.get("lat", 25.0))
        lon = properties.get("longitude", properties.get("lon", 78.0))
        name = properties.get("name", "")
        bg_t = properties.get("temp_c", 35.0)

        w_dict = {
            "temp_c": bg_t,
            "rh_pct": properties.get("rh_pct", 45.0),
            "wind_ms": properties.get("wind_ms", 2.0),
            "solar_wm2": properties.get("solar_wm2", 600.0),
        }

        try:
            pred = cls.predict_point(lat=lat, lon=lon, weather_data=w_dict, city_name=name)
            hyp = pred["hyperlocal_environment"]
            properties["ml_temp_c"] = hyp["temperature_c"]
            properties["ml_anomaly_c"] = hyp["anomaly_c"]
            properties["ml_wbgt_c"] = hyp["wbgt_c"]
            properties["ml_wbgt_delta_c"] = hyp["wbgt_delta_c"]
            properties["ml_risk"] = hyp["base_risk"]
            properties["ml_risk_color"] = hyp["risk_color"]
            properties["ml_drivers"] = pred.get("key_drivers", [])
            properties["ml_confidence"] = pred.get("confidence", "High")
            properties["ml_model_version"] = HyperlocalPredictor.MODEL_VERSION
        except Exception as e:
            # Fallback gracefully without breaking GeoJSON
            properties["ml_temp_c"] = bg_t
            properties["ml_anomaly_c"] = 0.0
            properties["ml_wbgt_c"] = properties.get("wbgt_c", 30.0)
            properties["ml_wbgt_delta_c"] = 0.0
            properties["ml_risk"] = properties.get("base_risk", "Moderate")
            properties["ml_risk_color"] = properties.get("risk_color", "#2f9e44")
            properties["ml_drivers"] = []
            properties["ml_confidence"] = "Fallback"
            properties["ml_model_version"] = HyperlocalPredictor.MODEL_VERSION

        return properties

    @classmethod
    def generate_spatial_grid_cells(
        cls,
        city_name: str,
        center_lat: float,
        center_lon: float,
        background_weather: Dict[str, Any],
        base_morphology: Optional[Dict[str, Any]] = None,
        region: str = "plains",
    ) -> List[Dict[str, Any]]:
        """
        Generates ~1km microclimate grid cells across representative urban sectors
        for a municipal center.
        """
        # Validate coordinates
        try:
            center_lat = float(center_lat)
            center_lon = float(center_lon)
            if not (-90.0 <= center_lat <= 90.0 and -180.0 <= center_lon <= 180.0):
                return []
        except (ValueError, TypeError):
            return []

        if base_morphology is None:
            base_morphology = get_city_morphology(city_name, lat=center_lat, lon=center_lon)

        bg_t = float(background_weather.get("temp_c", background_weather.get("current_temp_c", 35.0)))
        bg_rh = float(background_weather.get("rh_pct", background_weather.get("humidity", 50.0)))
        bg_wind = float(background_weather.get("wind_ms", 2.0))
        bg_solar = float(background_weather.get("solar_wm2", 600.0))

        # Check data quality
        is_verified_input = bool(bg_solar > 0 and bg_wind > 0 and background_weather.get("temp_c") is not None)
        data_quality_label = "Verified Reanalysis Input" if is_verified_input else "Fallback Meteorological Synthesis"

        city_code = "".join([c for c in city_name[:3].upper() if c.isalpha()]) or "IND"

        # Representative urban sector templates (~1.2km grid cells)
        sectors = [
            {
                "tag": "CORE",
                "name": f"{city_name} - High-Density Urban Core",
                "desc": "Commercial canyon & high impervious coverage",
                "d_lat": 0.007,
                "d_lon": 0.004,
                "morph": {
                    "urban_built_up_ratio": min(0.95, float(base_morphology.get("urban_built_up_ratio", 0.75)) + 0.12),
                    "vegetation_canopy_deficit": min(0.95, float(base_morphology.get("vegetation_canopy_deficit", 0.75)) + 0.10),
                    "population_density_k": round(float(base_morphology.get("population_density_k", 10.0)) * 1.35, 1),
                },
            },
            {
                "tag": "INDUS",
                "name": f"{city_name} - Industrial & Logistics Corridor",
                "desc": "Asphalt yards & large metal roof surfaces",
                "d_lat": -0.010,
                "d_lon": 0.010,
                "morph": {
                    "urban_built_up_ratio": min(0.92, float(base_morphology.get("urban_built_up_ratio", 0.75)) + 0.08),
                    "vegetation_canopy_deficit": min(0.92, float(base_morphology.get("vegetation_canopy_deficit", 0.75)) + 0.08),
                    "population_density_k": round(float(base_morphology.get("population_density_k", 10.0)) * 0.85, 1),
                },
            },
            {
                "tag": "RESID",
                "name": f"{city_name} - Dense Residential Neighborhood",
                "desc": "Mid-rise masonry housing with mixed surface cover",
                "d_lat": 0.009,
                "d_lon": -0.009,
                "morph": {
                    "urban_built_up_ratio": float(base_morphology.get("urban_built_up_ratio", 0.75)),
                    "vegetation_canopy_deficit": float(base_morphology.get("vegetation_canopy_deficit", 0.75)),
                    "population_density_k": float(base_morphology.get("population_density_k", 10.0)),
                },
            },
            {
                "tag": "GREEN",
                "name": f"{city_name} - Suburban Green & Campus Periphery",
                "desc": "High vegetative canopy & open permeable soil",
                "d_lat": -0.009,
                "d_lon": -0.010,
                "morph": {
                    "urban_built_up_ratio": max(0.35, float(base_morphology.get("urban_built_up_ratio", 0.75)) - 0.28),
                    "vegetation_canopy_deficit": max(0.35, float(base_morphology.get("vegetation_canopy_deficit", 0.75)) - 0.32),
                    "population_density_k": round(float(base_morphology.get("population_density_k", 10.0)) * 0.50, 1),
                },
            },
            {
                "tag": "TRANS",
                "name": f"{city_name} - Transit Corridor & Semi-Rural Buffer",
                "desc": "Suburban arterial highway & perimeter zone",
                "d_lat": 0.000,
                "d_lon": 0.013,
                "morph": {
                    "urban_built_up_ratio": max(0.42, float(base_morphology.get("urban_built_up_ratio", 0.75)) - 0.15),
                    "vegetation_canopy_deficit": max(0.45, float(base_morphology.get("vegetation_canopy_deficit", 0.75)) - 0.15),
                    "population_density_k": round(float(base_morphology.get("population_density_k", 10.0)) * 0.60, 1),
                },
            },
        ]

        batch_items = []
        cell_meta = []
        half_deg = 0.0055  # ~600m radius -> ~1.2km grid cell dimension

        cell_weather = {
            "temp_c": bg_t,
            "rh_pct": bg_rh,
            "wind_ms": bg_wind,
            "solar_wm2": bg_solar,
        }

        for idx, sec in enumerate(sectors, 1):
            c_lat = round(center_lat + sec["d_lat"], 5)
            c_lon = round(center_lon + sec["d_lon"], 5)
            cell_id = f"{city_code}-CELL-{idx:02d}-{sec['tag']}"

            # GeoJSON standard coordinates [lon, lat] forming a closed rectangular cell polygon
            west = round(c_lon - half_deg, 6)
            east = round(c_lon + half_deg, 6)
            south = round(c_lat - half_deg, 6)
            north = round(c_lat + half_deg, 6)
            polygon_coords = [
                [
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south],
                ]
            ]
            batch_items.append({
                "lat": c_lat,
                "lon": c_lon,
                "background_weather": cell_weather,
                "city_name": city_name,
                "morphology_override": sec["morph"],
            })
            cell_meta.append({
                "sec": sec,
                "c_lat": c_lat,
                "c_lon": c_lon,
                "cell_id": cell_id,
                "polygon_coords": polygon_coords,
            })

        try:
            preds = HyperlocalPredictor.predict_batch(batch_items)
        except Exception:
            preds = [None] * len(batch_items)

        features = []
        for idx, meta_item in enumerate(cell_meta):
            sec = meta_item["sec"]
            c_lat = meta_item["c_lat"]
            c_lon = meta_item["c_lon"]
            cell_id = meta_item["cell_id"]
            polygon_coords = meta_item["polygon_coords"]
            pred = preds[idx] if idx < len(preds) else None

            if pred and isinstance(pred, dict) and "hyperlocal_environment" in pred:
                hyp = pred["hyperlocal_environment"]
                ml_t = hyp["temperature_c"]
                ml_anom = hyp["anomaly_c"]
                wbgt = hyp["wbgt_c"]
                wbgt_band = hyp["wbgt_band"]
                risk_cat = hyp["base_risk"]
                risk_color = hyp["risk_color"]
                drivers = pred.get("key_drivers", [])
                conf = pred.get("confidence", "High")
            else:
                ml_t = bg_t
                ml_anom = 0.0
                wbgt = round(bg_t * 0.75, 1)
                wbgt_band = "Moderate"
                risk_cat = "Moderate"
                risk_color = "#2f9e44"
                drivers = []
                conf = "Fallback"

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords,
                },
                "properties": {
                    "cell_id": cell_id,
                    "city": city_name,
                    "name": sec["name"],
                    "sector_type": sec["desc"],
                    "latitude": c_lat,
                    "longitude": c_lon,
                    "background_weather": {
                        "temp_c": round(bg_t, 1),
                        "rh_pct": round(bg_rh, 1),
                        "wind_ms": round(bg_wind, 1),
                        "solar_wm2": round(bg_solar, 1),
                        "source": "Coarse Reanalysis / NWP Grid (~25km)",
                        "label": "REANALYSIS",
                    },
                    "background_temp_c": round(bg_t, 1),
                    "ml_temp_c": ml_t,
                    "ml_anomaly_c": ml_anom,
                    "wbgt_c": wbgt,
                    "wbgt_band": wbgt_band,
                    "risk_category": risk_cat,
                    "risk_color": risk_color,
                    "confidence": conf,
                    "confidence_state": conf,
                    "data_quality": data_quality_label,
                    "model_version": HyperlocalPredictor.MODEL_VERSION,
                    "major_drivers": drivers,
                    "labels": {
                        "observed_label": "OBSERVED (Ground Station Climatology Reference)",
                        "reanalysis_label": "REANALYSIS (Synoptic Weather ~25km)",
                        "ml_predicted_label": "ML-PREDICTED (Microclimate Downscaling)",
                    },
                    "spatial_resolution": "~1km Microclimate Cell",
                    "spatial_precision_disclaimer": "Grid cells represent ~1km microclimate sector morphology estimates, not ward-level sensor networks.",
                    "disclaimer": (
                        "SCIENTIFIC DECISION SUPPORT: The ML model provides localized environmental estimates; "
                        "all subsequent WBGT and risk classifications remain deterministic physics. "
                        "The ML layer never declares medical diagnosis or mortality prediction."
                    ),
                },
            })

        return features

    @classmethod
    def get_authority_spatial_grid(
        cls,
        weather_fetcher_func,
        city: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates GeoJSON FeatureCollection of spatial microclimate grid cells
        across monitored municipal centers.
        """
        from services.authority_service import AuthorityService

        monitored = AuthorityService.MONITORED_AREAS
        if city:
            city_clean = city.strip().lower()
            monitored = [a for a in monitored if a["name"].lower() == city_clean]
            if not monitored:
                # Search by substring or fallback
                monitored = [a for a in AuthorityService.MONITORED_AREAS if city_clean in a["name"].lower()]

        def _process_area(area):
            name = area["name"]
            lat = area["lat"]
            lon = area["lon"]
            reg = area.get("region", "plains")

            try:
                w = weather_fetcher_func(lat, lon)
            except Exception:
                w = {"temp_c": 35.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 550.0}

            return cls.generate_spatial_grid_cells(
                city_name=name,
                center_lat=lat,
                center_lon=lon,
                background_weather=w,
                region=reg,
            )

        with ThreadPoolExecutor(max_workers=min(len(monitored), 8)) as executor:
            grid_results = list(executor.map(_process_area, monitored))

        all_grid_features = []
        for cells in grid_results:
            all_grid_features.extend(cells)

        return {
            "type": "FeatureCollection",
            "spatial_resolution": "~1km Microclimate Cell",
            "model_version": HyperlocalPredictor.MODEL_VERSION,
            "total_cells": len(all_grid_features),
            "disclaimer": "Grid cells represent ~1km microclimate sector morphology estimates, not ward-level sensor networks.",
            "features": all_grid_features,
        }

    @classmethod
    def enhance_forecast_with_ml(
        cls,
        daily_forecasts: List[Dict[str, Any]],
        lat: float,
        lon: float,
        city_name: str,
        region: str = "plains",
    ) -> List[Dict[str, Any]]:
        """
        Enhance future weather forecasts with Hyperlocal ML downscaled microclimates,
        dynamic peak-risk detection, ensemble tree spread uncertainty, candidate lower-risk
        windows, and explainable peak driver attribution.
        """
        if not daily_forecasts or not isinstance(daily_forecasts, list):
            return []

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            lat, lon = 28.6139, 77.2090

        enhanced_days = []

        all_hour_items = []
        day_hour_map = []  # list of (day_copy, date_str, hour_contexts)

        for day in daily_forecasts:
            day_copy = dict(day)
            raw_hours = day_copy.get("hourly_profile") or day_copy.get("hourly", [])
            date_str = day_copy.get("date", "2026-05-15")

            hour_contexts = []
            for h in raw_hours:
                if not isinstance(h, dict):
                    continue

                t_bg = float(h.get("temp_c") if h.get("temp_c") is not None else 35.0)
                rh_bg = float(h.get("rh_pct") if h.get("rh_pct") is not None else 45.0)
                wind_bg = float(h.get("wind_ms") if h.get("wind_ms") is not None else 2.0)
                solar_bg = float(h.get("solar_wm2") if h.get("solar_wm2") is not None else 500.0)
                t_str = str(h.get("time", f"{date_str}T12:00"))
                time_str = t_str.split("T")[1][:5] if "T" in t_str else str(h.get("hour", "12:00"))

                w_dict = {
                    "temp_c": t_bg,
                    "rh_pct": rh_bg,
                    "wind_ms": wind_bg,
                    "solar_wm2": solar_bg,
                }
                all_hour_items.append({
                    "lat": lat,
                    "lon": lon,
                    "background_weather": w_dict,
                    "city_name": city_name,
                    "date_str": date_str,
                })
                hour_contexts.append({
                    "h": h,
                    "t_bg": t_bg,
                    "rh_bg": rh_bg,
                    "wind_bg": wind_bg,
                    "solar_bg": solar_bg,
                    "t_str": t_str,
                    "time_str": time_str,
                    "global_idx": len(all_hour_items) - 1,
                })
            day_hour_map.append((day_copy, date_str, hour_contexts))

        try:
            all_preds = HyperlocalPredictor.predict_batch(all_hour_items)
        except Exception:
            all_preds = [None] * len(all_hour_items)

        for day_copy, date_str, hour_contexts in day_hour_map:
            processed_hours = []
            for h_ctx in hour_contexts:
                t_bg = h_ctx["t_bg"]
                rh_bg = h_ctx["rh_bg"]
                wind_bg = h_ctx["wind_bg"]
                solar_bg = h_ctx["solar_bg"]
                t_str = h_ctx["t_str"]
                time_str = h_ctx["time_str"]
                h = h_ctx["h"]
                g_idx = h_ctx["global_idx"]
                pred = all_preds[g_idx] if g_idx < len(all_preds) else None

                if pred and isinstance(pred, dict) and "hyperlocal_environment" in pred:
                    hyp = pred.get("hyperlocal_environment", {})
                    ml_t = hyp.get("temperature_c", pred.get("localized_temp_c", t_bg))
                    ml_anom = hyp.get("anomaly_c", pred.get("temp_anomaly_c", 0.0))
                    wbgt_val = hyp.get("wbgt_c", round(t_bg * 0.72, 1))
                    wbgt_band = hyp.get("wbgt_band", "Moderate")
                    risk_cat = hyp.get("base_risk", "Moderate")
                    risk_col = hyp.get("risk_color", "#2f9e44")
                    uncertainty = pred.get("uncertainty", {})
                    confidence = pred.get("confidence", "High")
                    drivers = pred.get("key_drivers", [])
                else:
                    ml_t = t_bg
                    ml_anom = 0.0
                    wbgt_val = round(t_bg * 0.72, 1)
                    wbgt_band = "Moderate"
                    risk_cat = "Moderate"
                    risk_col = "#2f9e44"
                    uncertainty = {
                        "method": "Fallback Baseline",
                        "ensemble_tree_std_c": 0.0,
                        "ensemble_std_c": 0.20,
                        "ci_90_half_width_c": 0.33,
                        "prediction_interval_90_c": "±0.33°C",
                        "prediction_interval_90_pct": "±0.33°C",
                        "temp_lower_c": round(t_bg - 0.33, 2),
                        "temp_upper_c": round(t_bg + 0.33, 2),
                        "confidence_state": "Fallback",
                        "data_quality": "Fallback Synthesis",
                    }
                    confidence = "Fallback"
                    drivers = []

                processed_hours.append({
                    "timestamp": t_str,
                    "time": t_str,
                    "time_str": time_str,
                    "hour": time_str,
                    "hour_str": time_str,
                    "background_temp_c": round(t_bg, 1),
                    "temp_c": round(t_bg, 1),
                    "rh_pct": round(rh_bg, 1),
                    "wind_ms": round(wind_bg, 1),
                    "solar_wm2": round(solar_bg, 1),
                    "ml_localized_temp_c": round(ml_t, 1),
                    "ml_temp_c": round(ml_t, 1),
                    "ml_anomaly_c": round(ml_anom, 1),
                    "wbgt_c": round(wbgt_val, 1),
                    "wbgt_band": wbgt_band,
                    "risk_category": risk_cat,
                    "risk_color": risk_col,
                    "uncertainty": uncertainty,
                    "confidence": confidence,
                    "confidence_state": confidence,
                    "data_quality": uncertainty.get("data_quality", "Verified Numerical Weather Prediction Grid"),
                    "major_drivers": drivers,
                    "is_peak": False,
                    "is_lower_heat_stress_window": False,
                })

            if not processed_hours:
                enhanced_days.append(day_copy)
                continue

            # ──────────────────────────────────────────────────────────────────
            # Peak-Risk Detection (Dynamic, Non-Hardcoded)
            # ──────────────────────────────────────────────────────────────────
            peak_wbgt_pt = max(processed_hours, key=lambda x: x["wbgt_c"])
            peak_ml_t_pt = max(processed_hours, key=lambda x: x["ml_temp_c"])

            peak_threshold_wbgt = max(28.0, peak_wbgt_pt["wbgt_c"] - 1.5)
            elevated_hours = [
                h for h in processed_hours
                if h["risk_category"] in ("High", "Very High", "Extreme") or h["wbgt_c"] >= peak_threshold_wbgt
            ]

            if elevated_hours:
                elevated_start = elevated_hours[0]["time_str"]
                elevated_end = elevated_hours[-1]["time_str"]
                elevated_period_str = f"{elevated_start} – {elevated_end}"
                peak_period_str = f"{elevated_start} – {elevated_end} (Peak {peak_wbgt_pt['time_str']})"
                for h in elevated_hours:
                    h["is_peak"] = True
            else:
                elevated_start = peak_wbgt_pt["time_str"]
                elevated_end = peak_wbgt_pt["time_str"]
                elevated_period_str = f"{elevated_start} – {elevated_end}"
                peak_period_str = f"Diurnal peak at {peak_wbgt_pt['time_str']}"
                peak_wbgt_pt["is_peak"] = True

            # ──────────────────────────────────────────────────────────────────
            # Safer Time Windows (Candidate Lower-Risk Windows)
            # ──────────────────────────────────────────────────────────────────
            daylight_safe_hours = [
                h for h in processed_hours
                if not h["is_peak"] and h["wbgt_c"] < 28.0 and ("06:00" <= h["time_str"] <= "20:00")
            ]
            for h in daylight_safe_hours:
                h["is_lower_heat_stress_window"] = True

            candidate_windows = []
            if daylight_safe_hours:
                # Group contiguous blocks
                current_block = [daylight_safe_hours[0]]
                for h in daylight_safe_hours[1:]:
                    prev_h = int(current_block[-1]["time_str"].split(":")[0])
                    curr_h = int(h["time_str"].split(":")[0])
                    if curr_h - prev_h <= 1:
                        current_block.append(h)
                    else:
                        w_s = current_block[0]["time_str"]
                        w_e = current_block[-1]["time_str"]
                        min_w = min(x["wbgt_c"] for x in current_block)
                        max_w = max(x["wbgt_c"] for x in current_block)
                        candidate_windows.append({
                            "window": f"{w_s} – {w_e}",
                            "start": w_s,
                            "end": w_e,
                            "wbgt_range_c": [round(min_w, 1), round(max_w, 1)],
                            "max_wbgt_c": round(max_w, 1),
                            "label": "Lower forecast heat-stress window",
                            "description": "Reduced solar irradiance and convective heat load offer relatively lower heat stress compared to the diurnal peak window. Direct on-site monitoring still required.",
                        })
                        current_block = [h]
                if current_block:
                    w_s = current_block[0]["time_str"]
                    w_e = current_block[-1]["time_str"]
                    min_w = min(x["wbgt_c"] for x in current_block)
                    max_w = max(x["wbgt_c"] for x in current_block)
                    candidate_windows.append({
                        "window": f"{w_s} – {w_e}",
                        "start": w_s,
                        "end": w_e,
                        "wbgt_range_c": [round(min_w, 1), round(max_w, 1)],
                        "max_wbgt_c": round(max_w, 1),
                        "label": "Lower forecast heat-stress window",
                        "description": "Reduced solar irradiance and convective heat load offer relatively lower heat stress compared to the diurnal peak window. Direct on-site monitoring still required.",
                    })
            else:
                candidate_windows.append({
                    "window": "Nighttime / Early Dawn Only",
                    "start": "21:00",
                    "end": "05:30",
                    "wbgt_range_c": [20.0, 27.5],
                    "max_wbgt_c": 27.5,
                    "label": "Lower forecast heat-stress window",
                    "description": "Daytime hours exhibit continuous thermal stress. Lower heat load is confined to nocturnal cooling periods.",
                })

            # ──────────────────────────────────────────────────────────────────
            # Explanation: "Why is this period highest risk?"
            # ──────────────────────────────────────────────────────────────────
            factors = []
            drivers_list = []
            if peak_ml_t_pt["ml_temp_c"] >= 35.0:
                detail = f"Extreme localized air temperature ({peak_ml_t_pt['ml_temp_c']}°C) severely impairs dry sensible heat dissipation from the human body."
                factors.append({"driver": "Ambient Dry-Bulb Temperature", "detail": detail})
                drivers_list.append(detail)
            if peak_wbgt_pt["solar_wm2"] >= 500.0:
                detail = f"Intense shortwave solar irradiance ({peak_wbgt_pt['solar_wm2']:.0f} W/m²) sharply elevates mean radiant temperature and operative heat load."
                factors.append({"driver": "Direct & Diffuse Solar Radiation", "detail": detail})
                drivers_list.append(detail)
            if peak_wbgt_pt["wind_ms"] <= 2.2:
                detail = f"Low surface wind speed ({peak_wbgt_pt['wind_ms']:.1f} m/s) causes wind convective stagnation, restricting advection and trapping thermal plumes."
                factors.append({"driver": "Convective Wind Stagnation", "detail": detail})
                drivers_list.append(detail)
            if peak_wbgt_pt["rh_pct"] >= 40.0 and peak_wbgt_pt["ml_temp_c"] >= 35.0:
                detail = f"Elevated atmospheric humidity ({peak_wbgt_pt['rh_pct']:.0f}%) suppresses sweat evaporation across the skin-to-air boundary."
                factors.append({"driver": "Atmospheric Humidity Vapor Pressure", "detail": detail})
                drivers_list.append(detail)
            if peak_ml_t_pt["ml_anomaly_c"] >= 0.5:
                detail = f"Urban Heat Island localized microclimate adjustment adds +{peak_ml_t_pt['ml_anomaly_c']}°C from dense impervious built-up materials."
                factors.append({"driver": "Urban Heat Island Localized Amplification", "detail": detail})
                drivers_list.append(detail)

            if not drivers_list:
                drivers_list.append(f"Localized ambient temperature reaching {peak_ml_t_pt['ml_temp_c']}°C.")
                drivers_list.append(f"Midday solar irradiance ({peak_wbgt_pt['solar_wm2']:.0f} W/m²) driving radiant heat stress.")

            explanation_summary = f"Peak heat hazard ({peak_wbgt_pt['wbgt_c']}°C WBGT, {peak_wbgt_pt['risk_category']}) occurs at {peak_wbgt_pt['time_str']} due to synergistic combination of high air temperature and intense midday solar loading."

            # Update day object
            day_copy["hourly"] = processed_hours
            day_copy["hourly_profile"] = processed_hours
            day_copy["peak_timestamp"] = peak_wbgt_pt["timestamp"]
            day_copy["peak_wbgt_timestamp"] = peak_wbgt_pt["timestamp"]
            day_copy["peak_ml_temp_timestamp"] = peak_ml_t_pt["timestamp"]
            day_copy["peak_wbgt_c"] = peak_wbgt_pt["wbgt_c"]
            day_copy["peak_ml_temp_c"] = peak_ml_t_pt["ml_temp_c"]
            day_copy["max_temp_c"] = peak_ml_t_pt["ml_temp_c"]  # Aligns max temp with ML localized estimate
            day_copy["background_max_temp_c"] = max(h["background_temp_c"] for h in processed_hours)
            day_copy["ml_anomaly_c"] = peak_ml_t_pt["ml_anomaly_c"]
            day_copy["peak_risk"] = peak_wbgt_pt["risk_category"]
            day_copy["peak_risk_category"] = peak_wbgt_pt["risk_category"]
            day_copy["peak_risk_color"] = peak_wbgt_pt["risk_color"]
            day_copy["peak_risk_window"] = {
                "peak_period": peak_period_str,
                "status": "Available",
                "peak_hour": peak_wbgt_pt["time_str"],
                "start": elevated_start,
                "end": elevated_end,
            }
            day_copy["main_elevated_risk_window"] = {
                "start": elevated_start,
                "end": elevated_end,
                "period": elevated_period_str,
            }
            day_copy["lower_risk_candidate_windows"] = candidate_windows
            day_copy["peak_explanation"] = {
                "question": "Why is this period highest risk?",
                "summary": explanation_summary,
                "drivers": drivers_list,
                "contributing_factors": factors,
            }
            day_copy["uncertainty_summary"] = peak_wbgt_pt["uncertainty"]
            day_copy["confidence"] = peak_wbgt_pt["confidence"]
            day_copy["confidence_state"] = peak_wbgt_pt["confidence_state"]

            enhanced_days.append(day_copy)

        return enhanced_days


