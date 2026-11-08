"""
HeatWave Guardian — Scenario Simulator & Decision Timeline Service
==================================================================
Deterministic and explainable What-If simulator service enabling users to modify
supported environmental and exposure variables and observe changes in:
1. Localized ML Temperature ($T_local$) and Ensemble Tree Spread Uncertainty
2. Liljegren Wet-Bulb Globe Temperature (WBGT)
3. IMD Heatwave Status & Dual-Sensor Environmental Risk
4. Personal & Occupational Exposure-Adjusted Risk (ISO 7243)
5. Decision Timeline with 4 Operational States:
   - NORMAL (Standard monitoring; explicitly never "guaranteed safe")
   - ENHANCED CONTROLS (Active hydration, shaded rest areas, buddy checks)
   - RESTRICT / MODIFY EXPOSURE (Pacing shifts e.g. 45/15 or 30/30, rotate out of sun)
   - STOP / POSTPONE (Halt strenuous outdoor labor, mandatory conditioned shelter)
6. Data Honesty Safeguards & Validation Range Warnings
"""

import math
from typing import Dict, Any, List, Optional, Tuple

from engine import (
    estimate_wbgt,
    calculate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    classify_imd,
    fuse_risk,
    assess_full_risk_profile,
    get_work_rest,
    normalize_workload,
    normalize_acclimatization,
    normalize_recovery_environment,
    WORKLOAD_METABOLIC_WATTS,
    CLOTHING_ADJUSTMENT_VALUES,
    CITIES,
)
from ml.src.predict import (
    HyperlocalPredictor,
    predict_hyperlocal,
    adjust_humidity_for_temperature_delta,
)


# ──────────────────────────────────────────────────────────────────────────────
# Validation Range Boundaries (Empirical Training & Physiological Limits)
# ──────────────────────────────────────────────────────────────────────────────
VALIDATION_LIMITS = {
    "temperature_c": {"min": 15.0, "max": 52.0, "unit": "°C", "name": "Ambient Temperature"},
    "relative_humidity_pct": {"min": 5.0, "max": 95.0, "unit": "%", "name": "Relative Humidity"},
    "wind_speed_ms": {"min": 0.1, "max": 25.0, "unit": "m/s", "name": "Wind Speed"},
    "solar_radiation_wm2": {"min": 0.0, "max": 1200.0, "unit": "W/m²", "name": "Solar Radiation"},
    "exposure_duration_hours": {"min": 0.5, "max": 16.0, "unit": "hours", "name": "Exposure Duration"},
}

# Operational States Mapping (ISO 7243 & ACGIH Alignment)
# Strictly never called "guaranteed safe"
OPERATIONAL_STATES = {
    "Low": {
        "state": "NORMAL",
        "badge_color": "emerald",
        "description": "Routine occupational health monitoring; ensure continuous ad-libitum hydration and standard resting facilities.",
        "safety_note": "Normal operational mode. Note: Absolute safety cannot be guaranteed for unconditioned or vulnerable individuals.",
    },
    "Moderate": {
        "state": "ENHANCED CONTROLS",
        "badge_color": "amber",
        "description": "Mandatory scheduled hydration breaks (minimum 250ml every 20 mins); ensure accessible shaded canopy rest stations; institute active peer buddy monitoring.",
        "safety_note": "Elevated thermal load. Early fatigue symptoms must trigger immediate rest under shade.",
    },
    "High": {
        "state": "RESTRICT / MODIFY EXPOSURE",
        "badge_color": "orange",
        "description": "Enforce strict rotational work pacing (e.g. 45/15 or 30/30 split); rotate workers away from direct solar exposure; postpone non-critical high-metabolic tasks.",
        "safety_note": "Significant thermal strain. High risk of heat exhaustion if pacing intervals are breached.",
    },
    "Very High": {
        "state": "STOP / POSTPONE",
        "badge_color": "red",
        "description": "Halt strenuous outdoor manual labor during peak thermal hours; shift work to cooler morning/evening shifts; mandate air-conditioned recovery shelter.",
        "safety_note": "Extreme hazard. Core body temperature increases rapidly without aggressive engineering controls.",
    },
    "Extreme": {
        "state": "STOP / POSTPONE",
        "badge_color": "purple",
        "description": "Immediate suspension of outdoor physical activities; enforce mandatory cool shelter recovery; activate site emergency thermal illness protocols.",
        "safety_note": "Critical physiological emergency threshold. High probability of life-threatening heat stroke.",
    },
}


class ScenarioSimulator:
    """Core simulation engine for What-If scenario analysis."""

    @classmethod
    def validate_inputs(
        cls,
        env_inputs: Dict[str, Any],
        exp_inputs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validates whether environmental and exposure inputs lie within the model's
        scientifically validated operating domain.
        Returns: (is_validated_range: bool, warnings: List[str])
        """
        warnings = []
        is_valid = True

        # Temperature
        t_val = env_inputs.get("temperature_c") or env_inputs.get("temp_c") or env_inputs.get("current_temp_c")
        if t_val is not None:
            try:
                t_float = float(t_val)
                lim = VALIDATION_LIMITS["temperature_c"]
                if t_float < lim["min"] or t_float > lim["max"]:
                    is_valid = False
                    warnings.append(
                        f"{lim['name']} ({t_float:.1f}{lim['unit']}) falls outside the validated model range "
                        f"({lim['min']:.0f} to {lim['max']:.0f}{lim['unit']}). Output is extrapolated and unvalidated."
                    )
            except (ValueError, TypeError):
                is_valid = False
                warnings.append("Temperature value could not be parsed as a numeric float.")

        # Relative Humidity
        rh_val = env_inputs.get("relative_humidity_pct") or env_inputs.get("humidity") or env_inputs.get("rh_pct")
        if rh_val is not None:
            try:
                rh_float = float(rh_val)
                lim = VALIDATION_LIMITS["relative_humidity_pct"]
                if rh_float < lim["min"] or rh_float > lim["max"]:
                    is_valid = False
                    warnings.append(
                        f"{lim['name']} ({rh_float:.1f}{lim['unit']}) is outside empirical limits "
                        f"({lim['min']:.0f} to {lim['max']:.0f}{lim['unit']})."
                    )
            except (ValueError, TypeError):
                is_valid = False
                warnings.append("Relative humidity value could not be parsed as a numeric float.")

        # Wind Speed
        w_val = env_inputs.get("wind_speed_ms") or env_inputs.get("wind_ms")
        if w_val is not None:
            try:
                w_float = float(w_val)
                lim = VALIDATION_LIMITS["wind_speed_ms"]
                if w_float < lim["min"] or w_float > lim["max"]:
                    is_valid = False
                    warnings.append(
                        f"{lim['name']} ({w_float:.1f}{lim['unit']}) is outside valid limits "
                        f"({lim['min']:.1f} to {lim['max']:.1f}{lim['unit']})."
                    )
            except (ValueError, TypeError):
                is_valid = False
                warnings.append("Wind speed value could not be parsed as a numeric float.")

        # Solar Radiation
        sol_val = env_inputs.get("solar_radiation_wm2") or env_inputs.get("solar_wm2")
        if sol_val is not None:
            try:
                sol_float = float(sol_val)
                lim = VALIDATION_LIMITS["solar_radiation_wm2"]
                if sol_float < lim["min"] or sol_float > lim["max"]:
                    is_valid = False
                    warnings.append(
                        f"{lim['name']} ({sol_float:.0f}{lim['unit']}) exceeds supported solar irradiance limits "
                        f"({lim['min']:.0f} to {lim['max']:.0f}{lim['unit']})."
                    )
            except (ValueError, TypeError):
                is_valid = False
                warnings.append("Solar radiation value could not be parsed as a numeric float.")

        # Exposure Duration
        if exp_inputs:
            dur_val = exp_inputs.get("exposure_duration_hours") or exp_inputs.get("duration_hours")
            if dur_val is not None:
                try:
                    dur_float = float(dur_val)
                    lim = VALIDATION_LIMITS["exposure_duration_hours"]
                    if dur_float < lim["min"] or dur_float > lim["max"]:
                        is_valid = False
                        warnings.append(
                            f"{lim['name']} ({dur_float:.1f}{lim['unit']}) exceeds supported shift durations "
                            f"({lim['min']:.1f} to {lim['max']:.1f}{lim['unit']})."
                        )
                except (ValueError, TypeError):
                    pass

        return is_valid, warnings

    @classmethod
    def resolve_environmental_inputs(
        cls,
        raw: Optional[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Normalizes and extracts environmental variables with robust fallbacks."""
        data = raw or {}
        fb = fallback or {}

        raw_t = data.get("temperature_c") if data.get("temperature_c") is not None else data.get("temp_c", data.get("current_temp_c", fb.get("temperature_c", fb.get("temp_c", 35.0))))
        try:
            temp_c = float(raw_t)
        except (ValueError, TypeError):
            temp_c = float(fb.get("temperature_c", 35.0))

        raw_rh = data.get("relative_humidity_pct") if data.get("relative_humidity_pct") is not None else data.get("humidity", data.get("rh_pct", fb.get("relative_humidity_pct", fb.get("rh_pct", 50.0))))
        try:
            rh_pct = float(raw_rh)
        except (ValueError, TypeError):
            rh_pct = float(fb.get("relative_humidity_pct", 50.0))

        raw_wind = data.get("wind_speed_ms") if data.get("wind_speed_ms") is not None else data.get("wind_ms")
        if raw_wind is None and data.get("wind_kph") is not None:
            try:
                raw_wind = float(data.get("wind_kph", 10.0)) / 3.6
            except (ValueError, TypeError):
                raw_wind = 2.0
        elif raw_wind is None:
            raw_wind = fb.get("wind_speed_ms", fb.get("wind_ms", 2.0))
        try:
            wind_ms = float(raw_wind)
        except (ValueError, TypeError):
            wind_ms = float(fb.get("wind_speed_ms", 2.0))

        raw_solar = data.get("solar_radiation_wm2") if data.get("solar_radiation_wm2") is not None else data.get("solar_wm2", fb.get("solar_radiation_wm2", fb.get("solar_wm2", 600.0)))
        try:
            solar_wm2 = float(raw_solar)
        except (ValueError, TypeError):
            solar_wm2 = float(fb.get("solar_radiation_wm2", 600.0))

        hour_val = data.get("forecast_time") or data.get("hour") or data.get("time") or fb.get("hour", 14)
        try:
            if isinstance(hour_val, str) and ":" in hour_val:
                hour = int(hour_val.split(":")[0].replace("T", " ").split()[-1])
            else:
                hour = int(hour_val)
        except (ValueError, TypeError):
            hour = int(fb.get("hour", 14))

        return {
            "temperature_c": temp_c,
            "relative_humidity_pct": rh_pct,
            "wind_speed_ms": wind_ms,
            "solar_radiation_wm2": solar_wm2,
            "hour": hour,
        }

    @classmethod
    def resolve_exposure_inputs(
        cls,
        raw: Optional[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normalizes and sanitizes operational/exposure variables."""
        d = raw or {}
        fb = fallback or {}

        workload = normalize_workload(d.get("workload") or d.get("workload_intensity") or fb.get("workload", "moderate"))
        clothing = str(d.get("clothing") or d.get("clothing_ppe") or fb.get("clothing", "light")).lower()
        acclimatization = normalize_acclimatization(d.get("acclimatization") or d.get("experience") or fb.get("acclimatization", "very_used"))
        recovery_env = normalize_recovery_environment(d.get("recovery_environment") or d.get("recovery") or fb.get("recovery_environment", "outdoor_direct"))

        dur = d.get("exposure_duration_hours") if d.get("exposure_duration_hours") is not None else d.get("duration_hours", fb.get("exposure_duration_hours", 4.0))
        try:
            duration_hours = max(0.5, min(16.0, float(dur)))
        except (ValueError, TypeError):
            duration_hours = 4.0

        start_t = d.get("work_start_time") or fb.get("work_start_time", "08:00")
        if isinstance(start_t, str) and ":" in start_t:
            work_start_time = start_t[:5]
        else:
            work_start_time = "08:00"

        is_sun = d.get("is_sun_exposed") if d.get("is_sun_exposed") is not None else fb.get("is_sun_exposed", recovery_env == "outdoor_direct")

        return {
            "workload": workload,
            "clothing": clothing,
            "acclimatization": acclimatization,
            "recovery_environment": recovery_env,
            "exposure_duration_hours": duration_hours,
            "work_start_time": work_start_time,
            "is_sun_exposed": bool(is_sun),
            "age": int(d.get("age", fb.get("age", 35))),
            "age_group": d.get("age_group", fb.get("age_group", "25-45")),
        }

    @classmethod
    def evaluate_condition_state(
        cls,
        env: Dict[str, float],
        exp: Dict[str, Any],
        location: str = "Hyderabad",
        region: str = "plains",
        is_baseline: bool = False,
        raw_env: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete pipeline:
        Inputs -> ML Localized Temp -> Liljegren WBGT -> IMD -> Base Risk -> Personal / Work Pacing -> Operational State.
        """
        t_bg = env["temperature_c"]
        rh = env["relative_humidity_pct"]
        wind = env["wind_speed_ms"]
        solar = env["solar_radiation_wm2"]
        hour = env["hour"]

        lat, lon = CITIES.get(location, (17.3850, 78.4867))

        # Check validation range against original raw inputs if provided
        is_valid_range, warnings = cls.validate_inputs(raw_env or env, exp)

        # 1. Hyperlocal ML Model prediction
        w_dict = {
            "temp_c": t_bg,
            "rh_pct": rh,
            "wind_ms": wind,
            "solar_wm2": solar,
            "hour": hour,
        }
        try:
            pred = HyperlocalPredictor.predict_hyperlocal(
                lat=lat,
                lon=lon,
                background_weather=w_dict,
                city_name=location,
            )
            hyp = pred.get("hyperlocal_environment", {})
            ml_temp_c = hyp.get("temperature_c", t_bg)
            ml_anomaly_c = hyp.get("anomaly_c", 0.0)
            uncertainty = pred.get("uncertainty", {})
            confidence_state = uncertainty.get("confidence_state", pred.get("confidence", "High"))
            data_quality = uncertainty.get("data_quality", "Verified Numerical Weather Prediction Grid")
            model_available = pred.get("model_available", True)
        except Exception:
            ml_temp_c = t_bg
            ml_anomaly_c = 0.0
            uncertainty = {
                "method": "Fallback Baseline",
                "ensemble_tree_std_c": 0.0,
                "ci_90_half_width_c": 0.0,
                "confidence_state": "Model Fallback",
                "data_quality": "Fallback Meteorological Synthesis",
            }
            confidence_state = "Model Fallback"
            data_quality = "Fallback Meteorological Synthesis"
            model_available = False

        # If outside validated range, honest suppression of high confidence
        if not is_valid_range:
            confidence_state = "Unvalidated (Out of Supported Range)"
            data_quality = "Unvalidated Hypothetical Scenario"
            uncertainty["confidence_state"] = confidence_state
            uncertainty["data_quality"] = data_quality

        # 2. Liljegren WBGT on localized temperature with atmospheric moisture conservation
        eff_solar = solar if exp.get("is_sun_exposed", True) else 0.0
        adjusted_local_rh = adjust_humidity_for_temperature_delta(t_bg, rh, ml_temp_c)
        wbgt_c = estimate_wbgt(
            temp_c=ml_temp_c,
            rh_pct=adjusted_local_rh,
            wind_ms=wind,
            solar_wm2=eff_solar,
        )
        wbgt_band = classify_wbgt_band(wbgt_c)
        thermal_stress = classify_thermal_stress(wbgt_c)

        # 3. IMD Heatwave classification
        imd_status = classify_imd(ml_temp_c, region)

        # 4. Environmental Base Risk
        base_risk = fuse_risk(wbgt_band, imd_status)

        # 5. Full Exposure-Adjusted Assessment
        engine_weather = {
            "temp_c": ml_temp_c,
            "current_temp_c": ml_temp_c,
            "rh_pct": adjusted_local_rh,
            "humidity": adjusted_local_rh,
            "wind_ms": wind,
            "solar_wm2": eff_solar,
            "wbgt_c": wbgt_c,
        }
        engine_personal = {
            "workload": exp.get("workload") or exp.get("workload_intensity", "moderate"),
            "clothing": exp.get("clothing") or exp.get("clothing_ppe", "light"),
            "experience": exp.get("acclimatization") or exp.get("experience", "very_used"),
            "acclimatization": exp.get("acclimatization") or exp.get("experience", "very_used"),
            "recovery_environment": exp.get("recovery_environment", "outdoor_direct"),
            "is_sun_exposed": bool(exp.get("is_sun_exposed", True)),
            "duration_hours": float(exp.get("exposure_duration_hours") or exp.get("duration_hours", 4.0)),
            "age": exp.get("age", 35),
            "age_group": exp.get("age_group", "25-45"),
        }

        full_profile = assess_full_risk_profile(
            weather_data=engine_weather,
            personal_inputs=engine_personal,
            region=region,
            location=location,
            include_sensitivities=False,
        )

        final_risk = full_profile.get("overall_action_level", full_profile.get("final_risk", base_risk))
        work_min = full_profile.get("work_minutes", 45)
        rest_min = full_profile.get("rest_minutes", 15)

        # 6. Operational State Determination
        state_info = OPERATIONAL_STATES.get(final_risk, OPERATIONAL_STATES["Moderate"])
        operational_state = state_info["state"]
        op_description = state_info["description"]
        safety_note = state_info["safety_note"]

        return {
            "label": "BASELINE FORECAST" if is_baseline else "SIMULATED SCENARIO",
            "source": "REANALYSIS_FORECAST" if is_baseline else "SIMULATED SCENARIO",
            "is_simulated": not is_baseline,
            "is_validated_range": is_valid_range,
            "validation_warnings": warnings,
            "environmental_conditions": {
                "background_temp_c": round(t_bg, 1),
                "relative_humidity_pct": round(rh, 1),
                "wind_speed_ms": round(wind, 1),
                "solar_radiation_wm2": round(solar, 0),
                "forecast_hour": hour,
            },
            "ml_local_estimate": {
                "localized_temp_c": round(ml_temp_c, 1),
                "ml_anomaly_c": round(ml_anomaly_c, 2),
                "model_available": model_available,
                "confidence_state": confidence_state,
                "data_quality": data_quality,
                "uncertainty": uncertainty,
            },
            "thermal_physics": {
                "wbgt_c": round(wbgt_c, 1),
                "wbgt_band": wbgt_band,
                "thermal_stress_level": thermal_stress.get("thermal_stress_level", "Moderate"),
                "imd_heatwave_status": imd_status,
                "environmental_base_risk": base_risk,
            },
            "exposure_assessment": {
                "workload_intensity": exp.get("workload") or exp.get("workload_intensity", "moderate"),
                "clothing_ppe": exp.get("clothing") or exp.get("clothing_ppe", "light"),
                "acclimatization": exp.get("acclimatization") or exp.get("experience", "very_used"),
                "recovery_environment": exp.get("recovery_environment", "outdoor_direct"),
                "exposure_duration_hours": float(exp.get("exposure_duration_hours") or exp.get("duration_hours", 4.0)),
                "work_start_time": exp.get("work_start_time", "08:00"),
                "is_sun_exposed": bool(exp.get("is_sun_exposed", True)),
                "overall_action_level": final_risk,
                "risk_category": final_risk,
                "iso_7243_work_rest": {
                    "work_minutes": work_min,
                    "rest_minutes": rest_min,
                    "pacing_label": f"{work_min} min work / {rest_min} min rest per hour",
                },
            },
            "operational_recommendation": {
                "operational_state": operational_state,
                "badge_color": state_info["badge_color"],
                "recommended_action": op_description,
                "safety_guideline": safety_note,
            },
            "full_profile": full_profile,
        }

    @classmethod
    def generate_decision_timeline(
        cls,
        base_env: Dict[str, float],
        scenario_env: Dict[str, float],
        exp_inputs: Dict[str, Any],
        location: str = "Hyderabad",
        region: str = "plains",
    ) -> List[Dict[str, Any]]:
        """
        Constructs an hour-by-hour decision timeline across the workday shift:
        Time -> Forecast Heat Stress -> ML Localized Estimate -> WBGT -> Risk -> Recommended Operational State.
        """
        start_str = exp_inputs.get("work_start_time", "08:00")
        try:
            start_hour = int(start_str.split(":")[0])
        except (ValueError, TypeError):
            start_hour = 8

        duration = int(math.ceil(float(exp_inputs.get("exposure_duration_hours", 8.0))))
        duration = max(1, min(14, duration))

        timeline = []
        for step in range(duration):
            curr_h = (start_hour + step) % 24
            time_label = f"{curr_h:02d}:00"

            # Diurnal atmospheric modulation around the scenario conditions
            # Solar curve peak at 13:00, temp peak at 14:30
            solar_factor = max(0.0, math.sin(math.pi * max(0, curr_h - 6) / 12)) if 6 <= curr_h <= 18 else 0.0
            temp_diurnal_offset = -3.5 * math.cos(math.pi * (curr_h - 4) / 11) if 4 <= curr_h <= 19 else -4.0
            
            # Step weather
            step_env = dict(scenario_env)
            step_env["hour"] = curr_h
            step_env["temperature_c"] = round(scenario_env["temperature_c"] + temp_diurnal_offset * 0.4, 1)
            step_env["solar_radiation_wm2"] = round(scenario_env["solar_radiation_wm2"] * solar_factor, 0)
            # RH inversely correlates with temp
            rh_mod = scenario_env["relative_humidity_pct"] - (temp_diurnal_offset * 1.5)
            step_env["relative_humidity_pct"] = max(10.0, min(95.0, round(rh_mod, 1)))

            step_eval = cls.evaluate_condition_state(
                env=step_env,
                exp=exp_inputs,
                location=location,
                region=region,
                is_baseline=False,
            )

            op_rec = step_eval["operational_recommendation"]
            wbgt_c = step_eval["thermal_physics"]["wbgt_c"]
            risk_cat = step_eval["exposure_assessment"]["risk_category"]
            pacing = step_eval["exposure_assessment"]["iso_7243_work_rest"]["pacing_label"]

            timeline.append({
                "time": time_label,
                "hour": curr_h,
                "forecast_heat_stress": f"{step_env['temperature_c']}°C ({step_env['relative_humidity_pct']}% RH)",
                "background_temp_c": step_env["temperature_c"],
                "ml_localized_temp_c": step_eval["ml_local_estimate"]["localized_temp_c"],
                "ml_anomaly_c": step_eval["ml_local_estimate"]["ml_anomaly_c"],
                "wbgt_c": wbgt_c,
                "risk_category": risk_cat,
                "operational_state": op_rec["operational_state"],
                "operational_badge_color": op_rec["badge_color"],
                "work_rest_pacing": pacing,
                "recommended_action": op_rec["recommended_action"],
                "is_peak": (12 <= curr_h <= 15),
            })

        return timeline

    @classmethod
    def compute_differences(
        cls,
        base_eval: Dict[str, Any],
        scen_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculates exact differences and metric deltas between two evaluated states."""
        b_env = base_eval["environmental_conditions"]
        s_env = scen_eval["environmental_conditions"]

        b_ml = base_eval["ml_local_estimate"]
        s_ml = scen_eval["ml_local_estimate"]

        b_phys = base_eval["thermal_physics"]
        s_phys = scen_eval["thermal_physics"]

        b_exp = base_eval["exposure_assessment"]
        s_exp = scen_eval["exposure_assessment"]

        # Category rankings
        rank_map = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Extreme": 4}
        b_rank = rank_map.get(b_exp["risk_category"], 1)
        s_rank = rank_map.get(s_exp["risk_category"], 1)
        delta_rank = s_rank - b_rank

        direction = "improved" if delta_rank < 0 else ("worsened" if delta_rank > 0 else "unchanged")

        b_wr = b_exp["iso_7243_work_rest"]
        s_wr = s_exp["iso_7243_work_rest"]

        return {
            "temperature_diff_c": round(s_env["background_temp_c"] - b_env["background_temp_c"], 1),
            "relative_humidity_diff_pct": round(s_env["relative_humidity_pct"] - b_env["relative_humidity_pct"], 1),
            "wind_speed_diff_ms": round(s_env["wind_speed_ms"] - b_env["wind_speed_ms"], 1),
            "solar_radiation_diff_wm2": round(s_env["solar_radiation_wm2"] - b_env["solar_radiation_wm2"], 0),
            "ml_localized_temp_diff_c": round(s_ml["localized_temp_c"] - b_ml["localized_temp_c"], 1),
            "ml_anomaly_diff_c": round(s_ml["ml_anomaly_c"] - b_ml["ml_anomaly_c"], 2),
            "wbgt_diff_c": round(s_phys["wbgt_c"] - b_phys["wbgt_c"], 1),
            "risk_category_from": b_exp["risk_category"],
            "risk_category_to": s_exp["risk_category"],
            "risk_direction": direction,
            "risk_steps": delta_rank,
            "operational_state_from": base_eval["operational_recommendation"]["operational_state"],
            "operational_state_to": scen_eval["operational_recommendation"]["operational_state"],
            "work_minutes_delta": s_wr["work_minutes"] - b_wr["work_minutes"],
            "rest_minutes_delta": s_wr["rest_minutes"] - b_wr["rest_minutes"],
            "exposure_duration_diff_hours": round(s_exp["exposure_duration_hours"] - b_exp["exposure_duration_hours"], 1),
        }

    @classmethod
    def generate_causal_narrative(
        cls,
        diffs: Dict[str, Any],
        base_eval: Dict[str, Any],
        scen_eval: Dict[str, Any],
    ) -> Tuple[List[str], str]:
        """
        Produces technically defensible 'What changed?' and 'Why did the risk change?' explanations
        derived strictly from underlying physical and physiological model components.
        """
        what_changed = []
        why_changed = []

        b_env = base_eval["environmental_conditions"]
        s_env = scen_eval["environmental_conditions"]
        b_exp = base_eval["exposure_assessment"]
        s_exp = scen_eval["exposure_assessment"]

        # Environmental levers
        if diffs["temperature_diff_c"] != 0:
            what_changed.append(f"Ambient temperature shifted by {diffs['temperature_diff_c']:+0.1f}°C ({b_env['background_temp_c']}°C → {s_env['background_temp_c']}°C).")
            if diffs["temperature_diff_c"] > 0:
                why_changed.append("Higher dry-bulb temperature increases sensible convective heat gain to the human body.")
            else:
                why_changed.append("Lower ambient dry-bulb temperature widens the gradient for convective heat dissipation.")

        if diffs["relative_humidity_diff_pct"] != 0:
            what_changed.append(f"Relative humidity shifted by {diffs['relative_humidity_diff_pct']:+0.1f}% ({b_env['relative_humidity_pct']}% → {s_env['relative_humidity_pct']}%).")
            if diffs["relative_humidity_diff_pct"] > 0:
                why_changed.append("Increased vapor pressure degrades skin evaporative cooling efficiency, driving higher Wet-Bulb temperature.")
            else:
                why_changed.append("Reduced relative humidity enhances sweat evaporation rate, facilitating core thermal equilibrium.")

        if diffs["wind_speed_diff_ms"] != 0:
            what_changed.append(f"Wind speed shifted by {diffs['wind_speed_diff_ms']:+0.1f} m/s ({b_env['wind_speed_ms']} → {s_env['wind_speed_ms']} m/s).")
            if diffs["wind_speed_diff_ms"] > 0:
                why_changed.append("Increased air velocity enhances skin boundary-layer convective and evaporative heat transfer coefficient.")
            else:
                why_changed.append("Stagnant airflow inhibits boundary-layer air replacement, trapping humid warm microclimates next to the skin.")

        if diffs["solar_radiation_diff_wm2"] != 0:
            what_changed.append(f"Solar radiation changed by {diffs['solar_radiation_diff_wm2']:+0.0f} W/m² ({b_env['solar_radiation_wm2']} → {s_env['solar_radiation_wm2']} W/m²).")
            if diffs["solar_radiation_diff_wm2"] < 0:
                why_changed.append("Reduced downward solar flux lowers black globe temperature (Tg), diminishing radiant heat stress.")
            else:
                why_changed.append("Elevated direct solar irradiance significantly charges surface and black globe temperatures.")

        # Exposure / Personal levers
        if b_exp["workload_intensity"] != s_exp["workload_intensity"]:
            b_w = WORKLOAD_METABOLIC_WATTS.get(b_exp["workload_intensity"], 250)
            s_w = WORKLOAD_METABOLIC_WATTS.get(s_exp["workload_intensity"], 250)
            what_changed.append(f"Workload adjusted from {b_exp['workload_intensity']} to {s_exp['workload_intensity']}.")
            if s_w < b_w:
                why_changed.append(f"Workload pacing reduction decreases internal metabolic heat production by ~{abs(s_w - b_w)}W (~{b_w}W → ~{s_w}W), lowering endogenous cardiac demand.")
            else:
                why_changed.append(f"Workload escalation increases metabolic heat output by ~{s_w - b_w}W (~{b_w}W → ~{s_w}W), accelerating internal core heat storage.")

        if b_exp["clothing_ppe"] != s_exp["clothing_ppe"]:
            what_changed.append(f"Clothing/PPE modified from {b_exp['clothing_ppe']} to {s_exp['clothing_ppe']}.")
            if any(k in s_exp["clothing_ppe"] for k in ("light", "standard")) and any(k in b_exp["clothing_ppe"] for k in ("heavy", "ppe")):
                why_changed.append("Switching to lightweight breathable clothing lowers thermal insulation (clo) and enhances sweat vapor permeability.")
            elif any(k in s_exp["clothing_ppe"] for k in ("heavy", "ppe")):
                why_changed.append("Utilizing impermeable PPE or heavy garments creates a microclimate vapor barrier, severely restricting latent heat rejection.")

        if b_exp["acclimatization"] != s_exp["acclimatization"]:
            what_changed.append(f"Acclimatization status updated from {b_exp['acclimatization']} to {s_exp['acclimatization']}.")
            if s_exp["acclimatization"] in ("acclimatized", "very_used"):
                why_changed.append("Acclimatized physiology triggers earlier sweating onset, higher sweat volume with lower salt loss, and stabilized heart rate.")
            else:
                why_changed.append("Unacclimatized status impairs sweating efficiency and elevates cardiovascular strain under identical thermal conditions.")

        if b_exp["recovery_environment"] != s_exp["recovery_environment"]:
            what_changed.append(f"Recovery environment transitioned from {b_exp['recovery_environment']} to {s_exp['recovery_environment']}.")
            if s_exp["recovery_environment"] in ("cool_indoor", "shade"):
                why_changed.append(f"Conditioned rest ({s_exp['recovery_environment']}) establishes an active negative thermal gradient during rest breaks, accelerating core cooling.")
            else:
                why_changed.append("Resting in unconditioned direct outdoor conditions prevents core body heat dissipation during mandatory rest cycles.")

        if diffs["exposure_duration_diff_hours"] != 0:
            what_changed.append(f"Shift duration modified by {diffs['exposure_duration_diff_hours']:+0.1f} hours ({b_exp['exposure_duration_hours']}h → {s_exp['exposure_duration_hours']}h).")
            if diffs["exposure_duration_diff_hours"] < 0:
                why_changed.append("Shortened shift duration reduces cumulative physiological strain and limits exposure to peak diurnal heat windows.")
            else:
                why_changed.append("Prolonged exposure increases cumulative thermal strain, compounding dehydration and electrolyte depletion.")

        if not what_changed:
            what_changed.append("Baseline and simulated scenario conditions are identical.")
            why_changed.append("No environmental or operational parameters were modified.")

        synthesis = " ".join(why_changed)
        return what_changed, synthesis

    @classmethod
    def simulate(
        cls,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main execution orchestrator supporting:
        - Baseline vs Scenario
        - Baseline vs Scenario A vs Scenario B
        - Decision timeline generation
        - Boundary validation and honesty labeling
        """
        location = payload.get("location", "Hyderabad")
        region = payload.get("region", "plains")

        # Resolve Baseline
        raw_base_env = (
            payload.get("baseline_weather")
            or payload.get("weather_data")
            or payload.get("environment")
            or payload.get("baseline")
            or {}
        )
        base_env = cls.resolve_environmental_inputs(raw_base_env)

        raw_base_exp = payload.get("baseline_exposure") or payload.get("baseline") or {}
        base_exp = cls.resolve_exposure_inputs(raw_base_exp)

        # Baseline evaluation
        base_eval = cls.evaluate_condition_state(
            env=base_env,
            exp=base_exp,
            location=location,
            region=region,
            is_baseline=True,
            raw_env=raw_base_env,
        )

        # Collect scenarios to run
        scenarios_input = {}
        if "scenarios" in payload and isinstance(payload["scenarios"], dict):
            scenarios_input = payload["scenarios"]
        elif "scenario_a" in payload or "scenario_b" in payload:
            if "scenario_a" in payload:
                scenarios_input["scenario_a"] = payload["scenario_a"]
            if "scenario_b" in payload:
                scenarios_input["scenario_b"] = payload["scenario_b"]
        else:
            # Single scenario (backward compatible)
            scen_raw = payload.get("scenario") or {}
            scenarios_input["scenario_a"] = scen_raw

        evaluated_scenarios = {}
        comparisons = {}
        all_warnings = list(base_eval["validation_warnings"])

        for scen_key, scen_data in scenarios_input.items():
            # Scenarios inherit baseline environmental & exposure inputs unless overridden
            raw_s_env = scen_data.get("environmental") or scen_data.get("weather_data") or scen_data
            s_env = cls.resolve_environmental_inputs(raw_s_env, fallback=base_env)

            raw_s_exp = scen_data.get("exposure") or scen_data
            s_exp = cls.resolve_exposure_inputs(raw_s_exp, fallback=base_exp)

            s_eval = cls.evaluate_condition_state(
                env=s_env,
                exp=s_exp,
                location=location,
                region=region,
                is_baseline=False,
                raw_env=raw_s_env,
            )

            diffs = cls.compute_differences(base_eval, s_eval)
            what_ch, why_ch = cls.generate_causal_narrative(diffs, base_eval, s_eval)

            s_eval["differences_from_baseline"] = diffs
            s_eval["what_changed"] = what_ch
            s_eval["why_did_risk_change"] = why_ch

            evaluated_scenarios[scen_key] = s_eval
            comparisons[scen_key] = diffs
            all_warnings.extend(s_eval["validation_warnings"])

        # Timeline generation for the primary simulated scenario (Scenario A or primary)
        primary_key = "scenario_a" if "scenario_a" in evaluated_scenarios else list(evaluated_scenarios.keys())[0]
        primary_scen = evaluated_scenarios[primary_key]
        primary_env = primary_scen["environmental_conditions"]
        primary_exp = primary_scen["exposure_assessment"]

        timeline = cls.generate_decision_timeline(
            base_env=base_env,
            scenario_env={
                "temperature_c": primary_env["background_temp_c"],
                "relative_humidity_pct": primary_env["relative_humidity_pct"],
                "wind_speed_ms": primary_env["wind_speed_ms"],
                "solar_radiation_wm2": primary_env["solar_radiation_wm2"],
                "hour": primary_env["forecast_hour"],
            },
            exp_inputs=primary_exp,
            location=location,
            region=region,
        )

        # Deduplicate warnings
        unique_warnings = list(dict.fromkeys(all_warnings))
        is_all_validated = (len(unique_warnings) == 0)

        # Format backward-compatible fields for existing test suite
        primary_diff = comparisons.get(primary_key, {})
        action_level_change = {
            "baseline": primary_diff.get("risk_category_from", base_eval["exposure_assessment"]["risk_category"]),
            "scenario": primary_diff.get("risk_category_to", primary_scen["exposure_assessment"]["risk_category"]),
            "delta_steps": primary_diff.get("risk_steps", 0),
            "direction": primary_diff.get("risk_direction", "unchanged"),
        }
        work_rest_change = {
            "baseline_work_minutes": base_eval["exposure_assessment"]["iso_7243_work_rest"]["work_minutes"],
            "scenario_work_minutes": primary_scen["exposure_assessment"]["iso_7243_work_rest"]["work_minutes"],
            "work_delta_minutes": primary_diff.get("work_minutes_delta", 0),
            "baseline_rest_minutes": base_eval["exposure_assessment"]["iso_7243_work_rest"]["rest_minutes"],
            "scenario_rest_minutes": primary_scen["exposure_assessment"]["iso_7243_work_rest"]["rest_minutes"],
            "rest_delta_minutes": primary_diff.get("rest_minutes_delta", 0),
        }

        return {
            "status": "success",
            "simulation_label": "SIMULATED SCENARIO",
            "is_simulated": True,
            "is_validated_range": is_all_validated,
            "validation_warnings": unique_warnings,
            "location": location,
            "region": region,
            "baseline": base_eval,
            "scenarios": evaluated_scenarios,
            "comparisons": comparisons,
            "decision_timeline": timeline,
            # Backward-compatible keys for existing tests & UI
            "baseline_profile": base_eval["full_profile"],
            "scenario_profile": primary_scen["full_profile"],
            "baseline_result": base_eval,
            "scenario_result": primary_scen,
            "action_level_change": action_level_change,
            "work_rest_change": work_rest_change,
            "parameter_changes": [
                {"parameter": ch.split(" ")[0].lower(), "label": ch.split(" (")[0], "detail": ch}
                for ch in primary_scen.get("what_changed", [])
            ],
            "causal_explanations": primary_scen.get("what_changed", []) + [primary_scen.get("why_did_risk_change", "")],
            "transition_explanation": primary_scen.get("why_did_risk_change", ""),
            "decision_trace_baseline": base_eval["full_profile"].get("decision_trace"),
            "decision_trace_scenario": primary_scen["full_profile"].get("decision_trace"),
            "summary": primary_scen.get("why_did_risk_change", ""),
            "disclaimer": (
                "DATA HONESTY NOTICE: Outputs represent a SIMULATED SCENARIO and do not constitute actual observed "
                "or forecast meteorological readings. No operational state guarantees zero risk. All outdoor physical "
                "activities require continuous hydration, shade availability, and buddy monitoring."
            ),
        }
