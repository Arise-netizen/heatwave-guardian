"""
HeatWave Guardian — Intervention Intelligence Engine
=====================================================
Step 6: Explainable, transparent decision-support system converting environmental,
forecast, and exposure intelligence into structured, prioritized operational actions.

The engine answers:
"Given the predicted heat conditions and exposure context, what controllable factors
could reduce exposure or move the task away from the highest-risk period?"

CORE SCIENTIFIC & SAFETY PRINCIPLES:
1. DECISION SUPPORT ONLY: Explicitly NOT medical diagnosis, medical advice, or a
   guarantee of safety.
2. CONTROL HIERARCHY: Prioritizes actions following occupational hygiene principles:
   Elimination / Peak Avoidance -> Work Modification -> Environmental Controls ->
   Recovery Facilities -> Administrative Controls -> PPE Considerations.
3. DATA HONESTY: Directional and physical effects are stated transparently without
   fabricated percentages unless directly derived from physical equations.
4. SAFETY PPE GUARDRAIL: Never instructs removal of mandatory occupational safety PPE.
5. ROLE-SPECIFIC SCOPES: Worker, Supervisor / Worksite, and Municipal Authority views
   derived from the identical underlying thermal physics.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
import math
import uuid

from engine import (
    CITIES,
    CITY_REGIONS,
    RISK_LEVELS,
    RISK_COLORS,
    estimate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    classify_imd,
    fuse_risk,
    detect_peak_risk_windows,
    build_exposure_context,
    evaluate_exposure_intervention,
    get_work_rest,
    assess_full_risk_profile,
    WORKLOAD_METABOLIC_WATTS,
    CLOTHING_ADJUSTMENT_VALUES,
    RECOMMENDATION_LEVELS,
)
from services.scenario_simulator import ScenarioSimulator
from services.cooling_recommender import CoolingRecommenderService
from services.authority_service import AuthorityService


# ──────────────────────────────────────────────────────────────────────────────
# Control Hierarchy Levels (Occupational Hygiene Standards)
# ──────────────────────────────────────────────────────────────────────────────
CONTROL_HIERARCHY = {
    1: {"name": "Elimination / Avoid Peak Exposure", "short": "Avoid Peak"},
    2: {"name": "Engineering / Task Modification", "short": "Modify Work"},
    3: {"name": "Environmental Engineering Controls", "short": "Environment"},
    4: {"name": "Recovery Facility Access", "short": "Recovery"},
    5: {"name": "Administrative Controls & Escalation", "short": "Administrative"},
    6: {"name": "PPE Considerations & Microclimate", "short": "PPE"},
}

# ──────────────────────────────────────────────────────────────────────────────
# Unified Decision Timeline Operational States
# Strictly never called "guaranteed safe"
# ──────────────────────────────────────────────────────────────────────────────
OPERATIONAL_STATES = {
    "Low": {
        "state": "ROUTINE MONITORING",
        "badge_color": "emerald",
        "description": "Standard operational monitoring with ad-libitum hydration and ventilated rest facilities available.",
        "safety_note": "Routine monitoring active. Note: Absolute safety cannot be guaranteed for unconditioned, elderly, or vulnerable individuals.",
    },
    "Moderate": {
        "state": "ENHANCED CONTROLS",
        "badge_color": "amber",
        "description": "Scheduled hydration breaks, accessible shaded recovery zones, and peer buddy monitoring active.",
        "safety_note": "Elevated heat burden. Workers must not bypass scheduled breaks or hydrate solely on sensation of thirst.",
    },
    "High": {
        "state": "MODIFY / RESTRICT EXPOSURE",
        "badge_color": "orange",
        "description": "Enforce rotational work pacing (e.g. 45/15 or 30/30), shift heavy tasks away from direct sunlight, deploy spot cooling.",
        "safety_note": "Significant thermal strain. Core temperature rises rapidly during continuous heavy physical labor.",
    },
    "Very High": {
        "state": "SUPERVISOR REVIEW",
        "badge_color": "red",
        "description": "Mandatory supervisor review. Suspend non-essential strenuous work; enforce 15/45 pacing; mandate air-conditioned recovery.",
        "safety_note": "High hazard threshold. Immediate supervisor intervention required before assigning or continuing outdoor physical exertion.",
    },
    "Extreme": {
        "state": "AUTHORITY REVIEW",
        "badge_color": "purple",
        "description": "Authority and site emergency escalation. Immediate suspension of outdoor physical tasks; activate emergency cooling response.",
        "safety_note": "Critical physiological emergency threshold. High probability of heat exhaustion or life-threatening heat stroke.",
    },
}


class InterventionIntelligenceEngine:
    """
    Explainable Intervention Intelligence Engine converting meteorological,
    exposure, and forecast data into prioritized, role-tailored decision support.
    """

    ENGINE_VERSION = "v1.0-intervention-intelligence"
    DISCLAIMER = (
        "DECISION SUPPORT NOTICE: Recommendations are occupational health and safety "
        "screening guidance derived from ISO 7243:2017, ACGIH TLVs, and NDMA Heat Action "
        "guidelines. This system does NOT provide medical diagnosis, individualized clinical "
        "prescriptions, or guarantees of absolute safety."
    )

    @classmethod
    def evaluate(
        cls,
        weather_data: Optional[Dict[str, Any]] = None,
        hourly_forecast: Optional[List[Dict[str, Any]]] = None,
        exposure_context: Optional[Dict[str, Any]] = None,
        personal_inputs: Optional[Dict[str, Any]] = None,
        location: str = "Hyderabad",
        region: Optional[str] = None,
        role: str = "all",
        simulate_intervention: bool = False,
        intervention_modifications: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main decision-support pipeline evaluating controllable exposure factors.
        Returns prioritized interventions, explainable causal narratives, role-specific views,
        unified decision timeline, and structured alert objects.
        """
        # 1. Fallback & normalization of environmental parameters
        region = region or CITY_REGIONS.get(location, "plains")
        weather = dict(weather_data or {})
        temp_c = float(weather.get("temp_c") or weather.get("current_temp_c") or 35.0)
        rh_pct = float(weather.get("rh_pct") or weather.get("humidity") or 50.0)
        wind_ms = float(weather.get("wind_ms") or 2.0)
        solar_wm2 = float(weather.get("solar_wm2") or 600.0)
        
        # Calculate WBGT and base risk if not provided
        wbgt_c = weather.get("wbgt_c")
        if wbgt_c is None or not isinstance(wbgt_c, (int, float)):
            wbgt_c = estimate_wbgt(temp_c=temp_c, rh_pct=rh_pct, wind_ms=wind_ms, solar_wm2=solar_wm2)
        wbgt_c = round(float(wbgt_c), 1)

        thermal_stress = classify_thermal_stress(wbgt_c)
        wbgt_band = thermal_stress["wbgt_band"]
        imd_status = classify_imd(temp_c, region)
        base_risk = fuse_risk(wbgt_band, imd_status)

        # 2. Exposure Context normalization
        exp = dict(exposure_context or personal_inputs or {})
        workload = (exp.get("workload") or exp.get("workload_intensity") or "moderate").lower()
        acclimatization = (exp.get("acclimatization") or exp.get("experience") or "very_used").lower()
        clothing = (exp.get("clothing") or exp.get("clothing_ppe") or "light").lower()
        ppe_category = exp.get("ppe_category")
        recovery_env = (exp.get("recovery_environment") or "shade").lower()
        is_sun_exposed = bool(exp.get("is_sun_exposed", True))
        duration_hours = float(exp.get("duration_hours") or exp.get("exposure_duration_hours") or 4.0)
        work_start_time = exp.get("work_start_time") or "08:00"

        # 3. Peak-risk detection from hourly forecast series
        hourly_series = hourly_forecast or []
        peak_window = detect_peak_risk_windows(hourly_series) if hourly_series else {
            "peak_period": "12:30–15:30",
            "peak_hour": 14,
            "peak_wbgt_c": round(wbgt_c + 1.8, 1),
            "elevated_risk_start": 12,
            "elevated_risk_end": 16,
            "has_elevated_risk": wbgt_c >= 28.0,
            "is_flat_curve": False,
            "lower_exposure_windows": [
                {"label": "Early Morning Window", "window": "06:00–10:30", "start_hour": 6, "end_hour": 10},
                {"label": "Late Afternoon Window", "window": "16:30–19:00", "start_hour": 16, "end_hour": 19},
            ],
            "drivers": ["Diurnal peak solar irradiance and convective heating"],
            "confidence": "Medium",
        }

        # 4. Compute full baseline profile and work/rest schedule
        core_exp_ctx = build_exposure_context(
            wbgt_c=wbgt_c,
            environmental_risk_level=base_risk,
            wind_ms=wind_ms,
            solar_wm2=solar_wm2 if is_sun_exposed else 0.0,
            is_direct_sun=is_sun_exposed,
            workload=workload,
            exposure_duration_hours=duration_hours,
            acclimatization=acclimatization,
            clothing=clothing,
            ppe_category=ppe_category,
            recovery_environment=recovery_env,
        )
        base_work_rest = evaluate_exposure_intervention(
            exposure_context=core_exp_ctx,
            peak_risk_window=peak_window,
            is_peak_window=cls._check_is_peak(work_start_time, duration_hours, peak_window),
        )

        overall_risk = base_work_rest.get("recommendation_level", base_risk)
        # Normalize action tier for operational state
        risk_category = cls._map_recommendation_to_risk(overall_risk, base_risk)
        op_state_info = OPERATIONAL_STATES.get(risk_category, OPERATIONAL_STATES["Moderate"])

        # 5. Candidate shift window evaluation (Scheduling Intelligence)
        scheduling_candidates = cls._evaluate_candidate_shift_windows(
            hourly_series=hourly_series,
            peak_window=peak_window,
            duration_hours=duration_hours,
            current_start=work_start_time,
            baseline_wbgt=wbgt_c,
        )

        # 6. Generate structured interventions across all categories A-G
        interventions = cls._generate_structured_interventions(
            wbgt_c=wbgt_c,
            base_risk=base_risk,
            risk_category=risk_category,
            workload=workload,
            acclimatization=acclimatization,
            clothing=clothing,
            ppe_category=ppe_category,
            recovery_env=recovery_env,
            is_sun_exposed=is_sun_exposed,
            duration_hours=duration_hours,
            work_start_time=work_start_time,
            peak_window=peak_window,
            scheduling_candidates=scheduling_candidates,
            location=location,
            work_rest=base_work_rest,
        )

        # Sort interventions by control hierarchy (Level 1 to 6)
        interventions.sort(key=lambda x: (x["control_hierarchy_level"], x.get("priority_rank", 5)))

        # 7. Role-Specific Views (Worker, Supervisor, Authority)
        role_views = cls._build_role_views(
            interventions=interventions,
            weather={"temp_c": temp_c, "rh_pct": rh_pct, "wbgt_c": wbgt_c, "risk": risk_category},
            peak_window=peak_window,
            scheduling_candidates=scheduling_candidates,
            work_rest=base_work_rest,
            location=location,
            op_state=op_state_info,
        )

        # 8. Decision Timeline Generation
        timeline = cls._build_unified_timeline(
            temp_c=temp_c,
            rh_pct=rh_pct,
            wbgt_c=wbgt_c,
            workload=workload,
            peak_window=peak_window,
            interventions=interventions,
            duration_hours=duration_hours,
            start_time=work_start_time,
            hourly_series=hourly_series,
        )

        # 9. Structured Alert Object Preparation
        alerts = cls._prepare_structured_alerts(
            location=location,
            risk_category=risk_category,
            peak_window=peak_window,
            interventions=interventions,
            op_state=op_state_info,
        )

        # 10. Optional What-If Simulation of Primary Recommended Intervention
        simulation_result = None
        if simulate_intervention or intervention_modifications:
            simulation_result = cls._simulate_intervention(
                weather={"temp_c": temp_c, "rh_pct": rh_pct, "wind_ms": wind_ms, "solar_wm2": solar_wm2},
                baseline_exp={
                    "workload": workload,
                    "acclimatization": acclimatization,
                    "clothing": clothing,
                    "recovery_environment": recovery_env,
                    "exposure_duration_hours": duration_hours,
                    "work_start_time": work_start_time,
                    "is_sun_exposed": is_sun_exposed,
                },
                modifications=intervention_modifications or scheduling_candidates.get("recommended_modifications", {}),
                location=location,
                region=region,
            )

        # 11. Format final response
        selected_role_view = role_views.get(role, role_views)

        return {
            "status": "success",
            "is_decision_support": True,
            "disclaimer": cls.DISCLAIMER,
            "location": location,
            "region": region,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environmental_condition": {
                "temperature_c": temp_c,
                "relative_humidity_pct": rh_pct,
                "wbgt_c": wbgt_c,
                "wbgt_band": wbgt_band,
                "imd_heatwave_status": imd_status,
                "base_environmental_risk": base_risk,
                "data_source": weather.get("source", "Open-Meteo Atmospheric Services + Hyperlocal ML"),
                "is_observed": weather.get("is_observed", False),
                "is_forecast": weather.get("is_forecast", True),
            },
            "exposure_context": {
                "workload": workload,
                "acclimatization": acclimatization,
                "clothing": clothing,
                "ppe_category": ppe_category,
                "recovery_environment": recovery_env,
                "is_sun_exposed": is_sun_exposed,
                "duration_hours": duration_hours,
                "work_start_time": work_start_time,
                "overall_action_level": risk_category,
                "iso_7243_work_rest": {
                    "work_minutes": base_work_rest["work_minutes"],
                    "rest_minutes": base_work_rest["rest_minutes"],
                    "instruction": base_work_rest["instruction"],
                },
            },
            "peak_risk_window": peak_window,
            "scheduling_candidates": scheduling_candidates,
            "operational_state": {
                "state": op_state_info["state"],
                "badge_color": op_state_info["badge_color"],
                "description": op_state_info["description"],
                "safety_note": op_state_info["safety_note"],
            },
            "control_hierarchy_summary": cls._build_hierarchy_summary(interventions),
            "interventions": interventions,
            "role_views": role_views,
            "active_role_view": selected_role_view if role != "all" else None,
            "decision_timeline": timeline,
            "prepared_alerts": alerts,
            "simulated_intervention": simulation_result,
            "data_honesty": {
                "observed_vs_modelled_vs_simulated": "Observed meteorological inputs fused with Liljegren WBGT physics; counterfactual interventions simulated via ISO 7243 deterministic engine.",
                "model_version": cls.ENGINE_VERSION,
                "confidence_state": peak_window.get("confidence", "Medium"),
                "no_guaranteed_safety": "Absolute physiological safety cannot be guaranteed under thermal stress. Interventions are screening controls.",
            },
        }

    @staticmethod
    def _parse_hour_int(val: Any, default: int = 12) -> int:
        """Safely extracts integer hour (0-23) from int, float, or string like '12:30' or '2026-06-15T12:00'."""
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return int(val) % 24
        val_str = str(val).strip()
        if "T" in val_str:
            val_str = val_str.split("T")[1]
        if ":" in val_str:
            try:
                return int(val_str.split(":")[0]) % 24
            except (ValueError, TypeError):
                return default
        try:
            return int(float(val_str)) % 24
        except (ValueError, TypeError):
            return default

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: Check Peak Window Overlap
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _check_is_peak(cls, start_str: Any, duration: float, peak_window: Dict[str, Any]) -> bool:
        """Determines if the planned work shift overlaps the diurnal peak window."""
        sh = cls._parse_hour_int(start_str, 8)
        eh = sh + duration
        pk_sh = cls._parse_hour_int(peak_window.get("elevated_risk_start"), 12)
        pk_eh = cls._parse_hour_int(peak_window.get("elevated_risk_end"), 16)
        return max(sh, pk_sh) < min(eh, pk_eh)

    @staticmethod
    def _map_recommendation_to_risk(rec_level: str, base_risk: str) -> str:
        """Maps recommendation level strings to standard 5-tier risk categories."""
        if "Suspend" in rec_level or "Site-specific" in rec_level:
            return "Very High" if base_risk != "Extreme" else "Extreme"
        elif "High-exposure" in rec_level:
            return "High"
        elif "Frequent recovery" in rec_level:
            return "High"
        elif "Enhanced" in rec_level:
            return "Moderate"
        elif "Normal" in rec_level:
            return "Low"
        return base_risk

    # ──────────────────────────────────────────────────────────────────────────
    # Category A: Scheduling Intelligence
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _evaluate_candidate_shift_windows(
        cls,
        hourly_series: List[Dict[str, Any]],
        peak_window: Dict[str, Any],
        duration_hours: float,
        current_start: str,
        baseline_wbgt: float,
    ) -> Dict[str, Any]:
        """
        Analyzes diurnal forecast to identify lower-heat-stress candidate periods
        that avoid the diurnal peak thermal window.
        """
        candidates = []
        pk_start = cls._parse_hour_int(peak_window.get("elevated_risk_start"), 12)
        pk_end = cls._parse_hour_int(peak_window.get("elevated_risk_end"), 16)
        pk_period = peak_window.get("peak_period", "12:30–15:30")

        # Fallback candidate schedule when hourly series is absent or sparse
        if not hourly_series or len(hourly_series) < 6:
            candidates.append({
                "window_label": "Early Morning Shift",
                "recommended_start": "06:30",
                "recommended_end": f"{int(6.5 + duration_hours):02d}:00",
                "estimated_avg_wbgt_c": round(max(22.0, baseline_wbgt - 4.5), 1),
                "wbgt_reduction_direction": "Significant reduction vs midday peak",
                "avoids_peak": True,
                "rationale": f"Shifting start time to 06:30 completes work before peak heat window ({pk_period}) commences.",
                "confidence": "Medium (Diurnal Climatological Estimate)",
            })
            candidates.append({
                "window_label": "Late Afternoon Shift",
                "recommended_start": "16:30",
                "recommended_end": f"{int(16.5 + duration_hours):02d}:00",
                "estimated_avg_wbgt_c": round(max(24.0, baseline_wbgt - 3.2), 1),
                "wbgt_reduction_direction": "Moderate reduction as solar irradiance subsides",
                "avoids_peak": True,
                "rationale": "Scheduling after 16:30 avoids maximum solar radiation and declining ambient temperatures.",
                "confidence": "Medium (Diurnal Climatological Estimate)",
            })
        else:
            # Evaluate morning window (06:00 to pk_start)
            m_points = [p for p in hourly_series if 6 <= cls._get_hour(p) < pk_start]
            if m_points:
                avg_m_wbgt = round(sum(p.get("peak_wbgt_c", p.get("wbgt_c", 25.0)) for p in m_points) / len(m_points), 1)
                candidates.append({
                    "window_label": "Early Morning Window",
                    "recommended_start": "06:00",
                    "recommended_end": f"{min(pk_start, int(6 + duration_hours)):02d}:00",
                    "estimated_avg_wbgt_c": avg_m_wbgt,
                    "wbgt_reduction_direction": f"Lower WBGT by ~{round(baseline_wbgt - avg_m_wbgt, 1)}°C compared to peak",
                    "avoids_peak": True,
                    "rationale": f"Morning hours (06:00–{pk_start:02d}:00) exhibit lowest diurnal radiant flux and air temperatures.",
                    "confidence": "High (From Hourly NWP Forecast)",
                })

            # Evaluate late afternoon window (pk_end to 20:00)
            e_points = [p for p in hourly_series if pk_end <= cls._get_hour(p) <= 20]
            if e_points:
                avg_e_wbgt = round(sum(p.get("peak_wbgt_c", p.get("wbgt_c", 26.0)) for p in e_points) / len(e_points), 1)
                candidates.append({
                    "window_label": "Late Afternoon Window",
                    "recommended_start": f"{pk_end:02d}:00",
                    "recommended_end": f"{min(21, int(pk_end + duration_hours)):02d}:00",
                    "estimated_avg_wbgt_c": avg_e_wbgt,
                    "wbgt_reduction_direction": f"Lower WBGT by ~{round(baseline_wbgt - avg_e_wbgt, 1)}°C compared to peak",
                    "avoids_peak": True,
                    "rationale": f"Solar radiation drops significantly after {pk_end:02d}:00, substantially lowering black globe temperature.",
                    "confidence": "High (From Hourly NWP Forecast)",
                })

        # Recommended modification payload for simulation
        primary_candidate = candidates[0] if candidates else {}
        return {
            "has_candidate_windows": len(candidates) > 0,
            "peak_risk_period": pk_period,
            "candidate_windows": candidates,
            "recommended_modifications": {
                "work_start_time": primary_candidate.get("recommended_start", "06:30"),
                "exposure_duration_hours": min(duration_hours, 3.0),
                "workload": "moderate",
            },
        }

    @classmethod
    def _get_hour(cls, point: Dict[str, Any]) -> int:
        """Extracts hour integer from a forecast point."""
        if "hour" in point:
            return cls._parse_hour_int(point["hour"])
        if "time" in point:
            return cls._parse_hour_int(point["time"])
        if "time_str" in point:
            return cls._parse_hour_int(point["time_str"])
        return 12

    # ──────────────────────────────────────────────────────────────────────────
    # Structured Interventions Builder across Categories A-G
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _generate_structured_interventions(
        cls,
        wbgt_c: float,
        base_risk: str,
        risk_category: str,
        workload: str,
        acclimatization: str,
        clothing: str,
        ppe_category: Optional[str],
        recovery_env: str,
        is_sun_exposed: bool,
        duration_hours: float,
        work_start_time: str,
        peak_window: Dict[str, Any],
        scheduling_candidates: Dict[str, Any],
        location: str,
        work_rest: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Creates fully traceable, explainable interventions across the 7 categories:
        A. Scheduling
        B. Exposure Duration
        C. Workload
        D. Environment
        E. Clothing / PPE
        F. Cooling Access
        G. Operational Escalation
        """
        interventions: List[Dict[str, Any]] = []

        is_peak_overlapping = cls._check_is_peak(work_start_time, duration_hours, peak_window)
        pk_period = peak_window.get("peak_period", "12:30–15:30")
        pk_wbgt = peak_window.get("peak_wbgt_c", wbgt_c)

        # ─── Category A: SCHEDULING (Hierarchy Level 1: Elimination/Avoidance) ───
        if is_peak_overlapping and wbgt_c >= 28.0:
            cands = scheduling_candidates.get("candidate_windows", [])
            cand_desc = f" (e.g. {cands[0]['window_label']} starting {cands[0]['recommended_start']})" if cands else ""
            interventions.append({
                "id": "SCHED-01",
                "category": "SCHEDULING",
                "control_hierarchy_level": 1,
                "control_hierarchy_name": CONTROL_HIERARCHY[1]["name"],
                "intervention": "Reschedule strenuous outdoor tasks to cooler morning/evening windows",
                "reason": f"Forecast indicates diurnal heat burden peaks during {pk_period} with WBGT reaching {pk_wbgt}°C.",
                "triggering_condition": f"Planned shift ({work_start_time}, {duration_hours}h) overlaps the forecast peak heat stress period ({pk_period}).",
                "affected_exposure_factor": "diurnal_timing",
                "expected_direction_of_effect": f"Avoids continuous physical exertion during the peak radiative solar and ambient heat window{cand_desc}.",
                "confidence": peak_window.get("confidence", "Medium"),
                "source_reference": "ISO 7243:2017 Clause 6.2 (Organizational Measures)",
                "operational_scope": ["worker", "supervisor", "authority"],
                "priority_rank": 1,
                "candidate_window": cands[0] if cands else None,
            })
        elif not is_peak_overlapping and wbgt_c >= 28.0:
            interventions.append({
                "id": "SCHED-03",
                "category": "SCHEDULING",
                "control_hierarchy_level": 1,
                "control_hierarchy_name": CONTROL_HIERARCHY[1]["name"],
                "intervention": "Planned work schedule successfully avoids peak diurnal thermal window",
                "reason": f"Planned shift ({work_start_time}, {duration_hours}h) is scheduled outside the peak heat window ({pk_period}).",
                "triggering_condition": "Work shift successfully timed outside diurnal peak heat stress period.",
                "affected_exposure_factor": "diurnal_timing",
                "expected_direction_of_effect": "Eliminates coincidence with peak ambient temperature and maximum downward solar irradiance.",
                "confidence": "High",
                "source_reference": "ISO 7243:2017 Elimination / Peak Avoidance Principle",
                "operational_scope": ["worker", "supervisor"],
                "priority_rank": 4,
            })
        elif wbgt_c < 28.0:
            interventions.append({
                "id": "SCHED-02",
                "category": "SCHEDULING",
                "control_hierarchy_level": 1,
                "control_hierarchy_name": CONTROL_HIERARCHY[1]["name"],
                "intervention": "Maintain standard operational work schedule with routine hydration pacing",
                "reason": f"Current environmental heat burden (WBGT {wbgt_c}°C) does not breach acute shift rescheduling thresholds.",
                "triggering_condition": f"WBGT ({wbgt_c}°C) is below high-precaution screening thresholds.",
                "affected_exposure_factor": "diurnal_timing",
                "expected_direction_of_effect": "No schedule shift necessary under standard seasonal conditions.",
                "confidence": "High",
                "source_reference": "ACGIH Action Limit Screening",
                "operational_scope": ["supervisor"],
                "priority_rank": 5,
            })

        # ─── Category B: EXPOSURE DURATION (Hierarchy Level 2/5: Duration & Pacing) ───
        work_min = work_rest["work_minutes"]
        rest_min = work_rest["rest_minutes"]
        if rest_min > 0 or duration_hours >= 4.0:
            interventions.append({
                "id": "DUR-01",
                "category": "EXPOSURE_DURATION",
                "control_hierarchy_level": 2,
                "control_hierarchy_name": CONTROL_HIERARCHY[2]["name"],
                "intervention": f"Enforce hourly pacing split: {work_min} min work / {rest_min} min rest per hour",
                "reason": f"Under WBGT {wbgt_c}°C and {workload} workload, continuous exertion without structured rest drives uncompensable core body heat storage.",
                "triggering_condition": f"Thermal stress level requires {rest_min} min mandatory recovery per hour per ISO 7243 screening guidance.",
                "affected_exposure_factor": "continuous_exposure_duration",
                "expected_direction_of_effect": f"Restricts continuous heat exposure bouts to {work_min} minutes, preventing cumulative core temperature rise.",
                "confidence": "High",
                "source_reference": "ISO 7243:2017 Table C.1 & ACGIH TLV Work-Rest Schedule",
                "operational_scope": ["worker", "supervisor"],
                "priority_rank": 2,
            })
        if duration_hours >= 6.0 and wbgt_c >= 29.0:
            interventions.append({
                "id": "DUR-02",
                "category": "EXPOSURE_DURATION",
                "control_hierarchy_level": 5,
                "control_hierarchy_name": CONTROL_HIERARCHY[5]["name"],
                "intervention": "Limit cumulative shift exposure to maximum 6 hours; introduce task rotation",
                "reason": f"Prolonged shift length ({duration_hours}h) in hot environments compounds dehydration, electrolyte deficit, and cardiovascular fatigue.",
                "triggering_condition": f"Total planned exposure duration ({duration_hours} hours) exceeds 6 hours during elevated thermal conditions.",
                "affected_exposure_factor": "total_shift_duration",
                "expected_direction_of_effect": "Lowers cumulative physiological fatigue and cumulative cardiovascular strain.",
                "confidence": "Medium",
                "source_reference": "NIOSH Criteria for a Recommended Standard: Occupational Exposure to Heat",
                "operational_scope": ["supervisor"],
                "priority_rank": 3,
            })

        # ─── Category C: WORKLOAD (Hierarchy Level 2: Task Modification) ───
        if workload in ("heavy", "very_heavy"):
            watts = WORKLOAD_METABOLIC_WATTS.get(workload, 400)
            interventions.append({
                "id": "WORK-01",
                "category": "WORKLOAD",
                "control_hierarchy_level": 2,
                "control_hierarchy_name": CONTROL_HIERARCHY[2]["name"],
                "intervention": "Pace task intensity and substitute manual lifting with mechanical equipment",
                "reason": f"{workload.replace('_', ' ').capitalize()} exertion generates ~{watts}W of internal metabolic heat, multiplying environmental thermal load.",
                "triggering_condition": f"Workload is classified as {workload} (>350W metabolic heat production).",
                "affected_exposure_factor": "metabolic_rate",
                "expected_direction_of_effect": "Reduces internal endogenous body heat production, lowering cardiac output requirements.",
                "confidence": "High",
                "source_reference": "ISO 8996 (Determination of Metabolic Rate)",
                "operational_scope": ["worker", "supervisor"],
                "priority_rank": 2,
            })

        # ─── Category D: ENVIRONMENT (Hierarchy Level 3: Environmental Engineering) ───
        if is_sun_exposed:
            interventions.append({
                "id": "ENV-01",
                "category": "ENVIRONMENT",
                "control_hierarchy_level": 3,
                "control_hierarchy_name": CONTROL_HIERARCHY[3]["name"],
                "intervention": "Erect portable sunshades, tarpaulins, or move tasks to shaded staging areas",
                "reason": "Direct downward solar radiation substantially increases radiant black globe temperature (Tg).",
                "triggering_condition": "Work is performed under unshielded direct solar exposure.",
                "affected_exposure_factor": "radiant_solar_flux",
                "expected_direction_of_effect": "Shading eliminates direct solar irradiance on the body and tools. Note: Shade is a potential exposure-control intervention; absolute percentage reduction depends on canopy reflectivity.",
                "confidence": "High",
                "source_reference": "OSHA-NIOSH Heat Illness Prevention: Engineering Controls",
                "operational_scope": ["worker", "supervisor"],
                "priority_rank": 2,
            })
        if wbgt_c >= 29.0:
            interventions.append({
                "id": "ENV-02",
                "category": "ENVIRONMENT",
                "control_hierarchy_level": 3,
                "control_hierarchy_name": CONTROL_HIERARCHY[3]["name"],
                "intervention": "Deploy auxiliary industrial spot-cooling fans or misting fans in staging areas",
                "reason": "Air movement increases boundary-layer convective and evaporative heat transfer away from skin surfaces.",
                "triggering_condition": f"Ambient heat burden (WBGT {wbgt_c}°C) degrades natural sweat evaporation in stagnant air.",
                "affected_exposure_factor": "air_velocity_convective_cooling",
                "expected_direction_of_effect": "Promotes evaporative sweat cooling provided relative humidity allows latent evaporation.",
                "confidence": "Medium",
                "source_reference": "ACGIH Heat Stress and Strain: Engineering Controls",
                "operational_scope": ["supervisor"],
                "priority_rank": 3,
            })

        # ─── Category D / 4: RECOVERY ACCESS (Hierarchy Level 4: Recovery Facilities) ───
        if recovery_env in ("outdoor_direct", "warm_indoor"):
            interventions.append({
                "id": "REC-01",
                "category": "ENVIRONMENT",
                "control_hierarchy_level": 4,
                "control_hierarchy_name": CONTROL_HIERARCHY[4]["name"],
                "intervention": "Establish dedicated shaded or air-conditioned rest shelter for all recovery intervals",
                "reason": f"Resting in {recovery_env.replace('_', ' ')} prevents physiological core temperature recovery during breaks.",
                "triggering_condition": f"Current recovery setting is '{recovery_env}', offering zero or compromised thermal dissipation.",
                "affected_exposure_factor": "recovery_dissipation_rate",
                "expected_direction_of_effect": "Establishes a favorable thermal gradient during rest breaks to facilitate core body cooling.",
                "confidence": "High",
                "source_reference": "ISO 7243:2017 Clause 6.3 (Recovery Enclosures)",
                "operational_scope": ["worker", "supervisor"],
                "priority_rank": 1,
            })

        # ─── Category E: CLOTHING / PPE (Hierarchy Level 6: PPE Considerations & Safety Guardrail) ───
        if clothing in ("heavy", "ppe"):
            is_vapor = (clothing == "ppe") and (
                ppe_category and any(k in str(ppe_category).lower() for k in ("chemical", "encapsulated", "barrier", "impermeable"))
            )
            interventions.append({
                "id": "PPE-01",
                "category": "CLOTHING_PPE",
                "control_hierarchy_level": 6,
                "control_hierarchy_name": CONTROL_HIERARCHY[6]["name"],
                "intervention": "Flag heat-retaining protective equipment; deploy microclimate cooling vests (DO NOT remove mandatory safety PPE)",
                "reason": "Protective clothing reduces sweat vapor permeability and traps body heat inside the microclimate.",
                "triggering_condition": f"Clothing/PPE category is '{clothing}' with thermal insulation penalty +{CLOTHING_ADJUSTMENT_VALUES.get(clothing, 2.0)}°C WBGT.",
                "affected_exposure_factor": "clothing_vapor_resistance_clo",
                "expected_direction_of_effect": "Identifies PPE heat trapping as a key risk multiplier. SAFETY GUARDRAIL: Never instruct or authorize removal of mandatory safety PPE (e.g. chemical suits, respirators, fall protection). Instead shorten continuous exposure cycles or deploy passive ice/phase-change cooling vests under gear.",
                "confidence": "High",
                "source_reference": "ACGIH TLVs Clothing Adjustment Values & OSHA PPE Heat Guidance",
                "operational_scope": ["worker", "supervisor"],
                "priority_rank": 2,
            })
        elif clothing == "standard":
            interventions.append({
                "id": "PPE-02",
                "category": "CLOTHING_PPE",
                "control_hierarchy_level": 6,
                "control_hierarchy_name": CONTROL_HIERARCHY[6]["name"],
                "intervention": "Select lightweight, loose-fitting, light-colored breathable fabrics where safety protocols permit",
                "reason": "Light-colored breathable fabrics reflect radiant heat and permit unhindered sweat evaporation.",
                "triggering_condition": "Standard workwear configuration.",
                "affected_exposure_factor": "clothing_permeability",
                "expected_direction_of_effect": "Facilitates air circulation next to the skin without compromising workplace physical protection.",
                "confidence": "High",
                "source_reference": "NIOSH Work Clothing Guidance",
                "operational_scope": ["worker", "supervisor"],
                "priority_rank": 5,
            })

        # ─── Category F: COOLING ACCESS (Hierarchy Level 4: Community & Site Cooling) ───
        city_cooling_recs = CoolingRecommenderService.get_recommendations(target_city=location)
        if city_cooling_recs:
            top_rec = city_cooling_recs[0]
            interventions.append({
                "id": "COOL-01",
                "category": "COOLING_ACCESS",
                "control_hierarchy_level": 4,
                "control_hierarchy_name": CONTROL_HIERARCHY[4]["name"],
                "intervention": f"Utilize nearby cooling facilities / shelters in {location} ({top_rec.get('zone_name')})",
                "reason": f"Sited priority cooling corridor based on demographic exposure proxy and hospital proximity ({top_rec.get('nearest_facility', 'Municipal Center')}).",
                "triggering_condition": f"Municipal cooling infrastructure identified in active urban corridor ({top_rec.get('city')}).",
                "affected_exposure_factor": "proximate_cooling_infrastructure",
                "expected_direction_of_effect": "Provides rapid access to potable water and shaded/air-conditioned recovery zones for outdoor laborers.",
                "confidence": "Medium (Municipal Siting Advisory)",
                "source_reference": "HeatWave Guardian Cooling Recommender Service & NDMA Guidelines",
                "operational_scope": ["worker", "supervisor", "authority"],
                "priority_rank": 3,
                "facility_info": {
                    "zone_name": top_rec.get("zone_name"),
                    "nearest_facility": top_rec.get("nearest_facility"),
                    "distance_km": top_rec.get("distance_km"),
                },
            })

        # ─── Category G: OPERATIONAL ESCALATION (Hierarchy Level 5: Administrative) ───
        if risk_category in ("Very High", "Extreme") or wbgt_c >= 33.0:
            interventions.append({
                "id": "ESC-01",
                "category": "OPERATIONAL_ESCALATION",
                "control_hierarchy_level": 5,
                "control_hierarchy_name": CONTROL_HIERARCHY[5]["name"],
                "intervention": "Mandatory Supervisor Review: Suspend non-critical strenuous outdoor activities",
                "reason": f"Heat conditions have breached critical occupational screening limits (WBGT {wbgt_c}°C, Action Level: {risk_category}).",
                "triggering_condition": f"Overall Action Level reached '{risk_category}'.",
                "affected_exposure_factor": "operational_governance",
                "expected_direction_of_effect": "Escalates operational decision to site supervision; halts non-essential strenuous outdoor work.",
                "confidence": "High",
                "source_reference": "National Disaster Management Authority (NDMA) Heat Action Plan 2024",
                "operational_scope": ["supervisor", "authority"],
                "priority_rank": 1,
            })
        elif risk_category == "High":
            interventions.append({
                "id": "ESC-02",
                "category": "OPERATIONAL_ESCALATION",
                "control_hierarchy_level": 5,
                "control_hierarchy_name": CONTROL_HIERARCHY[5]["name"],
                "intervention": "Site-Level Escalation: Activate heat illness buddy checks and continuous water station verification",
                "reason": f"High heat stress (WBGT {wbgt_c}°C) elevates risk of early heat exhaustion without active surveillance.",
                "triggering_condition": "High thermal stress tier active.",
                "affected_exposure_factor": "supervisory_oversight",
                "expected_direction_of_effect": "Ensures prompt peer detection of dizziness, disorientation, or cessation of sweating.",
                "confidence": "High",
                "source_reference": "OSHA Buddy System Guidelines",
                "operational_scope": ["supervisor"],
                "priority_rank": 3,
            })

        # Mandatory Non-Prescriptive Hydration Reminder (Administrative)
        interventions.append({
            "id": "HYD-01",
            "category": "EXPOSURE_DURATION",
            "control_hierarchy_level": 5,
            "control_hierarchy_name": CONTROL_HIERARCHY[5]["name"],
            "intervention": "Maintain continuous potable water access; encourage 1 cup (250ml) every 20 minutes",
            "reason": "Profuse thermoregulatory sweating induces continuous fluid depletion before thirst sensation is triggered.",
            "triggering_condition": f"Thermal strain is active (WBGT {wbgt_c}°C).",
            "affected_exposure_factor": "hydration_fluid_balance",
            "expected_direction_of_effect": "Maintains blood volume and cardiovascular stability. Note: Workers with clinical fluid restrictions must follow medical provider instructions.",
            "confidence": "High",
            "source_reference": "ACGIH / OSHA Heat Illness Prevention Hydration Standard",
            "operational_scope": ["worker", "supervisor"],
            "priority_rank": 2,
        })

        # Acclimatization Guidance if unacclimatized (Administrative)
        if acclimatization in ("not_used", "somewhat_used", "unacclimatized"):
            interventions.append({
                "id": "ACC-01",
                "category": "OPERATIONAL_ESCALATION",
                "control_hierarchy_level": 5,
                "control_hierarchy_name": CONTROL_HIERARCHY[5]["name"],
                "intervention": "Implement progressive exposure ramp-up schedule for unacclimatized / returning personnel",
                "reason": "Unacclimatized individuals have delayed sweating onset, higher sweat electrolyte loss, and elevated heart rate.",
                "triggering_condition": f"Worker status is '{acclimatization.replace('_', ' ')}'.",
                "affected_exposure_factor": "physiological_acclimatization",
                "expected_direction_of_effect": "Limits Day 1 shift exposure to 20–50% to stimulate physiological adaptation without acute decompensation.",
                "confidence": "High",
                "source_reference": "NIOSH Acclimatization Protocol (20% Day 1, +20% per day)",
                "operational_scope": ["supervisor"],
                "priority_rank": 2,
            })

        return interventions

    # ──────────────────────────────────────────────────────────────────────────
    # Role-Specific Views (Worker, Supervisor, Authority)
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _build_role_views(
        cls,
        interventions: List[Dict[str, Any]],
        weather: Dict[str, Any],
        peak_window: Dict[str, Any],
        scheduling_candidates: Dict[str, Any],
        work_rest: Dict[str, Any],
        location: str,
        op_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Translates identical scientific thermal calculations into role-tailored views:
        1. WORKER: simple, direct, personal safety steps.
        2. SUPERVISOR / WORKSITE: shift scheduling, workforce exposure, engineering controls, escalation.
        3. AUTHORITY: municipal zones, event timing, cooling shelter deployment, HAP triggers.
        """
        # Worker View: Simple, direct, actionable
        worker_actions = []
        for i in interventions:
            if "worker" in i.get("operational_scope", []):
                worker_actions.append({
                    "action": i["intervention"],
                    "why": i["reason"],
                    "priority": "Immediate" if i["control_hierarchy_level"] <= 2 else "Routine",
                })

        worker_view = {
            "role": "worker",
            "title": "Worker Heat Safety Guidance",
            "current_condition": f"WBGT {weather['wbgt_c']}°C ({weather['risk']} Action Level)",
            "peak_period": peak_window.get("peak_period", "12:30–15:30"),
            "hourly_pacing": f"{work_rest['work_minutes']} min work / {work_rest['rest_minutes']} min rest",
            "simple_actions": worker_actions[:5],
            "safety_reminder": "Drink 1 cup of cool water every 20 minutes. Stop work and notify your buddy immediately if feeling faint, dizzy, or nauseous. Do not remove mandatory safety PPE.",
        }

        # Supervisor / Worksite View: Tactical workforce management
        sup_actions = [i for i in interventions if "supervisor" in i.get("operational_scope", [])]
        supervisor_view = {
            "role": "supervisor",
            "title": "Worksite & Supervisor Decision Support",
            "location": location,
            "operational_state": op_state["state"],
            "badge_color": op_state["badge_color"],
            "task_scheduling_options": {
                "peak_period": peak_window.get("peak_period", "12:30–15:30"),
                "recommended_candidate_windows": scheduling_candidates.get("candidate_windows", []),
            },
            "workforce_exposure_controls": {
                "work_rest_cycle": f"{work_rest['work_minutes']} min work / {work_rest['rest_minutes']} min rest",
                "instruction": work_rest["instruction"],
                "water_provision": "Verify cold potable water points within 50 meters of active worksites.",
            },
            "prioritized_interventions": sup_actions,
            "escalation_threshold": op_state["description"],
        }

        # Authority View: Municipal, zonal, and resource planning
        authority_actions = [
            {
                "action": "Monitor High-Vulnerability Municipal Zones",
                "detail": f"Active heat tracking for monitored municipal centers across {location}.",
                "source": "State Disaster Management Authority (SDMA) Guidelines",
            },
            {
                "action": "Public Event Timing Adjustment",
                "detail": f"Recommend rescheduling outdoor public gatherings away from peak window ({peak_window.get('peak_period')}).",
                "source": "NDMA National Heat Action Guidelines 2024",
            },
            {
                "action": "Cooling Shelter & Hydration Kiosk Activation",
                "detail": f"Deploy emergency water kiosks and open shaded transit shelters in dense informal labor hubs.",
                "source": "Municipal Disaster Management Plan",
            },
            {
                "action": "Field Verification Request",
                "detail": "Request on-site occupational safety compliance inspections for heavy outdoor construction corridors.",
                "source": "Department of Labor Standards",
            },
            {
                "action": "Review Vulnerable Institutional Facilities",
                "detail": "Issue heat advisory bulletins to hospitals, schools, and transit terminals.",
                "source": "Integrated Health & HAP Network",
            },
        ]

        authority_view = {
            "role": "authority",
            "title": "Municipal Authority & Disaster Management Planning",
            "location": location,
            "peak_risk_window": peak_window.get("peak_period", "12:30–15:30"),
            "operational_state": op_state["state"],
            "structured_authority_actions": authority_actions,
            "cooling_station_siting": CoolingRecommenderService.get_recommendations(target_city=location),
            "disclaimer": "Municipal advisory decision support. Resource prioritization reflects multi-criteria spatial indicators.",
        }

        return {
            "worker": worker_view,
            "supervisor": supervisor_view,
            "authority": authority_view,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Unified Decision Timeline
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _build_unified_timeline(
        cls,
        temp_c: float,
        rh_pct: float,
        wbgt_c: float,
        workload: str,
        peak_window: Dict[str, Any],
        interventions: List[Dict[str, Any]],
        duration_hours: float,
        start_time: str,
        hourly_series: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Creates an hour-by-hour decision timeline mapping:
        TIME -> FORECAST -> ML LOCALIZED CONDITION -> WBGT -> RISK -> PEAK PERIOD -> INTERVENTION -> STATUS.
        """
        start_hour = cls._parse_hour_int(start_time, 8)
        hours_to_show = max(6, min(14, int(math.ceil(duration_hours + 2))))
        timeline = []

        pk_start = cls._parse_hour_int(peak_window.get("elevated_risk_start"), 12)
        pk_end = cls._parse_hour_int(peak_window.get("elevated_risk_end"), 16)

        for step in range(hours_to_show):
            curr_h = (start_hour + step) % 24
            time_label = f"{curr_h:02d}:00"

            # Use real hourly forecast point if available, else diurnal curve
            pt = next((p for p in hourly_series if cls._get_hour(p) == curr_h), None)
            if pt:
                step_temp = float(pt.get("temp_c", temp_c))
                step_wbgt = float(pt.get("peak_wbgt_c", pt.get("wbgt_c", wbgt_c)))
                step_rh = float(pt.get("rh_pct", rh_pct))
            else:
                # Diurnal atmospheric offset
                offset = -3.5 * math.cos(math.pi * (curr_h - 4) / 11) if 4 <= curr_h <= 19 else -4.0
                step_temp = round(temp_c + offset * 0.4, 1)
                step_rh = max(15.0, min(90.0, round(rh_pct - offset * 1.2, 1)))
                step_wbgt = round(estimate_wbgt(temp_c=step_temp, rh_pct=step_rh, wind_ms=2.0, solar_wm2=600.0 if 7 <= curr_h <= 17 else 0.0), 1)

            step_band = classify_wbgt_band(step_wbgt)
            step_risk = fuse_risk(step_band, "Normal")
            is_peak = (pk_start <= curr_h < pk_end)

            # Assign operational status based on WBGT and peak
            if step_wbgt >= 33.0 or (step_risk in ("Very High", "Extreme")):
                status = "AUTHORITY REVIEW" if step_wbgt >= 35.0 else "SUPERVISOR REVIEW"
                badge_color = "red" if status == "SUPERVISOR REVIEW" else "purple"
                action_text = "Suspend non-critical outdoor labor; continuous conditioned recovery."
            elif step_wbgt >= 30.0 or is_peak:
                status = "MODIFY / RESTRICT EXPOSURE"
                badge_color = "orange"
                action_text = "Enforce 30m work / 30m rest; rotate workers into shade."
            elif step_wbgt >= 28.0:
                status = "ENHANCED CONTROLS"
                badge_color = "amber"
                action_text = "Provide 15m rest per hour; mandatory hydration breaks."
            else:
                status = "ROUTINE MONITORING"
                badge_color = "emerald"
                action_text = "Standard work pacing; continuous water access."

            # Primary matching intervention for this step
            step_int = "Avoid peak sun; hydrate" if is_peak else "Routine precautions"

            timeline.append({
                "time": time_label,
                "hour": curr_h,
                "forecast_heat": f"{step_temp:.1f}°C ({step_rh:.0f}% RH)",
                "ml_localized_temp_c": step_temp,
                "wbgt_c": step_wbgt,
                "risk": step_risk,
                "is_peak_period": is_peak,
                "primary_intervention": step_int,
                "operational_status": status,
                "badge_color": badge_color,
                "recommended_action": action_text,
            })

        return timeline

    # ──────────────────────────────────────────────────────────────────────────
    # Structured Alert Object Preparation
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _prepare_structured_alerts(
        cls,
        location: str,
        risk_category: str,
        peak_window: Dict[str, Any],
        interventions: List[Dict[str, Any]],
        op_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Prepares structured alert objects for downstream messaging adapters
        (SMS, WhatsApp, dashboard push, webhooks) with full provenance metadata.
        """
        roles = ["worker", "supervisor", "authority"]
        alerts = []
        pk_period = peak_window.get("peak_period", "12:30–15:30")
        primary_int = interventions[0]["intervention"] if interventions else "Maintain hydration and shaded recovery."
        primary_reason = interventions[0]["reason"] if interventions else f"Elevated thermal stress ({risk_category})."

        for role in roles:
            alert_obj = {
                "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
                "recipient_role": role,
                "location": location,
                "risk": risk_category,
                "peak_period": pk_period,
                "intervention": primary_int,
                "reason": primary_reason,
                "operational_state": op_state["state"],
                "confidence": peak_window.get("confidence", "Medium"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_version_provenance": {
                    "engine_version": cls.ENGINE_VERSION,
                    "ml_model_version": "v1.2-ensemble-rf",
                    "wbgt_physics": "Liljegren (2008)",
                    "occupational_standard": "ISO 7243:2017 / ACGIH",
                    "hap_standard": "NDMA National Heat Wave Guidelines 2024",
                    "data_source": "Open-Meteo NWP + Hyperlocal ML Grid",
                    "spatial_resolution": "Point / ~1km downscaled",
                    "status": "decision_support_advisory",
                },
            }
            alerts.append(alert_obj)

        return alerts

    # ──────────────────────────────────────────────────────────────────────────
    # What-If Simulator Connection (Intervention Simulation)
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _simulate_intervention(
        cls,
        weather: Dict[str, Any],
        baseline_exp: Dict[str, Any],
        modifications: Dict[str, Any],
        location: str,
        region: str,
    ) -> Dict[str, Any]:
        """
        Connects proposed interventions directly to the What-If ScenarioSimulator.
        Generates baseline vs simulated scenario comparison labeled 'SIMULATED SCENARIO'.
        """
        # Baseline inputs
        sim_payload = {
            "location": location,
            "region": region,
            "baseline_weather": {
                "temperature_c": weather["temp_c"],
                "relative_humidity_pct": weather["rh_pct"],
                "wind_speed_ms": weather.get("wind_ms", 2.0),
                "solar_radiation_wm2": weather.get("solar_wm2", 600.0),
            },
            "baseline_exposure": baseline_exp,
            "scenario": {
                "workload": modifications.get("workload", baseline_exp["workload"]),
                "clothing": modifications.get("clothing", baseline_exp["clothing"]),
                "acclimatization": modifications.get("acclimatization", baseline_exp["acclimatization"]),
                "recovery_environment": modifications.get("recovery_environment", baseline_exp["recovery_environment"]),
                "exposure_duration_hours": modifications.get("exposure_duration_hours", baseline_exp["exposure_duration_hours"]),
                "work_start_time": modifications.get("work_start_time", baseline_exp.get("work_start_time", "08:00")),
                "is_sun_exposed": modifications.get("is_sun_exposed", baseline_exp.get("is_sun_exposed", True)),
                # If shifting window, solar is attenuated
                "solar_radiation_wm2": 150.0 if not modifications.get("is_sun_exposed", baseline_exp.get("is_sun_exposed", True)) else weather.get("solar_wm2", 600.0),
            },
        }

        try:
            res = ScenarioSimulator.simulate(sim_payload)
            # Ensure honest labeling
            res["simulation_label"] = "SIMULATED SCENARIO"
            res["is_simulated"] = True
            return res
        except Exception as e:
            return {
                "status": "simulation_error",
                "error": str(e),
                "simulation_label": "SIMULATED SCENARIO",
                "is_simulated": True,
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: Control Hierarchy Summary
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_hierarchy_summary(interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarizes intervention distribution across the 6 hierarchy levels."""
        summary = {}
        for level, info in CONTROL_HIERARCHY.items():
            level_recs = [i for i in interventions if i.get("control_hierarchy_level") == level]
            summary[f"level_{level}"] = {
                "level": level,
                "name": info["name"],
                "short": info["short"],
                "count": len(level_recs),
                "has_recommendations": len(level_recs) > 0,
            }
        return summary
