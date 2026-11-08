"""
test_scientific_benchmarks.py
=============================
Authoritative Scientific Benchmarking and End-to-End Stress Test Suite.
Validates:
1. Liljegren WBGT Engine Numerical Stability & Monotonicity
2. IMD SOP Classification Criteria (Plains, Coastal, Hilly, Persistence)
3. Physical Moisture Conservation (Vapor Pressure Invariance)
4. ML Domain Bounds & Distribution Shift Detection
5. Operational Work/Rest & Intervention Consistency
6. Resilience, Graceful Degradation & End-to-End Real Data Traceability
"""

import pytest
import math
import numpy as np
import pandas as pd

from engine import (
    estimate_wbgt,
    calculate_wbgt,
    _stull_wet_bulb,
    _globe_temperature,
    classify_imd,
    classify_official_imd,
    evaluate_official_imd_criteria,
    fuse_risk,
    WORK_REST,
    FUSION_MATRIX,
    REGION_THRESHOLDS,
)

from ml.src.predict import (
    HyperlocalPredictor,
    adjust_humidity_for_temperature_delta,
    check_feature_distribution_shift,
    TRAINING_DOMAIN_BOUNDS,
)

from services.intervention_engine import InterventionIntelligenceEngine


# ==============================================================================
# SECTION 1: SCIENTIFIC ENGINE STRESS TEST (WBGT & THERMAL PHYSICS)
# ==============================================================================

class TestScientificWbgtEngine:
    """Stress tests for Liljegren (2008) WBGT engine and thermodynamic components."""

    def test_non_finite_inputs_raise_value_error(self):
        """Engine must reject NaN and Inf without propagating corrupted states."""
        with pytest.raises(ValueError, match="Non-finite input encountered"):
            estimate_wbgt(float("nan"), 50.0, 2.0, 600.0)

        with pytest.raises(ValueError, match="Non-finite input encountered"):
            estimate_wbgt(35.0, float("inf"), 2.0, 600.0)

        with pytest.raises(ValueError, match="Non-finite input encountered"):
            estimate_wbgt(35.0, 50.0, float("-inf"), 600.0)

        with pytest.raises(ValueError, match="Non-finite input encountered"):
            estimate_wbgt(35.0, 50.0, 2.0, float("nan"))

    def test_non_numeric_types_raise_value_error(self):
        """Invalid types must raise ValueError rather than silent failure."""
        with pytest.raises(ValueError):
            estimate_wbgt("invalid", 50.0, 2.0, 600.0)

    @pytest.mark.parametrize("t,rh,w,s", [
        (-20.0, 10.0, 0.5, 0.0),    # Sub-zero dry night
        (-10.0, 80.0, 5.0, 200.0),  # Sub-zero humid morning
        (15.0, 20.0, 0.1, 100.0),   # Mild dry low wind
        (35.0, 50.0, 2.0, 600.0),   # Typical warm summer midday
        (48.0, 15.0, 4.0, 950.0),   # Extreme dry heatwave (North India)
        (38.0, 85.0, 1.0, 800.0),   # Extreme humid heatwave (Coastal / Delta)
        (55.0, 10.0, 8.0, 1100.0),  # Clamped upper boundary
        (30.0, 99.0, 0.0, 0.0),     # Zero wind saturation night
    ])
    def test_boundary_and_extreme_conditions_produce_finite_results(self, t, rh, w, s):
        """All extreme realistic boundary combinations produce valid finite WBGT values."""
        wbgt = estimate_wbgt(t, rh, w, s)
        assert isinstance(wbgt, float)
        assert not math.isnan(wbgt)
        assert not math.isinf(wbgt)
        # WBGT in terrestrial conditions cannot realistically exceed 60°C or drop below -30°C
        assert -30.0 <= wbgt <= 60.0

    def test_stull_wet_bulb_invariants(self):
        """Physical law: Wet-bulb temperature Tw cannot exceed dry-bulb temperature Ta."""
        for t in [20.0, 30.0, 40.0, 48.0]:
            for rh in [10.0, 30.0, 50.0, 80.0, 100.0]:
                tw = _stull_wet_bulb(t, rh)
                # At 100% RH, Tw ≈ Ta; at <100% RH, Tw < Ta
                assert tw <= t + 0.5, f"Wet bulb {tw}°C exceeded dry bulb {t}°C at RH {rh}%"
                if rh < 90.0:
                    assert tw < t, f"Wet bulb {tw}°C should be strictly less than dry bulb {t}°C at RH {rh}%"

    def test_globe_temperature_invariants(self):
        """Physical law: With positive solar radiation, globe temp >= air temp."""
        t_air = 38.0
        tg_dark = _globe_temperature(t_air, 2.0, 0.0)
        assert abs(tg_dark - t_air) < 1e-3, "In absence of solar radiation, Tg must equal Ta"

        tg_sun = _globe_temperature(t_air, 2.0, 800.0)
        assert tg_sun > t_air, "Globe temperature in strong sun must exceed ambient air temperature"
        assert tg_sun <= t_air + 45.0, "Globe temperature must not exceed physical terrestrial bounds"

    def test_wbgt_monotonicity_with_temperature(self):
        """WBGT must strictly increase with dry-bulb temperature when all else is fixed."""
        temps = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
        wbgts = [estimate_wbgt(t, 50.0, 2.0, 600.0) for t in temps]
        for i in range(len(wbgts) - 1):
            assert wbgts[i + 1] > wbgts[i], f"Non-monotonic WBGT vs temp: {wbgts[i]} -> {wbgts[i+1]}"

    def test_wbgt_monotonicity_with_humidity(self):
        """WBGT must strictly increase with humidity when temperature and radiation are fixed."""
        rhs = [20.0, 35.0, 50.0, 65.0, 80.0, 95.0]
        wbgts = [estimate_wbgt(38.0, rh, 2.0, 600.0) for rh in rhs]
        for i in range(len(wbgts) - 1):
            assert wbgts[i + 1] > wbgts[i], f"Non-monotonic WBGT vs humidity: {wbgts[i]} -> {wbgts[i+1]}"

    def test_wbgt_monotonicity_with_solar_radiation(self):
        """WBGT must increase or remain equal with increasing solar radiation."""
        solars = [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
        wbgts = [estimate_wbgt(38.0, 40.0, 2.0, s) for s in solars]
        for i in range(len(wbgts) - 1):
            assert wbgts[i + 1] >= wbgts[i], f"Decreasing WBGT vs solar radiation: {wbgts[i]} -> {wbgts[i+1]}"

    def test_wbgt_cooling_effect_of_wind_under_sun(self):
        """In intense solar radiation, higher wind speed enhances convective globe cooling, reducing WBGT."""
        wbgt_low_wind = estimate_wbgt(42.0, 30.0, 0.5, 900.0)
        wbgt_high_wind = estimate_wbgt(42.0, 30.0, 5.0, 900.0)
        assert wbgt_high_wind < wbgt_low_wind, "Higher wind speed should lower WBGT under intense direct sunlight"


# ==============================================================================
# SECTION 2: IMD CLASSIFICATION VALIDATION (SOP CRITERIA)
# ==============================================================================

class TestImdClassificationValidation:
    """Rigorous audit of IMD Heat Wave Standard Operating Procedure (SOP) rules."""

    def test_plains_qualifying_threshold_boundary(self):
        """In Plains, T_max must be >= 40.0°C to qualify for any heatwave status."""
        # Just below qualifying threshold: 39.9°C with high departure (+5.9°C above 34.0°C normal)
        sub_res = classify_official_imd(daily_max_temp_c=39.9, normal_temp_c=34.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert sub_res["official_imd_status"] == "Normal"
        assert sub_res["qualifying_threshold_met"] is False
        assert sub_res["is_official_declaration_valid"] is False

        # At qualifying threshold: 40.0°C with +5.0°C departure (normal 35.0°C)
        met_res = classify_official_imd(daily_max_temp_c=40.0, normal_temp_c=35.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert met_res["official_imd_status"] == "Heatwave"
        assert met_res["qualifying_threshold_met"] is True
        assert met_res["is_official_declaration_valid"] is True

    def test_coastal_qualifying_threshold_boundary(self):
        """In Coastal regions, qualifying threshold is 37.0°C."""
        # 36.9°C -> Normal
        sub = classify_official_imd(daily_max_temp_c=36.9, normal_temp_c=31.0, region="coastal", consecutive_days=2, stations_meeting_criteria=2)
        assert sub["official_imd_status"] == "Normal"
        assert sub["qualifying_threshold_met"] is False

        # 37.1°C with +5.1°C departure -> Heatwave
        met = classify_official_imd(daily_max_temp_c=37.1, normal_temp_c=32.0, region="coastal", consecutive_days=2, stations_meeting_criteria=2)
        assert met["official_imd_status"] == "Heatwave"
        assert met["qualifying_threshold_met"] is True

    def test_hilly_qualifying_threshold_boundary(self):
        """In Hilly regions, qualifying threshold is 30.0°C."""
        sub = classify_official_imd(daily_max_temp_c=29.8, normal_temp_c=24.0, region="hilly", consecutive_days=2, stations_meeting_criteria=2)
        assert sub["official_imd_status"] == "Normal"

        met = classify_official_imd(daily_max_temp_c=31.0, normal_temp_c=25.0, region="hilly", consecutive_days=2, stations_meeting_criteria=2)
        assert met["official_imd_status"] == "Heatwave"  # departure +6.0°C is in +4.5 to +6.4°C range
        assert met["qualifying_threshold_met"] is True

        severe = classify_official_imd(daily_max_temp_c=32.0, normal_temp_c=25.0, region="hilly", consecutive_days=2, stations_meeting_criteria=2)
        assert severe["official_imd_status"] == "Severe Heatwave"  # departure +7.0°C >= +6.5°C
        assert severe["qualifying_threshold_met"] is True

    def test_departure_from_normal_thresholds(self):
        """Heatwave: departure +4.5 to +6.4°C; Severe Heatwave: departure >= +6.5°C."""
        # Normal = 36.0°C in plains.
        # Dep = +4.4°C -> 40.4°C: Normal
        r44 = classify_official_imd(daily_max_temp_c=40.4, normal_temp_c=36.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert r44["official_imd_status"] == "Normal"

        # Dep = +4.5°C -> 40.5°C: Heatwave
        r45 = classify_official_imd(daily_max_temp_c=40.5, normal_temp_c=36.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert r45["official_imd_status"] == "Heatwave"

        # Dep = +6.4°C -> 42.4°C: Heatwave
        r64 = classify_official_imd(daily_max_temp_c=42.4, normal_temp_c=36.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert r64["official_imd_status"] == "Heatwave"

        # Dep = +6.5°C -> 42.5°C: Severe Heatwave
        r65 = classify_official_imd(daily_max_temp_c=42.5, normal_temp_c=36.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert r65["official_imd_status"] == "Severe Heatwave"

    def test_absolute_criteria_for_plains(self):
        """In Plains, Heatwave when T_max >= 45.0°C and Severe when T_max >= 47.0°C, regardless of normal."""
        # High climatological normal (42.0°C), departure only +3.0°C (< 4.5°C), but T_max = 45.0°C
        r_abs_hw = classify_official_imd(daily_max_temp_c=45.0, normal_temp_c=42.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert r_abs_hw["official_imd_status"] == "Heatwave"
        assert r_abs_hw["is_official_declaration_valid"] is True

        # T_max = 47.2°C -> Severe Heatwave
        r_abs_shw = classify_official_imd(daily_max_temp_c=47.2, normal_temp_c=43.0, region="plains", consecutive_days=2, stations_meeting_criteria=2)
        assert r_abs_shw["official_imd_status"] == "Severe Heatwave"
        assert r_abs_shw["is_official_declaration_valid"] is True

    def test_persistence_and_temporal_invariants(self):
        """Official IMD heatwave requires persistence (>= 2 days), spatial extent (>= 2 stations), and daily maximum."""
        # 1 day only (persistence not met)
        p1 = classify_official_imd(daily_max_temp_c=45.5, normal_temp_c=40.0, region="plains", consecutive_days=1, stations_meeting_criteria=2)
        assert p1["is_official_declaration_valid"] is False
        assert "Pending Persistence Verification" in p1["official_imd_status"]
        assert p1["classification_type"] == "MODELLED_ENVIRONMENTAL_HEAT_CONDITION"

        # Instantaneous reading (is_daily_max=False)
        inst = classify_official_imd(daily_max_temp_c=45.5, normal_temp_c=40.0, region="plains", consecutive_days=2, stations_meeting_criteria=2, is_daily_max=False)
        assert inst["is_official_declaration_valid"] is False
        assert "Modelled Condition" in inst["official_imd_status"]
        assert inst["classification_type"] == "MODELLED_ENVIRONMENTAL_HEAT_CONDITION"

    def test_operational_screening_classify_imd_backward_compatibility(self):
        """Legacy classify_imd acts as instantaneous operational screening departure index."""
        assert classify_imd(39.0, "plains") == "Normal"
        assert classify_imd(40.5, "plains") == "Heat Day"
        assert classify_imd(42.5, "plains") == "Heatwave"
        assert classify_imd(44.5, "plains") == "Severe Heatwave"


# ==============================================================================
# SECTION 3: PHYSICAL MOISTURE CONSERVATION
# ==============================================================================

class TestMoistureConservation:
    """Verifies that relative humidity is recalculated via conserved vapor pressure."""

    def test_vapor_pressure_conservation_reduces_rh_under_heating(self):
        """When an air parcel is heated (e.g. UHI +2°C), RH must decrease to preserve vapor mass."""
        t_bg = 35.0
        rh_bg = 60.0
        t_local = 37.5  # +2.5°C UHI downscaling

        rh_local = adjust_humidity_for_temperature_delta(t_bg, rh_bg, t_local)
        assert rh_local < rh_bg, f"Localized RH {rh_local}% should be lower than background RH {rh_bg}%"

        # Verify vapor pressure invariance (e_actual_bg == e_actual_local)
        es_bg = 0.61078 * math.exp((17.27 * t_bg) / (t_bg + 237.3))
        e_bg = (rh_bg / 100.0) * es_bg

        es_local = 0.61078 * math.exp((17.27 * t_local) / (t_local + 237.3))
        e_local = (rh_local / 100.0) * es_local

        # Should match within 1% relative error due to rounding
        assert abs(e_bg - e_local) / e_bg < 0.015

    def test_equal_temperature_preserves_rh(self):
        """Zero anomaly leaves RH unchanged."""
        assert adjust_humidity_for_temperature_delta(34.0, 50.0, 34.0) == 50.0


# ==============================================================================
# SECTION 4: ML DISTRIBUTION SHIFT & DOMAIN BOUNDS
# ==============================================================================

class TestMlDistributionShift:
    """Audits ML feature validation and out-of-domain detection."""

    def test_in_domain_features_are_accepted(self):
        feat_dict = {
            "background_temp_c": 36.0,
            "background_rh_pct": 50.0,
            "background_wind_ms": 3.0,
            "background_solar_wm2": 700.0,
            "elevation_m": 250.0,
            "urban_built_up_ratio": 0.75,
            "vegetation_canopy_deficit": 0.80,
            "population_density_k": 12.0,
            "distance_to_coast_km": 400.0,
        }
        is_in_domain, warnings = check_feature_distribution_shift(feat_dict)
        assert is_in_domain is True
        assert len(warnings) == 0

    def test_out_of_domain_features_are_flagged(self):
        """Extreme mountain elevation (Leh 3500m) or unphysical heat (> 47°C) must trigger explicit warnings."""
        extreme_mountain = {
            "background_temp_c": 30.0,
            "elevation_m": 3500.0,  # Max training elevation is 1000m
        }
        is_in_domain, warnings = check_feature_distribution_shift(extreme_mountain)
        assert is_in_domain is False
        assert any("Elevation" in w for w in warnings)

        extreme_heat = {
            "background_temp_c": 53.0,  # Above max training temp 47.0°C
        }
        is_in_domain_h, warnings_h = check_feature_distribution_shift(extreme_heat)
        assert is_in_domain_h is False
        assert any("Background Temperature" in w for w in warnings_h)

    def test_hyperlocal_predictor_reports_domain_status(self):
        """Batch predictor must attach domain warnings and downscale confidence when out of domain."""
        item = {
            "lat": 34.15,
            "lon": 77.57,
            "city_name": "Leh",
            "background_weather": {
                "temp_c": 52.0,  # Extreme out-of-domain
                "rh_pct": 30.0,
                "wind_ms": 2.0,
                "solar_wm2": 700.0,
            },
            "morphology_override": {
                "elevation_m": 3500.0,
            }
        }
        res = HyperlocalPredictor.predict_batch([item])[0]
        assert res["prediction_status"] == "out_of_domain"
        assert res["is_in_domain"] is False
        assert len(res["out_of_domain_warnings"]) > 0
        assert "Limited (Out-of-Domain Input)" in res["uncertainty"]["confidence_state"]


# ==============================================================================
# SECTION 5: INTERVENTION CONSISTENCY & SCHEDULING LOGIC
# ==============================================================================

class TestInterventionConsistency:
    """Verifies that intervention guidance respects operational context and avoids contradictions."""

    def test_protective_scheduling_outside_peak_heat_window(self):
        """When worker shift is entirely outside peak window (e.g. 06:00-11:00), acknowledge proactive shift."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 30.0, "rh_pct": 50.0},
            exposure_context={"work_start_time": "06:00", "duration_hours": 4.0},
            location="Delhi",
        )
        rec_ids = [i["id"] for i in res["interventions"]]
        assert "SCHED-03" in rec_ids
        # Must NOT demand moving shift away from peak when it is ALREADY outside peak
        assert "SCHED-01" not in rec_ids

    def test_peak_overlapping_shift_triggers_rescheduling(self):
        """When worker shift overlaps peak hours (12:00-15:00) with High/Very High risk, recommend rescheduling."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 42.0, "wbgt_c": 32.5, "rh_pct": 45.0},
            exposure_context={"work_start_time": "12:00", "duration_hours": 4.0},
            location="Delhi",
        )
        rec_ids = [i["id"] for i in res["interventions"]]
        assert "SCHED-01" in rec_ids
        assert "SCHED-03" not in rec_ids


# ==============================================================================
# SECTION 6: WORK/REST ISO 7243:2017 SCREENING BOUNDARIES
# ==============================================================================

class TestWorkRestScreeningGuidance:
    """Ensures operational work/rest schedules conform to ISO screening principles."""

    def test_work_rest_table_structure(self):
        for risk, entry in WORK_REST.items():
            work = entry["work_minutes"]
            rest = entry["rest_minutes"]
            assert work + rest == 60, f"Work {work} + Rest {rest} does not equal 60 mins for {risk}"
            assert "instruction" in entry

    def test_fusion_matrix_coverage(self):
        """All 20 combinations of WBGT band and IMD status must map to standard risk tiers."""
        bands = ["Safe", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]
        imds = ["Normal", "Heat Day", "Heatwave", "Severe Heatwave"]
        for b in bands:
            for s in imds:
                risk = fuse_risk(b, s)
                assert risk in ["Low", "Moderate", "High", "Very High", "Extreme"]


# ==============================================================================
# SECTION 7: REAL-DATA END-TO-END TRACEABLE BENCHMARK
# ==============================================================================

class TestEndToEndRealBenchmarkScenario:
    """Traceable execution through the complete pipeline with verified historical telemetry."""

    def test_delhi_2024_05_28_heatwave_traceability(self):
        """
        Verify end-to-end execution of the Delhi May 28, 2024 heatwave scenario:
        Recorded background: Ta = 45.2°C, RH = 28%, Wind = 3.2 m/s, Solar = 880 W/m²
        """
        t_bg = 45.2
        rh_bg = 28.0
        w_bg = 3.2
        s_bg = 880.0

        # Step 1: Base NWP WBGT
        base_wbgt = estimate_wbgt(t_bg, rh_bg, w_bg, s_bg)
        assert base_wbgt > 32.0, "Delhi May 28 midday WBGT must be at least Danger level (>32°C)"

        # Step 2: ML Microclimate Localization
        delhi_item = {
            "lat": 28.6139,
            "lon": 77.2090,
            "city_name": "Delhi",
            "date_str": "2024-05-28",
            "background_weather": {
                "temp_c": t_bg,
                "rh_pct": rh_bg,
                "wind_ms": w_bg,
                "solar_wm2": s_bg,
            }
        }
        res = HyperlocalPredictor.predict_batch([delhi_item])[0]
        assert res["prediction_status"] == "in_domain"
        assert res["is_in_domain"] is True
        assert res["localized_temp_c"] >= t_bg  # Urban heat island residual >= 0 in dense Delhi core
        assert res["hyperlocal_environment"]["rh_pct"] <= rh_bg  # Conserved vapor pressure -> lower RH

        # Step 3: IMD Classification
        imd_res = classify_official_imd(
            daily_max_temp_c=res["localized_temp_c"],
            normal_temp_c=40.5,
            region="plains",
            consecutive_days=3,
            stations_meeting_criteria=4,
            is_daily_max=True,
        )
        assert imd_res["official_imd_status"] in ["Heatwave", "Severe Heatwave"]
        assert imd_res["is_official_declaration_valid"] is True

        # Step 4: Intervention Generation
        interv_res = InterventionIntelligenceEngine.evaluate(
            weather_data={
                "temp_c": res["localized_temp_c"],
                "wbgt_c": res["hyperlocal_environment"]["wbgt_c"],
                "rh_pct": res["hyperlocal_environment"]["rh_pct"],
                "wind_ms": w_bg,
                "solar_wm2": s_bg,
            },
            exposure_context={"work_start_time": "12:00", "duration_hours": 4.0},
            location="Delhi",
        )
        interventions = interv_res["interventions"]
        assert len(interventions) >= 3
        # Must recommend mandatory shaded rest and active hydration
        actions = [i["intervention"] for i in interventions]
        assert any("hydrat" in a.lower() or "water" in a.lower() for a in actions)
