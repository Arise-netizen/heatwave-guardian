"""
test_forecast_simulator.py
==========================
Comprehensive test suite for Peak-Risk Forecast + What-If Simulator layer.
Covers 52 distinct scenarios across 10 functional domains.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import patch

import app
import engine
from engine import (
    calculate_hourly_risk_profile,
    detect_peak_risk_windows,
    evaluate_peak_risk_windows,
    simulate_exposure_scenario,
    resolve_scenario_inputs,
    DEFAULT_EXPOSURE_SCENARIO,
    assess_full_risk_profile,
    calculate_wbgt,
)


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as c:
        yield c


# ==============================================================================
# Domain 1: Hourly Risk Profile Parsing & Validation (Tests 1-6)
# ==============================================================================
class TestHourlyRiskProfileParsing:

    def test_01_hourly_points_valid_parsing(self):
        hourly_data = [
            {"time": f"2026-06-15T{h:02d}:00", "temp_c": 28.0 + (h * 0.5), "humidity": 50.0, "wind_kph": 12.0, "solar_wm2": 400.0}
            for h in range(24)
        ]
        profiles = calculate_hourly_risk_profile(hourly_data, region="plains", location="Delhi")
        assert len(profiles) == 24
        assert profiles[0]["hour_str"] == "00:00"
        assert profiles[12]["hour_str"] == "12:00"

    def test_02_empty_or_none_hourly_points_returns_empty_list(self):
        assert calculate_hourly_risk_profile([]) == []
        assert calculate_hourly_risk_profile(None) == []  # type: ignore

    def test_03_non_dict_elements_skipped_gracefully(self):
        mixed = [
            {"time": "2026-06-15T08:00", "temp_c": 30.0, "humidity": 50.0},
            "invalid_string_entry",
            None,
            12345,
            {"time": "2026-06-15T09:00", "temp_c": 31.0, "humidity": 50.0},
        ]
        profiles = calculate_hourly_risk_profile(mixed, region="plains")
        assert len(profiles) == 2
        assert profiles[0]["hour_str"] == "08:00"
        assert profiles[1]["hour_str"] == "09:00"

    def test_04_missing_or_iso_timestamps_formatted(self):
        pts = [
            {"time": "2026-06-15T07:30:00Z", "temp_c": 29.0},
            {"hour": "08:30", "temp_c": 30.0},
            {"time_str": "09:30", "temp_c": 31.0},
        ]
        profiles = calculate_hourly_risk_profile(pts)
        assert profiles[0]["hour_str"] == "07:30"
        assert profiles[1]["hour_str"] == "08:30"
        assert profiles[2]["hour_str"] == "09:30"

    def test_05_extreme_and_mild_hourly_inputs(self):
        pts = [
            {"time": "2026-06-15T05:00", "temp_c": 19.0, "humidity": 45.0},
            {"time": "2026-06-15T14:00", "temp_c": 46.0, "humidity": 65.0, "solar_wm2": 950.0},
        ]
        profiles = calculate_hourly_risk_profile(pts, region="plains")
        assert len(profiles) == 2
        assert profiles[0]["overall_action_level"] in ["Low", "Moderate"]
        assert profiles[1]["overall_action_level"] in ["High", "Very High", "Extreme"]

    def test_06_hourly_profile_contains_all_required_keys(self):
        pts = [{"time": "2026-06-15T12:00", "temp_c": 36.0, "humidity": 50.0}]
        prof = calculate_hourly_risk_profile(pts)[0]
        required_keys = [
            "time", "hour_str", "temp_c", "humidity", "wind_kph", "solar_wm2",
            "wbgt_c", "wbgt_band", "environmental_level", "operational_exposure_level",
            "overall_action_level", "recommendation_level", "work_minutes", "rest_minutes",
            "hydration_rate_ml_hr", "primary_driver", "is_peak", "is_elevated_risk", "is_lower_exposure"
        ]
        for rk in required_keys:
            assert rk in prof, f"Missing key: {rk}"


# ==============================================================================
# Domain 2: Hourly WBGT & Normalization (Tests 7-11)
# ==============================================================================
class TestHourlyWBGTNormalization:

    def test_07_hourly_wbgt_preserved_when_present(self):
        pts = [{"time": "2026-06-15T10:00", "temp_c": 35.0, "wbgt_c": 31.4}]
        prof = calculate_hourly_risk_profile(pts)[0]
        assert prof["wbgt_c"] == 31.4

    def test_08_hourly_wbgt_fallback_to_peak_wbgt_c(self):
        pts = [{"time": "2026-06-15T10:00", "temp_c": 35.0, "peak_wbgt_c": 30.8}]
        prof = calculate_hourly_risk_profile(pts)[0]
        assert prof["wbgt_c"] == 30.8

    def test_09_hourly_wbgt_calculated_when_missing(self):
        pts = [{"time": "2026-06-15T12:00", "temp_c": 38.0, "humidity": 55.0, "solar_wm2": 700.0, "wind_kph": 10.0}]
        prof = calculate_hourly_risk_profile(pts)[0]
        assert prof["wbgt_c"] is not None
        assert 28.0 <= prof["wbgt_c"] <= 38.0

    def test_10_hourly_wind_kph_and_ms_conversion(self):
        pts = [{"time": "2026-06-15T12:00", "temp_c": 35.0, "wind_ms": 5.0}]
        prof = calculate_hourly_risk_profile(pts)[0]
        assert abs(prof["wind_kph"] - 18.0) < 0.2

    def test_11_hourly_rh_pct_and_humidity_flexibility(self):
        pts1 = [{"time": "2026-06-15T12:00", "temp_c": 35.0, "humidity": 60.0}]
        pts2 = [{"time": "2026-06-15T12:00", "temp_c": 35.0, "rh_pct": 60.0}]
        prof1 = calculate_hourly_risk_profile(pts1)[0]
        prof2 = calculate_hourly_risk_profile(pts2)[0]
        assert prof1["humidity"] == 60.0
        assert prof2["humidity"] == 60.0


# ==============================================================================
# Domain 3: Peak-Risk Window Detection (Tests 12-18)
# ==============================================================================
class TestPeakRiskWindowDetection:

    def test_12_insufficient_points_returns_unavailable(self):
        res = detect_peak_risk_windows([{"time": "12:00", "wbgt_c": 30.0}])
        assert res["status"] == "unavailable"
        assert res["is_reliable"] is False
        assert "cannot be estimated reliably" in res["message"]
        assert res["peak_period"] is None

    def test_13_typical_diurnal_curve_detects_midday_peak(self):
        hourly = [
            {"time": "2026-06-15T06:00", "temp_c": 27.0, "wbgt_c": 23.0},
            {"time": "2026-06-15T08:00", "temp_c": 30.0, "wbgt_c": 25.5},
            {"time": "2026-06-15T10:00", "temp_c": 34.0, "wbgt_c": 29.0},
            {"time": "2026-06-15T12:00", "temp_c": 39.0, "wbgt_c": 32.5},
            {"time": "2026-06-15T14:00", "temp_c": 41.0, "wbgt_c": 34.0},
            {"time": "2026-06-15T16:00", "temp_c": 38.0, "wbgt_c": 31.5},
            {"time": "2026-06-15T18:00", "temp_c": 33.0, "wbgt_c": 27.0},
            {"time": "2026-06-15T20:00", "temp_c": 30.0, "wbgt_c": 24.5},
        ]
        res = detect_peak_risk_windows(hourly)
        assert res["status"] == "available"
        assert res["is_reliable"] is True
        assert res["peak_hour"] == "14:00"
        assert res["peak_wbgt_c"] == 34.0
        assert res["has_elevated_risk"] is True

    def test_14_peak_hour_matches_max_wbgt(self):
        hourly = [
            {"time": f"2026-06-15T{h:02d}:00", "wbgt_c": 25.0 + (5.0 if h == 13 else 1.0)}
            for h in range(8, 18)
        ]
        res = detect_peak_risk_windows(hourly)
        assert res["peak_hour"] == "13:00"
        assert res["peak_wbgt_c"] == 30.0

    def test_15_elevated_risk_start_and_end_detected(self):
        hourly = [
            {"time": "2026-06-15T08:00", "wbgt_c": 25.0},
            {"time": "2026-06-15T10:00", "wbgt_c": 27.0},
            {"time": "2026-06-15T11:00", "wbgt_c": 30.5},
            {"time": "2026-06-15T13:00", "wbgt_c": 33.0},
            {"time": "2026-06-15T15:00", "wbgt_c": 30.2},
            {"time": "2026-06-15T17:00", "wbgt_c": 27.0},
        ]
        res = detect_peak_risk_windows(hourly)
        assert res["elevated_risk_start"] == "11:00"
        assert res["elevated_risk_end"] == "15:00"

    def test_16_peak_period_contiguous_grouping(self):
        hourly = [
            {"time": "2026-06-15T08:00", "wbgt_c": 25.0},
            {"time": "2026-06-15T10:00", "wbgt_c": 28.0},
            {"time": "2026-06-15T12:00", "wbgt_c": 31.0},
            {"time": "2026-06-15T14:00", "wbgt_c": 33.0},
            {"time": "2026-06-15T16:00", "wbgt_c": 30.5},
            {"time": "2026-06-15T18:00", "wbgt_c": 26.0},
        ]
        res = detect_peak_risk_windows(hourly)
        assert "12:00" in res["peak_period"]
        assert "16:00" in res["peak_period"]

    def test_17_environmental_drivers_identified(self):
        hourly = [
            {"time": f"2026-06-15T{h:02d}:00", "temp_c": 42.0 if h == 13 else 30.0, "wbgt_c": 34.0 if h == 13 else 25.0, "solar_wm2": 850.0 if h == 13 else 200.0, "humidity": 70.0 if h == 13 else 40.0, "wind_kph": 5.0}
            for h in range(8, 17)
        ]
        res = detect_peak_risk_windows(hourly)
        drivers = " ".join(res["drivers"])
        assert "solar" in drivers.lower() or "850" in drivers
        assert "humidity" in drivers.lower()
        assert "wind" in drivers.lower() or "5.0" in drivers

    def test_18_uncertainty_notice_and_confidence_present(self):
        hourly = [{"time": f"2026-06-15T{h:02d}:00", "wbgt_c": 28.0} for h in range(24)]
        res = detect_peak_risk_windows(hourly)
        assert "Forecast conditions may change" in res["uncertainty_notice"]
        assert res["confidence"] == "High"


# ==============================================================================
# Domain 4: Flat Diurnal Curve & Low Hazard (Tests 19-22)
# ==============================================================================
class TestFlatDiurnalCurve:

    def test_19_flat_curve_detected_when_delta_under_1deg(self):
        hourly = [
            {"time": f"2026-06-15T{h:02d}:00", "temp_c": 23.0 + (h * 0.03), "wbgt_c": 20.0 + (h * 0.02)}
            for h in range(8, 18)
        ]
        res = detect_peak_risk_windows(hourly)
        assert res["is_flat_curve"] is True

    def test_20_flat_curve_message_explicit(self):
        hourly = [
            {"time": f"2026-06-15T{h:02d}:00", "temp_c": 24.0, "wbgt_c": 21.0}
            for h in range(8, 18)
        ]
        res = detect_peak_risk_windows(hourly)
        assert "Flat diurnal thermal curve observed" in res["message"]

    def test_21_flat_curve_no_false_elevated_risk(self):
        hourly = [
            {"time": f"2026-06-15T{h:02d}:00", "temp_c": 23.0, "wbgt_c": 20.0}
            for h in range(8, 18)
        ]
        res = detect_peak_risk_windows(hourly)
        assert res["has_elevated_risk"] is False
        assert res["elevated_risk_start"] is None
        assert res["elevated_risk_end"] is None

    def test_22_flat_curve_above_28deg_not_flagged_as_low_hazard(self):
        hourly = [
            {"time": f"2026-06-15T{h:02d}:00", "temp_c": 36.0, "wbgt_c": 32.0}
            for h in range(8, 18)
        ]
        res = detect_peak_risk_windows(hourly)
        assert res["is_flat_curve"] is False


# ==============================================================================
# Domain 5: Lower-Exposure Window Detection (Tests 23-27)
# ==============================================================================
class TestLowerExposureWindows:

    def test_23_morning_daytime_lower_exposure_window_detected(self):
        hourly = [
            {"time": "2026-06-15T06:00", "wbgt_c": 24.0},
            {"time": "2026-06-15T07:00", "wbgt_c": 25.0},
            {"time": "2026-06-15T08:00", "wbgt_c": 26.5},
            {"time": "2026-06-15T11:00", "wbgt_c": 31.0},
            {"time": "2026-06-15T13:00", "wbgt_c": 34.0},
            {"time": "2026-06-15T15:00", "wbgt_c": 32.0},
            {"time": "2026-06-15T17:00", "wbgt_c": 29.0},
            {"time": "2026-06-15T19:00", "wbgt_c": 27.5},
        ]
        res = detect_peak_risk_windows(hourly)
        low_windows = res["lower_exposure_windows"]
        assert len(low_windows) >= 1
        assert any("06:00" in w["window"] for w in low_windows)

    def test_24_late_afternoon_lower_exposure_window_detected(self):
        hourly = [
            {"time": "2026-06-15T10:00", "wbgt_c": 32.0},
            {"time": "2026-06-15T12:00", "wbgt_c": 34.0},
            {"time": "2026-06-15T14:00", "wbgt_c": 33.0},
            {"time": "2026-06-15T16:00", "wbgt_c": 30.0},
            {"time": "2026-06-15T18:00", "wbgt_c": 27.0},
            {"time": "2026-06-15T19:00", "wbgt_c": 26.0},
        ]
        res = detect_peak_risk_windows(hourly)
        low_windows = res["lower_exposure_windows"]
        assert len(low_windows) >= 1
        assert any("18:00" in w["window"] for w in low_windows)

    def test_25_no_lower_exposure_window_on_extreme_heat_day(self):
        hourly = [
            {"time": f"2026-06-15T{h:02d}:00", "wbgt_c": 31.0 + (h * 0.2)}
            for h in range(6, 20)
        ]
        res = detect_peak_risk_windows(hourly)
        assert len(res["lower_exposure_windows"]) == 0

    def test_26_lower_exposure_window_excludes_night_hours(self):
        hourly = [
            {"time": "2026-06-15T02:00", "wbgt_c": 22.0},
            {"time": "2026-06-15T04:00", "wbgt_c": 22.5},
            {"time": "2026-06-15T11:00", "wbgt_c": 32.0},
            {"time": "2026-06-15T13:00", "wbgt_c": 34.0},
            {"time": "2026-06-15T15:00", "wbgt_c": 33.0},
            {"time": "2026-06-15T17:00", "wbgt_c": 31.0},
            {"time": "2026-06-15T23:00", "wbgt_c": 23.0},
        ]
        res = detect_peak_risk_windows(hourly)
        low_windows = res["lower_exposure_windows"]
        # Night hours (02:00, 04:00, 23:00) should not form daytime windows
        assert len(low_windows) == 0

    def test_27_lower_exposure_window_wbgt_range_accurate(self):
        hourly = [
            {"time": "2026-06-15T06:00", "wbgt_c": 23.5},
            {"time": "2026-06-15T07:00", "wbgt_c": 25.1},
            {"time": "2026-06-15T08:00", "wbgt_c": 26.8},
            {"time": "2026-06-15T12:00", "wbgt_c": 33.0},
            {"time": "2026-06-15T14:00", "wbgt_c": 34.0},
            {"time": "2026-06-15T16:00", "wbgt_c": 32.0},
        ]
        res = detect_peak_risk_windows(hourly)
        w = res["lower_exposure_windows"][0]
        assert w["wbgt_range_c"] == (23.5, 26.8)


# ==============================================================================
# Domain 6: What-If Simulator — Baseline and Defaults (Tests 28-32)
# ==============================================================================
class TestSimulatorBaselineAndDefaults:

    @pytest.fixture
    def sample_weather(self):
        return {
            "current_temp_c": 38.0,
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "humidity": 50.0,
            "wind_ms": 2.5,
            "wind_kph": 9.0,
            "solar_wm2": 650.0,
            "wbgt_c": 32.5,
        }

    def test_28_simulator_resolves_default_scenario_when_none(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs=None,
            scenario_inputs={"workload": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        assert sim["is_baseline_default"] is True
        assert sim["baseline_inputs_used"]["workload"] == DEFAULT_EXPOSURE_SCENARIO["workload"]

    def test_29_simulator_flags_is_baseline_default_correctly(self, sample_weather):
        sim_custom = simulate_exposure_scenario(
            baseline_inputs={"workload": "heavy"},
            scenario_inputs={"workload": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        assert sim_custom["is_baseline_default"] is False

    def test_30_simulator_scenario_falls_back_to_baseline(self, sample_weather):
        baseline = {"workload": "heavy", "clothing": "heavy", "recovery_environment": "shade"}
        scenario = {"workload": "light"}  # clothing and recovery omitted
        sim = simulate_exposure_scenario(
            baseline_inputs=baseline,
            scenario_inputs=scenario,
            weather_data=sample_weather,
            region="plains",
        )
        assert sim["scenario_inputs_used"]["clothing"] == "heavy"
        assert sim["scenario_inputs_used"]["recovery_environment"] == "shade"

    def test_31_simulator_tracks_parameter_changes(self, sample_weather):
        baseline = {"workload": "heavy", "clothing": "heavy"}
        scenario = {"workload": "light", "clothing": "light"}
        sim = simulate_exposure_scenario(
            baseline_inputs=baseline,
            scenario_inputs=scenario,
            weather_data=sample_weather,
            region="plains",
        )
        changed_params = [p["parameter"] for p in sim["parameter_changes"]]
        assert "workload" in changed_params
        assert "clothing" in changed_params

    def test_32_simulator_identical_inputs_returns_unchanged(self, sample_weather):
        inputs = {"workload": "moderate", "clothing": "standard", "recovery_environment": "shade"}
        sim = simulate_exposure_scenario(
            baseline_inputs=inputs,
            scenario_inputs=inputs,
            weather_data=sample_weather,
            region="plains",
        )
        assert len(sim["parameter_changes"]) == 0
        assert sim["action_level_change"]["direction"] == "unchanged"
        assert sim["action_level_change"]["delta_steps"] == 0
        assert sim["work_rest_change"]["work_delta_minutes"] == 0
        assert any("identical" in c.lower() for c in sim["causal_explanations"])


# ==============================================================================
# Domain 7: What-If Simulator — Causal Attribution (Tests 33-38)
# ==============================================================================
class TestSimulatorCausalAttribution:

    @pytest.fixture
    def sample_weather(self):
        return {
            "current_temp_c": 39.0,
            "temp_c": 39.0,
            "rh_pct": 52.0,
            "humidity": 52.0,
            "wind_ms": 2.0,
            "wind_kph": 7.2,
            "solar_wm2": 700.0,
            "wbgt_c": 33.2,
        }

    def test_33_workload_reduction_explains_metabolic_watts(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "heavy"},
            scenario_inputs={"workload": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        exps = " ".join(sim["causal_explanations"])
        assert "metabolic" in exps.lower()
        assert "400w" in exps.lower() or "150w" in exps.lower()

    def test_34_workload_increase_explains_metabolic_watts(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "light"},
            scenario_inputs={"workload": "very_heavy"},
            weather_data=sample_weather,
            region="plains",
        )
        exps = " ".join(sim["causal_explanations"])
        assert "metabolic" in exps.lower()
        assert "550w" in exps.lower() or "escalation" in exps.lower()

    def test_35_recovery_upgrade_explains_thermal_gradient(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"recovery_environment": "outdoor_direct"},
            scenario_inputs={"recovery_environment": "cool_indoor"},
            weather_data=sample_weather,
            region="plains",
        )
        exps = " ".join(sim["causal_explanations"])
        assert "gradient" in exps.lower() or "dissipation" in exps.lower() or "cooling" in exps.lower()

    def test_36_acclimatization_change_explains_sweat_and_cardiac(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"acclimatization": "not_used"},
            scenario_inputs={"acclimatization": "very_used"},
            weather_data=sample_weather,
            region="plains",
        )
        exps = " ".join(sim["causal_explanations"])
        assert "sweat" in exps.lower()
        assert "cardiovascular" in exps.lower() or "strain" in exps.lower()

    def test_37_clothing_change_explains_evaporative_resistance(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"clothing": "ppe"},
            scenario_inputs={"clothing": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        exps = " ".join(sim["causal_explanations"])
        assert "evaporative" in exps.lower() or "breathable" in exps.lower()

    def test_38_sun_exposure_elimination_explains_radiant_load(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"is_sun_exposed": True},
            scenario_inputs={"is_sun_exposed": False},
            weather_data=sample_weather,
            region="plains",
        )
        exps = " ".join(sim["causal_explanations"])
        assert "solar" in exps.lower() or "radiant" in exps.lower()


# ==============================================================================
# Domain 8: What-If Simulator — Safety, Disclaimers & Integrity (Tests 39-44)
# ==============================================================================
class TestSimulatorSafetyAndIntegrity:

    @pytest.fixture
    def sample_weather(self):
        return {
            "current_temp_c": 38.0,
            "temp_c": 38.0,
            "rh_pct": 50.0,
            "humidity": 50.0,
            "wind_ms": 2.5,
            "wind_kph": 9.0,
            "solar_wm2": 650.0,
            "wbgt_c": 32.5,
        }

    def test_39_no_fake_percentages_in_simulation_output(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "heavy"},
            scenario_inputs={"workload": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        all_text = " ".join(sim["causal_explanations"]).lower()
        prohibited_phrases = ["risk reduced by", "percent safer", "risk decreased by", "chance of heat stroke"]
        for p in prohibited_phrases:
            assert p not in all_text

    def test_40_no_fake_medical_or_clinical_guarantees(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "heavy"},
            scenario_inputs={"workload": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        all_text = (sim["disclaimer"] + " " + " ".join(sim["causal_explanations"])).lower()
        assert "guarantee" not in all_text or "not constitute individualized clinical diagnoses" in all_text
        assert "100% safe" not in all_text
        assert "prevents heat stroke" not in all_text

    def test_41_identical_weather_used_for_both_runs(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "heavy"},
            scenario_inputs={"workload": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        b_prof = sim["baseline_profile"]
        s_prof = sim["scenario_profile"]
        assert b_prof["current_temp_c"] == s_prof["current_temp_c"]
        assert b_prof["wbgt_c"] == s_prof["wbgt_c"]
        assert b_prof["base_risk"] == s_prof["base_risk"]

    def test_42_both_runs_use_exact_same_assess_full_risk_profile(self, sample_weather):
        with patch("engine.assess_full_risk_profile", wraps=assess_full_risk_profile) as mock_assess:
            simulate_exposure_scenario(
                baseline_inputs={"workload": "heavy"},
                scenario_inputs={"workload": "light"},
                weather_data=sample_weather,
                region="plains",
            )
            assert mock_assess.call_count == 2

    def test_43_disclaimer_present_in_output(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "heavy"},
            scenario_inputs={"workload": "light"},
            weather_data=sample_weather,
            region="plains",
        )
        assert "disclaimer" in sim
        assert "ISO 7243" in sim["disclaimer"]
        assert "decision support" in sim["disclaimer"].lower()

    def test_44_protective_factor_changes_tracked(self, sample_weather):
        sim = simulate_exposure_scenario(
            baseline_inputs={"recovery_environment": "outdoor_direct", "acclimatization": "not_used"},
            scenario_inputs={"recovery_environment": "cool_indoor", "acclimatization": "very_used"},
            weather_data=sample_weather,
            region="plains",
        )
        prot = sim["protective_factor_changes"]
        assert "added" in prot
        assert "removed" in prot


# ==============================================================================
# Domain 9: API Endpoints Contract & Intelligence View (Tests 45-48)
# ==============================================================================
class TestAPIEndpointsContract:

    @patch("app.fetch_forecast_weather")
    def test_45_api_forecast_default_view_returns_5_day_list(self, mock_forecast, client):
        mock_forecast.return_value = [
            {
                "date": f"2026-06-1{i}",
                "max_temp_c": 38.0,
                "peak_wbgt_c": 32.0,
                "hourly": [{"time": f"2026-06-1{i}T{h:02d}:00", "temp_c": 38.0, "wbgt_c": 32.0} for h in range(12)],
            }
            for i in range(5)
        ]
        resp = client.get("/api/forecast?location=Hyderabad")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) == 5

    @patch("app.fetch_forecast_weather")
    def test_46_api_forecast_enriched_with_simulator_and_peak_keys(self, mock_forecast, client):
        mock_forecast.return_value = [
            {
                "date": f"2026-06-1{i}",
                "max_temp_c": 38.0,
                "peak_wbgt_c": 32.0,
                "hourly": [{"time": f"2026-06-1{i}T{h:02d}:00", "temp_c": 38.0, "wbgt_c": 32.0} for h in range(12)],
            }
            for i in range(5)
        ]
        resp = client.get("/api/forecast?location=Hyderabad")
        day0 = resp.get_json()[0]
        assert "hourly_profile" in day0
        assert "peak_risk_window" in day0
        assert "lower_exposure_windows" in day0
        assert "peak_hour" in day0
        assert "peak_period" in day0
        assert "is_flat_curve" in day0

    @patch("app.fetch_forecast_weather")
    def test_47_api_forecast_view_intelligence_returns_wrapped_object(self, mock_forecast, client):
        mock_forecast.return_value = [
            {
                "date": f"2026-06-1{i}",
                "max_temp_c": 38.0,
                "peak_wbgt_c": 32.0,
                "hourly": [{"time": f"2026-06-1{i}T{h:02d}:00", "temp_c": 38.0, "wbgt_c": 32.0} for h in range(12)],
            }
            for i in range(5)
        ]
        resp = client.get("/api/forecast?location=Hyderabad&view=intelligence")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "forecast_days" in data
        assert len(data["forecast_days"]) == 5
        assert "current_action_level" in data
        assert "disclaimer" in data

    def test_48_api_simulate_scenario_post_endpoint(self, client):
        payload = {
            "baseline": {"workload": "heavy", "clothing": "heavy"},
            "scenario": {"workload": "light", "clothing": "light"},
            "weather_data": {
                "current_temp_c": 37.0,
                "temp_c": 37.0,
                "wbgt_c": 31.0,
                "humidity": 50.0,
                "rh_pct": 50.0,
                "wind_kph": 10.0,
                "wind_ms": 2.8,
                "solar_wm2": 600.0,
            }
        }
        resp = client.post("/api/simulate-scenario", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "action_level_change" in data
        assert "work_rest_change" in data
        assert "causal_explanations" in data
        assert data["action_level_change"]["direction"] == "improved"


# ==============================================================================
# Domain 10: Full Regression & System Integrity (Tests 49-52)
# ==============================================================================
class TestRegressionAndSystemIntegrity:

    def test_49_backward_compatibility_with_evaluate_peak_risk_windows(self):
        hourly_series = [
            {"time": "2026-06-15T08:00", "temp_c": 30.0, "peak_wbgt_c": 27.0},
            {"time": "2026-06-15T10:00", "temp_c": 34.0, "peak_wbgt_c": 29.5},
            {"time": "2026-06-15T12:00", "temp_c": 41.0, "peak_wbgt_c": 33.5},
            {"time": "2026-06-15T14:00", "temp_c": 42.0, "peak_wbgt_c": 34.0},
            {"time": "2026-06-15T16:00", "temp_c": 38.0, "peak_wbgt_c": 31.0},
            {"time": "2026-06-15T18:00", "temp_c": 33.0, "peak_wbgt_c": 28.0},
        ]
        res = evaluate_peak_risk_windows(hourly_series)
        assert res["status"].lower() == "available"
        assert len(res["windows"]) >= 2
        assert "peak_wbgt_c" in res

    def test_50_mild_heat_scenario_explains_unchanged_action_level(self):
        mild_weather = {
            "current_temp_c": 22.0,
            "temp_c": 22.0,
            "humidity": 45.0,
            "rh_pct": 45.0,
            "wind_ms": 3.0,
            "solar_wm2": 300.0,
            "wbgt_c": 19.5,
        }
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "moderate"},
            scenario_inputs={"workload": "light"},
            weather_data=mild_weather,
            region="plains",
        )
        assert sim["action_level_change"]["direction"] == "unchanged"
        assert any("mild" in c.lower() for c in sim["causal_explanations"])

    def test_51_extreme_heat_scenario_explains_sustained_curtailment(self):
        extreme_weather = {
            "current_temp_c": 47.0,
            "temp_c": 47.0,
            "humidity": 60.0,
            "rh_pct": 60.0,
            "wind_ms": 1.5,
            "solar_wm2": 950.0,
            "wbgt_c": 38.0,
        }
        sim = simulate_exposure_scenario(
            baseline_inputs={"workload": "very_heavy"},
            scenario_inputs={"workload": "heavy"},
            weather_data=extreme_weather,
            region="plains",
        )
        assert any("severe" in c.lower() or "extreme" in c.lower() for c in sim["causal_explanations"])

    def test_52_unrecognized_extra_keys_ignored_without_crash(self):
        weather = {
            "current_temp_c": 36.0,
            "temp_c": 36.0,
            "humidity": 50.0,
            "rh_pct": 50.0,
            "wind_ms": 2.0,
            "solar_wm2": 500.0,
            "favorite_animal": "tiger",
        }
        sim = simulate_exposure_scenario(
            baseline_inputs={"random_key_1": "abc"},
            scenario_inputs={"random_key_2": "xyz"},
            weather_data=weather,
            region="plains",
        )
        assert sim is not None
        assert "action_level_change" in sim
