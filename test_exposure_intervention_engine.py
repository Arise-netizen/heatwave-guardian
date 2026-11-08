"""
test_exposure_intervention_engine.py
====================================
Comprehensive test suite for the HeatWave Guardian Work/Rest + Exposure
Intervention Engine.

Validates:
1. Workload variations (Light, Moderate, Heavy, Very Heavy).
2. Acclimatization status variations (Acclimatized, Partially Acclimatized, Unacclimatized).
3. Clothing and PPE variations (Light, Heavy, Impermeable PPE, Unknown PPE).
4. Recovery environment variations (Cool Indoor, Shaded, Direct Sun, Warm Indoor, Unknown).
5. Environmental heat variations across WBGT spectrum.
6. Incomplete / missing data handling ("Site-specific assessment required").
7. Sensitivity checks (Workload, Acclimatization, Recovery environment, Irrelevant fields).
8. Peak-risk window detection from forecast vs fallback.
9. Safety language and framing audit (prohibition of unsafe claims).
10. API endpoints schema and backward compatibility contract.
"""

import pytest
import json
from unittest.mock import patch

from engine import (
    RECOMMENDATION_LEVELS,
    RECOVERY_ENVIRONMENTS,
    WORKLOAD_METABOLIC_WATTS,
    normalize_workload,
    normalize_acclimatization,
    normalize_clothing,
    normalize_recovery_environment,
    build_exposure_context,
    evaluate_peak_risk_windows,
    evaluate_exposure_intervention,
    get_work_rest,
    assess_full_risk_profile,
)
import app as flask_app


# ==============================================================================
# 1. Workload Variations
# ==============================================================================
class TestWorkloadVariations:
    def test_light_workload_allows_more_work_time(self):
        ctx_light = build_exposure_context(wbgt_c=31.0, workload="light")
        res_light = evaluate_exposure_intervention(ctx_light)

        ctx_heavy = build_exposure_context(wbgt_c=31.0, workload="heavy")
        res_heavy = evaluate_exposure_intervention(ctx_heavy)

        assert res_light["work_minutes"] >= res_heavy["work_minutes"]
        assert any("heavy" in b.lower() or "400w" in b.lower() for b in res_heavy["schedule_context"]["basis"])
        assert any(c["category"] == "WORKLOAD" for c in res_heavy["recommended_controls"])

    def test_very_heavy_workload_requires_substantial_recovery(self):
        ctx = build_exposure_context(wbgt_c=30.0, workload="very_heavy")
        res = evaluate_exposure_intervention(ctx)

        assert res["rest_minutes"] >= 30
        assert res["recommendation_level"] in [
            "Frequent recovery needed",
            "High-exposure controls required",
            "Suspend/defer strenuous exposure pending site assessment"
        ]
        assert any("metabolic" in c["reason"].lower() or "500w" in c["reason"].lower() for c in res["recommended_controls"])


# ==============================================================================
# 2. Acclimatization Status
# ==============================================================================
class TestAcclimatizationVariations:
    def test_unacclimatized_worker_triggers_ramp_up_intervention(self):
        ctx = build_exposure_context(wbgt_c=29.0, acclimatization="not_used")
        res = evaluate_exposure_intervention(ctx)

        acc_controls = [c for c in res["recommended_controls"] if c["category"] == "ACCLIMATIZATION"]
        assert len(acc_controls) > 0
        assert any("ramp" in c["action"].lower() or "acclimatization" in c["action"].lower() for c in acc_controls)
        assert any("unacclimatized" in b.lower() for b in res["recommendation_basis"])

    def test_partially_acclimatized_worker_guidance(self):
        ctx = build_exposure_context(wbgt_c=29.0, acclimatization="somewhat_used")
        res = evaluate_exposure_intervention(ctx)

        acc_controls = [c for c in res["recommended_controls"] if c["category"] == "ACCLIMATIZATION"]
        assert len(acc_controls) > 0
        assert any("progressive" in c["action"].lower() or "gradual" in c["action"].lower() or "partially" in c["action"].lower() for c in acc_controls)

    def test_acclimatized_worker_has_standard_pacing(self):
        ctx = build_exposure_context(wbgt_c=28.0, acclimatization="very_used")
        res = evaluate_exposure_intervention(ctx)
        # Acclimatized workers at mild heat should not be forced into high-exposure tiers
        assert res["recommendation_level"] in ["Normal precautions", "Enhanced precautions"]


# ==============================================================================
# 3. Clothing & PPE Variations
# ==============================================================================
class TestClothingAndPPEVariations:
    def test_light_clothing_baseline(self):
        ctx = build_exposure_context(wbgt_c=29.0, clothing="light")
        assert ctx["clothing_and_ppe"]["clothing_adjustment_c"] == 0.0

    def test_heavy_clothing_adds_thermal_penalty(self):
        ctx = build_exposure_context(wbgt_c=29.0, clothing="heavy")
        assert ctx["clothing_and_ppe"]["clothing_adjustment_c"] == 2.0
        res = evaluate_exposure_intervention(ctx)
        assert any(c["category"] == "CLOTHING_PPE" for c in res["recommended_controls"])

    def test_ppe_vapor_barrier_triggers_mandatory_monitoring(self):
        ctx = build_exposure_context(wbgt_c=29.0, clothing="ppe", ppe_category="encapsulated_chemical_suit")
        assert ctx["clothing_and_ppe"]["is_vapor_barrier"] is True
        assert ctx["clothing_and_ppe"]["clothing_adjustment_c"] >= 10.0
        res = evaluate_exposure_intervention(ctx)
        assert res["recommendation_level"] in [
            "High-exposure controls required",
            "Suspend/defer strenuous exposure pending site assessment"
        ]
        assert any(c["category"] == "MONITORING" and c["priority"] == "mandatory" for c in res["recommended_controls"])

    def test_unknown_ppe_flags_uncertainty(self):
        ctx = build_exposure_context(wbgt_c=28.0, clothing="unknown")
        assert ctx["clothing_and_ppe"]["has_uncertainty"] is True
        res = evaluate_exposure_intervention(ctx)
        assert any("clothing" in lim.lower() or "ppe" in lim.lower() for lim in res["limitations"])


# ==============================================================================
# 4. Recovery Environment Variations
# ==============================================================================
class TestRecoveryEnvironmentVariations:
    def test_cool_indoor_recovery_provides_optimal_dissipation(self):
        ctx = build_exposure_context(wbgt_c=31.0, recovery_environment="cool_indoor")
        res = evaluate_exposure_intervention(ctx)
        assert "air-conditioned" in res["recovery_guidance"].lower() or "cool" in res["recovery_guidance"].lower()

    def test_direct_sun_recovery_escalates_precaution(self):
        ctx = build_exposure_context(wbgt_c=30.0, recovery_environment="outdoor_direct")
        res = evaluate_exposure_intervention(ctx)
        # Direct sun recovery provides ineffective heat removal
        assert res["rest_minutes"] >= 30
        assert any(c["category"] == "RECOVERY" for c in res["recommended_controls"])
        assert any("direct sun" in c["reason"].lower() or "compromised" in c["action"].lower() for c in res["recommended_controls"])

    def test_warm_unconditioned_recovery_triggers_air_movement_advice(self):
        ctx = build_exposure_context(wbgt_c=30.0, recovery_environment="warm_indoor")
        res = evaluate_exposure_intervention(ctx)
        assert any("airflow" in c["action"].lower() or "ventilation" in c["action"].lower() or "misting" in c["action"].lower()
                   for c in res["recommended_controls"])

    def test_unknown_recovery_setting_flagged(self):
        ctx = build_exposure_context(wbgt_c=30.0, recovery_environment="unknown")
        assert ctx["recovery_environment"]["environment_type"] == "unknown"
        res = evaluate_exposure_intervention(ctx)
        assert any("recovery area" in lim.lower() or "recovery" in lim.lower() for lim in res["limitations"])


# ==============================================================================
# 5. Environmental Heat Spectrum Variations
# ==============================================================================
class TestEnvironmentalHeatSpectrum:
    @pytest.mark.parametrize("wbgt,expected_level", [
        (25.0, "Normal precautions"),
        (28.5, "Enhanced precautions"),
        (30.5, "Frequent recovery needed"),
        (32.5, "High-exposure controls required"),
        (35.0, "Suspend/defer strenuous exposure pending site assessment"),
    ])
    def test_recommendation_levels_across_wbgt(self, wbgt, expected_level):
        ctx = build_exposure_context(wbgt_c=wbgt, workload="moderate", acclimatization="very_used", clothing="light")
        res = evaluate_exposure_intervention(ctx)
        assert res["recommendation_level"] == expected_level

    def test_recommendation_level_constants_exist(self):
        for lvl in [
            "Normal precautions",
            "Enhanced precautions",
            "Frequent recovery needed",
            "High-exposure controls required",
            "Suspend/defer strenuous exposure pending site assessment",
            "Site-specific assessment required"
        ]:
            assert lvl in RECOMMENDATION_LEVELS


# ==============================================================================
# 6. Incomplete / Missing Data Handling
# ==============================================================================
class TestIncompleteMissingData:
    def test_missing_wbgt_yields_site_assessment_required(self):
        ctx = build_exposure_context(wbgt_c=None)
        res = evaluate_exposure_intervention(ctx)
        assert res["recommendation_level"] == "Site-specific assessment required"
        assert any("site-specific assessment required" in res["instruction"].lower() or "missing" in res["instruction"].lower() for _ in [1])
        assert len(res["limitations"]) >= 2

    def test_legacy_get_work_rest_with_none_risk(self):
        res = get_work_rest(None)
        assert res["recommendation_level"] == "Site-specific assessment required"
        assert res["work_minutes"] == 30
        assert res["rest_minutes"] == 30


# ==============================================================================
# 7. Sensitivity Analysis
# ==============================================================================
class TestSensitivityAnalysis:
    def test_workload_change_alters_recommendation(self):
        ctx_l = build_exposure_context(wbgt_c=31.0, workload="light")
        ctx_h = build_exposure_context(wbgt_c=31.0, workload="heavy")
        res_l = evaluate_exposure_intervention(ctx_l)
        res_h = evaluate_exposure_intervention(ctx_h)

        assert res_l["work_minutes"] != res_h["work_minutes"]
        assert res_l["work_guidance"] != res_h["work_guidance"]

    def test_unacclimatized_change_escalates_guidance(self):
        ctx_acc = build_exposure_context(wbgt_c=29.0, acclimatization="very_used")
        ctx_unacc = build_exposure_context(wbgt_c=29.0, acclimatization="not_used")
        res_acc = evaluate_exposure_intervention(ctx_acc)
        res_unacc = evaluate_exposure_intervention(ctx_unacc)

        assert len(res_unacc["recommended_controls"]) > len(res_acc["recommended_controls"])
        assert any(c["category"] == "ACCLIMATIZATION" for c in res_unacc["recommended_controls"])

    def test_recovery_setting_change_modifies_advice(self):
        ctx_cool = build_exposure_context(wbgt_c=31.0, recovery_environment="cool_indoor")
        ctx_sun = build_exposure_context(wbgt_c=31.0, recovery_environment="outdoor_direct")
        res_cool = evaluate_exposure_intervention(ctx_cool)
        res_sun = evaluate_exposure_intervention(ctx_sun)

        assert res_cool["recovery_guidance"] != res_sun["recovery_guidance"]
        assert res_sun["rest_minutes"] >= res_cool["rest_minutes"]

    def test_irrelevant_field_change_does_not_change_controls(self):
        ctx1 = build_exposure_context(wbgt_c=30.0, workload="moderate")
        ctx2 = build_exposure_context(wbgt_c=30.0, workload="moderate")
        ctx2["metadata_note"] = "completely_irrelevant_tag_123"

        res1 = evaluate_exposure_intervention(ctx1)
        res2 = evaluate_exposure_intervention(ctx2)

        assert res1["recommendation_level"] == res2["recommendation_level"]
        assert res1["work_minutes"] == res2["work_minutes"]
        assert res1["rest_minutes"] == res2["rest_minutes"]


# ==============================================================================
# 8. Peak-Risk Window Detection
# ==============================================================================
class TestPeakRiskWindows:
    def test_hourly_forecast_detects_peak_midday_window(self):
        hourly_series = [
            {"time": "2026-06-15T08:00", "temp_c": 30.0, "peak_wbgt_c": 27.0},
            {"time": "2026-06-15T10:00", "temp_c": 34.0, "peak_wbgt_c": 29.5},
            {"time": "2026-06-15T12:00", "temp_c": 41.0, "peak_wbgt_c": 33.5},
            {"time": "2026-06-15T14:00", "temp_c": 42.0, "peak_wbgt_c": 34.0},
            {"time": "2026-06-15T16:00", "temp_c": 38.0, "peak_wbgt_c": 31.0},
            {"time": "2026-06-15T18:00", "temp_c": 33.0, "peak_wbgt_c": 28.0},
        ]
        res = evaluate_peak_risk_windows(hourly_series)
        assert res["status"].lower() in ["available", "midday peak identified", "peak risk window detected"]
        windows = res.get("windows") or res.get("peak_windows", [])
        assert len(windows) >= 2
        # Morning should be present in detected windows
        assert any("08:00" in (w.get("window") or w.get("time_window", "")) for w in windows)

    def test_empty_forecast_series_returns_graceful_fallback(self):
        res = evaluate_peak_risk_windows([])
        assert res["status"].lower() == "unavailable"
        assert "cannot be estimated reliably" in res["message"]
        assert len(res.get("windows", [])) == 0


# ==============================================================================
# 9. Safety Language and Framing Audit
# ==============================================================================
class TestSafetyLanguageAudit:
    PROHIBITED_TERMS = [
        "guaranteed safe",
        "guarantee safety",
        "safe work/rest ratio",
        "drink exactly",
        "prevents heat illness",
        "cures heat exhaustion",
        "medical prescription",
        "will prevent all heat stress",
    ]

    def test_all_tiers_comply_with_safety_language(self):
        for lvl in ["Low", "Moderate", "High", "Very High", "Extreme", None]:
            for wl in ["light", "moderate", "heavy", "very_heavy"]:
                for acc in ["very_used", "somewhat_used", "not_used"]:
                    for rec in ["cool_indoor", "shade", "outdoor_direct"]:
                        res = get_work_rest(lvl, workload=wl, acclimatized=acc, recovery_environment=rec)
                        dumped = json.dumps(res).lower()
                        for term in self.PROHIBITED_TERMS:
                            assert term not in dumped, f"Prohibited term '{term}' found in output!"

    def test_decision_support_flag_is_always_true(self):
        res = get_work_rest("Moderate")
        assert res.get("is_decision_support") is True


# ==============================================================================
# 10. API Endpoints Contract & Backward Compatibility
# ==============================================================================
class TestAPIEndpointsContract:
    @pytest.fixture
    def client(self):
        flask_app.app.config["TESTING"] = True
        with flask_app.app.test_client() as client:
            yield client

    def test_api_check_risk_schema(self, client):
        mock_weather = {
            "temp_c": 36.5,
            "rh_pct": 52.0,
            "wind_ms": 2.2,
            "solar_wm2": 720.0,
            "wbgt_c": 31.4,
            "wbgt_band": "High",
            "thermal_stress_level": "High",
            "thermal_stress_description": "High thermal stress.",
            "is_day": 1,
            "condition": "Sunny",
            "timestamp": "2026-06-15T13:00",
            "source": "Mock"
        }

        with patch("app.fetch_current_weather", return_value=mock_weather):
            resp = client.post("/api/check-risk", json={
                "location": "Hyderabad",
                "latitude": 17.3850,
                "longitude": 78.4867,
                "age_group": "18–40",
                "experience": "very_used",
                "work_intensity": "heavy",
                "clothing": "heavy",
                "recovery_environment": "shade",
            })

            assert resp.status_code == 200
            data = resp.get_json()

            # Redesigned Schema checks
            assert "recommendation_level" in data
            assert data["recommendation_level"] in RECOMMENDATION_LEVELS
            assert "work_guidance" in data
            assert "recovery_guidance" in data
            assert "schedule_context" in data
            assert "recommended_controls" in data
            assert "recommendation_reasons" in data
            assert "limitations" in data
            assert "exposure_context" in data
            assert "peak_risk_window" in data

            # Legacy Schema backward compatibility checks
            assert "work_minutes" in data
            assert "rest_minutes" in data
            assert data["work_minutes"] + data["rest_minutes"] == 60
            assert "instruction" in data
            assert "hydration_guidance" in data

    def test_api_forecast_schema(self, client):
        mock_forecast = [
            {
                "date": "2026-06-16",
                "max_temp_c": 39.0,
                "peak_wbgt_c": 32.0,
                "hourly": [
                    {"time": "2026-06-16T10:00", "temp_c": 35.0, "peak_wbgt_c": 29.0},
                    {"time": "2026-06-16T13:00", "temp_c": 39.0, "peak_wbgt_c": 32.0},
                    {"time": "2026-06-16T17:00", "temp_c": 36.0, "peak_wbgt_c": 30.0},
                ]
            }
        ]

        with patch("app.fetch_forecast_weather", return_value=mock_forecast):
            resp = client.post("/api/forecast", json={
                "location": "Hyderabad",
                "latitude": 17.3850,
                "longitude": 78.4867,
                "age_group": "18–40",
                "work_intensity": "heavy",
                "experience": "not_used",
                "clothing": "ppe",
                "recovery_environment": "outdoor_direct",
            })

            assert resp.status_code == 200
            days = resp.get_json()
            assert len(days) == 1
            day = days[0]

            assert "recommendation_level" in day
            assert day["recommendation_level"] in RECOMMENDATION_LEVELS
            assert "work_guidance" in day
            assert "recovery_guidance" in day
            assert "schedule_context" in day
            assert "peak_risk_window" in day
            assert day["work_minutes"] + day["rest_minutes"] == 60
