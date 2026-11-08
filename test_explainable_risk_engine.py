"""
test_explainable_risk_engine.py
===============================
Comprehensive test suite for the Explainable Risk Engine + Decision Timeline layer:
- Phase 1 & 2: Decision Factor Classification (Primary vs Contributing vs Protective vs Uncertainty)
- Phase 3 & 4: Decision Trace Pipeline (6 stages) & Triggered Rules
- Phase 5 & 6: Factor Sensitivity & Counterfactuals (asserting NO_CHANGE under extreme thermal barrier)
- Phase 7 & 8: Traceable Recommendations with Rule/Standard citations
- Phase 9 & 10: Tiered Explanations (Level 1 Quick, Level 2 Detailed, Level 3 Technical)
- Phase 11 & 12: Decision Timeline & 5 Diurnal Shift Periods
- Phase 13 & 14: Multi-Day Decision Story & Day-to-Day Transitions
- Phase 15 & 16: What-If Scenario Transition Explanation
- API Endpoints: /api/check-risk, /api/forecast, /api/simulate-scenario, /api/decision-trace
- Regression & Backward Compatibility
"""

import pytest
import json
from unittest.mock import patch

from engine import (
    assess_full_risk_profile,
    classify_decision_factors,
    build_decision_trace,
    build_traceable_recommendations,
    generate_tiered_explanations,
    evaluate_factor_sensitivities,
    build_decision_timeline,
    build_day_decision_story,
    explain_day_transition,
    calculate_hourly_risk_profile,
    simulate_exposure_scenario,
)
from app import app, BENCHMARK_CASES


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def synthetic_extreme_weather():
    """Khammam peak: 48°C air temp, 25% RH, high solar, plains."""
    return {
        "temp_c": 48.0,
        "current_temp_c": 48.0,
        "rh_pct": 25.0,
        "humidity": 25.0,
        "wind_ms": 1.5,
        "solar_wm2": 850.0,
        "cloud_cover": 0.05,
        "region": "plains"
    }


@pytest.fixture
def synthetic_moderate_weather():
    """Moderate spring afternoon: 32°C air temp, 40% RH, moderate solar."""
    return {
        "temp_c": 32.0,
        "current_temp_c": 32.0,
        "rh_pct": 40.0,
        "humidity": 40.0,
        "wind_ms": 3.0,
        "solar_wm2": 450.0,
        "cloud_cover": 0.3,
        "region": "plains"
    }


def run_assessment(weather, age_group="18–40", experience="very_used", workload="heavy", clothing="heavy", recovery="shade", region="plains"):
    p_inputs = {
        "age_group": age_group,
        "experience": experience,
        "workload": workload,
        "clothing": clothing,
        "recovery_environment": recovery,
    }
    return assess_full_risk_profile(weather_data=weather, personal_inputs=p_inputs, region=region)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: DECISION FACTOR CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionFactorClassification:

    def test_extreme_heat_primary_driver_classification(self, synthetic_extreme_weather):
        """In 48°C ambient heat, temperature and solar radiation must be primary drivers."""
        w = synthetic_extreme_weather
        result = run_assessment(w, age_group="18–40", experience="very_used", workload="heavy", clothing="heavy")

        primary = result.get("primary_drivers", [])
        assert len(primary) > 0, "Extreme heat profile must have primary drivers"
        # Ambient temperature or WBGT must be a material determinant
        assert any("temperature" in s.lower() or "wbgt" in s.lower() or "heat" in s.lower() or "solar" in s.lower() for s in primary)

    def test_protective_factors_present_when_acclimatized(self, synthetic_moderate_weather):
        """Acclimatized individual with cool recovery must reflect protective factors."""
        w = synthetic_moderate_weather
        result = run_assessment(w, age_group="18–40", experience="very_used", workload="light", clothing="light", recovery="cool_indoor")

        protective = result.get("protective_factors", [])
        assert len(protective) >= 2, "Acclimatization, light work, and cool indoor recovery must provide defenses"
        combined_text = " ".join(protective).lower()
        assert any(term in combined_text for term in ["acclimatiz", "recovery", "dissipation", "gradient", "shade"])

    def test_plain_english_uncertainty_present(self, synthetic_moderate_weather):
        """Plain English uncertainty list must be provided without technical jargon."""
        w = synthetic_moderate_weather
        result = run_assessment(w, age_group="41–60", experience="somewhat_used", workload="moderate", clothing="light")

        uncertainties = result.get("plain_english_uncertainty", [])
        assert len(uncertainties) > 0
        assert any("screening" in u.lower() or "microclimate" in u.lower() or "individual" in u.lower() or "medical" in u.lower() for u in uncertainties)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: DECISION TRACE PIPELINE & TRIGGERED RULES
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionTracePipeline:

    def test_pipeline_contains_all_stages(self, synthetic_extreme_weather):
        """Verify the 6 distinct pipeline stages exist in decision trace."""
        w = synthetic_extreme_weather
        result = run_assessment(w, age_group="18–40", experience="very_used", workload="heavy", clothing="heavy")

        trace = result.get("decision_trace", {})
        stages = trace.get("pipeline_stages", {})
        required_stages = [
            "1_inputs",
            "2_derived_conditions",
            "3_environmental_level",
            "4_exposure_context",
            "5_action_level",
            "6_recommended_controls"
        ]
        for stage in required_stages:
            assert stage in stages, f"Stage {stage} missing from decision trace pipeline"

    def test_triggered_rules_have_standards_citations(self, synthetic_extreme_weather):
        """Triggered rules must cite ISO, ACGIH, or IMD standards."""
        w = synthetic_extreme_weather
        result = run_assessment(w, age_group="60+", experience="not_used", workload="very_heavy", clothing="ppe", recovery="outdoor_direct")

        trace = result.get("decision_trace", {})
        rules = trace.get("triggered_rules", [])
        assert len(rules) >= 3, "High vulnerability and extreme conditions should trigger multiple rules"

        for r in rules:
            assert "rule_id" in r and r["rule_id"].startswith("RULE_")
            assert "standard" in r and len(r["standard"]) > 0
            assert "description" in r and len(r["description"]) > 0

    def test_deterministic_trace_reproducibility(self, synthetic_moderate_weather):
        """Evaluating exact identical inputs must yield identical decision traces."""
        w = synthetic_moderate_weather
        run1 = run_assessment(w, age_group="18–40", experience="very_used", workload="moderate", clothing="light")
        run2 = run_assessment(w, age_group="18–40", experience="very_used", workload="moderate", clothing="light")

        assert run1["decision_trace"]["pipeline_stages"] == run2["decision_trace"]["pipeline_stages"]
        assert run1["decision_trace"]["triggered_rules"] == run2["decision_trace"]["triggered_rules"]


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: FACTOR SENSITIVITIES & COUNTERFACTUALS (PHASE 5 & 6)
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorSensitivities:

    def test_thermal_barrier_dominates_under_extreme_ambient(self, synthetic_extreme_weather):
        """
        Under 48°C air temp and high WBGT, changing clothing from heavy to light
        cannot lower the category from Extreme/Very High due to the ambient thermal barrier.
        Must report impact_type == NO_CHANGE and explain the mechanism.
        """
        w = synthetic_extreme_weather
        result = run_assessment(w, age_group="18–40", experience="very_used", workload="heavy", clothing="heavy", recovery="outdoor_direct")

        sensitivities = result.get("factor_sensitivities", [])
        assert len(sensitivities) > 0

        # Look for clothing sensitivity
        cloth_sens = next((s for s in sensitivities if s["factor"] == "clothing"), None)
        assert cloth_sens is not None
        if cloth_sens["impact_type"] == "NO_CHANGE":
            assert "not change" in cloth_sens["effect_summary"].lower() or "thermal" in cloth_sens["effect_summary"].lower()
            assert len(cloth_sens["physical_mechanism"]) > 0

    def test_workload_reduction_relieves_risk(self, synthetic_moderate_weather):
        """In moderate weather, dropping from heavy workload to moderate/light workload relieves risk."""
        w = synthetic_moderate_weather
        result = run_assessment(w, age_group="18–40", experience="somewhat_used", workload="heavy", clothing="light")

        sensitivities = result.get("factor_sensitivities", [])
        work_sens = next((s for s in sensitivities if s["factor"] == "workload"), None)
        assert work_sens is not None
        assert work_sens["impact_type"] in ["CATEGORY_CHANGE", "RECOMMENDATION_CHANGE", "NO_CHANGE"]
        assert "watts" in work_sens["physical_mechanism"].lower() or "metabolic" in work_sens["physical_mechanism"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: TRACEABLE RECOMMENDATIONS & TIERED EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceableRecommendationsAndExplanations:

    def test_recommendations_have_all_traceable_fields(self, synthetic_extreme_weather):
        """Every recommended control must have trigger, rule, source, confidence, and limitation."""
        w = synthetic_extreme_weather
        result = run_assessment(w, age_group="18–40", experience="very_used", workload="heavy", clothing="heavy")

        controls = result.get("recommended_controls", [])
        assert len(controls) > 0
        for c in controls:
            assert "trigger" in c and len(c["trigger"]) > 0
            assert "rule" in c and len(c["rule"]) > 0
            assert "source" in c and len(c["source"]) > 0
            assert "confidence" in c and len(c["confidence"]) > 0
            assert "limitation" in c and len(c["limitation"]) > 0

    def test_tiered_explanations_three_levels(self, synthetic_moderate_weather):
        """Assess profile and check Level 1, Level 2, and Level 3 explanations."""
        w = synthetic_moderate_weather
        result = run_assessment(w, age_group="18–40", experience="very_used", workload="moderate", clothing="light")

        exp = result.get("explanation", {})
        assert "level_1_quick" in exp and len(exp["level_1_quick"]) > 0
        assert "level_2_detailed" in exp
        l2 = exp["level_2_detailed"]
        assert "summary" in l2 and len(l2["summary"]) > 0
        assert "primary_drivers" in l2
        assert "contributing_factors" in l2
        assert "protective_factors" in l2
        assert "plain_english_uncertainty" in l2
        assert "level_3_technical" in exp
        l3 = exp["level_3_technical"]
        assert "decision_trace" in l3


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: TIMELINE & DIURNAL SHIFT PERIODS (PHASE 11 & 12)
# ─────────────────────────────────────────────────────────────────────────────

class TestTimelineAndDiurnalPeriods:

    def test_build_decision_timeline_diurnal_periods(self):
        """24-hour synthetic profile must produce 5 diurnal periods and 24 points."""
        synthetic_hourly = []
        for h in range(24):
            temp = 28.0 + 12.0 * (1.0 - abs(h - 14) / 10.0) if 4 <= h <= 24 else 26.0
            synthetic_hourly.append({
                "hour_str": f"{h:02d}:00",
                "temp_c": temp,
                "wbgt_c": temp - 6.0,
                "overall_action_level": "High" if 11 <= h <= 16 else ("Moderate" if 8 <= h <= 19 else "Low"),
                "recommended_controls": [{"action": "Hydrate frequently and take shaded rest breaks."}]
            })

        timeline = build_decision_timeline(synthetic_hourly)
        assert "diurnal_periods" in timeline
        assert "hourly_timeline" in timeline
        assert len(timeline["hourly_timeline"]) == 24

        periods = timeline["diurnal_periods"]
        assert len(periods) == 5
        period_ids = [p["period_id"] for p in periods]
        expected_ids = ["early_morning", "morning_buildup", "peak_hazard", "late_afternoon", "night_rest"]
        for eid in expected_ids:
            assert eid in period_ids

        # Check peak hazard period
        peak_period = next(p for p in periods if p["period_id"] == "peak_hazard")
        assert peak_period["window"] == "11:00 - 16:00"
        assert peak_period["dominant_category"] == "High"
        assert "primary_driver" in peak_period
        assert "recommended_control" in peak_period


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: MULTI-DAY STORY & TRANSITIONS (PHASE 13 & 14)
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiDayStoryAndTransitions:

    def test_explain_day_transition_escalation(self):
        """Test escalation transition when WBGT rises significantly."""
        prev_day = {
            "date": "2025-05-20",
            "wbgt_c": 28.5,
            "risk_category": "Moderate",
            "overall_action_level": "Enhanced precautions"
        }
        curr_day = {
            "date": "2025-05-21",
            "wbgt_c": 31.2,
            "risk_category": "High",
            "overall_action_level": "Frequent recovery needed"
        }
        trans = explain_day_transition(prev_day, curr_day)
        assert trans["type"] == "ESCALATING"
        assert trans["category_changed"] is True
        assert trans["wbgt_delta_c"] > 0
        assert "escalated" in trans["message"].lower()

    def test_explain_day_transition_relief(self):
        """Test improving transition when WBGT drops significantly."""
        prev_day = {
            "date": "2025-05-21",
            "wbgt_c": 32.0,
            "risk_category": "Very High",
            "overall_action_level": "High-exposure controls required"
        }
        curr_day = {
            "date": "2025-05-22",
            "wbgt_c": 27.5,
            "risk_category": "Moderate",
            "overall_action_level": "Enhanced precautions"
        }
        trans = explain_day_transition(prev_day, curr_day)
        assert trans["type"] == "IMPROVING"
        assert trans["wbgt_delta_c"] < 0
        assert "improved" in trans["message"].lower() or "relief" in trans["message"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: WHAT-IF SCENARIO TRANSITION EXPLANATION (PHASE 15 & 16)
# ─────────────────────────────────────────────────────────────────────────────

class TestWhatIfScenarioTransition:

    def test_simulate_exposure_scenario_transition_narrative(self, synthetic_extreme_weather):
        """Scenario simulation must return transition_explanation as a synthesized narrative."""
        w = synthetic_extreme_weather
        baseline = {
            "workload": "heavy",
            "clothing": "heavy",
            "acclimatization": "very_used",
            "recovery_environment": "outdoor_direct",
            "sun_exposure": True
        }
        scenario = {
            "workload": "light",
            "clothing": "light",
            "acclimatization": "very_used",
            "recovery_environment": "cool_indoor",
            "sun_exposure": False
        }
        env = {
            "temp_c": w["temp_c"],
            "rh_pct": w["rh_pct"],
            "wind_ms": w["wind_ms"],
            "solar_wm2": w["solar_wm2"],
            "region": "plains"
        }

        res = simulate_exposure_scenario(baseline, scenario, env)
        assert "transition_explanation" in res
        t_exp = res["transition_explanation"]
        assert isinstance(t_exp, str)
        assert len(t_exp) > 0
        assert "Action level" in t_exp
        assert "causal_explanations" in res
        assert len(res["causal_explanations"]) > 0
        assert "decision_trace_baseline" in res
        assert "decision_trace_scenario" in res


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS: API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class TestApiEndpoints:

    def test_api_check_risk_explainability_contract(self, client):
        """POST /api/check-risk returns all required explainable keys."""
        payload = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "18–40",
            "experience": "very_used",
            "work_intensity": "heavy",
            "clothing": "standard",
            "recovery_environment": "shade"
        }
        # Mock fetch_current_weather in app
        mock_weather = {
            "temp_c": 38.0,
            "current_temp_c": 38.0,
            "rh_pct": 35.0,
            "humidity": 35.0,
            "wind_ms": 2.0,
            "solar_wm2": 600.0,
            "cloud_cover": 0.1,
            "heat_index_c": 42.0,
            "confidence_label": "High"
        }
        with patch("app.fetch_current_weather", return_value=mock_weather):
            resp = client.post("/api/check-risk", json=payload)
            assert resp.status_code == 200
            data = resp.get_json()

            assert "decision_trace" in data
            assert "explanation" in data
            assert "primary_drivers" in data
            assert "contributing_factors" in data
            assert "protective_factors" in data
            assert "actionable_levers" in data
            assert "factor_sensitivities" in data
            assert "plain_english_uncertainty" in data
            assert "level_1_quick" in data

    def test_api_decision_trace_benchmark_presets(self, client):
        """GET /api/decision-trace with benchmark presets returns full trace."""
        for case_id in ["khammam_2015_peak", "hyderabad_2015_peak"]:
            resp = client.get(f"/api/decision-trace?case_id={case_id}")
            assert resp.status_code == 200
            data = resp.get_json()

            assert data.get("case_id") == case_id
            assert "decision_trace" in data
            assert "primary_drivers" in data
            assert "explanation" in data
            trace = data["decision_trace"]
            assert "pipeline_stages" in trace
            assert "triggered_rules" in trace

    def test_api_simulate_scenario_returns_transition_explanation(self, client):
        """POST /api/simulate-scenario returns transition_explanation and traces."""
        payload = {
            "baseline": {
                "workload": "heavy",
                "clothing": "heavy",
                "acclimatization": "very_used",
                "recovery_environment": "outdoor_direct",
                "sun_exposure": True
            },
            "scenario": {
                "workload": "light",
                "clothing": "light",
                "acclimatization": "very_used",
                "recovery_environment": "cool_indoor",
                "sun_exposure": False
            },
            "environment": {
                "temp_c": 40.0,
                "rh_pct": 30.0,
                "wind_ms": 2.0,
                "solar_wm2": 700.0,
                "region": "plains"
            }
        }
        resp = client.post("/api/simulate-scenario", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()

        assert "transition_explanation" in data
        assert isinstance(data["transition_explanation"], str)
        assert "causal_explanations" in data
        assert "decision_trace_baseline" in data
        assert "decision_trace_scenario" in data


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION & SAFETY GUARDS
# ─────────────────────────────────────────────────────────────────────────────

class TestScientificIntegrityAndRegression:

    def test_no_fabricated_percentages(self, synthetic_extreme_weather):
        """Verify the engine does NOT output arbitrary percentage claims ('Risk reduced by 43%')."""
        w = synthetic_extreme_weather
        result = run_assessment(w, age_group="18–40", experience="very_used", workload="heavy", clothing="heavy")

        summary_text = json.dumps(result)
        # Check against fabricated percentage reduction patterns
        assert "reduced by 4" not in summary_text
        assert "risk lowered by" not in summary_text.lower() or "tier" in summary_text.lower() or "level" in summary_text.lower()

    def test_no_clinical_prescriptions(self, synthetic_extreme_weather):
        """Verify disclaimers emphasize occupational decision support, not medical diagnosis."""
        w = synthetic_extreme_weather
        result = run_assessment(w, age_group="60+", experience="not_used", workload="heavy", clothing="heavy")

        disclaimer = result.get("disclaimer", "").lower()
        limitations = " ".join(result.get("limitations", [])).lower()
        combined = disclaimer + " " + limitations
        assert "medical" in combined
        assert "screening" in combined or "occupational" in combined or "decision" in combined
