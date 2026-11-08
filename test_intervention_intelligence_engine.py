"""
test_intervention_intelligence_engine.py
=========================================
Comprehensive test suite for HeatWave Guardian Step 6:
Intervention Intelligence Engine.

Validates:
1. Intervention trigger logic across all categories (A through G).
2. Occupational control hierarchy ordering (Levels 1 to 6).
3. Safety PPE guardrail enforcement (never advise removal of mandatory PPE).
4. Scheduling recommendations & lower-heat-stress candidate periods.
5. Workload, duration, environment, recovery, and cooling siting suggestions.
6. Role-specific outputs (Worker, Supervisor/Worksite, Municipal Authority).
7. What-If simulator connection and 'SIMULATED SCENARIO' labeling.
8. Data honesty, safety disclaimers, and prohibition of fake % claims.
9. Structured alert object generation and provenance metadata.
10. Robustness against missing forecast, invalid inputs, and out-of-range parameters.
11. REST API endpoints contract & backward compatibility.
"""

import pytest
import json
from unittest.mock import patch

from services.intervention_engine import (
    InterventionIntelligenceEngine,
    CONTROL_HIERARCHY,
    OPERATIONAL_STATES,
)
import app as flask_app


# ==============================================================================
# 1. Intervention Trigger Logic (Categories A - G)
# ==============================================================================
class TestInterventionTriggerLogic:
    def test_scheduling_trigger_on_peak_heat_overlap(self):
        """Category A: Peak overlap under high WBGT triggers task rescheduling."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 41.0, "wbgt_c": 33.0, "rh_pct": 50.0},
            exposure_context={"work_start_time": "12:00", "duration_hours": 4.0},
            location="Hyderabad",
        )
        sched_ints = [i for i in res["interventions"] if i["category"] == "SCHEDULING"]
        assert len(sched_ints) > 0
        assert any(i["id"] == "SCHED-01" for i in sched_ints)
        assert any("reschedule" in i["intervention"].lower() or "shift" in i["intervention"].lower() for i in sched_ints)

    def test_scheduling_maintains_normal_under_mild_conditions(self):
        """Category A: Under mild heat, normal schedule is maintained."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 26.0, "wbgt_c": 23.0, "rh_pct": 40.0},
            exposure_context={"work_start_time": "08:00", "duration_hours": 4.0},
            location="Pune",
        )
        sched_ints = [i for i in res["interventions"] if i["category"] == "SCHEDULING"]
        assert len(sched_ints) > 0
        assert any(i["id"] == "SCHED-02" for i in sched_ints)
        assert any("standard" in i["intervention"].lower() for i in sched_ints)

    def test_exposure_duration_pacing_trigger(self):
        """Category B: Elevated heat triggers structured hourly work/rest pacing."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 31.0, "rh_pct": 50.0},
            exposure_context={"workload": "heavy", "duration_hours": 4.0},
            location="Delhi",
        )
        dur_ints = [i for i in res["interventions"] if i["category"] == "EXPOSURE_DURATION"]
        assert len(dur_ints) > 0
        assert any(i["id"] == "DUR-01" for i in dur_ints)
        pacing_int = next(i for i in dur_ints if i["id"] == "DUR-01")
        assert "pacing" in pacing_int["intervention"].lower()
        assert "min work" in pacing_int["intervention"].lower()

    def test_prolonged_shift_duration_limit_trigger(self):
        """Category B: Shift >= 6h in hot conditions triggers shift duration curtailment."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 39.0, "wbgt_c": 30.5, "rh_pct": 45.0},
            exposure_context={"duration_hours": 8.0},
            location="Ahmedabad",
        )
        dur_ints = [i for i in res["interventions"] if i["id"] == "DUR-02"]
        assert len(dur_ints) > 0
        assert any("6 hours" in i["intervention"].lower() or "rotation" in i["intervention"].lower() for i in dur_ints)

    def test_heavy_workload_modification_trigger(self):
        """Category C: Heavy or Very Heavy metabolic exertion triggers task modification."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 37.0, "wbgt_c": 30.0, "rh_pct": 45.0},
            exposure_context={"workload": "very_heavy"},
            location="Nagpur",
        )
        work_ints = [i for i in res["interventions"] if i["category"] == "WORKLOAD"]
        assert len(work_ints) > 0
        assert any("mechanical" in i["intervention"].lower() or "intensity" in i["intervention"].lower() for i in work_ints)
        assert any("500w" in i["reason"].lower() or "metabolic" in i["reason"].lower() for i in work_ints)

    def test_sun_exposure_shading_trigger(self):
        """Category D: Direct sun exposure triggers radiant shading control."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 36.0, "wbgt_c": 29.5},
            exposure_context={"is_sun_exposed": True},
            location="Jaipur",
        )
        env_ints = [i for i in res["interventions"] if i["category"] == "ENVIRONMENT" and i["id"] == "ENV-01"]
        assert len(env_ints) > 0
        assert any("shade" in i["intervention"].lower() or "sunshade" in i["intervention"].lower() for i in env_ints)
        # Ensure honest language without fabricated percentage
        assert "shade is a potential exposure-control intervention" in env_ints[0]["expected_direction_of_effect"].lower()

    def test_compromised_recovery_trigger(self):
        """Category D/Recovery: Direct outdoor or warm uncooled rest triggers dedicated shelter."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 36.0, "wbgt_c": 30.0},
            exposure_context={"recovery_environment": "outdoor_direct"},
            location="Khammam",
        )
        rec_ints = [i for i in res["interventions"] if i["id"] == "REC-01"]
        assert len(rec_ints) > 0
        assert any("dedicated" in i["intervention"].lower() or "shelter" in i["intervention"].lower() for i in rec_ints)

    def test_cooling_access_integration(self):
        """Category F: Proximate municipal cooling hub recommendations are surfaced."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 39.0, "wbgt_c": 31.0},
            location="Hyderabad",
        )
        cool_ints = [i for i in res["interventions"] if i["category"] == "COOLING_ACCESS"]
        assert len(cool_ints) > 0
        c_int = cool_ints[0]
        assert "cooling" in c_int["intervention"].lower()
        assert "facility_info" in c_int
        assert c_int["facility_info"]["zone_name"] is not None

    def test_operational_escalation_on_extreme_heat(self):
        """Category G: Critical thermal thresholds trigger Supervisor/Authority Review."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 44.0, "wbgt_c": 34.5},
            exposure_context={"workload": "heavy"},
            location="Delhi",
        )
        esc_ints = [i for i in res["interventions"] if i["category"] == "OPERATIONAL_ESCALATION" and i["id"] == "ESC-01"]
        assert len(esc_ints) > 0
        assert "supervisor review" in esc_ints[0]["intervention"].lower()
        assert "suspend" in esc_ints[0]["intervention"].lower()

    def test_unacclimatized_worker_ramp_up_trigger(self):
        """Category G/Admin: Unacclimatized worker triggers progressive ramp-up."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 35.0, "wbgt_c": 29.0},
            exposure_context={"acclimatization": "not_used"},
            location="Delhi",
        )
        acc_ints = [i for i in res["interventions"] if i["id"] == "ACC-01"]
        assert len(acc_ints) > 0
        assert any("ramp-up" in i["intervention"].lower() or "progressive" in i["intervention"].lower() for i in acc_ints)


# ==============================================================================
# 2. Control Hierarchy Organization
# ==============================================================================
class TestControlHierarchy:
    def test_interventions_sorted_by_control_hierarchy(self):
        """Interventions must strictly sort from Level 1 (Elimination) to Level 6 (PPE)."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 41.0, "wbgt_c": 32.5, "rh_pct": 55.0},
            exposure_context={
                "workload": "heavy",
                "clothing": "heavy",
                "is_sun_exposed": True,
                "recovery_environment": "outdoor_direct",
                "work_start_time": "12:00",
                "duration_hours": 6.0,
            },
            location="Hyderabad",
        )
        levels = [i["control_hierarchy_level"] for i in res["interventions"]]
        assert len(levels) >= 5
        # Assert non-decreasing order: Level 1 <= Level 2 <= ... <= Level 6
        for idx in range(len(levels) - 1):
            assert levels[idx] <= levels[idx + 1]

    def test_hierarchy_summary_present(self):
        """Hierarchy summary dict contains all 6 levels with recommendation counts."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 31.0},
            location="Delhi",
        )
        summary = res["control_hierarchy_summary"]
        for lvl in range(1, 7):
            key = f"level_{lvl}"
            assert key in summary
            assert "name" in summary[key]
            assert "count" in summary[key]


# ==============================================================================
# 3. Safety PPE Guardrail
# ==============================================================================
class TestSafetyPPEGuardrails:
    def test_ppe_intervention_prohibits_automatic_ppe_removal(self):
        """Under no circumstances should the engine instruct removal of mandatory safety PPE."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 31.0},
            exposure_context={"clothing": "ppe", "ppe_category": "chemical_suit"},
            location="Mumbai",
        )
        ppe_ints = [i for i in res["interventions"] if i["category"] == "CLOTHING_PPE"]
        assert len(ppe_ints) > 0
        ppe_text = ppe_ints[0]["expected_direction_of_effect"].lower()
        # Must explicitly warn never to instruct removal of mandatory safety PPE
        assert "never instruct or authorize removal of mandatory safety ppe" in ppe_text
        assert "cooling vests" in ppe_text or "shorten" in ppe_text

    def test_no_intervention_recommends_discarding_protective_gear(self):
        """Audit all generated recommendations across high heat to ensure zero unsafe instructions."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 45.0, "wbgt_c": 35.0},
            exposure_context={"clothing": "heavy", "workload": "very_heavy"},
            location="Jaisalmer",
        )
        for i in res["interventions"]:
            txt = (i["intervention"] + " " + i["reason"] + " " + i["expected_direction_of_effect"]).lower()
            assert "remove safety ppe" not in txt
            assert "discard safety" not in txt


# ==============================================================================
# 4. Scheduling & Candidate Windows
# ==============================================================================
class TestSchedulingAndCandidateWindows:
    def test_candidate_shift_windows_identified_from_hourly_forecast(self):
        """Hourly forecast detects early morning and late afternoon windows avoiding peak."""
        hourly = [
            {"hour": 6, "temp_c": 28.0, "peak_wbgt_c": 25.0, "rh_pct": 65.0},
            {"hour": 8, "temp_c": 32.0, "peak_wbgt_c": 27.5, "rh_pct": 55.0},
            {"hour": 10, "temp_c": 36.0, "peak_wbgt_c": 29.5, "rh_pct": 45.0},
            {"hour": 12, "temp_c": 41.0, "peak_wbgt_c": 33.5, "rh_pct": 38.0},
            {"hour": 14, "temp_c": 43.0, "peak_wbgt_c": 34.5, "rh_pct": 35.0},
            {"hour": 16, "temp_c": 40.0, "peak_wbgt_c": 32.0, "rh_pct": 38.0},
            {"hour": 18, "temp_c": 35.0, "peak_wbgt_c": 28.0, "rh_pct": 48.0},
        ]
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 43.0, "wbgt_c": 34.5},
            hourly_forecast=hourly,
            exposure_context={"work_start_time": "12:00", "duration_hours": 4.0},
            location="Hyderabad",
        )
        sched = res["scheduling_candidates"]
        assert sched["has_candidate_windows"] is True
        cands = sched["candidate_windows"]
        assert len(cands) >= 1
        # Morning candidate should have substantially lower WBGT than 34.5°C
        m_cand = next((c for c in cands if "morning" in c["window_label"].lower()), cands[0])
        assert m_cand["estimated_avg_wbgt_c"] < 30.0
        assert m_cand["avoids_peak"] is True

    def test_candidate_window_climatological_fallback(self):
        """When hourly forecast is missing, diurnal climatological candidate windows are provided."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 40.0, "wbgt_c": 32.0},
            hourly_forecast=None,
            location="Delhi",
        )
        sched = res["scheduling_candidates"]
        assert sched["has_candidate_windows"] is True
        assert any("morning" in c["window_label"].lower() for c in sched["candidate_windows"])


# ==============================================================================
# 5. Role-Specific Views (Worker, Supervisor, Authority)
# ==============================================================================
class TestRoleSpecificViews:
    def test_role_views_contain_expected_roles(self):
        """Engine produces distinct views for worker, supervisor, and authority."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 31.0},
            location="Hyderabad",
        )
        views = res["role_views"]
        assert "worker" in views
        assert "supervisor" in views
        assert "authority" in views

    def test_worker_view_is_simple_and_direct(self):
        """Worker view contains simple actions and safety reminder without managerial jargon."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 31.0},
            role="worker",
            location="Hyderabad",
        )
        w_view = res["role_views"]["worker"]
        assert "simple_actions" in w_view
        assert len(w_view["simple_actions"]) > 0
        assert "safety_reminder" in w_view
        assert "hourly_pacing" in w_view

    def test_supervisor_view_contains_workforce_controls(self):
        """Supervisor view contains task scheduling options and workforce exposure controls."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 31.0},
            role="supervisor",
            location="Delhi",
        )
        s_view = res["role_views"]["supervisor"]
        assert "task_scheduling_options" in s_view
        assert "workforce_exposure_controls" in s_view
        assert "escalation_threshold" in s_view

    def test_authority_view_contains_municipal_planning(self):
        """Authority view contains structured actions, event timing, and cooling station siting."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "wbgt_c": 31.0},
            role="authority",
            location="Hyderabad",
        )
        a_view = res["role_views"]["authority"]
        assert "structured_authority_actions" in a_view
        assert len(a_view["structured_authority_actions"]) >= 3
        assert "cooling_station_siting" in a_view


# ==============================================================================
# 6. What-If Simulator Connection (Intervention Simulation)
# ==============================================================================
class TestWhatIfSimulatorConnection:
    def test_simulate_intervention_triggers_scenario_simulator(self):
        """Simulating an intervention returns a SIMULATED SCENARIO comparison."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 38.0, "rh_pct": 50.0, "wbgt_c": 31.5},
            exposure_context={"workload": "heavy", "work_start_time": "12:00", "duration_hours": 4.0},
            simulate_intervention=True,
            intervention_modifications={"workload": "moderate", "work_start_time": "07:00"},
            location="Hyderabad",
        )
        sim = res["simulated_intervention"]
        assert sim is not None
        assert sim["is_simulated"] is True
        assert sim["simulation_label"] == "SIMULATED SCENARIO"
        assert "baseline" in sim
        assert "scenarios" in sim


# ==============================================================================
# 7. Unified Decision Timeline
# ==============================================================================
class TestDecisionTimeline:
    def test_unified_timeline_structure_and_states(self):
        """Timeline maps TIME -> FORECAST -> LOCALIZED -> WBGT -> RISK -> INTERVENTION -> STATUS."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 39.0, "wbgt_c": 32.0},
            exposure_context={"work_start_time": "08:00", "duration_hours": 6.0},
            location="Delhi",
        )
        tl = res["decision_timeline"]
        assert len(tl) >= 6
        for step in tl:
            assert "time" in step
            assert "hour" in step
            assert "forecast_heat" in step
            assert "wbgt_c" in step
            assert "operational_status" in step
            assert step["operational_status"] in [
                "ROUTINE MONITORING",
                "ENHANCED CONTROLS",
                "MODIFY / RESTRICT EXPOSURE",
                "SUPERVISOR REVIEW",
                "AUTHORITY REVIEW",
            ]
            assert "guaranteed safe" not in step["operational_status"].lower()


# ==============================================================================
# 8. Data Honesty & Provenance
# ==============================================================================
class TestDataHonestyAndProvenance:
    def test_disclaimer_and_no_guaranteed_safety(self):
        """All outputs disclose decision support status and prohibit guaranteed safety claims."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 28.0, "wbgt_c": 24.0},
            location="Pune",
        )
        assert res["is_decision_support"] is True
        assert "not provide medical diagnosis" in res["disclaimer"].lower()
        # Verify operational state never claims guaranteed zero risk
        op_state = res["operational_state"]
        assert "guaranteed safe" not in op_state["state"].lower()
        assert "absolute safety cannot be guaranteed" in op_state["safety_note"].lower()

    def test_no_fabricated_percentage_claims(self):
        """Verifies no unbacked claims like 'shade reduces risk by 30%' appear in outputs."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 42.0, "wbgt_c": 33.0},
            exposure_context={"is_sun_exposed": True},
            location="Jaipur",
        )
        for item in res["interventions"]:
            txt = (item["intervention"] + " " + item["expected_direction_of_effect"]).lower()
            assert "reduces risk by 30%" not in txt
            assert "reduces risk by 20%" not in txt
            assert "guaranteed" not in txt


# ==============================================================================
# 9. Structured Alert Object Preparation
# ==============================================================================
class TestStructuredAlerts:
    def test_prepared_alerts_contain_all_required_provenance_keys(self):
        """Structured alert objects must have recipient_role, risk, peak, intervention, provenance."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 40.0, "wbgt_c": 32.5},
            location="Hyderabad",
        )
        alerts = res["prepared_alerts"]
        assert len(alerts) == 3
        roles = [a["recipient_role"] for a in alerts]
        assert "worker" in roles
        assert "supervisor" in roles
        assert "authority" in roles

        for a in alerts:
            assert a["alert_id"].startswith("ALT-")
            assert a["location"] == "Hyderabad"
            assert "risk" in a
            assert "peak_period" in a
            assert "intervention" in a
            assert "reason" in a
            assert "timestamp" in a
            prov = a["model_version_provenance"]
            assert prov["wbgt_physics"] == "Liljegren (2008)"
            assert "ISO 7243" in prov["occupational_standard"]
            assert "NDMA" in prov["hap_standard"]


# ==============================================================================
# 10. Robustness & Invalid Inputs
# ==============================================================================
class TestRobustnessAndEdgeCases:
    def test_missing_weather_uses_safe_fallback(self):
        """Omitting weather_data triggers safe fallback without crashing."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data=None,
            location="Delhi",
        )
        assert res["status"] == "success"
        assert res["environmental_condition"]["wbgt_c"] > 0

    def test_empty_exposure_context_uses_defaults(self):
        """Empty exposure context defaults safely to moderate workload and acclimatized worker."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 35.0, "wbgt_c": 30.0},
            exposure_context={},
            location="Hyderabad",
        )
        assert res["status"] == "success"
        assert res["exposure_context"]["workload"] == "moderate"

    def test_out_of_range_temperature_handling(self):
        """Extreme temperatures (e.g. 52°C) execute without unhandled exceptions."""
        res = InterventionIntelligenceEngine.evaluate(
            weather_data={"temp_c": 52.0, "wbgt_c": 38.0},
            location="Jaisalmer",
        )
        assert res["status"] == "success"
        assert res["operational_state"]["state"] in ("SUPERVISOR REVIEW", "AUTHORITY REVIEW")


# ==============================================================================
# 11. REST API Endpoints Contract
# ==============================================================================
class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        flask_app.app.config["TESTING"] = True
        return flask_app.app.test_client()

    def test_post_interventions_evaluate(self, client):
        """POST /api/interventions/evaluate returns 200 and valid schema."""
        resp = client.post(
            "/api/interventions/evaluate",
            json={
                "location": "Hyderabad",
                "weather_data": {"temp_c": 38.0, "wbgt_c": 31.0, "rh_pct": 50.0},
                "exposure_context": {"workload": "heavy"},
                "role": "all",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "interventions" in data
        assert "control_hierarchy_summary" in data
        assert "decision_timeline" in data
        assert "prepared_alerts" in data

    def test_get_interventions_endpoint(self, client):
        """GET /api/interventions?location=Delhi&role=worker returns 200 with worker view."""
        resp = client.get("/api/interventions?location=Delhi&role=worker")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "active_role_view" in data
        assert data["active_role_view"]["role"] == "worker"

    def test_alias_endpoints(self, client):
        """Aliases /api/interventions (POST) and /api/intervention-plan (POST) function identically."""
        payload = {"location": "Mumbai", "workload": "light"}
        r1 = client.post("/api/interventions", json=payload)
        r2 = client.post("/api/intervention-plan", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.get_json()["status"] == r2.get_json()["status"]

    def test_forecast_endpoint_with_intelligence_view_attaches_intervention_plan(self, client):
        """GET /api/forecast?location=Hyderabad&view=intelligence includes intervention_plan."""
        resp = client.get("/api/forecast?location=Hyderabad&view=intelligence")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "forecast_days" in data
        assert "intervention_plan" in data
        assert data["intervention_plan"] is not None
        assert "interventions" in data["intervention_plan"]
