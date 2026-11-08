"""
HeatWave Guardian — Step 5 What-If Simulator & Decision Timeline Test Suite
==========================================================================
Comprehensive tests validating:
1. Baseline calculation
2. Scenario calculation
3. Delta calculation (temperature, ML temp, WBGT, work/rest, duration)
4. Multiple scenarios (Baseline vs Scenario A vs Scenario B)
5. Invalid inputs (graceful rejection or fallback)
6. Out-of-range ML features (validation warnings & unvalidated flags)
7. Missing values handling & default fallbacks
8. API schema validation (/api/scenario-analysis and /api/simulate-scenario)
9. Timeline generation (operational states: NORMAL, ENHANCED CONTROLS, RESTRICT, STOP)
10. Risk transition logic (improvement and escalation)
11. Data honesty safeguards: "SIMULATED SCENARIO" labeling & no "guaranteed safe" claims
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app
from services.scenario_simulator import ScenarioSimulator, VALIDATION_LIMITS, OPERATIONAL_STATES


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestWhatIfSimulatorCore:
    """Core simulation engine unit tests."""

    def test_baseline_calculation(self):
        """1. Verify baseline calculation properly flows through ML, WBGT, and risk engine."""
        payload = {
            "location": "Hyderabad",
            "baseline": {
                "temperature_c": 36.0,
                "relative_humidity_pct": 50.0,
                "wind_speed_ms": 2.0,
                "solar_radiation_wm2": 600.0,
                "workload": "moderate",
                "clothing": "light",
            },
            "scenario": {
                "workload": "moderate",
            },
        }
        res = ScenarioSimulator.simulate(payload)
        assert res["status"] == "success"
        assert res["is_simulated"] is True
        assert res["simulation_label"] == "SIMULATED SCENARIO"

        baseline = res["baseline"]
        assert baseline["label"] == "BASELINE FORECAST"
        assert baseline["is_simulated"] is False
        assert "ml_local_estimate" in baseline
        assert "localized_temp_c" in baseline["ml_local_estimate"]
        assert "thermal_physics" in baseline
        assert "wbgt_c" in baseline["thermal_physics"]
        assert "operational_recommendation" in baseline
        assert baseline["operational_recommendation"]["operational_state"] in [
            "NORMAL", "ENHANCED CONTROLS", "RESTRICT / MODIFY EXPOSURE", "STOP / POSTPONE"
        ]

    def test_scenario_calculation(self):
        """2. Verify scenario calculation with modified environmental and exposure conditions."""
        payload = {
            "location": "Delhi",
            "baseline": {
                "temperature_c": 42.0,
                "relative_humidity_pct": 30.0,
                "wind_speed_ms": 2.0,
                "solar_radiation_wm2": 800.0,
                "workload": "heavy",
                "clothing": "heavy",
                "recovery_environment": "outdoor_direct",
            },
            "scenario": {
                "temperature_c": 35.0,
                "relative_humidity_pct": 40.0,
                "workload": "light",
                "clothing": "light",
                "recovery_environment": "cool_indoor",
            },
        }
        res = ScenarioSimulator.simulate(payload)
        scen = res["scenarios"]["scenario_a"]

        assert scen["label"] == "SIMULATED SCENARIO"
        assert scen["is_simulated"] is True
        assert scen["environmental_conditions"]["background_temp_c"] == 35.0
        assert scen["exposure_assessment"]["workload_intensity"] == "light"
        assert scen["exposure_assessment"]["recovery_environment"] == "cool_indoor"

    def test_delta_calculation(self):
        """3. Verify mathematical correctness of delta metrics between baseline and scenario."""
        payload = {
            "location": "Jaipur",
            "baseline": {
                "temperature_c": 40.0,
                "relative_humidity_pct": 35.0,
                "wind_speed_ms": 1.5,
                "solar_radiation_wm2": 700.0,
                "workload": "heavy",
                "exposure_duration_hours": 8.0,
            },
            "scenario": {
                "temperature_c": 36.0,
                "relative_humidity_pct": 35.0,
                "wind_speed_ms": 2.5,
                "solar_radiation_wm2": 400.0,
                "workload": "light",
                "exposure_duration_hours": 6.0,
            },
        }
        res = ScenarioSimulator.simulate(payload)
        diff = res["comparisons"]["scenario_a"]

        assert diff["temperature_diff_c"] == -4.0
        assert diff["wind_speed_diff_ms"] == 1.0
        assert diff["solar_radiation_diff_wm2"] == -300.0
        assert diff["exposure_duration_diff_hours"] == -2.0
        assert diff["wbgt_diff_c"] < 0.0  # WBGT must drop with lower temp and solar
        assert diff["risk_direction"] == "improved"
        assert diff["work_minutes_delta"] >= 0  # Allowed work time increases or stays max

    def test_multiple_scenarios_comparison(self):
        """4. Verify simultaneous evaluation of Baseline vs Scenario A vs Scenario B."""
        payload = {
            "location": "Hyderabad",
            "baseline": {
                "temperature_c": 41.0,
                "relative_humidity_pct": 40.0,
                "workload": "heavy",
                "clothing": "ppe",
            },
            "scenarios": {
                "scenario_a": {
                    "temperature_c": 39.0,
                    "workload": "moderate",
                    "clothing": "light",
                    "recovery_environment": "shade",
                },
                "scenario_b": {
                    "temperature_c": 34.0,
                    "workload": "light",
                    "clothing": "light",
                    "recovery_environment": "cool_indoor",
                    "exposure_duration_hours": 4.0,
                },
            },
        }
        res = ScenarioSimulator.simulate(payload)

        assert "scenarios" in res
        assert "scenario_a" in res["scenarios"]
        assert "scenario_b" in res["scenarios"]
        assert "comparisons" in res
        assert "scenario_a" in res["comparisons"]
        assert "scenario_b" in res["comparisons"]

        # Scenario B has aggressive cooling and work reduction compared to A
        wbgt_a = res["scenarios"]["scenario_a"]["thermal_physics"]["wbgt_c"]
        wbgt_b = res["scenarios"]["scenario_b"]["thermal_physics"]["wbgt_c"]
        assert wbgt_b < wbgt_a

    def test_invalid_inputs_handled_gracefully(self):
        """5. Verify invalid data types (strings for floats, corrupt structures) fail safely."""
        payload = {
            "location": "Delhi",
            "baseline": {
                "temperature_c": "not_a_number",
                "relative_humidity_pct": None,
            },
            "scenario": {
                "temperature_c": 35.0,
            },
        }
        # Should not crash with unhandled exception; uses robust fallbacks
        res = ScenarioSimulator.simulate(payload)
        assert res["status"] == "success"
        assert "baseline" in res
        assert "validation_warnings" in res
        assert len(res["validation_warnings"]) > 0

    def test_out_of_range_features_trigger_warnings(self):
        """6. Verify out-of-range features trigger warnings and unvalidated status."""
        payload = {
            "location": "Jaipur",
            "baseline": {
                "temperature_c": 62.0,  # Exceeds max 52°C
                "relative_humidity_pct": 110.0,  # Exceeds 95%
                "solar_radiation_wm2": 1400.0,  # Exceeds 1200
            },
            "scenario": {
                "temperature_c": 55.0,
            },
        }
        res = ScenarioSimulator.simulate(payload)

        assert res["is_validated_range"] is False
        assert len(res["validation_warnings"]) >= 2
        # Check honest confidence suppression
        ml_est = res["baseline"]["ml_local_estimate"]
        assert "Unvalidated" in ml_est["confidence_state"]
        assert "Unvalidated" in ml_est["data_quality"]

    def test_missing_values_use_defaults(self):
        """7. Verify missing values apply sensible scientific defaults."""
        payload = {
            "location": "Hyderabad",
            "baseline": {},
            "scenario": {},
        }
        res = ScenarioSimulator.simulate(payload)

        assert res["status"] == "success"
        base_env = res["baseline"]["environmental_conditions"]
        assert base_env["background_temp_c"] > 0
        assert base_env["relative_humidity_pct"] > 0
        assert base_env["wind_speed_ms"] > 0

    def test_timeline_generation_operational_states(self):
        """8. Verify decision timeline generates step-by-step hours and valid operational states."""
        payload = {
            "location": "Khammam",
            "baseline": {
                "temperature_c": 44.0,
                "relative_humidity_pct": 35.0,
                "work_start_time": "09:00",
                "exposure_duration_hours": 6.0,
            },
            "scenario": {
                "temperature_c": 40.0,
                "workload": "moderate",
            },
        }
        res = ScenarioSimulator.simulate(payload)
        timeline = res["decision_timeline"]

        assert len(timeline) == 6
        assert timeline[0]["time"] == "09:00"
        assert timeline[5]["time"] == "14:00"

        valid_states = {"NORMAL", "ENHANCED CONTROLS", "RESTRICT / MODIFY EXPOSURE", "STOP / POSTPONE"}
        for step in timeline:
            assert step["operational_state"] in valid_states
            assert "forecast_heat_stress" in step
            assert "ml_localized_temp_c" in step
            assert "wbgt_c" in step
            assert "risk_category" in step
            assert "recommended_action" in step

    def test_risk_transition_logic(self):
        """9. Verify escalation and de-escalation direction tracking."""
        # De-escalation (improvement)
        payload_improve = {
            "baseline": {"temperature_c": 43.0, "workload": "very_heavy", "clothing": "ppe"},
            "scenario": {"temperature_c": 32.0, "workload": "light", "clothing": "light", "recovery_environment": "cool_indoor"},
        }
        res_imp = ScenarioSimulator.simulate(payload_improve)
        assert res_imp["comparisons"]["scenario_a"]["risk_direction"] == "improved"

        # Escalation (worsening)
        payload_worsen = {
            "baseline": {"temperature_c": 32.0, "workload": "light", "clothing": "light", "recovery_environment": "cool_indoor"},
            "scenario": {"temperature_c": 43.0, "workload": "very_heavy", "clothing": "ppe"},
        }
        res_worsen = ScenarioSimulator.simulate(payload_worsen)
        assert res_worsen["comparisons"]["scenario_a"]["risk_direction"] == "worsened"

    def test_no_guaranteed_safety_claims(self):
        """10. Verify no operational state or description claims 'guaranteed safe'."""
        for state_key, state_data in OPERATIONAL_STATES.items():
            assert "guaranteed safe" not in state_data["description"].lower()
            assert "guaranteed safe" not in state_data["state"].lower()

        payload = {"baseline": {"temperature_c": 22.0}, "scenario": {"temperature_c": 20.0}}
        res = ScenarioSimulator.simulate(payload)
        json_dump = json.dumps(res).lower()
        # Verify no operational state claims guaranteed safety
        assert "state is guaranteed safe" not in json_dump
        assert "100% safe" not in json_dump
        timeline_states = [s["operational_state"].lower() for s in res["decision_timeline"]]
        for s in timeline_states:
            assert "guaranteed safe" not in s
            assert "safe" not in s or "normal" in s

    def test_simulated_scenario_data_honesty_labels(self):
        """11. Verify prominent labeling as SIMULATED SCENARIO."""
        payload = {
            "location": "Delhi",
            "baseline": {"temperature_c": 38.0},
            "scenario": {"temperature_c": 36.0},
        }
        res = ScenarioSimulator.simulate(payload)
        assert res["simulation_label"] == "SIMULATED SCENARIO"
        assert res["is_simulated"] is True
        assert res["scenarios"]["scenario_a"]["label"] == "SIMULATED SCENARIO"
        assert res["scenarios"]["scenario_a"]["source"] == "SIMULATED SCENARIO"
        assert "DATA HONESTY NOTICE" in res["disclaimer"]


class TestWhatIfSimulatorAPI:
    """API endpoint contract integration tests."""

    def test_api_scenario_analysis_endpoint(self, client):
        """12. POST /api/scenario-analysis returns full schema."""
        payload = {
            "location": "Hyderabad",
            "baseline": {
                "temperature_c": 39.0,
                "relative_humidity_pct": 45.0,
                "workload": "heavy",
                "clothing": "heavy",
            },
            "scenarios": {
                "scenario_a": {"workload": "moderate", "clothing": "light"},
                "scenario_b": {"workload": "light", "recovery_environment": "cool_indoor"},
            },
        }
        resp = client.post("/api/scenario-analysis", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["status"] == "success"
        assert data["simulation_label"] == "SIMULATED SCENARIO"
        assert "baseline" in data
        assert "scenarios" in data
        assert "scenario_a" in data["scenarios"]
        assert "scenario_b" in data["scenarios"]
        assert "comparisons" in data
        assert "decision_timeline" in data
        assert "disclaimer" in data

    def test_api_what_if_analyze_alias_endpoint(self, client):
        """13. POST /api/what-if/analyze acts as functional alias."""
        payload = {
            "location": "Jaipur",
            "baseline": {"temperature_c": 40.0},
            "scenario": {"temperature_c": 36.0},
        }
        resp = client.post("/api/what-if/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "decision_timeline" in data

    def test_api_simulate_scenario_backward_compatibility(self, client):
        """14. POST /api/simulate-scenario retains all legacy keys and behavior."""
        payload = {
            "baseline": {"workload": "very_heavy", "clothing": "ppe"},
            "scenario": {"workload": "light", "clothing": "light"},
            "weather_data": {
                "temp_c": 39.0,
                "rh_pct": 45.0,
                "wind_ms": 2.0,
                "solar_wm2": 650.0,
            },
            "location": "Hyderabad",
        }
        resp = client.post("/api/simulate-scenario", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()

        # Check required backward-compatible keys
        assert "action_level_change" in data
        assert "work_rest_change" in data
        assert "causal_explanations" in data
        assert "transition_explanation" in data
        assert "baseline_profile" in data
        assert "scenario_profile" in data
        assert "decision_timeline" in data  # Enriched with Step 5 timeline
        assert data["action_level_change"]["direction"] == "improved"
