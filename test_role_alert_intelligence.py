"""
HeatWave Guardian — Role-Based, Multilingual & Low-Connectivity Alert Intelligence Tests
=========================================================================================
Step 7 Verification Suite:
Tests the role model, structured alert policy, controlled multilingual templates,
low-connectivity compact SMS formatting, human-in-the-loop approval workflows,
consent/privacy enforcement, data honesty guardrails, and REST APIs.
"""

import pytest
import json
from datetime import datetime
from app import app
from services.alert_service import (
    AlertService,
    SUPPORTED_ROLES,
    SUPPORTED_LANGUAGES,
    SUPPORTED_CHANNELS,
    mask_contact,
    hash_contact,
)


@pytest.fixture(autouse=True)
def clean_alert_state():
    """Ensure in-memory alert logs, cooldowns, and preferences are cleared between tests."""
    AlertService.clear_all()
    yield
    AlertService.clear_all()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ==============================================================================
# 1. Role Model & Information Levels
# ==============================================================================
class TestRoleModelInformationLevels:
    def test_all_five_roles_supported(self):
        """All 5 roles must be valid in the alert engine."""
        expected_roles = {"worker", "supervisor", "site_manager", "authority", "healthcare_facility"}
        assert expected_roles.issubset(set(SUPPORTED_ROLES))

    def test_worker_role_content_focus(self):
        """Worker alert must focus on personal risk, peak window, hydration, and rest intervals."""
        alert = AlertService.create_alert(
            role="worker",
            location="Delhi",
            risk="High",
            wbgt_c=31.5,
            temp_c=40.0,
            work_minutes=30,
            rest_minutes=30,
            peak_window="12:00–15:00",
            language="en",
        )
        assert alert["role"] == "worker"
        assert "Delhi" in alert["full_message"]
        assert "30m work / 30m rest" in alert["full_message"]
        assert "hydration" in alert["full_message"].lower() or "drink" in alert["full_message"].lower()

    def test_supervisor_role_content_focus(self):
        """Supervisor alert must include task rescheduling, pacing, and buddy checks."""
        alert = AlertService.create_alert(
            role="supervisor",
            location="Ahmedabad",
            risk="Very High",
            wbgt_c=32.8,
            temp_c=42.5,
            work_minutes=15,
            rest_minutes=45,
            peak_window="12:30–15:30",
            language="en",
        )
        assert alert["role"] == "supervisor"
        assert "buddy check" in alert["full_message"].lower() or "rotation" in alert["full_message"].lower() or "pacing" in alert["full_message"].lower()
        assert "15m work / 45m rest" in alert["full_message"]

    def test_site_manager_role_content_focus(self):
        """Site manager alert must include site operational directives and water logistics."""
        alert = AlertService.create_alert(
            role="site_manager",
            location="Nagpur",
            risk="Extreme",
            wbgt_c=33.5,
            temp_c=44.0,
            work_minutes=0,
            rest_minutes=60,
            peak_window="12:00–16:00",
            language="en",
        )
        assert alert["role"] == "site_manager"
        assert "suspension" in alert["full_message"].lower() or "water tanker" in alert["full_message"].lower() or "directive" in alert["full_message"].lower()

    def test_authority_role_content_focus(self):
        """Authority alert must include municipal heatwave warnings, shelters, and public action."""
        alert = AlertService.create_alert(
            role="authority",
            location="Hyderabad",
            risk="Very High",
            wbgt_c=32.2,
            temp_c=41.8,
            language="en",
        )
        assert alert["role"] == "authority"
        assert "municipal" in alert["full_message"].lower() or "cooling shelter" in alert["full_message"].lower()

    def test_healthcare_facility_guardrail_no_fake_mortality(self):
        """Healthcare facility alert must give operational preparedness guidance without claiming admission counts."""
        alert = AlertService.create_alert(
            role="healthcare_facility",
            location="Pune",
            risk="High",
            wbgt_c=31.8,
            temp_c=39.5,
            peak_window="12:00–15:00",
            language="en",
        )
        assert alert["role"] == "healthcare_facility"
        msg = alert["full_message"].lower()
        # Must contain operational guidance
        assert "iv hydration" in msg or "cooling beds" in msg or "preparedness" in msg
        # Must NOT claim admission counts or mortality
        assert "mortality rate" not in msg
        assert "admissions predicted" not in msg
        assert "not an epidemiological patient forecast" in msg or "not a patient forecast" in msg or "not predict" in msg

    def test_invalid_role_raises_error(self):
        """Passing an unrecognized role must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid role"):
            AlertService.create_alert(
                role="astronaut",
                location="Delhi",
                risk="High",
                wbgt_c=31.0,
                temp_c=39.0,
            )


# ==============================================================================
# 2. Multilingual Messages & Controlled Templates
# ==============================================================================
class TestMultilingualControlledTemplates:
    @pytest.mark.parametrize("lang", ["en", "hi", "mr"])
    def test_structured_layout_in_all_languages(self, lang):
        """All languages must maintain the required structured layout."""
        alert = AlertService.create_alert(
            role="worker",
            location="Mumbai",
            risk="Very High",
            wbgt_c=32.5,
            temp_c=37.5,
            work_minutes=15,
            rest_minutes=45,
            peak_window="12:00–15:30",
            language=lang,
        )
        msg = alert["full_message"]
        # Required blocks from prompt
        assert "RISK" in msg
        assert "LOCATION" in msg
        assert "PEAK WINDOW" in msg
        assert "ACTION" in msg
        assert "CONFIDENCE" in msg
        assert "DATA TIMESTAMP" in msg

    def test_critical_numbers_and_units_preserved_across_languages(self):
        """Temperature, WBGT, work/rest minutes, and peak window must be identical across EN, HI, MR."""
        params = {
            "role": "worker",
            "location": "Jaisalmer",
            "risk": "High",
            "wbgt_c": 31.8,
            "temp_c": 43.5,
            "work_minutes": 20,
            "rest_minutes": 40,
            "peak_window": "11:30–15:30",
        }
        msg_en = AlertService.create_alert(language="en", **params)["full_message"]
        msg_hi = AlertService.create_alert(language="hi", **params)["full_message"]
        msg_mr = AlertService.create_alert(language="mr", **params)["full_message"]

        for msg in [msg_en, msg_hi, msg_mr]:
            assert "31.8°C" in msg
            assert "43.5°C" in msg
            assert "20" in msg
            assert "40" in msg
            assert "11:30–15:30" in msg

    def test_hindi_vocabulary_integrity(self):
        """Hindi messages must use appropriate Hindi occupational terminology."""
        alert = AlertService.create_alert(
            role="worker",
            location="Lucknow",
            risk="Extreme",
            wbgt_c=33.0,
            temp_c=44.0,
            language="hi",
        )
        msg = alert["full_message"]
        assert "हीटवेव" in msg
        assert "छाया" in msg or "पानी" in msg or "विश्राम" in msg

    def test_marathi_vocabulary_integrity(self):
        """Marathi messages must use appropriate Marathi occupational terminology."""
        alert = AlertService.create_alert(
            role="worker",
            location="Pune",
            risk="Extreme",
            wbgt_c=33.0,
            temp_c=42.0,
            language="mr",
        )
        msg = alert["full_message"]
        assert "हीटवेव्ह" in msg
        assert "सावली" in msg or "विश्रांती" in msg or "पाणी" in msg

    def test_invalid_language_raises_error(self):
        """Unsupported language code must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid language"):
            AlertService.create_alert(
                role="worker",
                location="Delhi",
                risk="High",
                wbgt_c=31.0,
                temp_c=38.0,
                language="de",
            )


# ==============================================================================
# 3. Low-Connectivity Compact SMS Formatting
# ==============================================================================
class TestLowConnectivityCompactSMS:
    @pytest.mark.parametrize("role", SUPPORTED_ROLES)
    @pytest.mark.parametrize("lang", ["en", "hi", "mr"])
    def test_compact_sms_under_gsm_limit_or_concise(self, role, lang):
        """Compact SMS must be concise, actionable, and within standard payload limits."""
        compact = AlertService.format_compact_sms(
            location="Delhi",
            risk_category="Very High",
            wbgt_c=32.2,
            temp_c=41.5,
            work_minutes=15,
            rest_minutes=45,
            role=role,
            language=lang,
            peak_window="12:00–15:30",
            confidence="High",
        )
        assert len(compact) <= 160 or len(compact) <= 180  # Compact single/short SMS
        assert "Delhi" in compact
        assert "32.2" in compact
        assert "12:00–15:30" in compact

    def test_compact_sms_retains_minimum_actionable_information(self):
        """Compact message must retain Location, Risk, WBGT, Peak Window, Action, and Confidence."""
        compact = AlertService.format_compact_sms(
            location="Ahmedabad",
            risk_category="Extreme",
            wbgt_c=33.4,
            temp_c=43.0,
            work_minutes=0,
            rest_minutes=60,
            role="worker",
            language="en",
            peak_window="12:30–15:30",
            confidence="High",
        )
        assert "Ahmedabad" in compact
        assert "EXTREME" in compact
        assert "33.4C" in compact or "33.4" in compact
        assert "Peak" in compact
        assert "Conf:" in compact


# ==============================================================================
# 4. Human-in-the-Loop (HITL) Workflow
# ==============================================================================
class TestHumanInTheLoopWorkflow:
    def test_high_impact_authority_alert_requires_approval(self):
        """Authority alert at Very High or Extreme risk defaults to PENDING_APPROVAL."""
        alert = AlertService.create_alert(
            role="authority",
            location="Khammam",
            risk="Very High",
            wbgt_c=33.0,
            temp_c=43.5,
        )
        assert alert["requires_human_approval"] is True
        assert alert["approval_state"] == "PENDING"
        assert alert["status"] == "PENDING_APPROVAL"

    def test_dispatch_blocked_when_approval_pending(self):
        """Attempting to dispatch an unapproved high-impact alert must be blocked."""
        alert = AlertService.create_alert(
            role="authority",
            location="Delhi",
            risk="Extreme",
            wbgt_c=34.0,
            temp_c=45.0,
        )
        res = AlertService.dispatch_alert_by_id(alert["alert_id"], recipient_phone="+919876543210")
        assert res["success"] is False
        assert res["status"] == "APPROVAL_REQUIRED"

    def test_approval_lifecycle_draft_to_approved_to_dispatch(self):
        """Full HITL path: DRAFT -> APPROVE -> DISPATCH -> AUDIT."""
        alert = AlertService.create_alert(
            role="site_manager",
            location="Nagpur",
            risk="Very High",
            wbgt_c=32.5,
            temp_c=42.0,
        )
        aid = alert["alert_id"]
        assert alert["approval_state"] == "PENDING"

        # Approve
        appr_res = AlertService.approve_alert(aid, approver_name="Senior Safety Engineer", notes="Site reviewed")
        assert appr_res["success"] is True
        assert appr_res["alert"]["approval_state"] == "APPROVED"

        # Dispatch
        disp_res = AlertService.dispatch_alert_by_id(aid, recipient_phone="+919876543210")
        assert disp_res["success"] is True
        assert disp_res["status"] in ["DELIVERED", "SIMULATED"]

        # Audit trail must document CREATED, APPROVED, and DISPATCHED
        audit = AlertService.get_alert(aid)["audit_trail"]
        actions = [e["action"] for e in audit]
        assert "CREATED" in actions
        assert "APPROVED" in actions
        assert any("DISPATCH" in a for a in actions)

    def test_rejection_workflow(self):
        """Rejecting an alert updates status to REJECTED and blocks subsequent dispatch."""
        alert = AlertService.create_alert(
            role="authority",
            location="Mumbai",
            risk="Very High",
            wbgt_c=32.0,
            temp_c=38.0,
        )
        aid = alert["alert_id"]

        rej_res = AlertService.reject_alert(aid, rejected_by="Director SDMA", reason="Front passed; temp dropping")
        assert rej_res["success"] is True
        assert AlertService.get_alert(aid)["approval_state"] == "REJECTED"

        # Dispatch should fail
        disp_res = AlertService.dispatch_alert_by_id(aid, recipient_phone="+919876543210")
        assert disp_res["success"] is False
        assert disp_res["status"] == "ALERT_REJECTED"


# ==============================================================================
# 5. Consent, Preferences & Privacy Protection
# ==============================================================================
class TestConsentPreferencesPrivacy:
    def test_recipient_contact_masking(self):
        """Phone numbers and emails must be masked in audit records."""
        assert mask_contact("+919876543210") == "+91*** ***10"
        assert mask_contact("admin@heatwave.gov.in") == "ad***@heatwave.gov.in"
        assert mask_contact("123") == "***"

    def test_opt_out_consent_suppresses_dispatch(self):
        """If a recipient has opted out, dispatch must be suppressed unless forced."""
        phone = "+919123456780"
        AlertService.set_preferences(recipient_id=phone, opted_in=False)

        res = AlertService.dispatch_alert(
            recipient_phone=phone,
            location="Delhi",
            risk_category="Moderate",
            wbgt_c=30.0,
            temp_c=38.0,
            work_minutes=45,
            rest_minutes=15,
            instruction="Drink water",
            force=False,
        )
        assert res["success"] is False
        assert res["status"] == "SUPPRESSED_NO_CONSENT"

    def test_quiet_hours_suppression_and_critical_bypass(self):
        """Quiet hours suppress routine alerts, but critical emergency alerts bypass."""
        phone = "+919876543211"
        # Set quiet hours to 00:00 to 23:59 so it's always active
        AlertService.set_preferences(
            recipient_id=phone,
            quiet_hours={"enabled": True, "start": "00:00", "end": "23:59"},
            opted_in=True,
        )

        # High risk (not Extreme) should be suppressed
        res_routine = AlertService.dispatch_alert(
            recipient_phone=phone,
            location="Ahmedabad",
            risk_category="High",
            wbgt_c=31.5,
            temp_c=40.0,
            work_minutes=30,
            rest_minutes=30,
            instruction="Stay hydrated",
            force=False,
        )
        assert res_routine["success"] is False
        assert res_routine["status"] == "SUPPRESSED_QUIET_HOURS"

        # Extreme risk bypasses quiet hours
        res_critical = AlertService.dispatch_alert(
            recipient_phone=phone,
            location="Ahmedabad",
            risk_category="Extreme",
            wbgt_c=33.5,
            temp_c=44.0,
            work_minutes=0,
            rest_minutes=60,
            instruction="Emergency suspension",
            force=False,
        )
        assert res_critical["success"] is True
        assert res_critical["status"] in ["SIMULATED", "DELIVERED"]


# ==============================================================================
# 6. Data Honesty & Simulation Guardrails
# ==============================================================================
class TestDataHonestyAndSimulationGuardrails:
    def test_simulation_cannot_trigger_live_telecom(self):
        """Alerts flagged as is_simulated=True must NEVER make live external telecom calls."""
        res = AlertService.dispatch_alert(
            recipient_phone="+919876543210",
            location="Delhi",
            risk_category="High",
            wbgt_c=31.5,
            temp_c=40.0,
            work_minutes=30,
            rest_minutes=30,
            instruction="Test What-If",
            is_simulated=True,
            force=True,
        )
        assert res["success"] is True
        assert res["simulated"] is True
        assert res["status"] == "SIMULATED"
        # Full message must include prominent simulation banner
        assert "SIMULATION SCENARIO" in res["log_entry"]["message"]

    def test_alert_object_retains_full_provenance(self):
        """Standardized alert object must retain model version, Liljegren physics, and timestamp."""
        alert = AlertService.create_alert(
            role="worker",
            location="Varanasi",
            risk="High",
            wbgt_c=31.2,
            temp_c=39.5,
            data_timestamp="2026-09-24T12:00:00Z",
        )
        assert alert["data_timestamp"] == "2026-09-24T12:00:00Z"
        assert "Liljegren 2008" in alert["model_version"]
        assert alert["data_quality"] == "VALIDATED_NWP_HYPERLOCAL"


# ==============================================================================
# 7. REST API Integration & Contracts
# ==============================================================================
class TestAlertAPIEndpoints:
    def test_api_generate_alerts(self, client):
        """POST /api/alerts/generate creates role-based alerts."""
        payload = {
            "location": "Delhi",
            "risk": "High",
            "wbgt_c": 31.8,
            "temp_c": 40.5,
            "roles": ["worker", "supervisor", "healthcare_facility"],
            "language": "hi",
        }
        res = client.post("/api/alerts/generate", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["count"] == 3
        roles = [a["role"] for a in data["alerts"]]
        assert "worker" in roles
        assert "supervisor" in roles
        assert "healthcare_facility" in roles

    def test_api_preview_alert(self, client):
        """POST /api/alerts/preview returns full text and compact SMS."""
        payload = {
            "role": "supervisor",
            "location": "Ahmedabad",
            "risk": "Very High",
            "wbgt_c": 32.5,
            "temp_c": 42.0,
            "language": "mr",
        }
        res = client.post("/api/alerts/preview", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "full_message" in data
        assert "compact_sms" in data
        assert data["compact_char_count"] > 0
        assert data["role"] == "supervisor"
        assert data["language"] == "mr"

    def test_api_list_and_get_alert(self, client):
        """GET /api/alerts and GET /api/alerts/<id> operate correctly."""
        AlertService.create_alert(role="worker", location="Jaipur", risk="Moderate", wbgt_c=29.5, temp_c=38.0)
        AlertService.create_alert(role="authority", location="Jaipur", risk="Extreme", wbgt_c=33.5, temp_c=43.0)

        res = client.get("/api/alerts")
        assert res.status_code == 200
        data = res.get_json()
        assert data["count"] >= 2

        aid = data["alerts"][0]["alert_id"]
        res_single = client.get(f"/api/alerts/{aid}")
        assert res_single.status_code == 200
        assert res_single.get_json()["alert"]["alert_id"] == aid

    def test_api_approve_and_dispatch_workflow(self, client):
        """API endpoints for approve and dispatch operate end-to-end."""
        alert = AlertService.create_alert(role="authority", location="Delhi", risk="Extreme", wbgt_c=34.0, temp_c=45.0)
        aid = alert["alert_id"]

        # Approve via API
        res_appr = client.post(f"/api/alerts/{aid}/approve", json={"approver": "Chief Secretary", "notes": "Approved"})
        assert res_appr.status_code == 200
        assert res_appr.get_json()["success"] is True

        # Dispatch via API
        res_disp = client.post(f"/api/alerts/{aid}/dispatch", json={"phone": "+919876543210"})
        assert res_disp.status_code == 200
        assert res_disp.get_json()["success"] is True

    def test_api_preferences_endpoint(self, client):
        """GET & POST /api/alerts/preferences manages user preferences."""
        payload = {
            "phone": "+919876543299",
            "role": "supervisor",
            "preferred_language": "hi",
            "opted_in": True,
        }
        res_post = client.post("/api/alerts/preferences", json=payload)
        assert res_post.status_code == 200
        assert res_post.get_json()["preferences"]["preferred_language"] == "hi"

        res_get = client.get("/api/alerts/preferences?phone=+919876543299")
        assert res_get.status_code == 200
        assert res_get.get_json()["preferences"]["role"] == "supervisor"

    def test_alerts_html_page_loads(self, client):
        """GET /alerts renders the Alerts & Communications Command dashboard."""
        res = client.get("/alerts")
        assert res.status_code == 200
        assert b"Alerts &amp; Communications Command" in res.data or b"Alerts & Communications Command" in res.data
        assert b"Step 7: Alert Intelligence" in res.data


# ==============================================================================
# 8. 5-Tier Chain of Command Escalation Hierarchy
# WORKER -> SUPERVISOR -> SITE MANAGER -> AUTHORITY -> SYSTEM ADMIN
# ==============================================================================
class TestEscalationChainHierarchy:
    def test_escalation_ladder_sequence(self):
        """Escalation ladder must follow exact 5-tier sequence."""
        from services.alert_service import ESCALATION_LADDER
        assert ESCALATION_LADDER == ["worker", "supervisor", "site_manager", "authority", "system_admin"]

    def test_full_chain_step_by_step_escalation(self):
        """Alert must successfully climb from WORKER -> SUPERVISOR -> SITE MANAGER -> AUTHORITY -> SYSTEM ADMIN."""
        alert = AlertService.create_alert(
            role="worker",
            location="Ahmedabad",
            risk="High",
            wbgt_c=32.0,
            temp_c=41.0,
        )
        aid = alert["alert_id"]
        assert alert["role"] == "worker"
        assert alert["escalation_level"] == 3  # High risk base escalation level

        # 1. Worker -> Supervisor
        res1 = AlertService.escalate_alert(aid, escalated_by="Gang Leader", reason="Continuous sun work exceeds safe limits")
        assert res1["success"] is True
        assert res1["current_role"] == "supervisor"
        assert res1["alert"]["role"] == "supervisor"
        assert "SUPERVISOR" in res1["alert"]["full_message"]

        # 2. Supervisor -> Site Manager
        res2 = AlertService.escalate_alert(aid, escalated_by="Site Supervisor", reason="Shaded rest zone capacity exhausted")
        assert res2["success"] is True
        assert res2["current_role"] == "site_manager"
        assert res2["alert"]["role"] == "site_manager"
        assert "SITE MANAGER" in res2["alert"]["full_message"]

        # 3. Site Manager -> Authority
        res3 = AlertService.escalate_alert(aid, escalated_by="Project Site Manager", reason="Site-wide electrical outage disabled misting fans")
        assert res3["success"] is True
        assert res3["current_role"] == "authority"
        assert res3["alert"]["role"] == "authority"
        assert "AUTHORITY" in res3["alert"]["full_message"]

        # 4. Authority -> System Admin
        res4 = AlertService.escalate_alert(aid, escalated_by="Disaster Officer", reason="City-wide multi-ward emergency declared")
        assert res4["success"] is True
        assert res4["current_role"] == "system_admin"
        assert res4["alert"]["role"] == "system_admin"
        assert "SYSTEM ADMINISTRATOR" in res4["alert"]["full_message"]

        # Audit trail must document every escalation hop
        audit = AlertService.get_alert(aid)["audit_trail"]
        escalation_actions = [e for e in audit if e["action"] == "ESCALATED"]
        assert len(escalation_actions) == 4

    def test_escalation_cap_at_system_admin(self):
        """Attempting to escalate past SYSTEM ADMIN returns MAX_ESCALATION."""
        alert = AlertService.create_alert(
            role="system_admin",
            location="Delhi",
            risk="Extreme",
            wbgt_c=34.0,
            temp_c=45.0,
        )
        res = AlertService.escalate_alert(alert["alert_id"])
        assert res["success"] is False
        assert res["status"] == "MAX_ESCALATION"

    def test_api_escalate_endpoint(self, client):
        """POST /api/alerts/<id>/escalate escalates alert via REST API."""
        alert = AlertService.create_alert(role="worker", location="Nagpur", risk="High", wbgt_c=32.0, temp_c=40.0)
        aid = alert["alert_id"]

        res = client.post(f"/api/alerts/{aid}/escalate", json={
            "escalated_by": "Field Safety Inspector",
            "reason": "WBGT rose rapidly from 30C to 32C"
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["previous_role"] == "worker"
        assert data["current_role"] == "supervisor"

    def test_system_admin_multilingual_templates(self):
        """System admin alerts must render correctly in English, Hindi, and Marathi."""
        for lang in ["en", "hi", "mr"]:
            alert = AlertService.create_alert(
                role="system_admin",
                location="Hyderabad",
                risk="Extreme",
                wbgt_c=33.5,
                temp_c=43.0,
                language=lang,
            )
            assert alert["role"] == "system_admin"
            msg = alert["full_message"]
            assert "Hyderabad" in msg
            assert "33.5°C" in msg
            assert "43.0°C" in msg
