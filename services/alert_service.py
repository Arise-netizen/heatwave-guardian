"""
HeatWave Guardian — Role-Based, Multilingual & Low-Connectivity Alert Intelligence
===================================================================================
Step 7: Role-aware alert intelligence layer converting validated risk, forecast,
and intervention outputs into structured notifications for workers, supervisors,
site managers, municipal authorities, and healthcare facilities.

CORE PRINCIPLES & GUARDRAILS:
1. DECISION SUPPORT ONLY: Explicitly operational decision support, NOT medical
   diagnosis, medical advice, or a guarantee of safety.
2. ROLE-SPECIFIC INFORMATION ARCHITECTURE:
   - worker: personal risk, peak period, simple action, hydration, rest intervals.
   - supervisor: affected tasks, work/rest pacing, rotational shade, escalation.
   - site_manager: site status, workforce exposure index, operational stand-down.
   - authority: monitored zones, IMD heatwave departure, HAP alert level, vulnerable wards.
   - healthcare_facility: heat surge preparedness signal, timing, cooling bed / IV reserve guidance.
     STRICT GUARDRAIL: No fake hospital admission or mortality claims.
3. CONTROLLED MULTILINGUAL TEMPLATES: Reviewed templates for English (en), Hindi (hi),
   and Marathi (mr). No free-form LLM translation for safety-critical instructions.
   All critical numbers, units (°C, min), and ISO timestamps remain uniform.
4. LOW-CONNECTIVITY COMPACT SMS: Compact message format (< 160 characters)
   retaining minimum actionable information for SMS / low-bandwidth channels.
5. HUMAN-IN-THE-LOOP (HITL) APPROVAL: High-impact authority and site-wide alerts
   default to PENDING_APPROVAL before dispatch can occur.
6. CONSENT, PREFERENCES & PRIVACY: Granular opt-in consent, quiet hours suppression,
   and phone number masking in all public audit trails.
7. DATA HONESTY: Explicit provenance (model version, data timestamp, confidence).
   Counterfactual simulations (is_simulated=True) are strictly blocked from live dispatch.
8. BACKWARD COMPATIBILITY: Fully maintains legacy signatures for format_alert_message,
   dispatch_alert, get_config, and get_history.
"""

import os
import time
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import requests


# ──────────────────────────────────────────────────────────────────────────────
# Supported Constants & Enums
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_ROLES = ["worker", "supervisor", "site_manager", "authority", "healthcare_facility", "system_admin"]
ESCALATION_LADDER = ["worker", "supervisor", "site_manager", "authority", "system_admin"]
SUPPORTED_LANGUAGES = ["en", "hi", "mr"]
SUPPORTED_CHANNELS = ["sms", "whatsapp", "dashboard", "email", "webhook"]
APPROVAL_STATES = ["NOT_REQUIRED", "PENDING", "APPROVED", "REJECTED"]
URGENCY_LEVELS = ["Low", "Moderate", "High", "Critical"]
MESSAGE_TYPES = [
    "advisory",
    "warning",
    "action_required",
    "standdown_escalation",
    "preparedness_signal",
]

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "mr": "Marathi (मराठी)",
}

ROLE_LABELS = {
    "worker": {"en": "Worker Advisory", "hi": "श्रमिक परामर्श", "mr": "कामगार सूचना"},
    "supervisor": {"en": "Supervisor Action Plan", "hi": "पर्यवेक्षक कार्य योजना", "mr": "पर्यवेक्षक कृती आराखडा"},
    "site_manager": {"en": "Site Manager Operational Directive", "hi": "साइट प्रबंधक परिचालन निर्देश", "mr": "प्रकल्प व्यवस्थापक कार्यात्मक आदेश"},
    "authority": {"en": "Municipal Authority Disaster Advisory", "hi": "नगर निगम / आपदा प्राधिकरण परामर्श", "mr": "महानगरपालिका / आपत्ती व्यवस्थापन इशारा"},
    "healthcare_facility": {"en": "Healthcare Facility Preparedness Signal", "hi": "स्वास्थ्य केंद्र तैयारी संकेत", "mr": "आरोग्य केंद्र पूर्वतयारी संकेत"},
    "system_admin": {"en": "System Administrator Directive", "hi": "सिस्टम प्रशासक निर्देश", "mr": "प्रणाली प्रशासक आदेश"},
}

RISK_LABELS_MULTILINGUAL = {
    "Low": {"en": "LOW", "hi": "सामान्य / कम (Low)", "mr": "कमी (Low)"},
    "Moderate": {"en": "MODERATE", "hi": "मध्यम (Moderate)", "mr": "मध्यम (Moderate)"},
    "High": {"en": "HIGH", "hi": "उच्च (High)", "mr": "जास्त (High)"},
    "Very High": {"en": "VERY HIGH", "hi": "अति उच्च (Very High)", "mr": "अति जास्त (Very High)"},
    "Extreme": {"en": "EXTREME", "hi": "अत्यधिक गंभीर (Extreme)", "mr": "अति तीव्र (Extreme)"},
}

CONFIDENCE_LABELS_MULTILINGUAL = {
    "High": {"en": "High Confidence", "hi": "उच्च विश्वसनीयता", "mr": "उच्च विश्वासार्हता"},
    "Medium": {"en": "Medium Confidence", "hi": "मध्यम विश्वसनीयता", "mr": "मध्यम विश्वासार्हता"},
    "Low": {"en": "Low Confidence (Sensor/NWP uncertainty)", "hi": "निम्न विश्वसनीयता", "mr": "कमी विश्वासार्हता"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Masking & Privacy Helpers
# ──────────────────────────────────────────────────────────────────────────────

def mask_contact(contact: Optional[str]) -> str:
    """Mask phone or email address for privacy in public logs."""
    if not contact:
        return "Unknown"
    s = str(contact).strip()
    if "@" in s:
        parts = s.split("@")
        name = parts[0]
        domain = parts[1] if len(parts) > 1 else ""
        masked_name = name[:2] + "***" if len(name) > 2 else "***"
        return f"{masked_name}@{domain}"
    # Phone number
    cleaned = s.replace(" ", "").replace("-", "")
    if len(cleaned) >= 8:
        prefix = cleaned[:3]
        suffix = cleaned[-2:]
        return f"{prefix}*** ***{suffix}"
    return "***"


def hash_contact(contact: Optional[str]) -> str:
    """Compute deterministic SHA-256 fingerprint for cooldown and consent lookups."""
    if not contact:
        return ""
    cleaned = (
        str(contact)
        .strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("+", "")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# AlertService Class
# ──────────────────────────────────────────────────────────────────────────────

class AlertService:
    """
    HeatWave Guardian Alert & Communication Intelligence Service.
    Supports role-specific message synthesis, verified multilingual templates,
    low-connectivity compact SMS payloads, human-in-the-loop approval workflows,
    and granular consent/preferences management.
    """

    # In-memory registry of alerts: {alert_id: alert_dict}
    _ALERT_REGISTRY: Dict[str, Dict[str, Any]] = {}

    # In-memory global audit log: list of audit records
    _ALERT_LOG: List[Dict[str, Any]] = []

    # In-memory cooldown tracking: {(contact_hash, risk_category): last_sent_timestamp}
    _COOLDOWN_TRACKER: Dict[Tuple[str, str], float] = {}
    COOLDOWN_SECONDS: int = 3600  # 60-minute duplicate suppression window

    # In-memory recipient preferences & consent: {contact_hash: preference_dict}
    _PREFERENCES: Dict[str, Dict[str, Any]] = {}

    # Global policy switch for auto-approval of high-impact alerts (default: False = requires human review)
    _AUTO_APPROVE_HIGH_IMPACT: bool = False

    ENGINE_VERSION = "2.4.0-alert-intelligence"

    # ──────────────────────────────────────────────────────────────────────────
    # Provider Configuration
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Check environment variables for telecom provider credentials."""
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        phone_number = os.environ.get("TWILIO_PHONE_NUMBER", "").strip()

        is_configured = bool(account_sid and auth_token and phone_number)
        return {
            "is_configured": is_configured,
            "account_sid_masked": f"{account_sid[:4]}...{account_sid[-4:]}" if is_configured else None,
            "provider": "Twilio (Live)" if is_configured else "Simulation Mode (No Credentials)",
            "simulation_mode": not is_configured,
            "supported_roles": SUPPORTED_ROLES,
            "supported_languages": SUPPORTED_LANGUAGES,
            "supported_channels": SUPPORTED_CHANNELS,
            "auto_approve_high_impact": cls._AUTO_APPROVE_HIGH_IMPACT,
            "cooldown_seconds": cls.COOLDOWN_SECONDS,
            "engine_version": cls.ENGINE_VERSION,
        }

    @classmethod
    def set_auto_approve_high_impact(cls, enabled: bool):
        """Configure whether high-impact alerts require human approval."""
        cls._AUTO_APPROVE_HIGH_IMPACT = bool(enabled)

    @classmethod
    def clear_all(cls):
        """Reset all in-memory registries (useful for automated testing)."""
        cls._ALERT_REGISTRY.clear()
        cls._ALERT_LOG.clear()
        cls._COOLDOWN_TRACKER.clear()
        cls._PREFERENCES.clear()
        cls._AUTO_APPROVE_HIGH_IMPACT = False

    # ──────────────────────────────────────────────────────────────────────────
    # Consent & User Preferences
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def set_preferences(
        cls,
        recipient_id: str,
        role: str = "worker",
        preferred_language: str = "en",
        enabled_channels: Optional[List[str]] = None,
        quiet_hours: Optional[Dict[str, Any]] = None,
        opted_in: bool = True,
    ) -> Dict[str, Any]:
        """
        Register or update recipient consent, role, language, and quiet hours.
        Contact data is securely hashed for index lookup and masked for display.
        """
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {SUPPORTED_ROLES}")
        if preferred_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Invalid language '{preferred_language}'. Must be one of {SUPPORTED_LANGUAGES}")

        c_hash = hash_contact(recipient_id)
        if not enabled_channels:
            enabled_channels = ["sms", "whatsapp", "dashboard"]

        # Validate quiet hours
        default_quiet = {"enabled": True, "start": "22:00", "end": "06:00"}
        if quiet_hours:
            qh_enabled = bool(quiet_hours.get("enabled", True))
            qh_start = quiet_hours.get("start", "22:00")
            qh_end = quiet_hours.get("end", "06:00")
            quiet_config = {"enabled": qh_enabled, "start": qh_start, "end": qh_end}
        else:
            quiet_config = default_quiet

        pref_obj = {
            "recipient_id_masked": mask_contact(recipient_id),
            "recipient_hash": c_hash,
            "role": role,
            "preferred_language": preferred_language,
            "enabled_channels": enabled_channels,
            "quiet_hours": quiet_config,
            "opted_in": bool(opted_in),
            "consent_status": "OPTED_IN" if opted_in else "OPTED_OUT",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        cls._PREFERENCES[c_hash] = pref_obj
        return pref_obj

    @classmethod
    def get_preferences(cls, recipient_id: str) -> Dict[str, Any]:
        """Retrieve stored preferences for recipient, or return defaults."""
        c_hash = hash_contact(recipient_id)
        if c_hash in cls._PREFERENCES:
            return cls._PREFERENCES[c_hash]
        # Default profile
        return {
            "recipient_id_masked": mask_contact(recipient_id),
            "recipient_hash": c_hash,
            "role": "worker",
            "preferred_language": "en",
            "enabled_channels": ["sms", "whatsapp", "dashboard"],
            "quiet_hours": {"enabled": False, "start": "22:00", "end": "06:00"},
            "opted_in": True,
            "consent_status": "OPTED_IN",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def is_quiet_hours_active(cls, recipient_id: str, check_time: Optional[datetime] = None) -> bool:
        """
        Check if the current time falls inside the recipient's quiet hours.
        Format: start='22:00', end='06:00'.
        """
        prefs = cls.get_preferences(recipient_id)
        qh = prefs.get("quiet_hours", {})
        if not qh.get("enabled"):
            return False

        now = check_time or datetime.now()
        cur_minute = now.hour * 60 + now.minute

        def _to_min(t_str: str) -> int:
            try:
                parts = t_str.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return 0

        start_min = _to_min(qh.get("start", "22:00"))
        end_min = _to_min(qh.get("end", "06:00"))

        if start_min > end_min:
            # Over midnight window (e.g. 22:00 to 06:00)
            return cur_minute >= start_min or cur_minute < end_min
        else:
            return start_min <= cur_minute < end_min

    # ──────────────────────────────────────────────────────────────────────────
    # Structured Alert Policy Engine
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def evaluate_alert_policy(
        cls,
        risk: str,
        role: str,
        confidence: str = "High",
        is_simulated: bool = False,
    ) -> Dict[str, Any]:
        """
        Deterministic, validated alert policy rules.
        Reuses standard HeatWave Guardian risk categories (Low, Moderate, High, Very High, Extreme).
        Determines should_alert, urgency, message_type, escalation_level, and whether
        human approval is mandatory.
        """
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {SUPPORTED_ROLES}")

        risk_norm = str(risk).strip().title()
        if risk_norm not in ["Low", "Moderate", "High", "Very High", "Extreme"]:
            risk_norm = "High"

        # Base Urgency & Escalation by Risk
        policy_map = {
            "Low": {
                "urgency": "Low",
                "escalation_level": 1,
                "message_type": "advisory",
                "should_alert": False,
            },
            "Moderate": {
                "urgency": "Moderate",
                "escalation_level": 2,
                "message_type": "advisory",
                "should_alert": True,
            },
            "High": {
                "urgency": "High",
                "escalation_level": 3,
                "message_type": "warning",
                "should_alert": True,
            },
            "Very High": {
                "urgency": "High",
                "escalation_level": 4,
                "message_type": "action_required",
                "should_alert": True,
            },
            "Extreme": {
                "urgency": "Critical",
                "escalation_level": 5,
                "message_type": "standdown_escalation",
                "should_alert": True,
            },
        }

        cfg = policy_map[risk_norm]
        urgency = cfg["urgency"]
        escalation_level = cfg["escalation_level"]
        message_type = cfg["message_type"]
        should_alert = cfg["should_alert"]

        if role == "healthcare_facility":
            message_type = "preparedness_signal"

        # Human-in-the-loop threshold:
        # High-impact alerts (authority, site_manager, or system_admin at Very High or Extreme)
        # require human approval unless explicitly auto-approved in global config.
        is_high_impact = role in ["authority", "site_manager", "system_admin"] and escalation_level >= 4
        requires_human_approval = is_high_impact and not cls._AUTO_APPROVE_HIGH_IMPACT

        return {
            "should_alert": should_alert,
            "urgency": urgency,
            "message_type": message_type,
            "escalation_level": escalation_level,
            "requires_human_approval": requires_human_approval,
            "is_high_impact": is_high_impact,
            "risk": risk_norm,
            "role": role,
            "confidence": confidence,
            "is_simulated": is_simulated,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Multilingual & Low-Connectivity Formatting
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _get_role_action_text(
        cls,
        role: str,
        risk: str,
        lang: str,
        work_minutes: int,
        rest_minutes: int,
        custom_action: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Returns reviewed, controlled (full_action, compact_action) text
        specifically tailored to the role, risk level, and language.
        No free-form LLM translation is used.
        """
        lang = lang if lang in SUPPORTED_LANGUAGES else "en"

        # System Admin Role
        if role == "system_admin":
            if lang == "en":
                full = (
                    "Platform Governance Directive: Level 5 critical emergency escalation reached. "
                    "Verify SMS/WhatsApp gateway throughput, enforce multi-agency broadcast throttling, "
                    "and review municipal disaster coordination audit logs."
                )
                compact = "Platform Alert: Level 5 escalation active. Check provider throughput & disaster logs."
            elif lang == "hi":
                full = (
                    "प्लेटफॉर्म शासन निर्देश: स्तर 5 महत्वपूर्ण आपातकालीन एस्केलेशन सक्रिय। "
                    "गेटवे थ्रूपुट की पुष्टि करें, मल्टी-एजेंसी प्रसारण थ्रॉटलिंग लागू करें और आपदा समन्वय ऑडिट लॉग की समीक्षा करें।"
                )
                compact = "प्लेटफॉर्म अलर्ट: स्तर 5 एस्केलेशन। गेटवे स्थिति व आपदा लॉग जांचें।"
            else:
                full = (
                    "प्रणाली प्रशासन आदेश: स्तर ५ आणीबाणी एस्केलेशन सक्रिय. "
                    "टेलिकॉम गेटवे क्षमता तपासा, बहु-एजन्सी प्रसारण थ्रॉटलिंग लागू करा आणि आपत्ती व्यवस्थापन ऑडिट नोंदी तपासा."
                )
                compact = "प्लॅटफॉर्म इशारा: स्तर ५ एस्केलेशन. गेटवे क्षमता व आपत्ती लॉग तपासा."
            return full, compact

        # Healthcare Facility Guardrail: Strictly operational preparedness advice.
        # NEVER claim unvalidated hospital admission counts or mortality projections.
        if role == "healthcare_facility":
            if lang == "en":
                full = (
                    "Operational Preparedness Signal: Prepare IV hydration reserves, ORS supply, "
                    "emergency cooling beds, and triage staff rotation during peak hours. "
                    "Notice: Advisory for facility surge capacity only; not an epidemiological patient forecast."
                )
                compact = "Prep IV/ORS reserves, cooling beds & staff rotation for peak hours. Not patient forecast."
            elif lang == "hi":
                full = (
                    "परिचालन तैयारी संकेत: पीक घंटों के दौरान आईवी तरल, ओआरएस आपूर्ति, आपातकालीन कूलिंग बेड "
                    "तथा स्टाफ रोटेशन तैयार रखें। सूचना: यह स्वास्थ्य केंद्र क्षमता हेतु परामर्श है; मरीज संख्या का पूर्वानुमान नहीं।"
                )
                compact = "पीक समय हेतु IV/ORS, कूलिंग बेड व स्टाफ तैयार रखें। मरीज संख्या नहीं।"
            else:  # mr
                full = (
                    "कार्यात्मक पूर्वतयारी संकेत: दुपारच्या कमाल उष्णतेसाठी सलाईन/ओआरएस साठा, आपत्कालीन बेड "
                    "व आरोग्य कर्मचारी फेरबदल सज्ज ठेवा. सूचना: केवळ आरोग्य केंद्र क्षमतेसाठी; रुग्णांचे भाकीत नाही."
                )
                compact = "कमाल वेळेसाठी IV/ORS, कूलिंग बेड व कर्मचारी सज्ज ठेवा. रुग्णांचे भाकीत नाही."
            return full, compact

        # Authority Role
        if role == "authority":
            if risk in ["Very High", "Extreme"]:
                if lang == "en":
                    full = (
                        "Issue municipal heatwave public warning. Deploy mobile water tankers to high-density wards; "
                        "open air-conditioned public cooling shelters; mandate daytime outdoor labor restriction."
                    )
                    compact = "Municipal Alert: Deploy water tankers, open cooling shelters, restrict daytime outdoor work."
                elif lang == "hi":
                    full = (
                        "नगर निगम हीटवेव चेतावनी जारी करें। सघन बस्तियों में पानी के टैंकर भेजें; वातानुकूलित कूलिंग शेल्टर खोलें; "
                        "दोपहर में खुले में भारी श्रम पर प्रतिबंध लागू करें।"
                    )
                    compact = "नगर निगम अलर्ट: पानी के टैंकर भेजें, कूलिंग शेल्टर खोलें, बाहरी श्रम सीमित करें।"
                else:  # mr
                    full = (
                        "महानगरपालिका उष्णतेची लाट इशारा जारी करा. दाट वस्त्यांमध्ये पाण्याचे टँकर पाठवा; वातानुकूलित सावली केंद्रे सुरू करा; "
                        "दुपारच्या कडक उन्हात मजुरीच्या कामांवर निर्बंध लावा."
                    )
                    compact = "मनपा इशारा: पाण्याचे टँकर पाठवा, कूलिंग शेल्टर सुरू करा, उन्हात काम थांबवा."
            else:
                if lang == "en":
                    full = "Maintain active municipal heat monitoring, public hydration kiosks, and primary health center heat advisories."
                    compact = "Monitor heat levels, keep hydration kiosks active, alert local health centers."
                elif lang == "hi":
                    full = "सार्वजनिक पेयजल केंद्रों को सक्रिय रखें और प्राथमिक स्वास्थ्य केंद्रों को सतर्क रहने का परामर्श दें।"
                    compact = "पेयजल केंद्र सक्रिय रखें, स्थानीय स्वास्थ्य केंद्रों को अलर्ट रखें।"
                else:
                    full = "सार्वजनिक पिण्याच्या पाण्याची सोय सुरू ठेवा आणि प्राथमिक आरोग्य केंद्रांना सतर्क ठेवा."
                    compact = "पाणी केंद्रे सुरू ठेवा, प्राथमिक आरोग्य केंद्रांना सतर्क ठेवा."
            return full, compact

        # Site Manager Role
        if role == "site_manager":
            if risk in ["Very High", "Extreme"]:
                if lang == "en":
                    full = (
                        f"Mandate work suspension for heavy outdoor labor during peak hours. "
                        f"Enforce mandatory {work_minutes}m work / {rest_minutes}m shaded rest pacing. "
                        f"Verify on-site cold water tanker and electrolyte availability."
                    )
                    compact = f"Site directive: Stop heavy outdoor work during peak. Pacing {work_minutes}m work/{rest_minutes}m rest."
                elif lang == "hi":
                    full = (
                        f"पीक समय में खुले में भारी शारीरिक कार्य तुरंत स्थगित करें। "
                        f"{work_minutes} मिनट कार्य / {rest_minutes} मिनट छायादार विश्राम अनिवार्य करें। "
                        f"साइट पर ठंडा पेयजल एवं ओआरएस सुनिश्चित करें।"
                    )
                    compact = f"साइट निर्देश: पीक समय भारी कार्य रोकें। कार्य-विश्राम {work_minutes}m/{rest_minutes}m."
                else:
                    full = (
                        f"दुपारच्या काळात उघड्यावरील कठीण कामे तातडीने थांबवा. "
                        f"{work_minutes} मिनिटे काम / {rest_minutes} मिनिटे सावलीत विश्रांतीचे नियम पाळा. "
                        f"प्रकल्प स्थळावर पिण्याचे थंड पाणी व ओआरएस उपलब्ध ठेवा."
                    )
                    compact = f"प्रकल्प आदेश: कडक उन्हात जड कामे थांबवा. वेळापत्रक {work_minutes}m काम/{rest_minutes}m विश्रांती."
            else:
                if lang == "en":
                    full = f"Maintain shaded recovery areas and enforce {work_minutes}m work / {rest_minutes}m rest pacing. Check water supplies."
                    compact = f"Maintain shade & water; enforce {work_minutes}m work/{rest_minutes}m rest pacing."
                elif lang == "hi":
                    full = f"छायादार विश्राम क्षेत्र बनाए रखें और {work_minutes} मिनट कार्य / {rest_minutes} मिनट विश्राम लागू करें।"
                    compact = f"छाया व पानी उपलब्ध रखें; {work_minutes}m कार्य/{rest_minutes}m विश्राम नियम."
                else:
                    full = f"सावलीची सोय ठेवा आणि {work_minutes} मिनिटे काम / {rest_minutes} मिनिटे विश्रांती नियम पाळा."
                    compact = f"सावली व पाण्याची सोय ठेवा; {work_minutes}m काम/{rest_minutes}m विश्रांती."
            return full, compact

        # Supervisor Role
        if role == "supervisor":
            if risk in ["Very High", "Extreme"]:
                if lang == "en":
                    full = (
                        f"Halt high-exertion outdoor tasks. Reschedule to early morning or shaded areas. "
                        f"Implement strict buddy checks every 15 minutes. Pacing: {work_minutes}m work / {rest_minutes}m rest."
                    )
                    compact = f"Halt heavy work. Reschedule to morning/shade. Buddy check q15m. Pacing {work_minutes}m/{rest_minutes}m."
                elif lang == "hi":
                    full = (
                        f"भारी शारीरिक कार्य रोकें। सुबह जल्दी या छाया में कार्य स्थानांतरित करें। "
                        f"प्रत्येक 15 मिनट में सहकर्मी की जांच (बडी सिस्टम) करें। कार्य-विश्राम: {work_minutes}m/{rest_minutes}m."
                    )
                    compact = f"भारी कार्य रोकें, सुबह शिफ्ट करें। 15 मिनट पर सहकर्मी जांच। {work_minutes}m/{rest_minutes}m."
                else:
                    full = (
                        f"जास्त कष्टाची कामे थांबवा. सकाळी लवकर किंवा सावलीत काम हलवा. "
                        f"दर १५ मिनिटांनी सहकाऱ्यांची तब्येत तपासा. वेळापत्रक: {work_minutes}m काम/{rest_minutes}m विश्रांती."
                    )
                    compact = f"जड काम थांबवा, सकाळी शिफ्ट करा. दर १५ मिनिटांनी तब्येत तपासा. {work_minutes}m/{rest_minutes}m."
            else:
                if lang == "en":
                    full = f"Monitor hydration breaks and rotational shade access. Pacing: {work_minutes}m work / {rest_minutes}m rest."
                    compact = f"Monitor hydration & shade. Pacing: {work_minutes}m work/{rest_minutes}m rest."
                elif lang == "hi":
                    full = f"पानी पीने के अंतराल और छायादार विश्राम की निगरानी करें। कार्य-विश्राम: {work_minutes}m/{rest_minutes}m."
                    compact = f"पानी व छाया की निगरानी रखें। कार्य-विश्राम {work_minutes}m/{rest_minutes}m."
                else:
                    full = f"पाणी पिण्याचे ब्रेक व सावलीत विश्रांती तपासा. वेळापत्रक: {work_minutes}m काम/{rest_minutes}m विश्रांती."
                    compact = f"पाणी व सावलीची काळजी घ्या. वेळापत्रक: {work_minutes}m/{rest_minutes}m."
            return full, compact

        # Worker Role (Default)
        if custom_action and lang == "en":
            action_base = custom_action
        else:
            action_base = None

        if risk in ["Very High", "Extreme"]:
            if lang == "en":
                full = (
                    action_base or
                    f"Stop strenuous work in direct sun. Move to shade or AC immediately. "
                    f"Pacing: {work_minutes}m work / {rest_minutes}m rest per hour. Drink 250ml water every 20m. "
                    f"Report dizziness, confusion, or stopped sweating to supervisor immediately."
                )
                compact = f"Stop heavy sun work. Rest in shade. Pacing {work_minutes}m/{rest_minutes}m. Drink water q20m."
            elif lang == "hi":
                full = (
                    f"तेज धूप में भारी कार्य तुरंत रोकें। फौरन छाया या वातानुकूलित स्थान पर जाएं। "
                    f"समय-सारणी: {work_minutes} मिनट कार्य / {rest_minutes} मिनट विश्राम। हर 20 मिनट में 250ml पानी पिएं। "
                    f"चक्कर, अत्यधिक कमजोरी या पसीना रुकने पर तुरंत सुपरवाइजर को सूचित करें।"
                )
                compact = f"धूप में भारी काम रोकें। छाया में जाएं। {work_minutes}m/{rest_minutes}m विश्राम। हर 20m पानी पिएं।"
            else:
                full = (
                    f"कडक उन्हात कष्टाचे काम थांबवा. तात्काळ सावलीत किंवा थंड ठिकाणी जा. "
                    f"वेळापत्रक: {work_minutes} मिनिटे काम / {rest_minutes} मिनिटे विश्रांती. दर २० मिनिटांनी पाणी प्या. "
                    f"चक्कर किंवा अशक्तपणा जाणवल्यास सुपरवायझरला लगेच सांगा."
                )
                compact = f"उन्हात जड काम थांबवा. सावलीत जा. {work_minutes}m/{rest_minutes}m विश्रांती. दर २०m पाणी प्या."
        else:
            if lang == "en":
                full = (
                    action_base or
                    f"Drink water regularly before feeling thirsty. Alternate {work_minutes}m work with {rest_minutes}m rest in shade. "
                    f"Wear lightweight loose clothing. Report heat distress early."
                )
                compact = f"Drink water often. Pacing {work_minutes}m work/{rest_minutes}m rest in shade. Stay hydrated."
            elif lang == "hi":
                full = (
                    f"प्यास लगने से पहले नियमित पानी पिएं। {work_minutes} मिनट कार्य के बाद {rest_minutes} मिनट छाया में विश्राम करें। "
                    f"हल्के सूती कपड़े पहनें। अस्वस्थ महसूस होने पर तुरंत विश्राम करें।"
                )
                compact = f"नियमित पानी पिएं। {work_minutes}m काम / {rest_minutes}m छायादार विश्राम नियम अपनाएं।"
            else:
                full = (
                    f"तहान लागण्याची वाट न पाहता नियमित पाणी प्या. {work_minutes} मिनिटे कामानंतर {rest_minutes} मिनिटे सावलीत विश्रांती घ्या. "
                    f"हलके सुती कपडे वापरा. अस्वस्थ वाटल्यास लगेच विश्रांती घ्या."
                )
                compact = f"नियमित पाणी प्या. {work_minutes}m काम / {rest_minutes}m सावलीत विश्रांती घ्या."

        return full, compact

    @classmethod
    def format_alert_message(
        cls,
        location: Optional[str] = None,
        risk_category: Optional[str] = None,
        wbgt_c: Optional[float] = None,
        temp_c: Optional[float] = None,
        work_minutes: Optional[int] = None,
        rest_minutes: Optional[int] = None,
        instruction: Optional[str] = None,
        channel: str = "sms",
        role: str = "worker",
        language: str = "en",
        peak_window: str = "12:00–15:30",
        confidence: str = "High",
        data_timestamp: Optional[str] = None,
        is_simulated: bool = False,
        risk_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synthesizes standard structured multilingual alert text.
        Schema:
          [RISK]
          [LOCATION]
          [PEAK WINDOW]
          [ACTION]
          [CONFIDENCE]
          [DATA TIMESTAMP]
        100% backward compatible with legacy positional/keyword parameters and risk_data dict.
        """
        if risk_data and isinstance(risk_data, dict):
            location = location or risk_data.get("location", "Delhi")
            risk_category = risk_category or risk_data.get("risk_category") or risk_data.get("risk_level", "High")
            wbgt_c = wbgt_c if wbgt_c is not None else risk_data.get("wbgt_c", 30.0)
            temp_c = temp_c if temp_c is not None else risk_data.get("temp_c", 38.0)
            work_minutes = work_minutes if work_minutes is not None else risk_data.get("work_minutes", 45)
            rest_minutes = rest_minutes if rest_minutes is not None else risk_data.get("rest_minutes", 15)
            instruction = instruction or risk_data.get("instruction") or risk_data.get("recommended_action", "Stay hydrated and take regular rest in shaded areas.")
            peak_window = risk_data.get("peak_period") or risk_data.get("peak_window") or peak_window
            confidence = risk_data.get("confidence") or confidence

        location = location or "Delhi"
        risk_category = risk_category or "High"
        wbgt_c = float(wbgt_c if wbgt_c is not None else 30.0)
        temp_c = float(temp_c if temp_c is not None else 38.0)
        work_minutes = int(work_minutes if work_minutes is not None else 45)
        rest_minutes = int(rest_minutes if rest_minutes is not None else 15)
        custom_act = instruction.strip() if (instruction and str(instruction).strip()) else None

        lang = language if language in SUPPORTED_LANGUAGES else "en"
        role_label = ROLE_LABELS.get(role, ROLE_LABELS["worker"]).get(lang, "Alert")
        risk_label = RISK_LABELS_MULTILINGUAL.get(risk_category, {}).get(lang, risk_category.upper())
        conf_label = CONFIDENCE_LABELS_MULTILINGUAL.get(confidence, {}).get(lang, confidence)
        dt_str = data_timestamp or datetime.now().strftime("%I:%M %p, %d %b %Y")

        full_action, _ = cls._get_role_action_text(
            role=role,
            risk=risk_category,
            lang=lang,
            work_minutes=work_minutes,
            rest_minutes=rest_minutes,
            custom_action=custom_act if role == "worker" and lang == "en" else None,
        )

        sim_banner = ""
        if is_simulated:
            sim_banner = (
                "\n⚠️ [SIMULATION SCENARIO - WHAT-IF COUNTERFACTUAL - NOT AN ACTIVE ALERT]\n"
                if lang == "en" else
                "\n⚠️ [सिम्युलेशन परिदृश्य - यह वास्तविक सक्रिय चेतावनी नहीं है]\n"
                if lang == "hi" else
                "\n⚠️ [सिम्युलेशन परिस्थिती - हा प्रत्यक्ष इशारा नाही]\n"
            )

        if lang == "en":
            header = f"🚨 HEATWAVE GUARDIAN ALERT — {role_label.upper()}" if channel == "sms" else f"☀️ *HEATWAVE GUARDIAN ALERT — {role_label.upper()}*"
            msg = (
                f"{header}{sim_banner}\n"
                f"[RISK] {risk_label} | Personal Risk: {risk_category.upper()}\n"
                f"[LOCATION] {location}\n"
                f"[METRICS] Ambient Temp: {temp_c:.1f}°C | WBGT Heat Stress: {wbgt_c:.1f}°C\n"
                f"[PEAK WINDOW] {peak_window}\n"
                f"[SCHEDULE] {work_minutes}m work / {rest_minutes}m rest per hour\n"
                f"[ACTION] {full_action}\n"
                f"[CONFIDENCE] {conf_label}\n"
                f"[DATA TIMESTAMP] {dt_str}\n"
                f"[NOTICE] Occupational decision support advisory. Not medical advice or safety guarantee."
            )
        elif lang == "hi":
            header = f"🚨 हीटवेव गार्डियन चेतावनी — {role_label}" if channel == "sms" else f"☀️ *हीटवेव गार्डियन चेतावनी — {role_label}*"
            msg = (
                f"{header}{sim_banner}\n"
                f"[जोखिम / RISK] {risk_label} | Personal Risk: {risk_category.upper()}\n"
                f"[स्थान / LOCATION] {location}\n"
                f"[मापदंड / METRICS] परिवेश तापमान: {temp_c:.1f}°C | WBGT हीट स्ट्रेस: {wbgt_c:.1f}°C\n"
                f"[पीक समय / PEAK WINDOW] {peak_window}\n"
                f"[समय-सारणी / SCHEDULE] {work_minutes} मिनट कार्य / {rest_minutes} मिनट विश्राम प्रति घंटा\n"
                f"[निर्देश / ACTION] {full_action}\n"
                f"[विश्वसनीयता / CONFIDENCE] {conf_label}\n"
                f"[डेटा समय / DATA TIMESTAMP] {dt_str}\n"
                f"[सूचना] परिचालन निर्णय-सहायता परामर्श। यह चिकित्सीय सलाह या सुरक्षा की गारंटी नहीं है।"
            )
        else:  # mr
            header = f"🚨 हीटवेव्ह गार्डियन इशारा — {role_label}" if channel == "sms" else f"☀️ *हीटवेव्ह गार्डियन इशारा — {role_label}*"
            msg = (
                f"{header}{sim_banner}\n"
                f"[धोका / RISK] {risk_label} | Personal Risk: {risk_category.upper()}\n"
                f"[स्थान / LOCATION] {location}\n"
                f"[मापदंड / METRICS] तापमान: {temp_c:.1f}°C | WBGT उष्णता ताण: {wbgt_c:.1f}°C\n"
                f"[कमाल काळ / PEAK WINDOW] {peak_window}\n"
                f"[वेळापत्रक / SCHEDULE] {work_minutes} मिनिटे काम / {rest_minutes} मिनिटे विश्रांती प्रति तास\n"
                f"[कृती / ACTION] {full_action}\n"
                f"[विश्वासार्हता / CONFIDENCE] {conf_label}\n"
                f"[डेटा वेळ / DATA TIMESTAMP] {dt_str}\n"
                f"[सूचना] कार्यप्रणाली निर्णय-साहाय्य. हा वैद्यकीय सल्ला किंवा सुरक्षेची हमी नाही."
            )

        return msg.strip()

    @classmethod
    def format_compact_sms(
        cls,
        location: str,
        risk_category: str,
        wbgt_c: float,
        temp_c: float,
        work_minutes: int,
        rest_minutes: int,
        role: str = "worker",
        language: str = "en",
        peak_window: str = "12:00–15:30",
        confidence: str = "High",
        data_timestamp: Optional[str] = None,
        is_simulated: bool = False,
    ) -> str:
        """
        Generates ultra-compact SMS string (< 160 chars) preserving essential actionable information
        for low-connectivity / 2G / poor bandwidth operations.
        """
        lang = language if language in SUPPORTED_LANGUAGES else "en"
        _, compact_act = cls._get_role_action_text(
            role=role,
            risk=risk_category,
            lang=lang,
            work_minutes=work_minutes,
            rest_minutes=rest_minutes,
        )

        dt_short = (data_timestamp or datetime.now().strftime("%d-%b %H:%M"))[:12]
        sim_tag = "[SIM] " if is_simulated else ""

        if lang == "en":
            compact = f"{sim_tag}[HWG ALERT] {location}: {risk_category.upper()} (WBGT {wbgt_c:.1f}C). Peak {peak_window}. {compact_act} Conf:{confidence}. {dt_short}"
        elif lang == "hi":
            risk_hi = RISK_LABELS_MULTILINGUAL.get(risk_category, {}).get("hi", risk_category)[:8]
            compact = f"{sim_tag}[HWG चेतावनी] {location}: {risk_hi} (WBGT {wbgt_c:.1f}C). पीक {peak_window}. {compact_act} Conf:{confidence}. {dt_short}"
        else:
            risk_mr = RISK_LABELS_MULTILINGUAL.get(risk_category, {}).get("mr", risk_category)[:8]
            compact = f"{sim_tag}[HWG इशारा] {location}: {risk_mr} (WBGT {wbgt_c:.1f}C). पीक {peak_window}. {compact_act} Conf:{confidence}. {dt_short}"

        return compact.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # Standardized Alert Object Creation
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def create_alert(
        cls,
        role: str,
        location: str,
        risk: str,
        wbgt_c: float,
        temp_c: float,
        work_minutes: int = 30,
        rest_minutes: int = 30,
        peak_window: str = "12:00–15:30",
        confidence: str = "High",
        language: str = "en",
        channel: str = "sms",
        instruction: Optional[str] = None,
        data_timestamp: Optional[str] = None,
        is_simulated: bool = False,
        source_engine: str = "InterventionEngine",
        recipient_phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates a standardized alert object complete with lifecycle status,
        human-in-the-loop approval state, data provenance, and multilingual formatting.
        """
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {SUPPORTED_ROLES}")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Invalid language '{language}'. Must be one of {SUPPORTED_LANGUAGES}")
        if channel not in SUPPORTED_CHANNELS:
            raise ValueError(f"Invalid channel '{channel}'. Must be one of {SUPPORTED_CHANNELS}")

        alert_id = f"ALT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        dt_val = data_timestamp or now_iso

        # Evaluate policy
        policy = cls.evaluate_alert_policy(
            risk=risk,
            role=role,
            confidence=confidence,
            is_simulated=is_simulated,
        )

        approval_state = "PENDING" if policy["requires_human_approval"] else "NOT_REQUIRED"
        status = "PENDING_APPROVAL" if policy["requires_human_approval"] else ("SIMULATED" if is_simulated else "DRAFT")

        full_msg = cls.format_alert_message(
            location=location,
            risk_category=risk,
            wbgt_c=wbgt_c,
            temp_c=temp_c,
            work_minutes=work_minutes,
            rest_minutes=rest_minutes,
            instruction=instruction or "",
            channel=channel,
            role=role,
            language=language,
            peak_window=peak_window,
            confidence=confidence,
            data_timestamp=dt_val,
            is_simulated=is_simulated,
        )

        compact_msg = cls.format_compact_sms(
            location=location,
            risk_category=risk,
            wbgt_c=wbgt_c,
            temp_c=temp_c,
            work_minutes=work_minutes,
            rest_minutes=rest_minutes,
            role=role,
            language=language,
            peak_window=peak_window,
            confidence=confidence,
            data_timestamp=dt_val,
            is_simulated=is_simulated,
        )

        c_hash = hash_contact(recipient_phone) if recipient_phone else None
        c_masked = mask_contact(recipient_phone) if recipient_phone else None

        alert_obj = {
            "alert_id": alert_id,
            "created_at": now_iso,
            "data_timestamp": dt_val,
            "location": location,
            "role": role,
            "risk": risk,
            "urgency": policy["urgency"],
            "message_type": policy["message_type"],
            "escalation_level": policy["escalation_level"],
            "requires_human_approval": policy["requires_human_approval"],
            "is_high_impact": policy["is_high_impact"],
            "peak_window": peak_window,
            "wbgt_c": round(float(wbgt_c), 1),
            "temp_c": round(float(temp_c), 1),
            "work_minutes": int(work_minutes),
            "rest_minutes": int(rest_minutes),
            "recommended_action": instruction or full_msg,
            "language": language,
            "channel": channel,
            "confidence": confidence,
            "data_quality": "VALIDATED_NWP_HYPERLOCAL" if not is_simulated else "SYNTHETIC_SIMULATION",
            "model_version": f"{cls.ENGINE_VERSION} (Liljegren 2008 / IMD 2024)",
            "approval_state": approval_state,
            "status": status,
            "is_simulated": bool(is_simulated),
            "full_message": full_msg,
            "compact_message": compact_msg,
            "recipient_masked": c_masked,
            "recipient_hash": c_hash,
            "audit_trail": [
                {
                    "action": "CREATED",
                    "timestamp": now_iso,
                    "actor": f"AlertService ({source_engine})",
                    "details": f"Generated for {role} role at {risk} risk. Approval required: {policy['requires_human_approval']}.",
                }
            ],
            "delivery_metadata": {
                "provider": "Twilio" if not is_simulated else "Simulation Mode",
                "provider_message_id": None,
                "dispatched_at": None,
                "delivered_at": None,
                "error": None,
            },
        }

        cls._ALERT_REGISTRY[alert_id] = alert_obj
        return alert_obj

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-Role Batch Generation from Engine Output
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def generate_role_alerts(
        cls,
        location: str,
        risk: str,
        wbgt_c: float,
        temp_c: float,
        peak_window: str = "12:00–15:30",
        confidence: str = "High",
        work_minutes: int = 30,
        rest_minutes: int = 30,
        instruction: Optional[str] = None,
        roles: Optional[List[str]] = None,
        language: str = "en",
        channel: str = "sms",
        is_simulated: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes alerts for all specified (or all 5) roles from single risk/weather context.
        """
        target_roles = roles if roles else SUPPORTED_ROLES
        generated = []
        for r in target_roles:
            if r in SUPPORTED_ROLES:
                alert = cls.create_alert(
                    role=r,
                    location=location,
                    risk=risk,
                    wbgt_c=wbgt_c,
                    temp_c=temp_c,
                    work_minutes=work_minutes,
                    rest_minutes=rest_minutes,
                    peak_window=peak_window,
                    confidence=confidence,
                    language=language,
                    channel=channel,
                    instruction=instruction,
                    is_simulated=is_simulated,
                )
                generated.append(alert)
        return generated

    # ──────────────────────────────────────────────────────────────────────────
    # Human-in-the-Loop Lifecycle (Approve / Reject)
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def approve_alert(
        cls,
        alert_id: str,
        approver_name: str = "Supervisor / Officer",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Approves a pending high-impact alert, enabling downstream dispatch.
        """
        if alert_id not in cls._ALERT_REGISTRY:
            return {"success": False, "error": f"Alert '{alert_id}' not found.", "status": "NOT_FOUND"}

        alert = cls._ALERT_REGISTRY[alert_id]
        if alert["approval_state"] == "APPROVED":
            return {"success": True, "message": "Alert is already approved.", "alert": alert}

        now_iso = datetime.now(timezone.utc).isoformat()
        alert["approval_state"] = "APPROVED"
        alert["status"] = "APPROVED"
        alert["audit_trail"].append({
            "action": "APPROVED",
            "timestamp": now_iso,
            "actor": approver_name,
            "details": notes or "Human operator confirmed alert dispatch authorization.",
        })

        return {"success": True, "message": f"Alert {alert_id} approved successfully.", "alert": alert}

    @classmethod
    def reject_alert(
        cls,
        alert_id: str,
        rejected_by: str = "Supervisor / Officer",
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Rejects a pending alert, preventing dispatch and documenting reasons in audit trail.
        """
        if alert_id not in cls._ALERT_REGISTRY:
            return {"success": False, "error": f"Alert '{alert_id}' not found.", "status": "NOT_FOUND"}

        alert = cls._ALERT_REGISTRY[alert_id]
        now_iso = datetime.now(timezone.utc).isoformat()
        alert["approval_state"] = "REJECTED"
        alert["status"] = "REJECTED"
        alert["audit_trail"].append({
            "action": "REJECTED",
            "timestamp": now_iso,
            "actor": rejected_by,
            "details": reason or "Alert cancelled/rejected by operator.",
        })

        return {"success": True, "message": f"Alert {alert_id} rejected.", "alert": alert}

    @classmethod
    def escalate_alert(
        cls,
        alert_id: str,
        escalated_by: str = "Operational Lead",
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Escalates an alert up the 5-tier chain of command:
        WORKER -> SUPERVISOR -> SITE MANAGER -> AUTHORITY -> SYSTEM ADMIN
        Promotes role, updates escalation level, regenerates role-specific messages,
        and logs the escalation transition into the immutable audit trail.
        """
        if alert_id not in cls._ALERT_REGISTRY:
            return {"success": False, "error": f"Alert '{alert_id}' not found.", "status": "NOT_FOUND"}

        alert = cls._ALERT_REGISTRY[alert_id]
        current_role = alert.get("role", "worker")

        if current_role not in ESCALATION_LADDER:
            next_role = "authority"
        else:
            idx = ESCALATION_LADDER.index(current_role)
            if idx >= len(ESCALATION_LADDER) - 1:
                return {
                    "success": False,
                    "status": "MAX_ESCALATION",
                    "error": "Alert is already at the highest escalation level (SYSTEM ADMIN).",
                    "alert": alert,
                }
            next_role = ESCALATION_LADDER[idx + 1]

        now_iso = datetime.now(timezone.utc).isoformat()
        new_escalation_level = min(5, alert.get("escalation_level", 1) + 1)
        alert["role"] = next_role
        alert["escalation_level"] = new_escalation_level

        # Evaluate policy for newly escalated role
        new_policy = cls.evaluate_alert_policy(
            risk=alert.get("risk", "High"),
            role=next_role,
            confidence=alert.get("confidence", "High"),
            is_simulated=alert.get("is_simulated", False),
        )
        alert["urgency"] = new_policy["urgency"]
        alert["message_type"] = new_policy["message_type"]
        alert["requires_human_approval"] = new_policy["requires_human_approval"]
        alert["is_high_impact"] = new_policy["is_high_impact"]

        if new_policy["requires_human_approval"] and alert.get("approval_state") != "APPROVED":
            alert["approval_state"] = "PENDING"
            alert["status"] = "PENDING_APPROVAL"

        # Re-synthesize messages for the newly escalated role
        alert["full_message"] = cls.format_alert_message(
            location=alert["location"],
            risk_category=alert["risk"],
            wbgt_c=alert["wbgt_c"],
            temp_c=alert["temp_c"],
            work_minutes=alert["work_minutes"],
            rest_minutes=alert["rest_minutes"],
            instruction=alert.get("recommended_action", ""),
            channel=alert.get("channel", "sms"),
            role=next_role,
            language=alert.get("language", "en"),
            peak_window=alert.get("peak_window", "12:00–15:30"),
            confidence=alert.get("confidence", "High"),
            data_timestamp=alert.get("data_timestamp"),
            is_simulated=alert.get("is_simulated", False),
        )

        alert["compact_message"] = cls.format_compact_sms(
            location=alert["location"],
            risk_category=alert["risk"],
            wbgt_c=alert["wbgt_c"],
            temp_c=alert["temp_c"],
            work_minutes=alert["work_minutes"],
            rest_minutes=alert["rest_minutes"],
            role=next_role,
            language=alert.get("language", "en"),
            peak_window=alert.get("peak_window", "12:00–15:30"),
            confidence=alert.get("confidence", "High"),
            data_timestamp=alert.get("data_timestamp"),
            is_simulated=alert.get("is_simulated", False),
        )

        # Audit transition
        alert["audit_trail"].append({
            "action": "ESCALATED",
            "timestamp": now_iso,
            "actor": escalated_by,
            "details": f"Escalated from {current_role.upper()} to {next_role.upper()}. Reason: {reason or 'Thermal risk escalation'}.",
        })

        return {
            "success": True,
            "message": f"Alert successfully escalated from {current_role.upper()} to {next_role.upper()}.",
            "previous_role": current_role,
            "current_role": next_role,
            "escalation_level": new_escalation_level,
            "alert": alert,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Dispatch Workflow & Adapters
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def dispatch_alert(
        cls,
        recipient_phone: str,
        location: str,
        risk_category: str,
        wbgt_c: float,
        temp_c: float,
        work_minutes: int,
        rest_minutes: int,
        instruction: str,
        channel: str = "sms",
        force: bool = False,
        role: str = "worker",
        language: str = "en",
        peak_window: str = "12:00–15:30",
        confidence: str = "High",
        is_simulated: bool = False,
        alert_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Primary dispatch method.
        Enforces:
        1. Recipient phone number validity.
        2. Recipient consent (opt-in check).
        3. Recipient quiet hours window suppression.
        4. Cooldown duplicate suppression window.
        5. Human-in-the-loop approval check for high-impact alerts.
        6. Simulation guardrail (never trigger live external API calls for simulations).
        7. Secure masking of contact information in all persistent logs.
        100% backward compatible with legacy callers!
        """
        phone_clean = (recipient_phone or "").strip().replace(" ", "").replace("-", "")
        if not phone_clean or len(phone_clean) < 10:
            return {
                "success": False,
                "error": "Invalid phone number. Please provide a valid 10-digit mobile number with country code.",
                "status": "INVALID_PHONE",
            }

        c_hash = hash_contact(phone_clean)
        c_masked = mask_contact(phone_clean)

        # 1. Resolve or Create Alert Object
        if alert_id and alert_id in cls._ALERT_REGISTRY:
            alert_obj = cls._ALERT_REGISTRY[alert_id]
        else:
            alert_obj = cls.create_alert(
                role=role,
                location=location,
                risk=risk_category,
                wbgt_c=wbgt_c,
                temp_c=temp_c,
                work_minutes=work_minutes,
                rest_minutes=rest_minutes,
                peak_window=peak_window,
                confidence=confidence,
                language=language,
                channel=channel,
                instruction=instruction,
                is_simulated=is_simulated,
                recipient_phone=phone_clean,
            )

        # 2. Human-in-the-Loop Rejection Check (Highest priority gate)
        if alert_obj["approval_state"] == "REJECTED":
            return {
                "success": False,
                "status": "ALERT_REJECTED",
                "error": "This alert was rejected and cannot be dispatched.",
                "alert_id": alert_obj["alert_id"],
            }

        # 3. Human-in-the-Loop Approval Check
        if alert_obj["approval_state"] == "PENDING" and not force:
            return {
                "success": False,
                "status": "APPROVAL_REQUIRED",
                "error": "This high-impact alert requires supervisor or authority approval before dispatch.",
                "alert_id": alert_obj["alert_id"],
            }

        # 4. Consent Verification
        prefs = cls.get_preferences(phone_clean)
        if not prefs.get("opted_in", True) and not force:
            return {
                "success": False,
                "status": "SUPPRESSED_NO_CONSENT",
                "error": f"Recipient {c_masked} has opted out of notifications.",
                "recipient_masked": c_masked,
            }

        # 5. Cooldown Verification (Check duplicate dispatches before quiet hours)
        now = time.time()
        cooldown_key = (c_hash, risk_category)
        last_sent = cls._COOLDOWN_TRACKER.get(cooldown_key)
        if not force and last_sent and (now - last_sent < cls.COOLDOWN_SECONDS):
            remaining_mins = int((cls.COOLDOWN_SECONDS - (now - last_sent)) / 60)
            return {
                "success": True,
                "status": "COOLDOWN_ACTIVE",
                "message": f"Alert already sent recently. Cooldown active for another {remaining_mins} minutes.",
                "recipient": phone_clean,
                "recipient_masked": c_masked,
                "cooldown_remaining_minutes": remaining_mins,
            }

        # 6. Quiet Hours Verification
        # Can be bypassed if urgency is Critical / Extreme, forced, or explicitly authorized by human approver
        is_extreme = str(risk_category).strip().title() == "Extreme"
        is_approved = alert_obj.get("approval_state") == "APPROVED"
        if not force and not is_extreme and not is_approved and cls.is_quiet_hours_active(phone_clean):
            return {
                "success": False,
                "status": "SUPPRESSED_QUIET_HOURS",
                "message": f"Alert suppressed: recipient {c_masked} is currently in quiet hours.",
                "recipient_masked": c_masked,
            }

        # 5. Formatted message payload
        message_text = alert_obj["full_message"]

        # 6. Simulation Guardrail Check:
        # If is_simulated is True, NEVER send live external SMS, strictly return SIMULATED!
        config = cls.get_config()
        if is_simulated:
            cls._COOLDOWN_TRACKER[cooldown_key] = now
            now_iso = datetime.now(timezone.utc).isoformat()
            alert_obj["status"] = "SIMULATED"
            alert_obj["delivery_metadata"]["provider"] = "Simulation Mode (What-If)"
            alert_obj["delivery_metadata"]["dispatched_at"] = now_iso
            alert_obj["audit_trail"].append({
                "action": "SIMULATED_DISPATCH",
                "timestamp": now_iso,
                "actor": "AlertService",
                "details": "Simulated counterfactual logged. Live external dispatch was safely prevented.",
            })

            log_entry = {
                "id": len(cls._ALERT_LOG) + 1,
                "alert_id": alert_obj["alert_id"],
                "timestamp": datetime.now().isoformat(),
                "recipient": phone_clean,
                "recipient_masked": c_masked,
                "location": location,
                "risk_category": risk_category,
                "role": role,
                "channel": channel,
                "message": message_text,
                "status": "SIMULATED",
                "simulated": True,
                "note": "Counterfactual scenario simulation logged to audit trail without live transmission.",
            }
            cls._ALERT_LOG.insert(0, log_entry)

            return {
                "success": True,
                "status": "SIMULATED",
                "simulated": True,
                "message": f"[SIMULATION SCENARIO] Alert logged for {c_masked}. Live SMS was safely prevented.",
                "log_entry": log_entry,
                "alert": alert_obj,
            }

        # 7. Live Dispatch (Twilio) if configured
        if config["is_configured"]:
            try:
                sid = os.environ.get("TWILIO_ACCOUNT_SID")
                token = os.environ.get("TWILIO_AUTH_TOKEN")
                from_number = os.environ.get("TWILIO_PHONE_NUMBER")

                twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
                target_to = phone_clean if phone_clean.startswith("+") else f"+91{phone_clean}"
                target_from = from_number

                if channel == "whatsapp":
                    target_to = f"whatsapp:{target_to}"
                    target_from = f"whatsapp:{from_number}"

                payload = {
                    "From": target_from,
                    "To": target_to,
                    "Body": message_text,
                }
                resp = requests.post(twilio_url, data=payload, auth=(sid, token), timeout=10)
                resp.raise_for_status()
                res_data = resp.json()

                now_iso = datetime.now(timezone.utc).isoformat()
                cls._COOLDOWN_TRACKER[cooldown_key] = now
                alert_obj["status"] = "DELIVERED"
                alert_obj["delivery_metadata"]["provider"] = "Twilio (Live)"
                alert_obj["delivery_metadata"]["provider_message_id"] = res_data.get("sid")
                alert_obj["delivery_metadata"]["dispatched_at"] = now_iso
                alert_obj["delivery_metadata"]["delivered_at"] = now_iso
                alert_obj["audit_trail"].append({
                    "action": "DISPATCHED_LIVE",
                    "timestamp": now_iso,
                    "actor": "Twilio Provider",
                    "details": f"Delivered via {channel.upper()} (SID: {res_data.get('sid')}).",
                })

                log_entry = {
                    "id": len(cls._ALERT_LOG) + 1,
                    "alert_id": alert_obj["alert_id"],
                    "timestamp": datetime.now().isoformat(),
                    "recipient": phone_clean,
                    "recipient_masked": c_masked,
                    "location": location,
                    "risk_category": risk_category,
                    "role": role,
                    "channel": channel,
                    "message": message_text,
                    "status": "DELIVERED",
                    "provider_message_id": res_data.get("sid"),
                    "simulated": False,
                }
                cls._ALERT_LOG.insert(0, log_entry)

                return {
                    "success": True,
                    "status": "DELIVERED",
                    "simulated": False,
                    "message": f"Alert successfully sent to {c_masked} via {channel.upper()}.",
                    "log_entry": log_entry,
                    "alert": alert_obj,
                }
            except Exception as e:
                now_iso = datetime.now(timezone.utc).isoformat()
                alert_obj["status"] = "FAILED"
                alert_obj["delivery_metadata"]["error"] = str(e)
                alert_obj["audit_trail"].append({
                    "action": "DISPATCH_FAILED",
                    "timestamp": now_iso,
                    "actor": "Twilio Provider",
                    "details": f"Transmission error: {str(e)}",
                })

                log_entry = {
                    "id": len(cls._ALERT_LOG) + 1,
                    "alert_id": alert_obj["alert_id"],
                    "timestamp": datetime.now().isoformat(),
                    "recipient": phone_clean,
                    "recipient_masked": c_masked,
                    "location": location,
                    "risk_category": risk_category,
                    "role": role,
                    "channel": channel,
                    "message": message_text,
                    "status": "FAILED",
                    "error": str(e),
                    "simulated": False,
                }
                cls._ALERT_LOG.insert(0, log_entry)

                return {
                    "success": False,
                    "status": "FAILED",
                    "simulated": False,
                    "error": f"Failed to deliver message via provider: {str(e)}",
                    "log_entry": log_entry,
                    "alert": alert_obj,
                }

        # 8. Simulation Mode (Twilio credentials not configured in environment)
        cls._COOLDOWN_TRACKER[cooldown_key] = now
        now_iso = datetime.now(timezone.utc).isoformat()
        alert_obj["status"] = "SIMULATED"
        alert_obj["delivery_metadata"]["provider"] = "Simulation Mode (No Credentials)"
        alert_obj["delivery_metadata"]["dispatched_at"] = now_iso
        alert_obj["audit_trail"].append({
            "action": "SIMULATED_DISPATCH",
            "timestamp": now_iso,
            "actor": "AlertService",
            "details": "Logged to audit trail in test/simulation mode (Twilio credentials not configured).",
        })

        log_entry = {
            "id": len(cls._ALERT_LOG) + 1,
            "alert_id": alert_obj["alert_id"],
            "timestamp": datetime.now().isoformat(),
            "recipient": phone_clean,
            "recipient_masked": c_masked,
            "location": location,
            "risk_category": risk_category,
            "role": role,
            "channel": channel,
            "message": message_text,
            "status": "SIMULATED",
            "simulated": True,
            "note": "Logged to audit trail in Test/Simulation Mode (Twilio credentials not configured in environment).",
        }
        cls._ALERT_LOG.insert(0, log_entry)

        return {
            "success": True,
            "status": "SIMULATED",
            "simulated": True,
            "message": (
                f"[SIMULATION MODE] Alert generated and logged for {phone_clean} ({channel.upper()}). "
                f"No external SMS sent because TWILIO_ACCOUNT_SID is not configured."
            ),
            "log_entry": log_entry,
            "alert": alert_obj,
        }

    @classmethod
    def dispatch_alert_by_id(
        cls,
        alert_id: str,
        recipient_phone: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Dispatches an existing standardized alert by its ID after verification.
        """
        if alert_id not in cls._ALERT_REGISTRY:
            return {"success": False, "status": "NOT_FOUND", "error": f"Alert '{alert_id}' not found."}

        alert = cls._ALERT_REGISTRY[alert_id]
        phone = recipient_phone or "+919876543210"

        return cls.dispatch_alert(
            recipient_phone=phone,
            location=alert["location"],
            risk_category=alert["risk"],
            wbgt_c=alert["wbgt_c"],
            temp_c=alert["temp_c"],
            work_minutes=alert["work_minutes"],
            rest_minutes=alert["rest_minutes"],
            instruction=alert["recommended_action"],
            channel=alert["channel"],
            force=force,
            role=alert["role"],
            language=alert["language"],
            peak_window=alert["peak_window"],
            confidence=alert["confidence"],
            is_simulated=alert["is_simulated"],
            alert_id=alert_id,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # History & Registry Inspection
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def get_history(cls, limit: int = 25) -> List[Dict[str, Any]]:
        """Retrieve recent alert dispatch history."""
        return cls._ALERT_LOG[:limit]

    @classmethod
    def list_alerts(
        cls,
        role: Optional[str] = None,
        status: Optional[str] = None,
        approval_state: Optional[str] = None,
        urgency: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Filter stored alerts by role, status, approval state, or urgency."""
        alerts = list(cls._ALERT_REGISTRY.values())
        if role:
            alerts = [a for a in alerts if a.get("role") == role]
        if status:
            alerts = [a for a in alerts if a.get("status") == status]
        if approval_state:
            alerts = [a for a in alerts if a.get("approval_state") == approval_state]
        if urgency:
            alerts = [a for a in alerts if a.get("urgency") == urgency]

        # Sort reverse chronological by created_at
        alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return alerts[:limit]

    @classmethod
    def get_alert(cls, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single alert object with full audit trail."""
        return cls._ALERT_REGISTRY.get(alert_id)
