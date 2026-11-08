"""
HeatWave Guardian — Government Heat Action Plan (HAP) Integration Framework
===========================================================================
Translates environmental heat stress (Liljegren WBGT) and IMD regional departures
into operational government and municipal Heat Action Plan protocols.

Modelled after National Disaster Management Authority (NDMA) national heatwave guidelines
and Ahmedabad Municipal Corporation Heat Action Plan (HAP).
"""

from typing import Dict, Any, List


class HAPService:
    FRAMEWORK_TITLE = "Government Heat Action Plan (HAP) Integration Framework"
    FRAMEWORK_AUTHORITY = "Modelled on NDMA Guidelines & Ahmedabad HAP Architecture"

    # Standardized 3-tier HAP alert matrix based on risk level and IMD status
    @classmethod
    def evaluate_hap_status(cls, risk_category: str, imd_status: str, wbgt_c: float) -> Dict[str, Any]:
        """
        Evaluate Government Heat Action Plan alert level and actionable protocols.
        Risk Categories: Low, Moderate, High, Very High, Extreme.
        IMD Statuses: Normal, Heat Day, Heatwave, Severe Heatwave.
        """
        if risk_category in ("Very High", "Extreme") or imd_status == "Severe Heatwave":
            color_code = "RED"
            alert_name = "Red Alert — Severe Heatwave Emergency"
            description = (
                "Extreme thermal stress condition posing acute life-threatening danger to "
                "outdoor workers, vulnerable citizens, and elderly residents."
            )
            public_actions = [
                "Suspend all non-essential outdoor manual labor between 11:00 AM and 4:30 PM.",
                "Maintain continuous access to potable water; drink small amounts regularly throughout the shift and use ORS/electrolytes for heavy sweating.",
                "Stay indoors or in actively cooled public shelters.",
                "Immediately transfer individuals exhibiting heat stroke symptoms (confusion, cessation of sweating) to hospital emergency wards.",
            ]
            institutional_actions = [
                "Activate designated 'Cooling Shelters' in bus terminals, municipal buildings, and religious centers.",
                "Deploy municipal water tankers to high-density informal settlements and construction corridors.",
                "Enforce mandatory labor rest enforcement protocols under Factories & Labour Acts.",
                "Equip all Primary Health Centres (PHCs) and district hospitals with dedicated ice packs, IV fluids, and heat wards.",
                "Suspend power-grid maintenance shutdowns during peak afternoon hours to ensure uninterrupted water supply and fans.",
            ]
            departments = ["Health", "Labour", "Municipal Corporation", "Disaster Management (SDMA/DDMA)", "Transport"]

        elif risk_category in ("High",) or imd_status == "Heatwave":
            color_code = "ORANGE"
            alert_name = "Orange Alert — Heatwave Warning"
            description = (
                "Severe heat conditions with heightened risk of heat exhaustion, dehydration, "
                "and heat-related illness for outdoor workers."
            )
            public_actions = [
                "Limit intense physical work in direct sunlight; enforce 30-min work / 30-min rest rotations.",
                "Ensure continuous hydration with electrolytes, lemon water, and buttermilk.",
                "Wear light, loose-fitting cotton clothing and wide-brim head coverings.",
                "Keep domestic animals and cattle in shaded enclosures with adequate water troughs.",
            ]
            institutional_actions = [
                "Issue public warning broadcasts via local radio, social media, and SMS channels.",
                "Open municipal drinking water kiosks (Piaos) along major pedestrian transit routes.",
                "Mandate shade canopies at construction sites and open-air marketplaces.",
                "Put emergency medical services and ambulance fleets on high-heat standby.",
            ]
            departments = ["Municipal Corporation", "Labour Department", "Public Health", "Traffic Police"]

        elif risk_category == "Moderate" or imd_status == "Heat Day":
            color_code = "YELLOW"
            alert_name = "Yellow Alert — Heat Advisory / Watch"
            description = (
                "Elevated temperatures requiring caution and preventive awareness. "
                "High-risk individuals and unacclimatized workers should monitor symptoms."
            )
            public_actions = [
                "Maintain regular water intake during outdoor tasks.",
                "Take 15-minute shaded rest breaks every hour during strenuous tasks.",
                "Avoid prolonged midday exposure if unaccustomed to severe heat.",
            ]
            institutional_actions = [
                "Distribute heat advisory flyers at worker pickup points (nakas) and transit stations.",
                "Verify water supply readiness across municipal wards.",
                "Ensure school sports and outdoor assemblies are scheduled before 10:00 AM.",
            ]
            departments = ["Education Department", "Municipal Water Supply", "Public Information"]

        else:
            color_code = "GREEN"
            alert_name = "Normal — No Active Heatwave Alert"
            description = "Current environmental thermal stress is within manageable seasonal thresholds."
            public_actions = [
                "Normal outdoor activities with standard hydration habits.",
            ]
            institutional_actions = [
                "Standard seasonal surveillance and monitoring.",
            ]
            departments = ["State Disaster Management Authority"]

        return {
            "framework": cls.FRAMEWORK_TITLE,
            "authority_source": cls.FRAMEWORK_AUTHORITY,
            "alert_level": color_code,
            "alert_title": alert_name,
            "description": description,
            "trigger_factors": {
                "risk_category": risk_category,
                "imd_status": imd_status,
                "wbgt_c": wbgt_c,
            },
            "public_actions": public_actions,
            "institutional_actions": institutional_actions,
            "coordinating_departments": departments,
            "is_framework_standard": True,
        }
