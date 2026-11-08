"""
HeatWave Guardian — Experimental Mortality Risk Index (Proxy Indicator)
=======================================================================
Provides an experimental proxy heat-mortality vulnerability index for public
health surveillance and municipal prioritization.

CRITICAL DATA HONESTY & MEDICAL DISCLAIMER:
- This is strictly an EXPERIMENTAL PROXY INDICATOR.
- It is NOT a clinically validated mortality prediction or medical diagnosis.
- It does NOT estimate individual death counts.
- It calculates a relative 0.0 to 10.0 population vulnerability proxy based on
  combined environmental heat stress (Liljegren WBGT), IMD thermal departures,
  and demographic exposure variables.
"""

from typing import Dict, Any, List
from services.demographic_service import DemographicService


class MortalityRiskService:
    DISCLAIMER = (
        "EXPERIMENTAL DECISION-SUPPORT PROXY: This indicator is an experimental population "
        "heat-stress index computed from environmental WBGT, IMD departures, and demographic "
        "exposure proxies. It is NOT a clinically or epidemiologically validated mortality prediction, "
        "and must never be used as a substitute for professional medical assessment or official "
        "vital statistics reporting."
    )

    @classmethod
    def calculate_proxy_index(
        cls,
        wbgt_c: float,
        imd_status: str,
        temp_c: float,
        location: str,
        admin1: str = "",
        consecutive_heat_days: int = 1
    ) -> Dict[str, Any]:
        """
        Compute experimental proxy score on a 0.0 - 10.0 scale.
        Factors:
        1. WBGT Heat Stress Severity (0 - 4.5 pts)
        2. IMD Regional Heatwave Departure (0 - 2.5 pts)
        3. Heat Duration / Cumulative Exposure (0 - 1.5 pts)
        4. Demographic Vulnerability Proxy (0 - 1.5 pts)
        """
        # 1. WBGT factor (Safe < 27: 0.5, Caution: 1.5, Extreme Caution: 2.5, Danger: 3.5, Extreme Danger > 32.1: 4.5)
        if wbgt_c < 27.0:
            wbgt_pts = 0.5
            wbgt_band = "Safe"
        elif wbgt_c <= 29.3:
            wbgt_pts = 1.5
            wbgt_band = "Caution"
        elif wbgt_c <= 31.0:
            wbgt_pts = 2.5
            wbgt_band = "Extreme Caution"
        elif wbgt_c <= 32.1:
            wbgt_pts = 3.5
            wbgt_band = "Danger"
        else:
            wbgt_pts = 4.5
            wbgt_band = "Extreme Danger"

        # 2. IMD Departure factor
        imd_weights = {
            "Normal": 0.2,
            "Heat Day": 1.0,
            "Heatwave": 1.8,
            "Severe Heatwave": 2.5,
        }
        imd_pts = imd_weights.get(imd_status, 0.2)

        # 3. Consecutive extreme days (heat duration compounding strain)
        days = max(consecutive_heat_days, 1)
        duration_pts = min((days - 1) * 0.5, 1.5)

        # 4. Demographic exposure proxy factor
        demo = DemographicService.get_demographics(location, admin1)
        demo_vuln = DemographicService.get_vulnerability_score(demo)  # 0.0 - 1.0
        demo_pts = round(demo_vuln * 1.5, 2)

        raw_score = wbgt_pts + imd_pts + duration_pts + demo_pts
        score = min(round(raw_score, 1), 10.0)

        # Categorize proxy level
        if score < 3.0:
            level = "Baseline / Minimal"
            color = "#2f9e44"
            implication = "Thermal conditions within normal compensatory ranges for general population."
        elif score < 5.5:
            level = "Elevated Risk"
            color = "#f2a900"
            implication = "Mildly elevated cardiovascular strain; vulnerable elderly and unacclimatized workers require monitoring."
        elif score < 7.5:
            level = "High Risk"
            color = "#e8590c"
            implication = "Severe environmental strain; significant risk of heat exhaustion and emergency room visits."
        else:
            level = "Critical Population Stress"
            color = "#6a040f"
            implication = "Life-threatening acute heat emergency; severe physiological strain across elderly and outdoor workforce."

        contributing_factors = [
            {
                "factor": "Outdoor WBGT Heat Stress",
                "points": round(wbgt_pts, 1),
                "max_points": 4.5,
                "detail": f"{wbgt_c:.1f}°C ({wbgt_band} band)",
            },
            {
                "factor": "IMD Regional Departure",
                "points": round(imd_pts, 1),
                "max_points": 2.5,
                "detail": f"{imd_status} (Air temp: {temp_c:.1f}°C)",
            },
            {
                "factor": "Cumulative Heat Duration",
                "points": round(duration_pts, 1),
                "max_points": 1.5,
                "detail": f"{days} consecutive heat day(s)",
            },
            {
                "factor": "Demographic Exposure Proxy",
                "points": round(demo_pts, 1),
                "max_points": 1.5,
                "detail": f"Elderly: {demo['elderly_pct']}%, Outdoor labor: {demo['outdoor_worker_pct']}%",
                "is_demo_data": demo.get("is_demo_data", True),
            }
        ]

        return {
            "title": "Experimental Mortality Risk Indicator (Proxy Index)",
            "proxy_score": score,
            "max_score": 10.0,
            "severity_level": level,
            "color": color,
            "implication": implication,
            "location": location,
            "contributing_factors": contributing_factors,
            "demographic_context": demo,
            "disclaimer": cls.DISCLAIMER,
            "is_experimental": True,
            "is_medical_prediction": False,
        }
