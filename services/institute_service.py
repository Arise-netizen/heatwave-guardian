"""
HeatWave Guardian — Institute & Workplace Heat-Risk Monitoring Service
======================================================================
Provides multi-site occupational heat-stress monitoring for construction projects,
manufacturing plants, logistics facilities, and educational campuses.

PIPELINE INTEGRITY RULE:
All site risk assessments directly invoke the existing core HeatWave Guardian engine:
estimate_wbgt() -> classify_imd() -> fuse_risk() -> assess_vulnerability() -> escalate_risk() -> get_work_rest().
"""

from typing import Dict, Any, List
from engine import (
    estimate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    calculate_thermal_stress_index,
    classify_imd,
    fuse_risk,
    assess_vulnerability,
    escalate_risk,
    get_work_rest,
    generate_summary,
    RISK_COLORS,
    CITIES,
)

class InstituteService:
    # Supported preconfigured industrial & educational institutions
    INSTITUTES = {
        "infra_delhi": {
            "id": "infra_delhi",
            "name": "Delhi Metro Rail Expansion (Phase IV Corridors)",
            "city": "Delhi",
            "region": "plains",
            "base_lat": 28.6139,
            "base_lon": 77.2090,
            "type": "Heavy Civil Construction",
            "supervisor": "Chief Safety Officer (Northern Zone)",
            "sites": [
                {
                    "site_id": "DEL-S1",
                    "site_name": "Elevated Viaduct Casting Yard",
                    "zone_type": "Outdoor Open Slab (High Solar Exposure)",
                    "lat": 28.6500,
                    "lon": 77.2200,
                    "work_intensity": "heavy",
                    "clothing": "heavy",
                    "age_group": "18–40",
                    "experience": "very_used",
                    "worker_count": 140,
                },
                {
                    "site_id": "DEL-S2",
                    "site_name": "Steel Rebar Bending & Welding Yard",
                    "zone_type": "Heavy Fabrication & Metal Work (PPE Required)",
                    "lat": 28.6300,
                    "lon": 77.2500,
                    "work_intensity": "very_heavy",
                    "clothing": "ppe",
                    "age_group": "41–60",
                    "experience": "very_used",
                    "worker_count": 85,
                },
                {
                    "site_id": "DEL-S3",
                    "site_name": "Track Bed Laying & Ballast Tamping",
                    "zone_type": "Unshaded Open Alignment",
                    "lat": 28.5900,
                    "lon": 77.1900,
                    "work_intensity": "heavy",
                    "clothing": "heavy",
                    "age_group": "18–40",
                    "experience": "somewhat_used",
                    "worker_count": 110,
                },
                {
                    "site_id": "DEL-S4",
                    "site_name": "Underground Ventilation Shaft Excavation",
                    "zone_type": "Deep Pit / High Humidity & Confined Airflow",
                    "lat": 28.6200,
                    "lon": 77.2100,
                    "work_intensity": "heavy",
                    "clothing": "ppe",
                    "age_group": "18–40",
                    "experience": "not_used",
                    "worker_count": 65,
                },
                {
                    "site_id": "DEL-S5",
                    "site_name": "Central Material Storage & Heavy Rigging",
                    "zone_type": "Partially Shaded Logistics Depot",
                    "lat": 28.6400,
                    "lon": 77.1800,
                    "work_intensity": "moderate",
                    "clothing": "light",
                    "age_group": "18–40",
                    "experience": "very_used",
                    "worker_count": 45,
                },
            ]
        },
        "logistics_hyd": {
            "id": "logistics_hyd",
            "name": "Telangana Logistics Park & Cargo Gateway",
            "city": "Hyderabad",
            "region": "plains",
            "base_lat": 17.3850,
            "base_lon": 78.4867,
            "type": "Warehousing & Cold Chain Logistics",
            "supervisor": "Facility Operations Director",
            "sites": [
                {
                    "site_id": "HYD-S1",
                    "site_name": "Open Apron Container Loading Bay",
                    "zone_type": "Direct Concrete Apron & Tarmac Heat",
                    "lat": 17.3900,
                    "lon": 78.4900,
                    "work_intensity": "very_heavy",
                    "clothing": "heavy",
                    "age_group": "18–40",
                    "experience": "very_used",
                    "worker_count": 120,
                },
                {
                    "site_id": "HYD-S2",
                    "site_name": "Dry Goods Sorting & Pallet Stacking",
                    "zone_type": "Corrugated Metal Warehouse Interior",
                    "lat": 17.3800,
                    "lon": 78.4800,
                    "work_intensity": "moderate",
                    "clothing": "light",
                    "age_group": "18–40",
                    "experience": "somewhat_used",
                    "worker_count": 95,
                },
                {
                    "site_id": "HYD-S3",
                    "site_name": "Fleet Maintenance & Truck Inspection Bays",
                    "zone_type": "Semi-enclosed Mechanical Bay",
                    "lat": 17.4000,
                    "lon": 78.4700,
                    "work_intensity": "heavy",
                    "clothing": "heavy",
                    "age_group": "41–60",
                    "experience": "very_used",
                    "worker_count": 50,
                },
                {
                    "site_id": "HYD-S4",
                    "site_name": "Refrigerated Transfer Dock Shunting",
                    "zone_type": "Rapid Cold-to-Extreme Heat Transition",
                    "lat": 17.3700,
                    "lon": 78.4950,
                    "work_intensity": "moderate",
                    "clothing": "heavy",
                    "age_group": "18–40",
                    "experience": "not_used",
                    "worker_count": 40,
                },
            ]
        },
        "manufacturing_pune": {
            "id": "manufacturing_pune",
            "name": "Maharashtra Industrial Automotive Foundry & Assembly",
            "city": "Pune",
            "region": "plains",
            "base_lat": 18.5204,
            "base_lon": 73.8567,
            "type": "Foundry & Heavy Engineering",
            "supervisor": "Occupational Health & Safety Manager",
            "sites": [
                {
                    "site_id": "PUN-S1",
                    "site_name": "Furnace & Molten Metal Casting Floor",
                    "zone_type": "Extreme Radiant Heat + Full Molten PPE",
                    "lat": 18.5250,
                    "lon": 73.8600,
                    "work_intensity": "very_heavy",
                    "clothing": "ppe",
                    "age_group": "41–60",
                    "experience": "very_used",
                    "worker_count": 60,
                },
                {
                    "site_id": "PUN-S2",
                    "site_name": "Automotive Chassis Welding Line",
                    "zone_type": "Indoor High-Density Welding (Flame Retardant Gear)",
                    "lat": 18.5200,
                    "lon": 73.8500,
                    "work_intensity": "heavy",
                    "clothing": "ppe",
                    "age_group": "18–40",
                    "experience": "very_used",
                    "worker_count": 130,
                },
                {
                    "site_id": "PUN-S3",
                    "site_name": "Finished Vehicle Open Staging Lot",
                    "zone_type": "Unshaded Asphalt Staging",
                    "lat": 18.5150,
                    "lon": 73.8700,
                    "work_intensity": "moderate",
                    "clothing": "light",
                    "age_group": "18–40",
                    "experience": "somewhat_used",
                    "worker_count": 35,
                },
            ]
        },
        "campus_chennai": {
            "id": "campus_chennai",
            "name": "Southern Technological University Campus",
            "city": "Chennai",
            "region": "coastal",
            "base_lat": 13.0827,
            "base_lon": 80.2707,
            "type": "University Campus & Botanical Grounds",
            "supervisor": "Estate & Campus Facilities Dean",
            "sites": [
                {
                    "site_id": "CHE-S1",
                    "site_name": "Campus Grounds & Horticulture Crew",
                    "zone_type": "Outdoor Open Landscaping & Lawns",
                    "lat": 13.0850,
                    "lon": 80.2750,
                    "work_intensity": "heavy",
                    "clothing": "light",
                    "age_group": "41–60",
                    "experience": "very_used",
                    "worker_count": 45,
                },
                {
                    "site_id": "CHE-S2",
                    "site_name": "Sports Complex & Track Construction",
                    "zone_type": "Active Outdoor Civil Works",
                    "lat": 13.0800,
                    "lon": 80.2650,
                    "work_intensity": "heavy",
                    "clothing": "heavy",
                    "age_group": "18–40",
                    "experience": "not_used",
                    "worker_count": 55,
                },
                {
                    "site_id": "CHE-S3",
                    "site_name": "Student Outdoor Walkways & Shuttles",
                    "zone_type": "Pedestrian Transit Paths",
                    "lat": 13.0820,
                    "lon": 80.2710,
                    "work_intensity": "light",
                    "clothing": "light",
                    "age_group": "Under 18",
                    "experience": "somewhat_used",
                    "worker_count": 800,
                },
            ]
        }
    }

    @classmethod
    def get_institute_list(cls) -> List[Dict[str, Any]]:
        """Return list of available institutions for selector dropdown."""
        return [
            {
                "id": inst["id"],
                "name": inst["name"],
                "city": inst["city"],
                "type": inst["type"],
                "site_count": len(inst["sites"]),
            }
            for inst in cls.INSTITUTES.values()
        ]

    @classmethod
    def evaluate_institute(
        cls,
        institute_id: str,
        weather_provider_func
    ) -> Dict[str, Any]:
        """
        Evaluate real-time heat risk for all sites in an institution.
        Uses weather_provider_func(lat, lon) to obtain current weather,
        and applies the EXACT HeatWave Guardian core engine pipeline.
        """
        inst = cls.INSTITUTES.get(institute_id)
        if not inst:
            # Fallback to first
            inst = list(cls.INSTITUTES.values())[0]

        city = inst["city"]
        region = inst.get("region", "plains")

        # Fetch representative weather for base coordinates
        weather = weather_provider_func(inst["base_lat"], inst["base_lon"])
        if not weather or not isinstance(weather, dict) or weather.get("temp_c") is None:
            from services.weather_service import WeatherServiceUnavailableError
            raise WeatherServiceUnavailableError(
                f"Weather observations unavailable for institution '{inst.get('name')}'"
            )

        temp_c = weather["temp_c"]
        rh_pct = weather["rh_pct"]
        wind_ms = weather["wind_ms"]
        solar_wm2 = weather["solar_wm2"]

        site_evaluations = []
        counts = {
            "total": len(inst["sites"]),
            "safe": 0,
            "moderate": 0,
            "high": 0,
            "extreme": 0,
            "total_workers": 0,
        }

        for site in inst["sites"]:
            counts["total_workers"] += site.get("worker_count", 0)

            # Core Pipeline execution for this site:
            # 1. Authoritative Thermal Stress Index (Single source calculation)
            thermal_stress = calculate_thermal_stress_index(temp_c, rh_pct, wind_ms, solar_wm2)
            wbgt_c = thermal_stress["wbgt_c"]
            wbgt_band = thermal_stress["wbgt_band"]
            thermal_stress_level = thermal_stress["thermal_stress_level"]
            thermal_stress_desc = thermal_stress["thermal_stress_description"]

            # 2. IMD regional status
            imd_status = classify_imd(temp_c, region)

            # 3. Environmental Base Risk (Fusion)
            base_risk = fuse_risk(wbgt_band, imd_status)

            # 4. Site-specific vulnerability assessment
            vuln = assess_vulnerability(
                age_group=site["age_group"],
                acclimatization=site["experience"],
                workload=site["work_intensity"],
                clothing=site["clothing"],
            )

            # 5. Escalation to Personal/Workforce Risk
            final_risk = escalate_risk(base_risk, vuln)

            # 6. Work/rest & operational exposure guidance
            work_rest = get_work_rest(
                final_risk,
                workload=site["work_intensity"],
                acclimatized=site["experience"],
                clothing=site["clothing"],
                wbgt_c=wbgt_c,
            )

            # Summary explanation
            summary = generate_summary(
                location=site["site_name"],
                temp_c=temp_c,
                wbgt_c=wbgt_c,
                wbgt_band=wbgt_band,
                imd_status=imd_status,
                base_risk=base_risk,
                final_risk=final_risk,
                vulnerability=vuln,
                age_group=site["age_group"],
                experience=site["experience"],
                workload=site["work_intensity"],
                clothing=site["clothing"],
            )

            # Update counters
            if final_risk in ("Low",):
                counts["safe"] += 1
            elif final_risk in ("Moderate",):
                counts["moderate"] += 1
            elif final_risk in ("High",):
                counts["high"] += 1
            else:  # Very High or Extreme
                counts["extreme"] += 1

            site_evaluations.append({
                "site_id": site["site_id"],
                "site_name": site["site_name"],
                "zone_type": site["zone_type"],
                "worker_count": site.get("worker_count", 0),
                "work_intensity": site["work_intensity"],
                "clothing": site["clothing"],
                "age_group": site["age_group"],
                "experience": site["experience"],
                "temp_c": round(temp_c, 1),
                "wbgt_c": wbgt_c,
                "wbgt_band": wbgt_band,
                "thermal_stress": thermal_stress,
                "thermal_stress_index": wbgt_c,
                "thermal_stress_level": thermal_stress_level,
                "thermal_stress_description": thermal_stress_desc,
                "imd_status": imd_status,
                "environmental_risk": base_risk,
                "base_risk": base_risk,
                "personal_risk": final_risk,
                "risk_category": final_risk,
                "vulnerability": vuln,
                "work_minutes": work_rest["work_minutes"],
                "rest_minutes": work_rest["rest_minutes"],
                "instruction": work_rest["instruction"],
                "recommendation_level": work_rest["recommendation_level"],
                "work_guidance": work_rest["work_guidance"],
                "recovery_guidance": work_rest["recovery_guidance"],
                "recommended_controls": work_rest["recommended_controls"],
                "summary": summary,
                "risk_color": RISK_COLORS.get(final_risk, "#2f9e44"),
            })

        # Overall institute status
        if counts["extreme"] > 0:
            overall_status = "CRITICAL: Immediate Labor Work-Rest Rotations Enforced"
            overall_color = "#6a040f"
        elif counts["high"] > 0:
            overall_status = "HIGH ALERT: Active Heat-Stress Mitigation Required"
            overall_color = "#e8590c"
        elif counts["moderate"] > 0:
            overall_status = "MODERATE: Periodic Shaded Hydration Breaks Mandated"
            overall_color = "#f2a900"
        else:
            overall_status = "NORMAL: Standard Operational Protocol"
            overall_color = "#2f9e44"

        return {
            "institute": inst,
            "counts": counts,
            "overall_status": overall_status,
            "overall_color": overall_color,
            "weather": {
                "temp_c": round(temp_c, 1),
                "rh_pct": round(rh_pct, 1),
                "wind_ms": round(wind_ms, 1),
                "solar_wm2": round(solar_wm2, 1),
            },
            "sites": site_evaluations,
        }
