"""
HeatWave Guardian — Cooling-Station Siting Recommender
======================================================
Decision-support service for municipal planners and disaster management authorities.
Identifies high-need urban/semi-urban zones for emergency hydration kiosks,
shaded labor rest hubs, and cooling shelters.

TRANSPARENCY NOTE:
This module provides explainable decision-support recommendations based on
environmental heat intensity (WBGT), population density proxies, and proximity
to known cooling infrastructure. It is an advisory planning aid, not an
authoritative executive directive.
"""

from typing import Dict, Any, List
import math


class CoolingRecommenderService:
    DISCLAIMER = (
        "DECISION-SUPPORT ADVISORY: Siting priorities are generated via multi-criteria "
        "spatial-risk evaluation (WBGT thermal load + demographic exposure + facility distance). "
        "Intended for municipal disaster planning and NGO resource deployment."
    )

    # Candidate zone repository across representative Indian urban corridors
    # (coordinates, population density proxy, nearest known facility, baseline facility count)
    _CANDIDATE_ZONES = [
        {
            "zone_id": "DEL-01",
            "zone_name": "Anand Vihar Inter-State Transit Corridor",
            "city": "Delhi",
            "latitude": 28.6469,
            "longitude": 77.3160,
            "zone_type": "Transit Hub & Unorganized Labor Corridor",
            "nearest_facility": "Swami Dayanand Hospital",
            "distance_km": 3.8,
            "informal_worker_density": "Very High",
            "tree_canopy_pct": 8.0,
        },
        {
            "zone_id": "DEL-02",
            "zone_name": "Okhla Industrial Area Phase-II",
            "city": "Delhi",
            "latitude": 28.5300,
            "longitude": 77.2700,
            "zone_type": "Heavy Industrial & Manufacturing Zone",
            "nearest_facility": "ESI Hospital Okhla",
            "distance_km": 2.1,
            "informal_worker_density": "High",
            "tree_canopy_pct": 6.5,
        },
        {
            "zone_id": "HYD-01",
            "zone_name": "Afzal Gunj & Old City Market Belt",
            "city": "Hyderabad",
            "latitude": 17.3753,
            "longitude": 78.4795,
            "zone_type": "Dense Wholesale Market & Street Vendor Hub",
            "nearest_facility": "Osmania General Hospital",
            "distance_km": 1.4,
            "informal_worker_density": "Extreme",
            "tree_canopy_pct": 5.0,
        },
        {
            "zone_id": "HYD-02",
            "zone_name": "Gachibowli Construction Corridors",
            "city": "Hyderabad",
            "latitude": 17.4401,
            "longitude": 78.3489,
            "zone_type": "High-Rise Commercial Construction Site",
            "nearest_facility": "Kondapur Area Hospital",
            "distance_km": 4.5,
            "informal_worker_density": "Very High",
            "tree_canopy_pct": 12.0,
        },
        {
            "zone_id": "BOM-01",
            "zone_name": "Dharavi Transit & Artisanal Cluster",
            "city": "Mumbai",
            "latitude": 19.0400,
            "longitude": 72.8550,
            "zone_type": "High-Density Informal Industrial Ward",
            "nearest_facility": "Sion Municipal General Hospital",
            "distance_km": 1.8,
            "informal_worker_density": "Extreme",
            "tree_canopy_pct": 4.0,
        },
        {
            "zone_id": "PUN-01",
            "zone_name": "Hadapsar Industrial & Agricultural Market",
            "city": "Pune",
            "latitude": 18.5089,
            "longitude": 73.9260,
            "zone_type": "APMC Mandi & Heavy Logistics Hub",
            "nearest_facility": "Sassoon General Hospital",
            "distance_km": 6.2,
            "informal_worker_density": "High",
            "tree_canopy_pct": 9.0,
        },
        {
            "zone_id": "AMD-01",
            "zone_name": "Naroda Industrial Estate & Brick Kilns",
            "city": "Ahmedabad",
            "latitude": 23.0700,
            "longitude": 72.6500,
            "zone_type": "Chemical & Brick Kiln Labor Zone",
            "nearest_facility": "Shardaben Municipal Hospital",
            "distance_km": 5.1,
            "informal_worker_density": "Very High",
            "tree_canopy_pct": 4.5,
        },
        {
            "zone_id": "KHM-01",
            "zone_name": "Khammam Grain Market Yard & Quarry Belt",
            "city": "Khammam",
            "latitude": 17.2500,
            "longitude": 80.1500,
            "zone_type": "Open Grain Mandi & Granite Quarry",
            "nearest_facility": "Khammam District Headquarters Hospital",
            "distance_km": 4.0,
            "informal_worker_density": "Very High",
            "tree_canopy_pct": 7.0,
        }
    ]

    @classmethod
    def get_recommendations(
        cls,
        target_city: str = None,
        live_wbgt_map: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate candidate zones and generate a ranked list of cooling-station recommendations.
        Scoring Model (0 - 100):
        - Thermal Load Score (WBGT) [0 - 45 pts]
        - Proximity Gap (Distance to nearest hospital/cooling facility) [0 - 25 pts]
        - Labor Vulnerability / Worker Exposure [0 - 20 pts]
        - Tree Canopy Deficit [0 - 10 pts]
        """
        live_wbgt_map = live_wbgt_map or {}
        zones = cls._CANDIDATE_ZONES

        if target_city:
            clean_city = target_city.strip().lower()
            filtered = [z for z in zones if z["city"].lower() == clean_city]
            if filtered:
                zones = filtered

        recommendations = []

        for z in zones:
            city = z["city"]
            # Lookup live or default WBGT
            wbgt = live_wbgt_map.get(city, 31.5)  # default realistic warm season WBGT if unmapped

            # 1. Thermal load points (0 to 45)
            # WBGT 26°C -> 10 pts, 30°C -> 25 pts, 33°C+ -> 45 pts
            thermal_score = min(max((wbgt - 25.0) * 5.0, 5.0), 45.0)

            # 2. Distance gap points (0 to 25)
            # > 5 km = 25 pts, 1 km = 5 pts
            dist = z["distance_km"]
            dist_score = min(dist * 4.5, 25.0)

            # 3. Worker density factor (0 to 20)
            density_map = {"Extreme": 20.0, "Very High": 16.0, "High": 12.0, "Moderate": 8.0}
            labor_score = density_map.get(z["informal_worker_density"], 12.0)

            # 4. Canopy deficit (0 to 10): less canopy = higher need
            canopy = z["tree_canopy_pct"]
            canopy_score = max(10.0 - (canopy * 0.7), 2.0)

            total_score = round(thermal_score + dist_score + labor_score + canopy_score, 1)

            # Assign Priority Tier
            if total_score >= 80.0:
                priority = "IMMEDIATE (Priority 1)"
                badge_class = "badge-extreme"
                suggested_facility = "Emergency Misting Canopy + High-Volume Hydration Kiosk + ORS Station"
            elif total_score >= 65.0:
                priority = "URGENT (Priority 2)"
                badge_class = "badge-very-high"
                suggested_facility = "Drinking Water Piao + Shaded Rest Tent + Wet Sponge Cooling Point"
            elif total_score >= 50.0:
                priority = "HIGH (Priority 3)"
                badge_class = "badge-high"
                suggested_facility = "Mobile Water Tanker Dispatch + Shade Pavilion"
            else:
                priority = "MODERATE (Priority 4)"
                badge_class = "badge-moderate"
                suggested_facility = "Community Awareness Kiosk + Public Water Fountain"

            rationale = (
                f"WBGT thermal load of {wbgt:.1f}°C combined with {z['distance_km']} km distance "
                f"to {z['nearest_facility']} and {z['informal_worker_density'].lower()} informal labor density "
                f"creates a significant cooling coverage deficit."
            )

            recommendations.append({
                "zone_id": z["zone_id"],
                "zone_name": z["zone_name"],
                "city": z["city"],
                "latitude": z["latitude"],
                "longitude": z["longitude"],
                "zone_type": z["zone_type"],
                "wbgt_c": round(wbgt, 1),
                "siting_score": total_score,
                "priority": priority,
                "badge_class": badge_class,
                "suggested_facility": suggested_facility,
                "nearest_facility": z["nearest_facility"],
                "distance_km": z["distance_km"],
                "worker_density": z["informal_worker_density"],
                "tree_canopy_pct": z["tree_canopy_pct"],
                "rationale": rationale,
                "is_decision_support": True,
            })

        # Sort descending by siting_score
        recommendations.sort(key=lambda x: x["siting_score"], reverse=True)
        return recommendations
