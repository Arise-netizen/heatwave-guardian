"""
HeatWave Guardian — Demographic Data Service
=============================================
Provides demographic vulnerability indicators across Indian urban & rural districts.

DATA HONESTY COMPLIANCE:
This service currently operates in DEMONSTRATION MODE.
All demographic indicators are sample/proxy data curated for architecture
demonstration and decision-support simulation.
They are NOT official Census of India or Ministry of Health statistics.
"""

from typing import Dict, Any, Optional


class DemographicService:
    # Pluggable data store. In production, this connects to official Census / NFHS data pipelines.
    _DEMO_DATA: Dict[str, Dict[str, Any]] = {
        "delhi": {
            "district": "New Delhi / NCR",
            "state": "Delhi",
            "population": 16787941,
            "population_density_sqkm": 11320,
            "elderly_pct": 7.2,
            "children_pct": 11.4,
            "outdoor_worker_pct": 28.5,
            "informal_housing_pct": 32.0,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "hyderabad": {
            "district": "Hyderabad Urban",
            "state": "Telangana",
            "population": 6809970,
            "population_density_sqkm": 9400,
            "elderly_pct": 6.8,
            "children_pct": 10.9,
            "outdoor_worker_pct": 31.2,
            "informal_housing_pct": 28.4,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "mumbai": {
            "district": "Mumbai Suburban & City",
            "state": "Maharashtra",
            "population": 12442373,
            "population_density_sqkm": 20634,
            "elderly_pct": 8.1,
            "children_pct": 9.8,
            "outdoor_worker_pct": 34.0,
            "informal_housing_pct": 41.5,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "pune": {
            "district": "Pune District",
            "state": "Maharashtra",
            "population": 9429408,
            "population_density_sqkm": 603,
            "elderly_pct": 7.5,
            "children_pct": 11.0,
            "outdoor_worker_pct": 26.5,
            "informal_housing_pct": 22.1,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "ahmedabad": {
            "district": "Ahmedabad Municipal Corp",
            "state": "Gujarat",
            "population": 5577940,
            "population_density_sqkm": 11950,
            "elderly_pct": 7.9,
            "children_pct": 10.5,
            "outdoor_worker_pct": 33.5,
            "informal_housing_pct": 29.8,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "kolkata": {
            "district": "Kolkata Metropolitan",
            "state": "West Bengal",
            "population": 4496694,
            "population_density_sqkm": 24252,
            "elderly_pct": 9.4,
            "children_pct": 8.7,
            "outdoor_worker_pct": 30.8,
            "informal_housing_pct": 33.2,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "chennai": {
            "district": "Chennai District",
            "state": "Tamil Nadu",
            "population": 7088000,
            "population_density_sqkm": 17000,
            "elderly_pct": 8.7,
            "children_pct": 9.5,
            "outdoor_worker_pct": 29.0,
            "informal_housing_pct": 26.0,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "nagpur": {
            "district": "Nagpur Urban",
            "state": "Maharashtra",
            "population": 2405665,
            "population_density_sqkm": 11000,
            "elderly_pct": 7.1,
            "children_pct": 11.2,
            "outdoor_worker_pct": 35.2,
            "informal_housing_pct": 27.5,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
        "khammam": {
            "district": "Khammam District",
            "state": "Telangana",
            "population": 1401639,
            "population_density_sqkm": 321,
            "elderly_pct": 7.8,
            "children_pct": 11.8,
            "outdoor_worker_pct": 48.6,
            "informal_housing_pct": 25.3,
            "source": "Sample Proxy Data (Demo Architecture)",
            "is_demo_data": True,
        },
    }

    # Generic national proxy fallback for other locations
    _DEFAULT_NATIONAL_PROXY: Dict[str, Any] = {
        "district": "Regional Area (National Proxy)",
        "state": "India",
        "population": 1000000,
        "population_density_sqkm": 420,
        "elderly_pct": 7.0,
        "children_pct": 11.5,
        "outdoor_worker_pct": 32.0,
        "informal_housing_pct": 26.0,
        "source": "Generic National Demographic Baseline (Demo Architecture)",
        "is_demo_data": True,
    }

    @classmethod
    def get_demographics(cls, location_name: str, admin1: str = "") -> Dict[str, Any]:
        """
        Retrieve demographic indicators for a location.
        Returns data dictionary with guaranteed `is_demo_data: True` flag and clear attribution.
        """
        loc_key = (location_name or "").strip().lower()
        for city_key, demo in cls._DEMO_DATA.items():
            if city_key in loc_key:
                res = demo.copy()
                return res

        # Fallback to national proxy
        fallback = cls._DEFAULT_NATIONAL_PROXY.copy()
        fallback["district"] = f"{location_name} Area"
        if admin1:
            fallback["state"] = admin1
        return fallback

    @classmethod
    def get_vulnerability_score(cls, demographics: Dict[str, Any]) -> float:
        """
        Compute demographic exposure factor on a 0.0 - 1.0 scale:
        Weighted composite of:
        - Elderly % (30%)
        - Children % (20%)
        - Outdoor worker % (35%)
        - Informal housing % (15%)
        """
        elderly = demographics.get("elderly_pct", 7.0) / 15.0  # normalize up to 15%
        children = demographics.get("children_pct", 11.0) / 20.0
        workers = demographics.get("outdoor_worker_pct", 30.0) / 60.0
        housing = demographics.get("informal_housing_pct", 25.0) / 50.0

        score = (0.30 * min(elderly, 1.0) +
                 0.20 * min(children, 1.0) +
                 0.35 * min(workers, 1.0) +
                 0.15 * min(housing, 1.0))
        return round(score, 2)
