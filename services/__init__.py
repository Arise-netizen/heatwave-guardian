"""
HeatWave Guardian — Services Package
Provides modular service layers for weather & atmospheric pipeline, geocoding, demographic indicators,
Heat Action Plan integration, SMS/WhatsApp alert dispatch, experimental
mortality proxy index, cooling-station siting, and institutional monitoring.
"""

from .weather_service import (
    WeatherService,
    WeatherServiceError,
    WeatherServiceUnavailableError,
    WeatherServiceValidationError,
)
from .geo_service import GeoService
from .demographic_service import DemographicService
from .hap_service import HAPService
from .alert_service import AlertService
from .mortality_service import MortalityRiskService
from .cooling_recommender import CoolingRecommenderService
from .institute_service import InstituteService
from .authority_service import AuthorityService

__all__ = [
    "WeatherService",
    "WeatherServiceError",
    "WeatherServiceUnavailableError",
    "WeatherServiceValidationError",
    "GeoService",
    "DemographicService",
    "HAPService",
    "AlertService",
    "MortalityRiskService",
    "CoolingRecommenderService",
    "InstituteService",
    "AuthorityService",
]
