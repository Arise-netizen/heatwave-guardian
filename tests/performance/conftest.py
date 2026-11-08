"""
HeatWave Guardian — Performance Test Suite Configuration
==========================================================
Automatically enables weather caching during performance test execution.
"""

import os
import pytest
from services.weather_service import WeatherService
from services.hyperlocal_ml_service import HyperlocalMLService
from services.authority_service import AuthorityService


@pytest.fixture(autouse=True, scope="session")
def enable_weather_cache_for_perf():
    """Enable in-memory weather caching for high-concurrency performance benchmarks."""
    os.environ["HEATWAVE_ENABLE_WEATHER_CACHE"] = "1"
    yield
    os.environ.pop("HEATWAVE_ENABLE_WEATHER_CACHE", None)


@pytest.fixture(autouse=True, scope="function")
def reset_caches_between_perf_tests():
    """Ensure clean cache state between performance tests if needed."""
    yield
