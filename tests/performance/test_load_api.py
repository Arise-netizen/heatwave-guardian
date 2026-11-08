"""
HeatWave Guardian — API Endpoint Load & Concurrency Tests
==========================================================
Tests /api/check-risk, /api/forecast, /api/what-if, /api/authority/grid,
/api/ml/hyperlocal, /api/alerts, and related endpoints under sequential
and concurrent load using the Flask test client.

ALL PAYLOADS ARE SYNTHETIC BENCHMARK INPUTS — NOT SCIENTIFIC DATA.
"""

import os
import sys
import time
import json
import pytest
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tests.performance.perf_utils import (
    BenchmarkResult,
    measure_latency,
    run_concurrent_benchmark,
    BENCHMARK_LOCATIONS,
    make_check_risk_payload,
    make_forecast_params,
    make_whatif_payload,
    make_alert_payload,
    format_results_table,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a shared Flask test client."""
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="module")
def app_ctx():
    """Create an application context for the test module."""
    from app import app
    app.config["TESTING"] = True
    with app.app_context():
        yield app


# ──────────────────────────────────────────────────────────────────────────────
# 1. /api/check-risk — Personal Risk Assessment
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckRiskLoad:
    """Load test the personal risk assessment endpoint."""

    def _post_check_risk(self, client, payload: Dict) -> float:
        """Post to check-risk and return latency in ms."""
        start = time.perf_counter()
        resp = client.post(
            "/api/check-risk",
            data=json.dumps(payload),
            content_type="application/json",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Accept 200 or 502 (weather unavailable in test mode)
        assert resp.status_code in (200, 502), f"Unexpected status: {resp.status_code}"
        return elapsed_ms

    def test_single_request(self, client):
        """Baseline: single check-risk request."""
        payload = make_check_risk_payload(0, 0)
        latencies = []
        for _ in range(5):
            lat = self._post_check_risk(client, payload)
            latencies.append(lat)
        result = BenchmarkResult("check_risk_single", latencies)
        print(f"\n{result.summary_line()}")

    def test_10_sequential_requests(self, client):
        """10 sequential check-risk requests."""
        latencies = []
        for i in range(10):
            payload = make_check_risk_payload(i, i % 5)
            lat = self._post_check_risk(client, payload)
            latencies.append(lat)
        result = BenchmarkResult("check_risk_10_seq", latencies)
        print(f"\n{result.summary_line()}")

    def test_50_concurrent_requests(self, client):
        """50 concurrent check-risk requests via thread pool."""
        payloads = [make_check_risk_payload(i, i % 5) for i in range(50)]
        latencies = []
        errors = 0

        def _worker(payload):
            return self._post_check_risk(client, payload)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_worker, p): p for p in payloads}
            for f in as_completed(futures):
                try:
                    lat = f.result(timeout=30)
                    latencies.append(lat)
                except Exception:
                    errors += 1

        result = BenchmarkResult("check_risk_50_conc", latencies, errors)
        print(f"\n{result.summary_line()}")

    def test_100_concurrent_requests(self, client):
        """100 concurrent check-risk requests — stress test."""
        payloads = [make_check_risk_payload(i % 10, i % 5) for i in range(100)]
        latencies = []
        errors = 0

        def _worker(payload):
            return self._post_check_risk(client, payload)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_worker, p): p for p in payloads}
            for f in as_completed(futures):
                try:
                    lat = f.result(timeout=60)
                    latencies.append(lat)
                except Exception:
                    errors += 1

        result = BenchmarkResult("check_risk_100_conc", latencies, errors)
        print(f"\n{result.summary_line()}")
        # Error rate should stay below 5%
        assert result.error_rate < 5.0, f"Error rate {result.error_rate:.1f}% exceeds 5% threshold"


# ──────────────────────────────────────────────────────────────────────────────
# 2. /api/forecast — Multi-Day Forecast
# ──────────────────────────────────────────────────────────────────────────────

class TestForecastLoad:
    """Load test the forecast endpoint."""

    def _get_forecast(self, client, params: Dict) -> float:
        start = time.perf_counter()
        resp = client.get("/api/forecast", query_string=params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert resp.status_code in (200, 502), f"Unexpected status: {resp.status_code}"
        return elapsed_ms

    def test_single_forecast(self, client):
        """Baseline: single forecast request."""
        params = make_forecast_params(0)
        latencies = []
        for _ in range(3):
            lat = self._get_forecast(client, params)
            latencies.append(lat)
        result = BenchmarkResult("forecast_single", latencies)
        print(f"\n{result.summary_line()}")

    def test_multi_location_forecast(self, client):
        """Forecast for all benchmark locations sequentially."""
        latencies = []
        for i in range(len(BENCHMARK_LOCATIONS)):
            params = make_forecast_params(i)
            lat = self._get_forecast(client, params)
            latencies.append(lat)
        result = BenchmarkResult("forecast_multi_location", latencies)
        print(f"\n{result.summary_line()}")

    def test_concurrent_forecast_requests(self, client):
        """10 concurrent forecast requests."""
        all_params = [make_forecast_params(i) for i in range(10)]
        latencies = []
        errors = 0

        def _worker(params):
            return self._get_forecast(client, params)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_worker, p): p for p in all_params}
            for f in as_completed(futures):
                try:
                    lat = f.result(timeout=60)
                    latencies.append(lat)
                except Exception:
                    errors += 1

        result = BenchmarkResult("forecast_10_conc", latencies, errors)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. /api/scenario-analysis — What-If Simulator
# ──────────────────────────────────────────────────────────────────────────────

class TestWhatIfLoad:
    """Load test the What-If scenario analysis endpoint."""

    def _post_whatif(self, client, payload: Dict) -> float:
        start = time.perf_counter()
        resp = client.post(
            "/api/scenario-analysis",
            data=json.dumps(payload),
            content_type="application/json",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert resp.status_code in (200, 400, 502), f"Unexpected status: {resp.status_code}"
        return elapsed_ms

    def test_single_whatif(self, client):
        """Baseline: single What-If scenario."""
        payload = make_whatif_payload()
        latencies = []
        for _ in range(5):
            lat = self._post_whatif(client, payload)
            latencies.append(lat)
        result = BenchmarkResult("whatif_single", latencies)
        print(f"\n{result.summary_line()}")

    def test_repeated_scenario_evaluations(self, client):
        """30 sequential repeated scenario evaluations."""
        latencies = []
        temps = [35.0, 38.0, 40.0, 42.0, 44.0, 46.0]
        for i in range(30):
            payload = make_whatif_payload(temp_c=temps[i % len(temps)])
            lat = self._post_whatif(client, payload)
            latencies.append(lat)
        result = BenchmarkResult("whatif_30_repeated", latencies)
        print(f"\n{result.summary_line()}")

    def test_concurrent_whatif(self, client):
        """20 concurrent What-If evaluations."""
        payloads = [make_whatif_payload(temp_c=35 + i) for i in range(20)]
        latencies = []
        errors = 0

        def _worker(payload):
            return self._post_whatif(client, payload)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_worker, p): p for p in payloads}
            for f in as_completed(futures):
                try:
                    lat = f.result(timeout=30)
                    latencies.append(lat)
                except Exception:
                    errors += 1

        result = BenchmarkResult("whatif_20_conc", latencies, errors)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. /api/authority/grid — Spatial Grid
# ──────────────────────────────────────────────────────────────────────────────

class TestSpatialGridLoad:
    """Load test the authority spatial grid endpoint."""

    def _get_grid(self, client, city: str = None) -> float:
        params = {}
        if city:
            params["city"] = city
        start = time.perf_counter()
        resp = client.get("/api/authority/grid", query_string=params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert resp.status_code in (200, 500, 502), f"Unexpected status: {resp.status_code}"
        return elapsed_ms

    def test_single_city_grid(self, client):
        """Grid generation for a single city."""
        latencies = []
        for _ in range(3):
            lat = self._get_grid(client, city="Delhi")
            latencies.append(lat)
        result = BenchmarkResult("grid_single_city", latencies)
        print(f"\n{result.summary_line()}")

    def test_all_cities_grid(self, client):
        """Grid generation for all monitored cities."""
        latencies = []
        elapsed_ms = self._get_grid(client)  # No city filter = all
        latencies.append(elapsed_ms)
        result = BenchmarkResult("grid_all_cities", latencies)
        print(f"\n{result.summary_line()}")

    def test_repeated_grid_requests(self, client):
        """10 repeated grid requests — tests caching."""
        latencies = []
        for _ in range(10):
            lat = self._get_grid(client, city="Delhi")
            latencies.append(lat)
        result = BenchmarkResult("grid_10_repeated", latencies)
        print(f"\n{result.summary_line()}")
        # Later requests should be faster if caching works
        if len(latencies) > 2:
            first = latencies[0]
            last_avg = sum(latencies[2:]) / len(latencies[2:])
            print(f"  First request: {first:.1f}ms, Avg of requests 3-10: {last_avg:.1f}ms")


# ──────────────────────────────────────────────────────────────────────────────
# 5. /api/ml/hyperlocal — ML Hyperlocal Endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestMLHyperlocalEndpoint:
    """Load test the ML hyperlocal prediction endpoint."""

    def _get_hyperlocal(self, client, lat=28.6139, lon=77.2090, city="Delhi") -> float:
        start = time.perf_counter()
        resp = client.get(
            "/api/ml/hyperlocal",
            query_string={"lat": lat, "lon": lon, "city": city},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert resp.status_code in (200, 502), f"Unexpected status: {resp.status_code}"
        return elapsed_ms

    def test_sequential_predictions(self, client):
        """20 sequential ML hyperlocal predictions."""
        latencies = []
        for i, loc in enumerate(BENCHMARK_LOCATIONS[:5]):
            for _ in range(4):
                lat = self._get_hyperlocal(client, loc["lat"], loc["lon"], loc["name"])
                latencies.append(lat)
        result = BenchmarkResult("ml_hyperlocal_20_seq", latencies)
        print(f"\n{result.summary_line()}")

    def test_concurrent_predictions(self, client):
        """30 concurrent ML hyperlocal predictions."""
        locs = BENCHMARK_LOCATIONS * 3  # 30 requests
        latencies = []
        errors = 0

        def _worker(loc):
            return self._get_hyperlocal(client, loc["lat"], loc["lon"], loc["name"])

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_worker, loc): loc for loc in locs}
            for f in as_completed(futures):
                try:
                    lat = f.result(timeout=30)
                    latencies.append(lat)
                except Exception:
                    errors += 1

        result = BenchmarkResult("ml_hyperlocal_30_conc", latencies, errors)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. /api/alerts — Alert System
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertEndpointLoad:
    """Load test the alert generation and listing endpoints."""

    def test_alert_generation(self, client):
        """Test alert generation latency."""
        payload = {
            "risk_level": "High",
            "location": "Delhi",
            "wbgt_c": 33.5,
            "temp_c": 44.0,
            "role": "worker",
            "language": "en",
            "_benchmark_label": "SYNTHETIC_BENCHMARK_ALERT",
        }
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            resp = client.post(
                "/api/alerts/generate",
                data=json.dumps(payload),
                content_type="application/json",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            # Accept various status codes since alert behavior may vary
            assert resp.status_code < 500, f"Server error: {resp.status_code}"

        result = BenchmarkResult("alert_generation", latencies)
        print(f"\n{result.summary_line()}")

    def test_alert_listing(self, client):
        """Test alert listing latency."""
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            resp = client.get("/api/alerts")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            assert resp.status_code in (200,), f"Unexpected status: {resp.status_code}"

        result = BenchmarkResult("alert_listing", latencies)
        print(f"\n{result.summary_line()}")

    def test_alert_history(self, client):
        """Test alert history endpoint."""
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            resp = client.get("/api/alerts/history")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        result = BenchmarkResult("alert_history", latencies)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. /api/health — Health Check
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Baseline health check latency."""

    def test_health_check_performance(self, client):
        """Health endpoint should respond in <10ms."""
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            resp = client.get("/api/health")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            assert resp.status_code == 200

        result = BenchmarkResult("health_check", latencies)
        print(f"\n{result.summary_line()}")
        assert result.median_ms < 10, f"Health check median {result.median_ms:.1f}ms exceeds 10ms"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Request Safety Under Load
# ──────────────────────────────────────────────────────────────────────────────

class TestAPISafety:
    """Verify API doesn't crash under invalid or high-volume requests."""

    def test_invalid_json_payload(self, client):
        """Malformed JSON should not crash the server."""
        for _ in range(10):
            resp = client.post(
                "/api/check-risk",
                data="NOT_VALID_JSON{{{",
                content_type="application/json",
            )
            # Should return 400 or handle gracefully, not 500
            assert resp.status_code < 500, f"Server error on malformed JSON: {resp.status_code}"

    def test_oversized_payload_rejected(self, client):
        """Payloads exceeding MAX_CONTENT_LENGTH should be rejected."""
        # The app has 1MB limit
        large_data = json.dumps({"data": "x" * (2 * 1024 * 1024)})
        resp = client.post(
            "/api/check-risk",
            data=large_data,
            content_type="application/json",
        )
        assert resp.status_code in (413, 400), f"Expected 413, got {resp.status_code}"

    def test_empty_payload_handling(self, client):
        """Empty POST body should not crash."""
        resp = client.post(
            "/api/check-risk",
            data="{}",
            content_type="application/json",
        )
        # Should handle gracefully
        assert resp.status_code < 500

    def test_missing_parameters(self, client):
        """Missing required parameters handled gracefully."""
        resp = client.get("/api/forecast")
        assert resp.status_code < 500

    def test_extreme_coordinate_values(self, client):
        """Extreme lat/lon values should not crash."""
        resp = client.get(
            "/api/ml/hyperlocal",
            query_string={"lat": 999, "lon": -999, "city": "Test"},
        )
        assert resp.status_code < 500
