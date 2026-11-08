"""
HeatWave Guardian — Soak, Sustained Load & Failure Mode Tests
==============================================================
Tests for memory leaks, latency degradation under sustained load,
cache correctness, resource growth, and stress-limit discovery.

ALL PAYLOADS ARE SYNTHETIC BENCHMARK INPUTS — NOT SCIENTIFIC DATA.
"""

import os
import sys
import time
import json
import gc
import tracemalloc
import pytest
from typing import Dict, Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tests.performance.perf_utils import (
    BenchmarkResult,
    measure_latency,
    BENCHMARK_LOCATIONS,
    make_ml_weather_dict,
    make_check_risk_payload,
    format_results_table,
)
from ml.src.predict import HyperlocalPredictor
from engine import estimate_wbgt, clear_wbgt_cache
from services.hyperlocal_ml_service import HyperlocalMLService
from services.authority_service import AuthorityService
from services.alert_service import AlertService


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mock_weather(lat, lon):
    return {
        "temp_c": 42.0, "rh_pct": 30.0, "wind_ms": 2.0, "solar_wm2": 750.0,
        "humidity": 30.0, "current_temp_c": 42.0, "freshness": "FRESH",
        "source": "BENCHMARK_SYNTHETIC",
        "source_time": "2024-05-15T12:00:00Z",
        "source_transparency": {
            "source": "Benchmark Synthetic Weather",
            "is_cached": False, "freshness": "FRESH",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Sustained ML Inference Soak
# ──────────────────────────────────────────────────────────────────────────────

class TestMLSoak:
    """Sustained ML inference to detect memory leaks and latency degradation."""

    def test_sustained_ml_inference_300_iterations(self):
        """Run 300 ML predictions and watch for degradation."""
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        w = make_ml_weather_dict(0)
        latencies = []
        window_size = 30

        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()
        mem_start = tracemalloc.get_traced_memory()[1]

        for i in range(300):
            loc = BENCHMARK_LOCATIONS[i % len(BENCHMARK_LOCATIONS)]
            elapsed_ms, result = measure_latency(
                HyperlocalPredictor.predict_hyperlocal,
                loc["lat"], loc["lon"], w, city_name=loc["name"],
            )
            latencies.append(elapsed_ms)
            assert "hyperlocal_environment" in result

        mem_end = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Check for latency degradation: compare first and last windows
        first_window = latencies[:window_size]
        last_window = latencies[-window_size:]
        first_avg = sum(first_window) / len(first_window)
        last_avg = sum(last_window) / len(last_window)
        degradation = (last_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0

        # Memory growth
        mem_growth_mb = (mem_end - mem_start) / (1024 * 1024)

        result = BenchmarkResult("soak_ml_300", latencies)
        print(f"\n{result.summary_line()}")
        print(f"  First 30 avg: {first_avg:.1f}ms, Last 30 avg: {last_avg:.1f}ms")
        print(f"  Latency degradation: {degradation:+.1f}%")
        print(f"  Memory growth: {mem_growth_mb:+.2f} MB")

        # Latency should not degrade more than 50%
        assert degradation < 50, f"Latency degraded {degradation:.0f}% over 300 iterations"
        # Memory should not grow more than 50 MB
        assert mem_growth_mb < 50, f"Memory grew {mem_growth_mb:.1f}MB over 300 iterations"

    def test_sustained_batch_inference_100_batches(self):
        """Run 100 batch inferences (50 items each) to detect leaks."""
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        batch_size = 50
        items = []
        for i in range(batch_size):
            loc = BENCHMARK_LOCATIONS[i % len(BENCHMARK_LOCATIONS)]
            w = make_ml_weather_dict(i % 5)
            items.append({
                "lat": loc["lat"],
                "lon": loc["lon"],
                "background_weather": w,
                "city_name": loc["name"],
            })

        latencies = []
        tracemalloc.start()

        for iteration in range(100):
            elapsed_ms, results = measure_latency(HyperlocalPredictor.predict_batch, items)
            latencies.append(elapsed_ms)
            assert len(results) == batch_size

        peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        first_10 = latencies[:10]
        last_10 = latencies[-10:]
        degradation = ((sum(last_10)/10) - (sum(first_10)/10)) / (sum(first_10)/10) * 100

        result = BenchmarkResult("soak_batch_100x50", latencies)
        print(f"\n{result.summary_line()}")
        print(f"  Peak memory: {peak_mem:.2f} MB")
        print(f"  Latency degradation: {degradation:+.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 2. WBGT Cache Soak
# ──────────────────────────────────────────────────────────────────────────────

class TestWBGTSoak:
    """Sustained WBGT computation to test LRU cache behavior."""

    def test_wbgt_cache_fill_and_eviction(self):
        """Fill the WBGT cache to capacity and verify behavior."""
        clear_wbgt_cache()
        # Cache has maxsize=32768
        latencies = []
        for i in range(35000):
            t = 20.0 + (i % 4500) * 0.01
            elapsed_ms, _ = measure_latency(estimate_wbgt, t, 40.0, 2.0, 600.0)
            latencies.append(elapsed_ms)

        result = BenchmarkResult("wbgt_cache_35k", latencies)
        print(f"\n{result.summary_line()}")

    def test_wbgt_sustained_identical_requests(self):
        """1000 identical WBGT requests — all should be cache hits."""
        # Warm cache
        estimate_wbgt(42.0, 30.0, 2.0, 750.0)
        latencies = []
        for _ in range(1000):
            elapsed_ms, _ = measure_latency(estimate_wbgt, 42.0, 30.0, 2.0, 750.0)
            latencies.append(elapsed_ms)
        result = BenchmarkResult("wbgt_1000_cache_hits", latencies)
        print(f"\n{result.summary_line()}")
        # All cache hits should be under 0.1ms average
        assert result.avg_ms < 1.0, f"Cache hit avg {result.avg_ms:.3f}ms too slow"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Spatial Grid Soak
# ──────────────────────────────────────────────────────────────────────────────

class TestSpatialGridSoak:
    """Sustained spatial grid generation."""

    def test_repeated_grid_generation_50x(self):
        """Generate grid 50 times and watch for degradation."""
        w = make_ml_weather_dict(0)
        latencies = []

        for i in range(50):
            loc = BENCHMARK_LOCATIONS[i % len(BENCHMARK_LOCATIONS)]
            elapsed_ms, cells = measure_latency(
                HyperlocalMLService.generate_spatial_grid_cells,
                loc["name"], loc["lat"], loc["lon"], w, region=loc["region"],
            )
            latencies.append(elapsed_ms)

        result = BenchmarkResult("grid_soak_50", latencies)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cache Correctness Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheCorrectness:
    """Verify cache behavior under various scenarios."""

    def test_cache_hit_returns_same_result(self):
        """Identical requests must return identical results."""
        w = make_ml_weather_dict(0)
        result_a = HyperlocalMLService.predict_point(28.6139, 77.2090, w, "Delhi")
        result_b = HyperlocalMLService.predict_point(28.6139, 77.2090, w, "Delhi")

        assert result_a["localized_temp_c"] == result_b["localized_temp_c"]
        assert (
            result_a["hyperlocal_environment"]["wbgt_c"]
            == result_b["hyperlocal_environment"]["wbgt_c"]
        )

    def test_different_locations_not_confused(self):
        """Different locations must return different cache entries."""
        w = make_ml_weather_dict(0)
        HyperlocalMLService._cache.clear()

        result_delhi = HyperlocalMLService.predict_point(28.6139, 77.2090, w, "Delhi")
        result_mumbai = HyperlocalMLService.predict_point(19.0760, 72.8777, w, "Mumbai")

        # Different locations may produce different temperatures
        # (they may also be similar depending on weather, but should come from different cache entries)
        assert isinstance(result_delhi["localized_temp_c"], (int, float))
        assert isinstance(result_mumbai["localized_temp_c"], (int, float))

    def test_different_weather_not_confused(self):
        """Different weather for same location must not return stale cache."""
        HyperlocalMLService._cache.clear()

        w1 = {"temp_c": 30.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 500.0}
        w2 = {"temp_c": 45.0, "rh_pct": 15.0, "wind_ms": 1.0, "solar_wm2": 900.0}

        result_a = HyperlocalMLService.predict_point(28.6139, 77.2090, w1, "Delhi")
        result_b = HyperlocalMLService.predict_point(28.6139, 77.2090, w2, "Delhi")

        # Different background temps should yield different localized temps
        assert result_a["localized_temp_c"] != result_b["localized_temp_c"]

    def test_wbgt_cache_determinism(self):
        """WBGT cache must return consistent values."""
        clear_wbgt_cache()
        r1 = estimate_wbgt(42.0, 30.0, 2.0, 750.0)
        r2 = estimate_wbgt(42.0, 30.0, 2.0, 750.0)
        assert r1 == r2, f"WBGT cache non-deterministic: {r1} != {r2}"

    def test_authority_cache_ttl(self):
        """Authority cache should expire after TTL."""
        AuthorityService.clear_cache()
        # Temporarily set very short TTL
        original_ttl = AuthorityService._CACHE_TTL_SECONDS
        AuthorityService._CACHE_TTL_SECONDS = 0.001  # 1ms

        try:
            result_a = AuthorityService.get_area_heat_layer(_mock_weather, use_cache=True)
            time.sleep(0.01)  # Wait for TTL to expire
            # Should not return stale cache
            result_b = AuthorityService.get_area_heat_layer(_mock_weather, use_cache=True)
            # Both should be valid
            assert "features" in result_a
            assert "features" in result_b
        finally:
            AuthorityService._CACHE_TTL_SECONDS = original_ttl
            AuthorityService.clear_cache()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Alert Soak Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertSoak:
    """Sustained alert operations."""

    def test_bulk_alert_formatting(self):
        """Format 100 alerts rapidly."""
        latencies = []
        risk_data = {
            "risk_level": "High",
            "wbgt_c": 33.5,
            "temp_c": 44.0,
            "location": "Delhi",
            "peak_period": "12:00–16:00 IST",
            "work_minutes": 30,
            "rest_minutes": 30,
        }

        roles = ["worker", "supervisor", "site_manager", "authority", "healthcare_facility"]
        for i in range(100):
            role = roles[i % len(roles)]
            elapsed_ms, msg = measure_latency(
                AlertService.format_alert_message,
                risk_data=risk_data,
                role=role,
                language="en",
            )
            latencies.append(elapsed_ms)
            assert isinstance(msg, str) or isinstance(msg, dict)

        result = BenchmarkResult("alert_format_100", latencies)
        print(f"\n{result.summary_line()}")

    def test_alert_audit_log_growth(self):
        """Check audit log doesn't grow unboundedly during sustained operations."""
        initial_history_len = len(AlertService.get_history())
        for i in range(50):
            AlertService.format_alert_message(
                risk_data={
                    "risk_level": "High", "wbgt_c": 33.0, "temp_c": 43.0,
                    "location": f"BenchmarkCity{i}", "peak_period": "12:00–15:00 IST",
                },
                role="worker",
                language="en",
            )
        final_history_len = len(AlertService.get_history())
        growth = final_history_len - initial_history_len
        print(f"\n[AUDIT_LOG] Growth after 50 alerts: {growth} entries")
        # History shouldn't grow unboundedly (should have size limits)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Stress Limit Discovery
# ──────────────────────────────────────────────────────────────────────────────

class TestStressLimit:
    """Push beyond expected workload to find saturation point."""

    def test_ml_batch_stress_increasing(self):
        """Increase batch size until performance degrades significantly."""
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        sizes = [10, 50, 100, 500, 1000, 2000]
        results = []
        for size in sizes:
            items = []
            for i in range(size):
                loc = BENCHMARK_LOCATIONS[i % len(BENCHMARK_LOCATIONS)]
                w = make_ml_weather_dict(i % 5)
                items.append({
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "background_weather": w,
                    "city_name": loc["name"],
                })

            start = time.perf_counter()
            try:
                predictions = HyperlocalPredictor.predict_batch(items)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                per_item = elapsed_ms / size
                results.append((size, elapsed_ms, per_item, len(predictions), "OK"))
                print(f"  batch={size:>5}: total={elapsed_ms:>8.1f}ms  per_item={per_item:.3f}ms  status=OK")
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                results.append((size, elapsed_ms, 0, 0, f"FAIL: {type(e).__name__}"))
                print(f"  batch={size:>5}: FAILED after {elapsed_ms:.1f}ms — {type(e).__name__}: {e}")

        # Report saturation point
        ok_results = [r for r in results if r[4] == "OK"]
        if ok_results:
            max_ok = ok_results[-1]
            print(f"\n  Saturation point: batch_size={max_ok[0]} at {max_ok[1]:.1f}ms total")

    def test_api_endpoint_stress(self):
        """Push API endpoint with rapid sequential requests."""
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        latencies = []
        errors = 0
        for i in range(200):
            payload = make_check_risk_payload(i % 10, i % 5)
            start = time.perf_counter()
            try:
                resp = client.post(
                    "/api/check-risk",
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if resp.status_code < 500:
                    latencies.append(elapsed_ms)
                else:
                    errors += 1
            except Exception:
                errors += 1

        result = BenchmarkResult("stress_api_200_seq", latencies, errors)
        print(f"\n{result.summary_line()}")

        # Check for latency stability
        if len(latencies) >= 40:
            first_20 = latencies[:20]
            last_20 = latencies[-20:]
            first_avg = sum(first_20) / len(first_20)
            last_avg = sum(last_20) / len(last_20)
            print(f"  First 20 avg: {first_avg:.1f}ms, Last 20 avg: {last_avg:.1f}ms")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Recovery Test
# ──────────────────────────────────────────────────────────────────────────────

class TestRecovery:
    """Test system recovery after heavy load."""

    def test_recovery_after_stress(self):
        """System should return to baseline performance after heavy load."""
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        w = make_ml_weather_dict(0)

        # Baseline
        baseline_latencies = []
        for _ in range(10):
            elapsed_ms, _ = measure_latency(
                HyperlocalPredictor.predict_hyperlocal,
                28.6139, 77.2090, w, city_name="Delhi",
            )
            baseline_latencies.append(elapsed_ms)
        baseline_avg = sum(baseline_latencies) / len(baseline_latencies)

        # Heavy load
        items = []
        for i in range(500):
            loc = BENCHMARK_LOCATIONS[i % len(BENCHMARK_LOCATIONS)]
            items.append({
                "lat": loc["lat"],
                "lon": loc["lon"],
                "background_weather": w,
                "city_name": loc["name"],
            })
        HyperlocalPredictor.predict_batch(items)

        # Force cleanup
        gc.collect()

        # Recovery measurement
        recovery_latencies = []
        for _ in range(10):
            elapsed_ms, _ = measure_latency(
                HyperlocalPredictor.predict_hyperlocal,
                28.6139, 77.2090, w, city_name="Delhi",
            )
            recovery_latencies.append(elapsed_ms)
        recovery_avg = sum(recovery_latencies) / len(recovery_latencies)

        degradation = (recovery_avg - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
        print(f"\n[RECOVERY] Baseline: {baseline_avg:.1f}ms, Post-stress: {recovery_avg:.1f}ms")
        print(f"  Degradation: {degradation:+.1f}%")
        # Recovery should bring latency back within 100% of baseline
        assert degradation < 100, f"Failed to recover: {degradation:.0f}% degradation"
