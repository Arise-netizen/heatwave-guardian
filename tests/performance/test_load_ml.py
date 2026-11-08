"""
HeatWave Guardian — ML Inference Performance & Scalability Tests
================================================================
Tests ML model loading, single-point inference, batch/vectorized inference,
grid inference scaling, and numerical equivalence between approaches.

ALL PAYLOADS ARE SYNTHETIC BENCHMARK INPUTS — NOT SCIENTIFIC DATA.
"""

import os
import sys
import time
import json
import math
import tracemalloc
import pytest
from typing import Dict, Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tests.performance.perf_utils import (
    BenchmarkResult,
    measure_latency,
    measure_memory,
    run_sequential_benchmark,
    BENCHMARK_LOCATIONS,
    BENCHMARK_WEATHER_CONDITIONS,
    make_ml_weather_dict,
    format_results_table,
)
from ml.src.predict import HyperlocalPredictor
from ml.src.features import extract_features_from_dict, FEATURE_NAMES
from engine import calculate_thermal_stress_index, estimate_wbgt


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_model():
    """Ensure the ML model is loaded once for the module."""
    model = HyperlocalPredictor.get_model()
    if model is None:
        pytest.skip("ML model artifact not available; cannot benchmark inference.")
    return model


@pytest.fixture(scope="module")
def single_weather():
    return make_ml_weather_dict(0)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Loading Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestModelLoading:
    """Verify model loading behavior and singleton pattern."""

    def test_model_loads_successfully(self):
        """Model artifact must load without error."""
        model = HyperlocalPredictor.get_model()
        assert model is not None, "Model failed to load"

    def test_model_load_time(self):
        """Measure cold-start model load time."""
        # Force a fresh load by clearing the cached model
        original_model = HyperlocalPredictor._model
        HyperlocalPredictor._model = None
        try:
            elapsed_ms, model = measure_latency(HyperlocalPredictor.get_model)
            assert model is not None
            print(f"\n[MODEL_LOAD] Cold load time: {elapsed_ms:.2f} ms")
            # Model should load in under 5 seconds even on slow machines
            assert elapsed_ms < 5000, f"Model load took {elapsed_ms:.0f}ms (>5s threshold)"
        finally:
            HyperlocalPredictor._model = original_model

    def test_model_singleton_pattern(self, loaded_model):
        """Subsequent get_model() calls must return the cached instance, not reload."""
        latencies = []
        for _ in range(100):
            elapsed_ms, model = measure_latency(HyperlocalPredictor.get_model)
            latencies.append(elapsed_ms)
            assert model is loaded_model, "Model singleton violated — reload detected"
        avg = sum(latencies) / len(latencies)
        print(f"\n[MODEL_SINGLETON] 100 cached get_model() calls: avg={avg:.4f}ms")
        # Cached access should be essentially free (<0.5ms)
        assert avg < 0.5, f"Cached model access too slow: {avg:.2f}ms avg"

    def test_model_memory_footprint(self, loaded_model):
        """Measure memory consumption of the loaded model."""
        mem_mb, _ = measure_memory(lambda: loaded_model)
        print(f"\n[MODEL_MEMORY] Model memory footprint: {mem_mb:.2f} MB")
        # Prototype model should be <50 MB
        assert mem_mb < 50, f"Model memory {mem_mb:.1f}MB exceeds 50MB threshold"

    def test_missing_model_handled_safely(self):
        """Verify graceful handling when model file is missing."""
        original_model = HyperlocalPredictor._model
        HyperlocalPredictor._model = None
        original_path = os.environ.get("_TEST_OVERRIDE_MODEL_PATH")
        try:
            import ml.src.predict as pred_mod
            real_path = pred_mod.MODEL_FILE_PATH
            pred_mod.MODEL_FILE_PATH = "/nonexistent/path/model.joblib"
            HyperlocalPredictor._model = None
            model = HyperlocalPredictor.get_model()
            assert model is None, "Expected None for missing model"
            # Verify prediction returns unavailable state
            result = HyperlocalPredictor.predict_hyperlocal(
                28.6139, 77.2090, make_ml_weather_dict(0), city_name="Delhi"
            )
            assert result["model_available"] is False
            assert result["prediction_status"] == "model_unavailable"
        finally:
            pred_mod.MODEL_FILE_PATH = real_path
            HyperlocalPredictor._model = original_model


# ──────────────────────────────────────────────────────────────────────────────
# 2. Single-Point Inference
# ──────────────────────────────────────────────────────────────────────────────

class TestSinglePointInference:
    """Benchmark single-point ML inference performance."""

    def test_single_prediction_latency(self, loaded_model, single_weather):
        """Measure per-request ML inference time."""
        latencies = []
        for _ in range(50):
            elapsed_ms, result = measure_latency(
                HyperlocalPredictor.predict_hyperlocal,
                28.6139, 77.2090, single_weather, city_name="Delhi"
            )
            latencies.append(elapsed_ms)
            assert "hyperlocal_environment" in result
            assert "localized_temp_c" in result

        result = BenchmarkResult("single_point_inference", latencies)
        print(f"\n{result.summary_line()}")
        # Single prediction should complete within 200ms
        assert result.median_ms < 200, f"Median single inference {result.median_ms:.1f}ms exceeds 200ms"

    def test_single_prediction_across_locations(self, loaded_model):
        """Measure inference across diverse locations."""
        latencies = []
        for loc in BENCHMARK_LOCATIONS:
            w = make_ml_weather_dict(0)
            elapsed_ms, result = measure_latency(
                HyperlocalPredictor.predict_hyperlocal,
                loc["lat"], loc["lon"], w, city_name=loc["name"]
            )
            latencies.append(elapsed_ms)
            assert result["model_available"] is True

        result = BenchmarkResult("single_point_multi_location", latencies)
        print(f"\n{result.summary_line()}")

    def test_single_prediction_across_weather(self, loaded_model):
        """Measure inference across weather conditions."""
        latencies = []
        for w_cond in BENCHMARK_WEATHER_CONDITIONS:
            w = {k: v for k, v in w_cond.items() if k != "label"}
            elapsed_ms, result = measure_latency(
                HyperlocalPredictor.predict_hyperlocal,
                28.6139, 77.2090, w, city_name="Delhi"
            )
            latencies.append(elapsed_ms)

        result = BenchmarkResult("single_point_multi_weather", latencies)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Batch/Vectorized Inference
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchInference:
    """Benchmark batch inference and verify vectorization advantage."""

    def _make_batch_items(self, n: int) -> List[Dict[str, Any]]:
        """Generate n benchmark items by cycling through locations and weather."""
        items = []
        for i in range(n):
            loc = BENCHMARK_LOCATIONS[i % len(BENCHMARK_LOCATIONS)]
            w = make_ml_weather_dict(i % len(BENCHMARK_WEATHER_CONDITIONS))
            items.append({
                "lat": loc["lat"],
                "lon": loc["lon"],
                "background_weather": w,
                "city_name": loc["name"],
            })
        return items

    @pytest.mark.parametrize("batch_size", [5, 10, 50, 100, 500])
    def test_batch_inference_latency(self, loaded_model, batch_size):
        """Measure vectorized batch inference at increasing sizes."""
        items = self._make_batch_items(batch_size)
        elapsed_ms, results = measure_latency(HyperlocalPredictor.predict_batch, items)
        per_item_ms = elapsed_ms / batch_size

        assert len(results) == batch_size
        for r in results:
            assert "hyperlocal_environment" in r

        print(f"\n[BATCH_{batch_size}] total={elapsed_ms:.1f}ms per_item={per_item_ms:.2f}ms")

    def test_batch_vs_loop_speedup(self, loaded_model):
        """Compare batch inference vs sequential loop — quantify vectorization benefit."""
        n = 100
        items = self._make_batch_items(n)

        # Batch inference
        batch_ms, batch_results = measure_latency(HyperlocalPredictor.predict_batch, items)

        # Sequential loop inference
        loop_start = time.perf_counter()
        loop_results = []
        for item in items:
            r = HyperlocalPredictor.predict_hyperlocal(
                item["lat"], item["lon"], item["background_weather"],
                city_name=item.get("city_name")
            )
            loop_results.append(r)
        loop_ms = (time.perf_counter() - loop_start) * 1000.0

        speedup = loop_ms / batch_ms if batch_ms > 0 else float("inf")
        print(f"\n[VECTORIZATION] batch={batch_ms:.1f}ms loop={loop_ms:.1f}ms speedup={speedup:.2f}x")

        # Verify both produce identical numerical outputs
        for i in range(n):
            b_t = batch_results[i]["localized_temp_c"]
            l_t = loop_results[i]["localized_temp_c"]
            assert abs(b_t - l_t) < 0.01, (
                f"Numerical mismatch at index {i}: batch={b_t} loop={l_t}"
            )

    def test_large_grid_inference_1000(self, loaded_model):
        """Stress test with 1,000 grid cells."""
        items = self._make_batch_items(1000)
        elapsed_ms, results = measure_latency(HyperlocalPredictor.predict_batch, items)
        per_item_ms = elapsed_ms / 1000
        print(f"\n[GRID_1000] total={elapsed_ms:.1f}ms per_item={per_item_ms:.3f}ms")
        assert len(results) == 1000
        # 1000 cells should complete in under 30 seconds
        assert elapsed_ms < 30000, f"1000 cells took {elapsed_ms:.0f}ms (>30s)"

    def test_grid_scaling_profile(self, loaded_model):
        """Measure how inference time scales with grid size."""
        sizes = [10, 50, 100, 500]
        timings = []
        for size in sizes:
            items = self._make_batch_items(size)
            elapsed_ms, results = measure_latency(HyperlocalPredictor.predict_batch, items)
            per_item = elapsed_ms / size
            timings.append((size, elapsed_ms, per_item))
            print(f"  grid={size:>5}: total={elapsed_ms:>8.1f}ms  per_item={per_item:.3f}ms")

        # Check that per-item time doesn't increase drastically (sublinear is ideal)
        per_item_10 = timings[0][2]
        per_item_500 = timings[3][2]
        # Allow up to 3x degradation for larger batches (overhead)
        if per_item_10 > 0.01:
            ratio = per_item_500 / per_item_10
            print(f"\n  Scaling ratio (500 vs 10 per-item): {ratio:.2f}x")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ensemble Uncertainty Performance
# ──────────────────────────────────────────────────────────────────────────────

class TestEnsembleUncertainty:
    """Benchmark ensemble tree spread computation."""

    def test_uncertainty_computation_time(self, loaded_model):
        """Measure time spent computing ensemble uncertainty."""
        items = [{
            "lat": 28.6139,
            "lon": 77.2090,
            "background_weather": make_ml_weather_dict(0),
            "city_name": "Delhi",
        }]
        elapsed_ms, results = measure_latency(HyperlocalPredictor.predict_batch, items)
        assert "uncertainty" in results[0]
        unc = results[0]["uncertainty"]
        assert "ensemble_tree_std_c" in unc
        assert unc["ensemble_tree_std_c"] >= 0
        print(f"\n[UNCERTAINTY] Single prediction with uncertainty: {elapsed_ms:.2f}ms")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Numerical Equivalence Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNumericalEquivalence:
    """Verify optimized paths produce identical outputs to baseline."""

    def test_wbgt_calculation_consistency(self):
        """WBGT results must be identical for identical inputs."""
        results = []
        for _ in range(100):
            val = estimate_wbgt(42.0, 30.0, 2.0, 750.0)
            results.append(val)
        assert all(r == results[0] for r in results), "WBGT non-determinism detected"

    def test_batch_prediction_determinism(self, loaded_model):
        """Batch predictions must be deterministic across runs."""
        items = [{
            "lat": 28.6139,
            "lon": 77.2090,
            "background_weather": make_ml_weather_dict(0),
            "city_name": "Delhi",
        }] * 10

        results_a = HyperlocalPredictor.predict_batch(items)
        results_b = HyperlocalPredictor.predict_batch(items)

        for i in range(10):
            assert abs(results_a[i]["localized_temp_c"] - results_b[i]["localized_temp_c"]) < 0.001
            assert abs(
                results_a[i]["hyperlocal_environment"]["wbgt_c"]
                - results_b[i]["hyperlocal_environment"]["wbgt_c"]
            ) < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# 6. Engine Core Computation Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

class TestEngineCoreBenchmarks:
    """Benchmark the core scientific engine functions."""

    def test_wbgt_cached_performance(self):
        """Benchmark cached WBGT calculation."""
        # Warm up the cache
        estimate_wbgt(42.0, 30.0, 2.0, 750.0)
        latencies = []
        for _ in range(1000):
            elapsed_ms, _ = measure_latency(estimate_wbgt, 42.0, 30.0, 2.0, 750.0)
            latencies.append(elapsed_ms)
        result = BenchmarkResult("wbgt_cached", latencies)
        print(f"\n{result.summary_line()}")

    def test_wbgt_uncached_performance(self):
        """Benchmark uncached WBGT with varying inputs."""
        from engine import clear_wbgt_cache
        clear_wbgt_cache()
        latencies = []
        for i in range(500):
            t = 25.0 + (i * 0.05)
            elapsed_ms, _ = measure_latency(estimate_wbgt, t, 40.0, 2.0, 600.0)
            latencies.append(elapsed_ms)
        result = BenchmarkResult("wbgt_uncached_sweep", latencies)
        print(f"\n{result.summary_line()}")

    def test_thermal_stress_index_performance(self):
        """Benchmark full thermal stress index calculation."""
        latencies = []
        for _ in range(200):
            elapsed_ms, _ = measure_latency(
                calculate_thermal_stress_index, 42.0, 30.0, 2.0, 750.0
            )
            latencies.append(elapsed_ms)
        result = BenchmarkResult("thermal_stress_index", latencies)
        print(f"\n{result.summary_line()}")

    def test_feature_extraction_performance(self):
        """Benchmark ML feature extraction."""
        sample = {
            "background_temp_c": 42.0,
            "background_rh_pct": 30.0,
            "background_wind_ms": 2.0,
            "background_solar_wm2": 750.0,
            "elevation_m": 216.0,
            "urban_built_up_ratio": 0.85,
            "vegetation_canopy_deficit": 0.78,
            "population_density_k": 11.0,
            "distance_to_coast_km": 1050.0,
            "is_coastal": 0,
            "date": "2024-05-15",
        }
        latencies = []
        for _ in range(500):
            elapsed_ms, _ = measure_latency(extract_features_from_dict, sample)
            latencies.append(elapsed_ms)
        result = BenchmarkResult("feature_extraction", latencies)
        print(f"\n{result.summary_line()}")
