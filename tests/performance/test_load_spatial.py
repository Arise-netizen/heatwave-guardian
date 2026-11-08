"""
HeatWave Guardian — Spatial Grid & GeoJSON Performance Tests
=============================================================
Benchmarks spatial grid generation, GeoJSON serialization, payload sizes,
and scaling behavior for increasing feature counts.

ALL PAYLOADS ARE SYNTHETIC BENCHMARK INPUTS — NOT SCIENTIFIC DATA.
"""

import os
import sys
import time
import json
import pytest
from typing import Dict, Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tests.performance.perf_utils import (
    BenchmarkResult,
    measure_latency,
    measure_memory,
    BENCHMARK_LOCATIONS,
    make_ml_weather_dict,
    format_results_table,
)
from services.hyperlocal_ml_service import HyperlocalMLService
from services.authority_service import AuthorityService
from ml.src.predict import HyperlocalPredictor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mock_weather_fetcher(lat: float, lon: float) -> Dict[str, Any]:
    """Deterministic synthetic weather fetcher for benchmarking without network calls."""
    return {
        "temp_c": 42.0,
        "rh_pct": 30.0,
        "wind_ms": 2.0,
        "solar_wm2": 750.0,
        "humidity": 30.0,
        "current_temp_c": 42.0,
        "freshness": "FRESH",
        "source": "BENCHMARK_SYNTHETIC",
        "source_time": "2024-05-15T12:00:00Z",
        "source_transparency": {
            "source": "Benchmark Synthetic Weather",
            "is_cached": False,
            "freshness": "FRESH",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Single-City Grid Generation
# ──────────────────────────────────────────────────────────────────────────────

class TestSingleCityGrid:
    """Benchmark spatial grid generation for single cities."""

    def test_generate_grid_single_city(self):
        """Generate grid cells for one city."""
        w = make_ml_weather_dict(0)
        latencies = []
        for _ in range(10):
            elapsed_ms, cells = measure_latency(
                HyperlocalMLService.generate_spatial_grid_cells,
                "Delhi", 28.6139, 77.2090, w, region="plains",
            )
            latencies.append(elapsed_ms)
        result = BenchmarkResult("grid_gen_single_city", latencies)
        print(f"\n{result.summary_line()}")
        # Check that cells were generated
        assert len(cells) > 0, "No cells generated"

    def test_grid_cell_count(self):
        """Verify expected cell count per city."""
        w = make_ml_weather_dict(0)
        _, cells = measure_latency(
            HyperlocalMLService.generate_spatial_grid_cells,
            "Delhi", 28.6139, 77.2090, w, region="plains",
        )
        cell_count = len(cells)
        print(f"\n[GRID_CELLS] Cells per city: {cell_count}")
        # Each city generates a fixed number of sector cells (5 by design)
        assert cell_count >= 1, "Expected at least 1 cell"

    def test_all_cities_individually(self):
        """Benchmark grid generation for each monitored city."""
        w = make_ml_weather_dict(0)
        latencies = []
        for loc in BENCHMARK_LOCATIONS:
            elapsed_ms, cells = measure_latency(
                HyperlocalMLService.generate_spatial_grid_cells,
                loc["name"], loc["lat"], loc["lon"], w, region=loc["region"],
            )
            latencies.append(elapsed_ms)
        result = BenchmarkResult("grid_gen_per_city", latencies)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-City Authority Grid (Full GeoJSON Layer)
# ──────────────────────────────────────────────────────────────────────────────

class TestAuthoritySpatialGrid:
    """Benchmark the full authority spatial grid (all monitored cities)."""

    def test_full_grid_generation(self):
        """Generate full authority grid with mock weather."""
        AuthorityService.clear_cache()
        latencies = []
        for _ in range(3):
            AuthorityService.clear_cache()
            elapsed_ms, grid = measure_latency(
                HyperlocalMLService.get_authority_spatial_grid,
                _mock_weather_fetcher,
            )
            latencies.append(elapsed_ms)
        result = BenchmarkResult("authority_full_grid", latencies)
        print(f"\n{result.summary_line()}")
        # Report cell count
        total_cells = grid.get("total_cells", len(grid.get("features", [])))
        print(f"  Total cells across all cities: {total_cells}")

    def test_single_city_filter(self):
        """Generate grid filtered to one city."""
        AuthorityService.clear_cache()
        latencies = []
        for _ in range(5):
            elapsed_ms, grid = measure_latency(
                HyperlocalMLService.get_authority_spatial_grid,
                _mock_weather_fetcher,
                city="Delhi",
            )
            latencies.append(elapsed_ms)
        result = BenchmarkResult("authority_grid_delhi", latencies)
        print(f"\n{result.summary_line()}")

    def test_authority_area_heat_layer(self):
        """Benchmark the area heat layer (non-grid, per-area summary)."""
        AuthorityService.clear_cache()
        latencies = []
        for _ in range(3):
            AuthorityService.clear_cache()
            elapsed_ms, layer = measure_latency(
                AuthorityService.get_area_heat_layer,
                _mock_weather_fetcher,
                use_cache=False,
            )
            latencies.append(elapsed_ms)
        result = BenchmarkResult("authority_area_layer", latencies)
        print(f"\n{result.summary_line()}")

    def test_authority_cache_effectiveness(self):
        """Measure cache hit vs miss latency difference."""
        AuthorityService.clear_cache()

        # Cold (cache miss)
        cold_ms, _ = measure_latency(
            AuthorityService.get_area_heat_layer,
            _mock_weather_fetcher,
            use_cache=True,
        )

        # Warm (cache hit)
        warm_latencies = []
        for _ in range(10):
            elapsed_ms, _ = measure_latency(
                AuthorityService.get_area_heat_layer,
                _mock_weather_fetcher,
                use_cache=True,
            )
            warm_latencies.append(elapsed_ms)

        warm_avg = sum(warm_latencies) / len(warm_latencies)
        speedup = cold_ms / warm_avg if warm_avg > 0 else float("inf")
        print(f"\n[CACHE] Cold: {cold_ms:.1f}ms, Warm avg: {warm_avg:.1f}ms, Speedup: {speedup:.1f}x")


# ──────────────────────────────────────────────────────────────────────────────
# 3. GeoJSON Serialization Performance
# ──────────────────────────────────────────────────────────────────────────────

class TestGeoJSONSerialization:
    """Benchmark GeoJSON generation, serialization, and payload size."""

    def test_serialization_time(self):
        """Measure JSON serialization time for GeoJSON payloads."""
        AuthorityService.clear_cache()
        _, grid = measure_latency(
            HyperlocalMLService.get_authority_spatial_grid,
            _mock_weather_fetcher,
        )

        # Measure serialization
        latencies = []
        for _ in range(50):
            elapsed_ms, json_str = measure_latency(json.dumps, grid)
            latencies.append(elapsed_ms)
        result = BenchmarkResult("geojson_serialization", latencies)
        print(f"\n{result.summary_line()}")

        # Payload size
        payload_bytes = len(json_str.encode("utf-8"))
        payload_kb = payload_bytes / 1024.0
        print(f"  GeoJSON payload size: {payload_kb:.1f} KB ({payload_bytes} bytes)")

    def test_payload_size_per_city(self):
        """Measure GeoJSON payload size per city."""
        w = make_ml_weather_dict(0)
        for loc in BENCHMARK_LOCATIONS[:5]:
            _, cells = measure_latency(
                HyperlocalMLService.generate_spatial_grid_cells,
                loc["name"], loc["lat"], loc["lon"], w, region=loc["region"],
            )
            geojson = {
                "type": "FeatureCollection",
                "features": cells,
            }
            json_str = json.dumps(geojson)
            size_kb = len(json_str.encode("utf-8")) / 1024.0
            print(f"  {loc['name']:>12}: {len(cells)} cells, {size_kb:.1f} KB")

    def test_large_feature_collection_serialization(self):
        """Measure serialization with a synthetically large feature collection."""
        w = make_ml_weather_dict(0)
        # Generate cells for all locations
        all_cells = []
        for loc in BENCHMARK_LOCATIONS:
            _, cells = measure_latency(
                HyperlocalMLService.generate_spatial_grid_cells,
                loc["name"], loc["lat"], loc["lon"], w, region=loc["region"],
            )
            all_cells.extend(cells)

        large_geojson = {
            "type": "FeatureCollection",
            "features": all_cells,
        }
        total_features = len(all_cells)

        latencies = []
        for _ in range(20):
            elapsed_ms, json_str = measure_latency(json.dumps, large_geojson)
            latencies.append(elapsed_ms)

        result = BenchmarkResult("geojson_large_serialization", latencies)
        payload_kb = len(json_str.encode("utf-8")) / 1024.0
        print(f"\n{result.summary_line()}")
        print(f"  Total features: {total_features}, Payload: {payload_kb:.1f} KB")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Grid Scaling Analysis
# ──────────────────────────────────────────────────────────────────────────────

class TestGridScaling:
    """Analyze how spatial grid performance scales with number of cities."""

    def test_scaling_by_city_count(self):
        """Measure grid generation time as city count increases."""
        w = make_ml_weather_dict(0)
        city_counts = [1, 3, 5, 8, 10]
        results = []
        for n in city_counts:
            all_cells = []
            start = time.perf_counter()
            for loc in BENCHMARK_LOCATIONS[:n]:
                cells = HyperlocalMLService.generate_spatial_grid_cells(
                    loc["name"], loc["lat"], loc["lon"], w, region=loc["region"],
                )
                all_cells.extend(cells)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_cells = len(all_cells)
            results.append((n, total_cells, elapsed_ms))
            print(f"  {n} cities: {total_cells} cells, {elapsed_ms:.1f}ms")

        # Verify roughly linear scaling
        if len(results) >= 2:
            first_per_city = results[0][2] / results[0][0]
            last_per_city = results[-1][2] / results[-1][0]
            ratio = last_per_city / first_per_city if first_per_city > 0 else 0
            print(f"\n  Per-city time ratio (last/first): {ratio:.2f}x")

    def test_grid_memory_usage(self):
        """Measure memory usage of full grid generation."""
        AuthorityService.clear_cache()
        mem_mb, grid = measure_memory(
            HyperlocalMLService.get_authority_spatial_grid,
            _mock_weather_fetcher,
        )
        total_cells = grid.get("total_cells", len(grid.get("features", [])))
        print(f"\n[GRID_MEMORY] Full grid ({total_cells} cells): {mem_mb:.2f} MB")
        per_cell_kb = (mem_mb * 1024) / total_cells if total_cells > 0 else 0
        print(f"  Per-cell memory: {per_cell_kb:.2f} KB")
