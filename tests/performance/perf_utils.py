"""
HeatWave Guardian — Performance Test Utilities & Shared Benchmark Infrastructure
=================================================================================
Shared helpers, synthetic benchmark payloads, timing decorators, and statistics
collectors used across all performance/load tests.

ALL PAYLOADS IN THIS MODULE ARE SYNTHETIC BENCHMARK INPUTS ONLY.
They are labelled as such and must NEVER be used as scientific training/validation data.
"""

import time
import json
import math
import statistics
import os
import sys
import tracemalloc
from typing import Dict, Any, List, Optional, Callable, Tuple
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark Timing & Statistics
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkResult:
    """Collects and computes statistics from a sequence of latency samples."""

    def __init__(self, name: str, latencies_ms: List[float], errors: int = 0, timeouts: int = 0):
        self.name = name
        self.latencies_ms = sorted(latencies_ms)
        self.errors = errors
        self.timeouts = timeouts
        self.total = len(latencies_ms) + errors + timeouts

    @property
    def success_count(self) -> int:
        return len(self.latencies_ms)

    @property
    def error_rate(self) -> float:
        return (self.errors / self.total * 100.0) if self.total > 0 else 0.0

    @property
    def timeout_rate(self) -> float:
        return (self.timeouts / self.total * 100.0) if self.total > 0 else 0.0

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def requests_per_sec(self) -> float:
        if not self.latencies_ms:
            return 0.0
        total_duration_s = sum(self.latencies_ms) / 1000.0
        if total_duration_s <= 0:
            return 0.0
        return len(self.latencies_ms) / total_duration_s

    def _percentile(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        k = (pct / 100.0) * (len(self.latencies_ms) - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return self.latencies_ms[int(k)]
        return self.latencies_ms[f] * (c - k) + self.latencies_ms[c] * (k - f)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_requests": self.total,
            "successful": self.success_count,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "error_rate_pct": round(self.error_rate, 2),
            "timeout_rate_pct": round(self.timeout_rate, 2),
            "avg_ms": round(self.avg_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "throughput_rps": round(self.requests_per_sec, 2),
        }

    def summary_line(self) -> str:
        return (
            f"[{self.name}] n={self.total} ok={self.success_count} err={self.errors} "
            f"avg={self.avg_ms:.1f}ms med={self.median_ms:.1f}ms "
            f"p95={self.p95_ms:.1f}ms p99={self.p99_ms:.1f}ms max={self.max_ms:.1f}ms "
            f"err%={self.error_rate:.1f}%"
        )


def measure_latency(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """Execute func and return (latency_ms, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result


def run_sequential_benchmark(
    name: str,
    func: Callable,
    iterations: int,
    args_list: Optional[List[tuple]] = None,
    kwargs_list: Optional[List[dict]] = None,
    timeout_ms: float = 30000.0,
) -> BenchmarkResult:
    """Run func sequentially N times, collecting latencies."""
    latencies = []
    errors = 0
    timeouts = 0
    for i in range(iterations):
        a = args_list[i % len(args_list)] if args_list else ()
        kw = kwargs_list[i % len(kwargs_list)] if kwargs_list else {}
        try:
            elapsed_ms, _ = measure_latency(func, *a, **kw)
            if elapsed_ms > timeout_ms:
                timeouts += 1
            else:
                latencies.append(elapsed_ms)
        except Exception:
            errors += 1
    return BenchmarkResult(name, latencies, errors, timeouts)


def run_concurrent_benchmark(
    name: str,
    func: Callable,
    total_requests: int,
    concurrency: int,
    args_list: Optional[List[tuple]] = None,
    kwargs_list: Optional[List[dict]] = None,
    timeout_ms: float = 30000.0,
) -> BenchmarkResult:
    """Run func with thread-pool concurrency, collecting latencies."""
    latencies = []
    errors = 0
    timeouts = 0

    def _worker(idx):
        a = args_list[idx % len(args_list)] if args_list else ()
        kw = kwargs_list[idx % len(kwargs_list)] if kwargs_list else {}
        return measure_latency(func, *a, **kw)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_worker, i): i for i in range(total_requests)}
        for future in as_completed(futures):
            try:
                elapsed_ms, _ = future.result(timeout=timeout_ms / 1000.0)
                if elapsed_ms > timeout_ms:
                    timeouts += 1
                else:
                    latencies.append(elapsed_ms)
            except Exception:
                errors += 1

    return BenchmarkResult(name, latencies, errors, timeouts)


def measure_memory(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """Execute func and return (peak_memory_mb, result)."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024), result


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Benchmark Payloads  (BENCHMARK INPUTS ONLY — NOT SCIENTIFIC DATA)
# ──────────────────────────────────────────────────────────────────────────────

# Representative Indian city benchmark locations
BENCHMARK_LOCATIONS = [
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "region": "plains"},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "region": "plains"},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "region": "coastal"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "region": "coastal"},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "region": "plains"},
    {"name": "Jaisalmer", "lat": 26.9157, "lon": 70.9083, "region": "plains"},
    {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882, "region": "plains"},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "region": "plains"},
    {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "region": "coastal"},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "region": "plains"},
]

# Synthetic weather conditions for benchmark inputs (not real observation data)
BENCHMARK_WEATHER_CONDITIONS = [
    {"label": "BENCHMARK_Hot_Dry",   "temp_c": 44.0, "rh_pct": 18.0, "wind_ms": 1.5, "solar_wm2": 850.0},
    {"label": "BENCHMARK_Humid_Hot", "temp_c": 37.0, "rh_pct": 75.0, "wind_ms": 2.0, "solar_wm2": 600.0},
    {"label": "BENCHMARK_Moderate",  "temp_c": 33.0, "rh_pct": 50.0, "wind_ms": 3.5, "solar_wm2": 500.0},
    {"label": "BENCHMARK_Evening",   "temp_c": 30.0, "rh_pct": 60.0, "wind_ms": 2.5, "solar_wm2": 100.0},
    {"label": "BENCHMARK_Night",     "temp_c": 28.0, "rh_pct": 65.0, "wind_ms": 1.8, "solar_wm2": 0.0},
]

def make_check_risk_payload(location_idx: int = 0, weather_idx: int = 0) -> Dict[str, Any]:
    """Generate a synthetic POST payload for /api/check-risk benchmarking."""
    loc = BENCHMARK_LOCATIONS[location_idx % len(BENCHMARK_LOCATIONS)]
    w = BENCHMARK_WEATHER_CONDITIONS[weather_idx % len(BENCHMARK_WEATHER_CONDITIONS)]
    return {
        "location": loc["name"],
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "age": 35,
        "workload": "moderate",
        "acclimatization": "acclimatized",
        "clothing": "standard_work",
        "is_sun_exposed": True,
        "_benchmark_label": f"SYNTHETIC_BENCHMARK_INPUT_{loc['name']}_{w['label']}",
    }


def make_forecast_params(location_idx: int = 0) -> Dict[str, str]:
    """Generate synthetic GET params for /api/forecast benchmarking."""
    loc = BENCHMARK_LOCATIONS[location_idx % len(BENCHMARK_LOCATIONS)]
    return {
        "location": loc["name"],
        "lat": str(loc["lat"]),
        "lon": str(loc["lon"]),
    }


def make_whatif_payload(temp_c: float = 42.0, rh_pct: float = 30.0) -> Dict[str, Any]:
    """Generate a synthetic What-If scenario payload for benchmarking."""
    return {
        "temperature_c": temp_c,
        "relative_humidity_pct": rh_pct,
        "wind_speed_ms": 2.0,
        "solar_radiation_wm2": 750.0,
        "location": "Delhi",
        "workload": "heavy",
        "acclimatization": "partially",
        "clothing": "standard_work",
        "exposure_duration_hours": 4.0,
        "_benchmark_label": "SYNTHETIC_BENCHMARK_WHAT_IF",
    }


def make_ml_weather_dict(weather_idx: int = 0) -> Dict[str, Any]:
    """Generate a weather dictionary for ML inference benchmarking."""
    w = BENCHMARK_WEATHER_CONDITIONS[weather_idx % len(BENCHMARK_WEATHER_CONDITIONS)]
    return {
        "temp_c": w["temp_c"],
        "rh_pct": w["rh_pct"],
        "wind_ms": w["wind_ms"],
        "solar_wm2": w["solar_wm2"],
    }


def make_alert_payload(risk_level: str = "High", role: str = "worker") -> Dict[str, Any]:
    """Generate a synthetic alert request payload for benchmarking."""
    return {
        "role": role,
        "risk_level": risk_level,
        "location": "Delhi",
        "wbgt_c": 33.5,
        "language": "en",
        "channel": "dashboard",
        "temp_c": 44.0,
        "_benchmark_label": "SYNTHETIC_BENCHMARK_ALERT",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Flask Test Client Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_test_client():
    """Create a Flask test client for benchmarking without starting a server."""
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


# ──────────────────────────────────────────────────────────────────────────────
# Report Helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as a readable text table."""
    lines = []
    lines.append(f"{'Benchmark':<45} {'N':>5} {'Avg(ms)':>10} {'Med(ms)':>10} "
                 f"{'P95(ms)':>10} {'P99(ms)':>10} {'Max(ms)':>10} {'Err%':>7} {'RPS':>8}")
    lines.append("-" * 120)
    for r in results:
        lines.append(
            f"{r.name:<45} {r.total:>5} {r.avg_ms:>10.2f} {r.median_ms:>10.2f} "
            f"{r.p95_ms:>10.2f} {r.p99_ms:>10.2f} {r.max_ms:>10.2f} "
            f"{r.error_rate:>6.1f}% {r.requests_per_sec:>8.1f}"
        )
    return "\n".join(lines)
