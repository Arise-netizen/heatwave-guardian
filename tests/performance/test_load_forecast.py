"""
HeatWave Guardian — Forecast Engine Performance & Scalability Tests
====================================================================
Benchmarks multi-day forecast generation, per-location forecast time,
hourly profile computation, ML forecast enhancement, and concurrent
forecast workloads.

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
from engine import (
    estimate_wbgt,
    classify_wbgt_band,
    classify_thermal_stress,
    classify_imd,
    fuse_risk,
    assess_vulnerability,
    escalate_risk,
    detect_peak_risk_windows,
    calculate_hourly_risk_profile,
    build_exposure_context,
    get_work_rest,
    evaluate_data_quality,
    generate_structured_explanation,
    build_decision_timeline,
)
from services.hyperlocal_ml_service import HyperlocalMLService
from ml.src.predict import HyperlocalPredictor


# ──────────────────────────────────────────────────────────────────────────────
# Test Data Generators
# ──────────────────────────────────────────────────────────────────────────────

def _make_hourly_profile(date_str: str = "2024-05-15") -> List[Dict[str, Any]]:
    """Generate a synthetic 24-hour diurnal profile for benchmarking."""
    hours = []
    for h in range(24):
        # Sinusoidal temperature pattern peaking at 15:00
        base_t = 30.0 + 12.0 * max(0, (1 - abs(h - 15) / 9.0))
        rh = max(15, min(85, 70 - base_t * 0.5))
        solar = max(0, 800 * max(0, 1 - abs(h - 12) / 7.0))
        wind = 1.5 + 0.5 * abs(h - 12) / 12.0
        hours.append({
            "time": f"{date_str}T{h:02d}:00",
            "hour": h,
            "temp_c": round(base_t, 1),
            "rh_pct": round(rh, 1),
            "wind_ms": round(wind, 1),
            "solar_wm2": round(solar, 1),
        })
    return hours


def _make_daily_forecasts(n_days: int = 5) -> List[Dict[str, Any]]:
    """Generate n days of synthetic daily forecast data for benchmarking."""
    days = []
    for d in range(n_days):
        date_str = f"2024-05-{15 + d:02d}"
        hourly = _make_hourly_profile(date_str)
        max_temp = max(h["temp_c"] for h in hourly)
        peak_wbgt = estimate_wbgt(max_temp, 30.0, 2.0, 800.0)
        days.append({
            "date": date_str,
            "max_temp_c": max_temp,
            "peak_wbgt_c": peak_wbgt,
            "hourly": hourly,
            "hourly_profile": hourly,
        })
    return days


# ──────────────────────────────────────────────────────────────────────────────
# 1. Forecast Pipeline Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

class TestForecastPipeline:
    """Benchmark the internal forecast processing pipeline."""

    def test_peak_risk_window_detection(self):
        """Benchmark peak risk window detection on 24h profiles."""
        hourly = _make_hourly_profile()
        latencies = []
        for _ in range(200):
            elapsed_ms, result = measure_latency(detect_peak_risk_windows, hourly)
            latencies.append(elapsed_ms)
        result_bench = BenchmarkResult("peak_risk_detection", latencies)
        print(f"\n{result_bench.summary_line()}")

    def test_hourly_risk_profile_computation(self):
        """Benchmark hourly risk profile computation."""
        hourly = _make_hourly_profile()
        parsed = {
            "age": 35, "age_group": "adult", "acclimatization": "acclimatized",
            "experience": "acclimatized", "workload": "moderate",
            "clothing": "standard_work",
        }
        latencies = []
        for _ in range(50):
            elapsed_ms, profile = measure_latency(
                calculate_hourly_risk_profile,
                hourly_points=hourly,
                region="plains",
                personal_inputs=parsed,
                location="Delhi",
            )
            latencies.append(elapsed_ms)
        result = BenchmarkResult("hourly_risk_profile", latencies)
        print(f"\n{result.summary_line()}")

    def test_full_day_pipeline(self):
        """Benchmark the full single-day forecast pipeline."""
        hourly = _make_hourly_profile()
        max_temp = 42.0
        wbgt_c = estimate_wbgt(max_temp, 30.0, 2.0, 800.0)
        parsed = {
            "age": 35, "age_group": "adult", "acclimatization": "acclimatized",
            "experience": "acclimatized", "workload": "moderate",
            "clothing": "standard_work", "is_sun_exposed": True,
            "duration_hours": 8, "exposure_pattern": "continuous",
            "ppe_category": None, "recovery_environment": "shaded_outdoor",
            "day_in_heat": 5,
        }

        def _full_day():
            ts = classify_thermal_stress(wbgt_c)
            imd = classify_imd(max_temp, "plains")
            base_risk = fuse_risk(ts["wbgt_band"], imd)
            vuln = assess_vulnerability(35, "acclimatized", "moderate", "standard_work", "adult")
            personal_risk = escalate_risk(base_risk, vuln)
            peak = detect_peak_risk_windows(hourly)
            profile = calculate_hourly_risk_profile(hourly, "plains", parsed, "Delhi")
            exp_ctx = build_exposure_context(
                wbgt_c=wbgt_c,
                environmental_risk_level=base_risk,
                wind_ms=2.0,
                solar_wm2=800.0,
                is_direct_sun=True,
                workload="moderate",
                exposure_duration_hours=8,
                exposure_pattern="continuous",
                acclimatization="acclimatized",
                clothing="standard_work",
                ppe_category=None,
                recovery_environment="shaded_outdoor",
            )
            work_rest = get_work_rest(
                personal_risk, workload="moderate", acclimatized="acclimatized",
                clothing="standard_work", recovery_environment="shaded_outdoor",
                exposure_duration_hours=8, wbgt_c=wbgt_c, is_direct_sun=True,
            )
            conf = evaluate_data_quality(
                {"temp_c": max_temp, "rh_pct": 30.0, "wind_ms": 2.0, "solar_wm2": 800.0},
                True, "Open-Meteo", "Delhi"
            )
            explanation = generate_structured_explanation(
                "Delhi", max_temp, wbgt_c, ts["wbgt_band"], imd, base_risk,
                personal_risk, vuln, "adult", "acclimatized", "moderate",
                "standard_work", {}, conf,
            )
            timeline = build_decision_timeline(profile)
            return {"risk": personal_risk, "timeline": timeline}

        latencies = []
        for _ in range(30):
            elapsed_ms, _ = measure_latency(_full_day)
            latencies.append(elapsed_ms)
        result = BenchmarkResult("full_day_pipeline", latencies)
        print(f"\n{result.summary_line()}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. ML Forecast Enhancement Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

class TestMLForecastEnhancement:
    """Benchmark ML-enhanced forecast generation."""

    def test_enhance_single_day(self):
        """Benchmark ML enhancement of a single day's hourly forecast."""
        days = _make_daily_forecasts(1)
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        latencies = []
        for _ in range(5):
            elapsed_ms, result = measure_latency(
                HyperlocalMLService.enhance_forecast_with_ml,
                days, 28.6139, 77.2090, "Delhi", "plains",
            )
            latencies.append(elapsed_ms)
        result_bench = BenchmarkResult("ml_enhance_1_day", latencies)
        print(f"\n{result_bench.summary_line()}")

    def test_enhance_5_day_forecast(self):
        """Benchmark ML enhancement of a 5-day forecast."""
        days = _make_daily_forecasts(5)
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        latencies = []
        for _ in range(3):
            elapsed_ms, result = measure_latency(
                HyperlocalMLService.enhance_forecast_with_ml,
                days, 28.6139, 77.2090, "Delhi", "plains",
            )
            latencies.append(elapsed_ms)
        result_bench = BenchmarkResult("ml_enhance_5_day", latencies)
        print(f"\n{result_bench.summary_line()}")
        # 5 days * 24 hours = 120 ML predictions — should be feasible
        total_hours = 5 * 24
        avg_per_hour = result_bench.avg_ms / total_hours if total_hours > 0 else 0
        print(f"  Total hours: {total_hours}, Avg per hour: {avg_per_hour:.2f}ms")

    def test_enhance_multi_location(self):
        """Benchmark ML-enhanced forecasts across multiple locations."""
        days = _make_daily_forecasts(3)
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        latencies = []
        for loc in BENCHMARK_LOCATIONS[:5]:
            elapsed_ms, result = measure_latency(
                HyperlocalMLService.enhance_forecast_with_ml,
                days, loc["lat"], loc["lon"], loc["name"], loc["region"],
            )
            latencies.append(elapsed_ms)
        result_bench = BenchmarkResult("ml_enhance_multi_loc", latencies)
        print(f"\n{result_bench.summary_line()}")

    def test_forecast_enhancement_memory(self):
        """Measure memory impact of forecast ML enhancement."""
        days = _make_daily_forecasts(5)
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        mem_mb, result = measure_memory(
            HyperlocalMLService.enhance_forecast_with_ml,
            days, 28.6139, 77.2090, "Delhi", "plains",
        )
        print(f"\n[FORECAST_MEMORY] 5-day ML enhancement peak memory: {mem_mb:.2f} MB")
        # Should not consume more than 100 MB
        assert mem_mb < 100, f"Forecast enhancement used {mem_mb:.1f}MB (>100MB threshold)"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Forecast Scaling Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestForecastScaling:
    """Verify forecast generation scales acceptably."""

    @pytest.mark.parametrize("n_days", [1, 3, 5, 7])
    def test_forecast_scaling_by_days(self, n_days):
        """Measure how forecast computation scales with number of days."""
        days = _make_daily_forecasts(n_days)
        model = HyperlocalPredictor.get_model()
        if model is None:
            pytest.skip("ML model unavailable")

        elapsed_ms, result = measure_latency(
            HyperlocalMLService.enhance_forecast_with_ml,
            days, 28.6139, 77.2090, "Delhi", "plains",
        )
        per_day = elapsed_ms / n_days
        print(f"\n[FORECAST_SCALE] {n_days} days: total={elapsed_ms:.1f}ms per_day={per_day:.1f}ms")
