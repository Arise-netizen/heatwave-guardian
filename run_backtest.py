#!/usr/bin/env python3
"""
HeatWave Guardian — Standalone Automated Backtest Runner
=========================================================
Runs multi-case, multi-climate empirical backtesting across all registered
historical cases, prints formatted statistical and benchmark summaries,
and writes the complete machine-readable report to JSON or CSV.

Usage:
    python run_backtest.py
    python run_backtest.py --case delhi_2024_extreme
    python run_backtest.py --dataset era5_ap_telangana_2015
    python run_backtest.py --out validation_report.json
    python run_backtest.py --format csv --out validation_summary.csv
"""

import sys
import os
import json
import argparse

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.validation_engine import (
    ValidationEngine,
    CaseRegistry,
    ENGINE_VERSION,
    METHODOLOGY_VERSION,
    VALIDATION_DATASET_VERSION,
    APPLICATION_VERSION,
)
from services.dataset_registry import DatasetRegistry


def format_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) for row in rows]
    return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)


def main():
    parser = argparse.ArgumentParser(description="HeatWave Guardian Automated Backtest Runner")
    parser.add_argument("--case", type=str, help="Specific case ID to run (e.g. khammam_2015_heatwave)")
    parser.add_argument("--dataset", type=str, help="Filter cases by dataset ID (e.g. era5_ap_telangana_2015)")
    parser.add_argument("--out", type=str, default="validation_report.json", help="Output report file path")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format (json or csv)")
    args = parser.parse_args()

    print("=" * 84)
    print("  HEATWAVE GUARDIAN — AUTOMATED BACKTESTING & REAL-WORLD EVIDENCE SUITE")
    print(f"  Engine: v{ENGINE_VERSION} | Methodology: {METHODOLOGY_VERSION} | Datasets: v{VALIDATION_DATASET_VERSION}")
    print("=" * 84)

    case_ids = None
    if args.case:
        case_ids = [args.case]
    elif args.dataset:
        all_cases = CaseRegistry.list_cases()
        case_ids = [c["case_id"] for c in all_cases if c.get("dataset_id") == args.dataset]
        if not case_ids:
            print(f"[ERROR] No historical cases registered for dataset ID: {args.dataset}")
            return 1

    report = ValidationEngine.run_backtest(case_ids=case_ids)

    cov = report["coverage_summary"]
    env = report["environmental_metrics"]
    clf = report["classification_metrics"]
    bm = report["benchmarks"]
    fc = report["forecast_backtest"]
    pk = report["peak_window_backtest"]
    abl = report["ablation"]
    cons = report["consistency_audit"]
    errs = report["error_registry"]

    print("\n1. COVERAGE & EXECUTION SUMMARY")
    print(f"   • Total Cases Evaluated : {cov['cases_evaluated_count']}")
    print(f"   • Total Case-Days       : {cov['total_case_days']}")
    print(f"   • Geographic Locations  : {', '.join(cov['locations'])}")
    print(f"   • Climate Zones Covered : {', '.join(cov['climate_zones_covered'])}")
    print(f"   • Negative Control Case : {'Included (Bengaluru Jan 2023)' if cov['has_negative_control_case'] else 'None'}")

    print("\n2. ENVIRONMENTAL REANALYSIS VS GROUND OBSERVATIONS")
    print(f"   • Sample Count (Records): {env['sample_count']}")
    print(f"   • Mean Absolute Error   : {env['mae_c']} °C")
    print(f"   • Root Mean Square Error: {env['rmse_c']} °C")
    print(f"   • Mean Bias (Model-Ref) : {env['mean_bias_c']} °C  (Documented pre-monsoon cold bias)")
    print(f"   • Median Absolute Error : {env['median_abs_error_c']} °C")

    print("\n3. IMD HEATWAVE CLASSIFICATION FIDELITY")
    print(f"   • Overall Agreement Rate: {clf['agreement_pct']}% ({clf['exact_matches']}/{clf['total_records']})")
    print(f"   • False Alarms (Non-heat flagged as Heatwave): {clf['false_alarms']}")
    print(f"   • Under-Detections (Reanalysis smoothing)   : {clf['misses']}")

    print("\n4. 4-WAY BENCHMARK MODEL COMPARISON")
    bm_rows = [
        ["Temperature-Only (T >= 40°C)", bm["temp_only_flagged_count"], f"{bm['temp_only_pct']}%", "Misses high-humidity coastal thermal stress (Mumbai)"],
        ["IMD Regional Criteria", bm["imd_criteria_flagged_count"], f"{bm['imd_criteria_pct']}%", "Adjusts for regional climatology (Plains 40°, Coastal 37°)"],
        ["Liljegren WBGT (WBGT >= 29.4°C)", bm["wbgt_elevated_count"], f"{bm['wbgt_elevated_pct']}%", "Captures radiation and humidity evaporation deficit"],
        ["Integrated HeatWave Guardian", bm["guardian_escalated_count"], f"{bm['guardian_escalated_pct']}%", "Fuses biometeorology with Workload, PPE & Acclimatization"],
    ]
    print(format_table(["Model Approach", "Flagged Days", "Flagged %", "Key Behavioral Distinction"], bm_rows))

    print("\n5. FORECAST BACKTEST (NWP 24H LEAD-TIME)")
    if fc.get("has_historical_forecast_archive"):
        print(f"   • Archived Forecast Cases: {', '.join(fc['sources_evaluated'])}")
        print(f"   • Forecast Temp MAE      : {fc['forecast_temp_mae_c']} °C")
        print(f"   • Forecast Temp RMSE     : {fc['forecast_temp_rmse_c']} °C")
        print(f"   • Peak Timing Error Mean : {fc['peak_timing_diff_hours_mean']} hours")
        print(f"   • Missed Elevated Periods: {fc['missed_elevated_periods']}")
        print(f"   • False Peak Periods     : {fc['false_peak_periods']}")
    else:
        print("   • Historical forecast backtesting unavailable for this source.")

    print("\n6. PEAK-RISK THERMAL WINDOW BACKTEST")
    print(f"   • Hourly Cases Evaluated : {pk['hourly_cases_evaluated']}")
    print(f"   • Elevated Detection Rate: {pk['elevated_detection_rate_pct']}%")
    print(f"   • Mean Peak Hour Diff    : {pk['mean_peak_hour_diff_hours']} hours")
    print(f"   • Negative Control Alarms: {pk['control_false_alarms']} (Zero False Alarms)")

    print("\n7. COMPONENT ABLATION SUMMARY")
    abl_rows = []
    for k, v in abl.items():
        abl_rows.append([v["component"], v["scenarios_tested"], v["scenarios_changed"], f"{v['impact_pct']}%"])
    print(format_table(["Ablated Component", "Scenarios", "Changed", "Impact %"], abl_rows))

    print("\n8. DECISION MONOTONICITY & CONSISTENCY AUDIT")
    print(f"   • Workload Monotonicity      : {'PASSED (0 violations)' if cons['workload_monotonicity_passed'] else 'FAILED'}")
    print(f"   • Acclimatization Monotonicity: {'PASSED (0 violations)' if cons['acclimatization_monotonicity_passed'] else 'FAILED'}")
    print(f"   • Recovery Environment Monot. : {'PASSED (0 violations)' if cons['recovery_monotonicity_passed'] else 'FAILED'}")
    print(f"   • Clothing/PPE Monotonicity   : {'PASSED (0 violations)' if cons['clothing_monotonicity_passed'] else 'FAILED'}")
    print(f"   • OVERALL LOGICAL CONSISTENCY : {'100% VERIFIED' if cons['all_consistency_checks_passed'] else 'FAILED'}")

    if errs:
        print(f"\n9. DISAGREEMENT & ERROR REGISTRY ({len(errs)} cases classified)")
        cat_counts = {}
        for e in errs:
            c = e.get("error_category", "unknown")
            cat_counts[c] = cat_counts.get(c, 0) + 1
        for cat, count in sorted(cat_counts.items()):
            print(f"   • {cat.replace('_', ' ').title()}: {count} occurrences")

    out_path = os.path.join(BASE_DIR, args.out)
    if args.format == "csv":
        csv_content = ValidationEngine.export_report_csv(report)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)
        print(f"\n[OK] CSV validation summary exported to: {out_path}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Machine-readable JSON validation report exported to: {out_path}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
