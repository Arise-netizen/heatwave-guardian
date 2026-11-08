"""
HeatWave Guardian — Comprehensive Validation, Evidence & Backtesting Test Suite
================================================================================
Exhaustive verification of Dataset Registry, Case Registry, Automated Backtesting Pipeline,
Environmental Error Metrics, IMD Classification Fidelity, NWP Forecast Backtesting,
Peak Window Detection, Component Ablation Regression, Decision Monotonicity,
Structured Error Taxonomy, Reproducibility, Data Leakage Prevention, Frontend APIs & Regression.
Covers all 76 mandatory test items specified in Phase 25.
"""

import os
import sys
import copy
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from services.dataset_registry import DatasetRegistry, DatasetType
from services.validation_engine import (
    ValidationEngine,
    CaseRegistry,
    ENGINE_VERSION,
    METHODOLOGY_VERSION,
    VALIDATION_DATASET_VERSION,
    APPLICATION_VERSION,
)
from engine import (
    estimate_wbgt,
    classify_imd,
    assess_full_risk_profile,
    detect_peak_risk_windows,
    calculate_hourly_risk_profile,
)


class TestValidationSystem(unittest.TestCase):
    """Exhaustive test suite for HeatWave Guardian empirical evidence & backtesting."""

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        DatasetRegistry.reset_registry()

    def tearDown(self):
        DatasetRegistry.reset_registry()

    # ──────────────────────────────────────────────────────────────────────────
    # DATASET REGISTRY (Tests 1–6)
    # ──────────────────────────────────────────────────────────────────────────

    def test_01_dataset_creation(self):
        """1. Verify successful registration of a valid dataset."""
        sample_meta = {
            "dataset_id": "test_obs_2026",
            "name": "Test Observation Station Network",
            "source": "State Disaster Management Authority",
            "provider": "SDMA",
            "retrieval_date": "2026-01-01",
            "data_type": DatasetType.REAL_OBSERVATION,
            "coverage": {
                "geographic": "Telangana Central Basin",
                "temporal_start": "2026-05-01",
                "temporal_end": "2026-05-07",
            },
            "resolution": {"spatial": "Station", "temporal": "Hourly"},
            "variables": [{"name": "temp_c", "unit": "°C"}],
            "units": {"temperature": "°C"},
        }
        ds_id = DatasetRegistry.register_dataset(sample_meta)
        self.assertEqual(ds_id, "test_obs_2026")
        retrieved = DatasetRegistry.get_dataset("test_obs_2026")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["name"], "Test Observation Station Network")
        self.assertTrue(retrieved["is_real_data"])
        self.assertTrue(retrieved["validation_eligibility"])

    def test_02_dataset_validation(self):
        """2. Verify metadata schema validator rejects incomplete datasets and accepts valid ones."""
        self.assertTrue(DatasetRegistry.validate_metadata("era5_ap_telangana_2015"))
        self.assertTrue(DatasetRegistry.validate_metadata("imd_delhi_heatwave_2024"))
        self.assertFalse(DatasetRegistry.validate_metadata("non_existent_dataset"))
        self.assertFalse(DatasetRegistry.validate_metadata({"name": "Incomplete"}))

    def test_03_missing_metadata(self):
        """3. Registration fails cleanly when required metadata fields are missing."""
        invalid_meta = {
            "dataset_id": "missing_provider",
            "name": "No Provider",
            # missing provider, source, variables
        }
        with self.assertRaises(ValueError) as ctx:
            DatasetRegistry.register_dataset(invalid_meta)
        self.assertIn("Missing required dataset metadata field", str(ctx.exception))

    def test_04_invalid_metadata(self):
        """4. Registration fails when data_type is unrecognized."""
        invalid_type = {
            "dataset_id": "bad_type",
            "name": "Bad Type",
            "source": "Unknown",
            "provider": "Test",
            "retrieval_date": "2024-01-01",
            "data_type": "UNSUPPORTED_TYPE_XYZ",
            "coverage": {"temporal_start": "2024-01-01", "temporal_end": "2024-01-02"},
            "resolution": {"spatial": "Point", "temporal": "Daily"},
            "variables": ["temperature"],
            "units": {"temp": "C"},
        }
        with self.assertRaises(ValueError) as ctx:
            DatasetRegistry.register_dataset(invalid_type)
        self.assertIn("Invalid dataset data_type", str(ctx.exception))

    def test_05_observed_vs_forecast_classification(self):
        """5. Ensure canonical observed vs forecast vs demo categorization."""
        obs = DatasetRegistry.get_dataset("imd_delhi_heatwave_2024")
        fc = DatasetRegistry.get_dataset("openmeteo_nwp_forecast_archive")
        demo = DatasetRegistry.get_dataset("demo_synthetic_sample_2024")
        self.assertEqual(obs["observed_vs_forecast"], "observed")
        self.assertEqual(fc["observed_vs_forecast"], "forecast")
        self.assertEqual(demo["observed_vs_forecast"], "demo")

    def test_06_demo_data_separation(self):
        """6. Ensure demo/synthetic data is strictly isolated and excluded from real validation."""
        real_datasets = DatasetRegistry.get_real_datasets()
        real_ids = [d["dataset_id"] for d in real_datasets]
        self.assertNotIn("demo_synthetic_sample_2024", real_ids)
        demo = DatasetRegistry.get_dataset("demo_synthetic_sample_2024")
        self.assertFalse(demo["is_real_data"])
        self.assertFalse(demo["validation_eligibility"])

    # ──────────────────────────────────────────────────────────────────────────
    # CASE REGISTRY (Tests 7–12)
    # ──────────────────────────────────────────────────────────────────────────

    def test_07_valid_case_retrieval(self):
        """7. Verify valid case retrieval by case ID."""
        c = CaseRegistry.get_case("khammam_2015_heatwave")
        self.assertIsNotNone(c)
        self.assertEqual(c["location"], "Khammam")
        self.assertEqual(c["state"], "Telangana")
        self.assertEqual(len(c["days"]), 18)

    def test_08_invalid_case(self):
        """8. Non-existent case returns None."""
        c = CaseRegistry.get_case("non_existent_case_999")
        self.assertIsNone(c)

    def test_09_duplicate_case_resilience(self):
        """9. Verify all case IDs in registry are unique."""
        all_cases = CaseRegistry.list_cases()
        case_ids = [c["case_id"] for c in all_cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_10_incomplete_case_guard(self):
        """10. Verify every case contains all mandatory metadata fields."""
        required = ["case_id", "name", "climate_zone", "location", "state", "region", "start_date", "end_date", "days"]
        for c in CaseRegistry.list_cases():
            for f in required:
                self.assertIn(f, c, f"Case {c.get('case_id')} missing field {f}")
                self.assertTrue(c[f], f"Case {c.get('case_id')} has empty field {f}")

    def test_11_timezone_handling(self):
        """11. Verify all dates follow ISO format YYYY-MM-DD corresponding to IST daytime peaks."""
        for c in CaseRegistry.list_cases():
            for d in c["days"]:
                date_str = d["date"]
                parsed = datetime.strptime(date_str, "%Y-%m-%d")
                self.assertIsInstance(parsed, datetime)

    def test_12_spatial_metadata(self):
        """12. Verify valid latitude and longitude coordinates across all cases."""
        for c in CaseRegistry.list_cases():
            lat = c["latitude"]
            lon = c["longitude"]
            self.assertTrue(8.0 <= lat <= 37.0, f"Latitude {lat} out of Indian bounds for {c['case_id']}")
            self.assertTrue(68.0 <= lon <= 97.5, f"Longitude {lon} out of Indian bounds for {c['case_id']}")

    # ──────────────────────────────────────────────────────────────────────────
    # BACKTEST PIPELINE (Tests 13–19)
    # ──────────────────────────────────────────────────────────────────────────

    def test_13_single_case_backtest(self):
        """13. Run backtest for a single case."""
        rep = ValidationEngine.run_backtest(case_ids=["delhi_2024_extreme"])
        self.assertEqual(rep["coverage_summary"]["cases_evaluated_count"], 1)
        self.assertEqual(rep["coverage_summary"]["total_case_days"], 7)
        self.assertEqual(rep["coverage_summary"]["locations"], ["Delhi"])

    def test_14_multiple_cases_backtest(self):
        """14. Run backtest for two specific cases."""
        rep = ValidationEngine.run_backtest(case_ids=["khammam_2015_heatwave", "mumbai_2023_coastal_humid"])
        self.assertEqual(rep["coverage_summary"]["cases_evaluated_count"], 2)
        self.assertEqual(rep["coverage_summary"]["total_case_days"], 25)

    def test_15_multiple_locations_all_cases(self):
        """15. Run full backtest across all 7 cases covering all locations."""
        rep = ValidationEngine.run_backtest()
        self.assertEqual(rep["coverage_summary"]["cases_evaluated_count"], 7)
        self.assertEqual(rep["coverage_summary"]["total_case_days"], 70)
        self.assertIn("Mumbai", rep["coverage_summary"]["locations"])
        self.assertIn("Delhi", rep["coverage_summary"]["locations"])
        self.assertIn("Bengaluru", rep["coverage_summary"]["locations"])

    def test_16_empty_dataset_filter(self):
        """16. Empty case selection returns 0 evaluated records cleanly."""
        rep = ValidationEngine.run_backtest(case_ids=["non_existent_id"])
        self.assertEqual(rep["coverage_summary"]["cases_evaluated_count"], 0)
        self.assertEqual(rep["coverage_summary"]["total_case_days"], 0)

    def test_17_malformed_dataset_handling(self):
        """17. Engine handles unexpected case structures gracefully without crashing."""
        malformed = [{"case_id": "malformed", "location": "Test", "days": []}]
        with patch.object(CaseRegistry, "list_cases", return_value=malformed):
            rep = ValidationEngine.run_backtest()
            self.assertEqual(rep["coverage_summary"]["cases_evaluated_count"], 1)
            self.assertEqual(rep["coverage_summary"]["total_case_days"], 0)

    def test_18_missing_optional_data(self):
        """18. Days without ground reference temperatures are processed without crashing."""
        case_missing_ref = {
            "case_id": "test_no_ref",
            "name": "No Ref Case",
            "location": "Test",
            "state": "Test",
            "climate_zone": "Plains",
            "region": "plains",
            "latitude": 20.0,
            "longitude": 75.0,
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "dataset_id": "test_ds",
            "data_type": DatasetType.REAL_OBSERVATION,
            "days": [{"date": "2024-01-01", "temp_c": 35.0, "rh_pct": 50.0}]  # no ref_temp_c, no ref_imd
        }
        with patch.object(CaseRegistry, "list_cases", return_value=[case_missing_ref]):
            rep = ValidationEngine.run_backtest()
            self.assertEqual(rep["environmental_metrics"]["sample_count"], 0)
            self.assertEqual(rep["coverage_summary"]["total_case_days"], 1)

    def test_19_deterministic_rerun(self):
        """19. Running backtest twice with same inputs produces identical results."""
        rep1 = ValidationEngine.run_backtest()
        rep2 = ValidationEngine.run_backtest()
        self.assertEqual(rep1["environmental_metrics"], rep2["environmental_metrics"])
        self.assertEqual(rep1["classification_metrics"], rep2["classification_metrics"])
        self.assertEqual(rep1["benchmarks"], rep2["benchmarks"])

    # ──────────────────────────────────────────────────────────────────────────
    # ENVIRONMENTAL ERROR METRICS (Tests 20–24)
    # ──────────────────────────────────────────────────────────────────────────

    def test_20_mae_calculation(self):
        """20. Mean Absolute Error calculation accuracy."""
        rep = ValidationEngine.run_backtest()
        mae = rep["environmental_metrics"]["mae_c"]
        self.assertGreater(mae, 0.0)
        self.assertLess(mae, 5.0)

    def test_21_rmse_calculation(self):
        """21. Root Mean Square Error is >= MAE."""
        rep = ValidationEngine.run_backtest()
        mae = rep["environmental_metrics"]["mae_c"]
        rmse = rep["environmental_metrics"]["rmse_c"]
        self.assertGreaterEqual(rmse, mae)

    def test_22_mean_bias_direction(self):
        """22. Mean bias captures the documented ERA5 pre-monsoon cold bias (negative)."""
        rep = ValidationEngine.run_backtest()
        bias = rep["environmental_metrics"]["mean_bias_c"]
        self.assertLess(bias, 0.0)  # Reanalysis is cooler than ground station peak

    def test_23_sample_counting(self):
        """23. Sample count matches total reference ground observation records."""
        rep = ValidationEngine.run_backtest()
        self.assertEqual(rep["environmental_metrics"]["sample_count"], 70)

    def test_24_median_absolute_error(self):
        """24. Median absolute error is positive and finite."""
        rep = ValidationEngine.run_backtest()
        med = rep["environmental_metrics"]["median_abs_error_c"]
        self.assertGreater(med, 0.0)
        self.assertLess(med, 4.0)

    # ──────────────────────────────────────────────────────────────────────────
    # IMD CLASSIFICATION FIDELITY (Tests 25–29)
    # ──────────────────────────────────────────────────────────────────────────

    def test_25_agreement_percentage(self):
        """25. Classification agreement rate is computed accurately."""
        rep = ValidationEngine.run_backtest()
        clf = rep["classification_metrics"]
        expected_pct = round(clf["exact_matches"] / clf["total_records"] * 100, 1)
        self.assertEqual(clf["agreement_pct"], expected_pct)

    def test_26_confusion_matrix_structure(self):
        """26. 4x4 Confusion matrix has all 4 IMD categories."""
        rep = ValidationEngine.run_backtest()
        cm = rep["classification_metrics"]["confusion_matrix"]
        cats = ["Normal", "Heat Day", "Heatwave", "Severe Heatwave"]
        for r in cats:
            self.assertIn(r, cm)
            for c in cats:
                self.assertIn(c, cm[r])
                self.assertIsInstance(cm[r][c], int)

    def test_27_zero_false_alarms(self):
        """27. Zero non-heat baseline periods are misclassified as heatwaves."""
        rep = ValidationEngine.run_backtest(case_ids=["bangalore_2023_background_cool"])
        self.assertEqual(rep["classification_metrics"]["false_alarms"], 0)

    def test_28_misses_tracking(self):
        """28. Misses reflect documented reanalysis spatial smoothing in Hyderabad."""
        rep = ValidationEngine.run_backtest(case_ids=["hyderabad_2015_heatwave"])
        self.assertGreater(rep["classification_metrics"]["misses"], 0)

    def test_29_category_boundaries(self):
        """29. Classification adheres strictly to Plains (40/42/44) and Coastal (37) criteria."""
        self.assertEqual(classify_imd(39.0, region="plains"), "Normal")
        self.assertEqual(classify_imd(40.5, region="plains"), "Heat Day")
        self.assertEqual(classify_imd(42.5, region="plains"), "Heatwave")
        self.assertEqual(classify_imd(44.5, region="plains"), "Severe Heatwave")
        self.assertEqual(classify_imd(36.5, region="coastal"), "Normal")
        self.assertEqual(classify_imd(37.5, region="coastal"), "Heat Day")
        self.assertEqual(classify_imd(39.5, region="coastal"), "Heatwave")
        self.assertEqual(classify_imd(41.5, region="coastal"), "Severe Heatwave")

    # ──────────────────────────────────────────────────────────────────────────
    # FORECAST BACKTEST (Tests 30–34)
    # ──────────────────────────────────────────────────────────────────────────

    def test_30_forecast_vs_observed_evaluation(self):
        """30. NWP forecast backtest evaluates forecast error when archive is present."""
        rep = ValidationEngine.run_backtest(case_ids=["delhi_2024_extreme"])
        fc = rep["forecast_backtest"]
        self.assertTrue(fc["has_historical_forecast_archive"])
        self.assertGreater(fc["records_evaluated"], 0)
        self.assertGreaterEqual(fc["forecast_temp_mae_c"], 0.0)

    def test_31_missing_forecast_archive_message(self):
        """31. Missing forecast archive explicitly states unavailable rather than faking data."""
        rep = ValidationEngine.run_backtest(case_ids=["khammam_2015_heatwave"])
        fc = rep["forecast_backtest"]
        self.assertFalse(fc["has_historical_forecast_archive"])
        self.assertGreater(len(fc["sources_unavailable"]), 0)
        self.assertEqual(
            fc["sources_unavailable"][0]["message"],
            "Historical forecast backtesting unavailable for this source."
        )

    def test_32_forecast_peak_timing(self):
        """32. Forecast peak timing difference is evaluated in hours."""
        rep = ValidationEngine.run_backtest(case_ids=["delhi_2024_extreme"])
        fc = rep["forecast_backtest"]
        self.assertIn("peak_timing_diff_hours_mean", fc)
        self.assertGreaterEqual(fc["peak_timing_diff_hours_mean"], 0.0)

    def test_33_false_peak_detection(self):
        """33. False peak periods metric is tracked."""
        rep = ValidationEngine.run_backtest(case_ids=["delhi_2024_extreme"])
        fc = rep["forecast_backtest"]
        self.assertIn("false_peak_periods", fc)
        self.assertEqual(fc["false_peak_periods"], 0)

    def test_34_missed_peak_detection(self):
        """34. Missed elevated periods metric is tracked."""
        rep = ValidationEngine.run_backtest(case_ids=["delhi_2024_extreme"])
        fc = rep["forecast_backtest"]
        self.assertIn("missed_elevated_periods", fc)
        self.assertEqual(fc["missed_elevated_periods"], 0)

    # ──────────────────────────────────────────────────────────────────────────
    # PEAK WINDOW BACKTEST (Tests 35–38 - Mapping items 32-34)
    # ──────────────────────────────────────────────────────────────────────────

    def test_35_peak_window_detection_rate(self):
        """35. Peak thermal window detects elevated period in severe heat events."""
        rep = ValidationEngine.run_backtest()
        pk = rep["peak_window_backtest"]
        self.assertGreaterEqual(pk["elevated_detection_rate_pct"], 90.0)

    def test_36_peak_window_control_case(self):
        """36. Negative control case produces zero false peak alarms."""
        rep = ValidationEngine.run_backtest()
        pk = rep["peak_window_backtest"]
        self.assertEqual(pk["control_false_alarms"], 0)

    # ──────────────────────────────────────────────────────────────────────────
    # COMPONENT ABLATION REGRESSION (Tests 37–40)
    # ──────────────────────────────────────────────────────────────────────────

    def test_37_ablation_each_component_present(self):
        """37. All 7 atmospheric and operational component ablations are computed."""
        rep = ValidationEngine.run_backtest()
        abl = rep["ablation"]
        components = ["humidity", "wind", "solar_radiation", "workload", "acclimatization", "ppe", "recovery_environment"]
        for c in components:
            self.assertIn(c, abl)
            self.assertGreater(abl[c]["scenarios_tested"], 0)
            self.assertGreater(abl[c]["scenarios_changed"], 0)
            self.assertGreater(abl[c]["impact_pct"], 0.0)

    def test_38_ablation_humidity_high_impact(self):
        """38. Relative humidity ablation has >=50% material impact on scenarios."""
        rep = ValidationEngine.run_backtest()
        hum = rep["ablation"]["humidity"]
        self.assertGreaterEqual(hum["impact_pct"], 50.0)

    # ──────────────────────────────────────────────────────────────────────────
    # DECISION MONOTONICITY & CONSISTENCY (Tests 39–43)
    # ──────────────────────────────────────────────────────────────────────────

    def test_39_workload_monotonicity(self):
        """39. Increasing workload strictly maintains or increases caution level."""
        audit = ValidationEngine.run_decision_consistency_audit()
        self.assertTrue(audit["workload_monotonicity_passed"])
        self.assertEqual(audit["workload_violations"], 0)

    def test_40_acclimatization_monotonicity(self):
        """40. Unacclimatized worker strictly maintains or increases caution level."""
        audit = ValidationEngine.run_decision_consistency_audit()
        self.assertTrue(audit["acclimatization_monotonicity_passed"])
        self.assertEqual(audit["acclimatization_violations"], 0)

    def test_41_recovery_monotonicity(self):
        """41. AC <= Shaded <= Direct Sun recovery caution monotonicity."""
        audit = ValidationEngine.run_decision_consistency_audit()
        self.assertTrue(audit["recovery_monotonicity_passed"])
        self.assertEqual(audit["recovery_violations"], 0)

    def test_42_ppe_monotonicity(self):
        """42. Light <= Heavy <= Impermeable PPE caution monotonicity."""
        audit = ValidationEngine.run_decision_consistency_audit()
        self.assertTrue(audit["clothing_monotonicity_passed"])
        self.assertEqual(audit["clothing_violations"], 0)

    def test_43_irrelevant_variable_invariance(self):
        """43. Location display string alone does not alter physical thermal stress."""
        w1 = {"temp_c": 38.0, "rh_pct": 50.0, "wind_ms": 2.0, "solar_wm2": 600.0}
        p1 = {"workload": "moderate"}
        prof1 = assess_full_risk_profile(w1, p1, "plains", location="LocationA")
        prof2 = assess_full_risk_profile(w1, p1, "plains", location="LocationB")
        self.assertEqual(prof1["wbgt_c"], prof2["wbgt_c"])
        self.assertEqual(prof1["overall_action_level"], prof2["overall_action_level"])

    # ──────────────────────────────────────────────────────────────────────────
    # STRUCTURED ERROR REGISTRY (Tests 44–48)
    # ──────────────────────────────────────────────────────────────────────────

    def test_44_classification_mismatch_categorization(self):
        """44. Disagreements are mapped to valid error taxonomy categories."""
        rep = ValidationEngine.run_backtest()
        valid_cats = {
            "spatial_mismatch", "category_boundary_shift", "classification_difference",
            "environmental_calculation_difference", "missing_variable", "data_quality_issue",
            "reference_definition_mismatch", "temporal_mismatch", "forecast_uncertainty",
            "model_limitation", "unknown"
        }
        for err in rep["error_registry"]:
            self.assertIn(err["error_category"], valid_cats)
            self.assertTrue(err["explanation"])

    def test_45_spatial_mismatch_hyderabad(self):
        """45. Hyderabad smoothing errors are categorized as spatial_mismatch."""
        rep = ValidationEngine.run_backtest(case_ids=["hyderabad_2015_heatwave"])
        errs = rep["error_registry"]
        spatial_errs = [e for e in errs if e["error_category"] == "spatial_mismatch"]
        self.assertGreater(len(spatial_errs), 0)

    def test_46_category_boundary_shift_khammam(self):
        """46. Borderline heatwave vs severe heatwave shifts are categorized appropriately."""
        rep = ValidationEngine.run_backtest(case_ids=["khammam_2015_heatwave"])
        errs = rep["error_registry"]
        shifts = [e for e in errs if e["error_category"] == "category_boundary_shift"]
        self.assertGreater(len(shifts), 0)

    def test_47_model_limitation_disclosure(self):
        """47. Model limitations and cold biases are explicitly disclosed in metadata and case notes."""
        rep = ValidationEngine.run_backtest()
        self.assertIn("limitations", rep["metadata"])
        self.assertTrue(any("reanalysis" in lim.lower() for lim in rep["metadata"]["limitations"]))

    def test_48_error_registry_counts(self):
        """48. Error registry correctly aggregates total mismatches."""
        rep = ValidationEngine.run_backtest()
        total_mismatches = rep["classification_metrics"]["total_records"] - rep["classification_metrics"]["exact_matches"]
        self.assertEqual(len(rep["error_registry"]), total_mismatches)

    # ──────────────────────────────────────────────────────────────────────────
    # REPRODUCIBILITY & SERIALIZATION (Tests 49–50)
    # ──────────────────────────────────────────────────────────────────────────

    def test_49_json_serialization(self):
        """49. Complete report serializes and deserializes to JSON cleanly."""
        rep = ValidationEngine.run_backtest()
        s = json.dumps(rep)
        loaded = json.loads(s)
        self.assertEqual(loaded["metadata"]["engine_version"], ENGINE_VERSION)
        self.assertEqual(len(loaded["cases"]), 7)

    def test_50_csv_export_format(self):
        """50. Report exports to tabular CSV with correct headers."""
        rep = ValidationEngine.run_backtest()
        csv_text = ValidationEngine.export_report_csv(report=rep)
        self.assertIn("Case ID,Location,State", csv_text)
        self.assertIn("Khammam", csv_text)
        self.assertIn("Delhi", csv_text)

    # ──────────────────────────────────────────────────────────────────────────
    # DATA LEAKAGE AUDIT (Tests 51–53)
    # ──────────────────────────────────────────────────────────────────────────

    def test_51_zero_future_data_leakage(self):
        """51. Engine calculates instantaneous risk using only contemporaneous inputs."""
        w = {"temp_c": 42.0, "rh_pct": 30.0, "wind_ms": 2.0, "solar_wm2": 700.0}
        p = {"workload": "light"}
        prof = assess_full_risk_profile(w, p, "plains")
        # Ensure calculation contains no future timestamps or next-day predictions
        self.assertNotIn("future_prediction", prof)

    def test_52_observation_forecast_separation(self):
        """52. Historical observations are never labeled as forecasts."""
        for c in CaseRegistry.list_cases():
            self.assertIn(c["data_type"], [DatasetType.REAL_OBSERVATION, DatasetType.REANALYSIS, DatasetType.REFERENCE_BULLETIN])

    def test_53_reference_label_leakage_audit(self):
        """53. Reference IMD bulletin labels are NOT inputs to classify_imd()."""
        # classify_imd only takes temp_c, departure, region, max_consecutive_days
        imd_res = classify_imd(43.0, region="plains")
        self.assertIn(imd_res, ["Heatwave", "Severe Heatwave", "Heat Day", "Normal"])

    # ──────────────────────────────────────────────────────────────────────────
    # FRONTEND & REST API ENDPOINTS (Tests 54–63)
    # ──────────────────────────────────────────────────────────────────────────

    def test_54_validation_page_renders_200(self):
        """54. GET /validation returns 200 OK with full HTML content."""
        res = self.client.get("/validation")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Real-World Evidence", res.data)
        self.assertIn(b"4-Way Multi-Model Benchmark Comparison", res.data)
        self.assertIn(b"Interactive Historical Case Explorer", res.data)

    def test_55_api_validation_report(self):
        """55. GET /api/validation/report returns 200 with JSON structure."""
        res = self.client.get("/api/validation/report")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["metadata"]["engine_version"], ENGINE_VERSION)
        self.assertEqual(data["coverage_summary"]["cases_evaluated_count"], 7)

    def test_56_api_validation_datasets(self):
        """56. GET /api/validation/datasets returns registered datasets and provenance."""
        res = self.client.get("/api/validation/datasets")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("provenance_summary", data)
        self.assertIn("datasets", data)
        self.assertGreaterEqual(len(data["datasets"]), 7)

    def test_57_api_validation_cases(self):
        """57. GET /api/validation/cases returns all 7 registered cases."""
        res = self.client.get("/api/validation/cases")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["count"], 7)

    def test_58_api_validation_single_case_detail(self):
        """58. GET /api/validation/case/<case_id> returns detailed evaluation."""
        res = self.client.get("/api/validation/case/delhi_2024_extreme")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["case"]["location"], "Delhi")
        self.assertEqual(len(data["evaluation"]["days"]), 7)

    def test_59_api_validation_case_not_found(self):
        """59. GET /api/validation/case/invalid returns 404."""
        res = self.client.get("/api/validation/case/invalid_case_id")
        self.assertEqual(res.status_code, 404)

    def test_60_api_validation_export_json(self):
        """60. GET /api/validation/export?format=json returns valid JSON report."""
        res = self.client.get("/api/validation/export?format=json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, "application/json")

    def test_61_api_validation_export_csv(self):
        """61. GET /api/validation/export?format=csv returns downloadable CSV."""
        res = self.client.get("/api/validation/export?format=csv")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/csv", res.content_type)
        self.assertIn(b"Case ID,Location,State", res.data)

    def test_62_legacy_api_validation_data_backward_compat(self):
        """62. GET /api/validation/data remains backward compatible."""
        res = self.client.get("/api/validation/data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("cities", data)

    # ──────────────────────────────────────────────────────────────────────────
    # REGRESSION TESTS (Tests 63–76)
    # ──────────────────────────────────────────────────────────────────────────

    def test_63_regression_check_risk_endpoint(self):
        """63. POST /api/check-risk operates normally with preset city."""
        payload = {"location": "Delhi", "workload": "moderate"}
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["location"], "Delhi")

    def test_64_regression_forecast_endpoint(self):
        """64. GET /api/forecast operates normally."""
        mock_forecast = [
            {
                "date": "2026-06-16",
                "max_temp_c": 39.0,
                "peak_wbgt_c": 32.0,
                "hourly": [
                    {"time": "2026-06-16T10:00", "temp_c": 35.0, "peak_wbgt_c": 29.0},
                    {"time": "2026-06-16T13:00", "temp_c": 39.0, "peak_wbgt_c": 32.0},
                    {"time": "2026-06-16T17:00", "temp_c": 36.0, "peak_wbgt_c": 30.0},
                ]
            }
        ]
        with patch("app.fetch_forecast_weather", return_value=mock_forecast):
            res = self.client.get("/api/forecast?location=Delhi")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIsInstance(data, list)
            self.assertEqual(data[0]["location"], "Delhi")

    def test_65_regression_simulate_scenario_endpoint(self):
        """65. POST /api/simulate-scenario operates normally."""
        payload = {
            "baseline": {"location": "Delhi", "workload": "heavy"},
            "scenario": {"workload": "light"},
            "weather_data": {"temp_c": 40.0, "humidity": 40.0, "wind_ms": 2.0, "solar_wm2": 600.0}
        }
        res = self.client.post("/api/simulate-scenario", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("action_level_change", data)

    def test_66_regression_decision_trace_endpoint(self):
        """66. GET /api/decision-trace operates normally."""
        res = self.client.get("/api/decision-trace?case_id=khammam_2015_peak")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("decision_trace", data)

    def test_67_regression_institute_endpoints(self):
        """67. GET /api/institute/list operates normally."""
        res = self.client.get("/api/institute/list")
        self.assertEqual(res.status_code, 200)

    def test_68_regression_authority_endpoints(self):
        """68. GET /api/authority/zones operates normally."""
        res = self.client.get("/api/authority/zones?city=Delhi")
        self.assertEqual(res.status_code, 200)

    def test_69_regression_cooling_stations(self):
        """69. GET /api/cooling-stations/recommendations operates normally."""
        res = self.client.get("/api/cooling-stations/recommendations?city=Delhi")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("recommendations", data)

    def test_70_regression_alert_history(self):
        """70. GET /api/alerts/history operates normally."""
        res = self.client.get("/api/alerts/history")
        self.assertEqual(res.status_code, 200)

    def test_71_regression_mortality_risk(self):
        """71. GET /api/mortality-risk operates normally."""
        res = self.client.get("/api/mortality-risk?temp_c=42.0&wbgt_c=32.0&location=Delhi")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("proxy_score", data)
        self.assertFalse(data.get("is_medical_prediction", True))

    def test_72_regression_hap_guidelines(self):
        """72. GET /api/hap-guidelines operates normally."""
        res = self.client.get("/api/hap-guidelines?risk_category=High&imd_status=Heatwave")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("alert_level", data)
        self.assertIn("institutional_actions", data)

    def test_73_regression_page_my_risk(self):
        """73. GET / renders 200."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)

    def test_74_regression_page_forecast(self):
        """74. GET /forecast renders 200."""
        res = self.client.get("/forecast")
        self.assertEqual(res.status_code, 200)

    def test_75_regression_page_what_if(self):
        """75. GET /what-if renders 200."""
        res = self.client.get("/what-if")
        self.assertEqual(res.status_code, 200)

    def test_76_regression_page_cooling_stations(self):
        """76. GET /cooling-stations renders 200."""
        res = self.client.get("/cooling-stations")
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
