"""
Comprehensive Scientific and Engineering Audit Test Suite
=========================================================
Tests:
1. WBGT thermodynamic calculations, energy balance, and boundary conditions
2. IMD regional qualifying heatwave classification
3. Categorical risk fusion matrix integrity
4. Operational exposure tiers (metabolic workload + clothing adjustment values)
5. Acclimatization as a first-class factor and NIOSH progressive schedules
6. Data quality and transparent qualitative confidence scoring
7. Explainability: environmental, operational, context drivers and protective factors
8. Work/rest decision support and non-prescriptive hydration guidance
9. Unified assessment pipeline (assess_full_risk_profile)
10. API endpoints (/api/check-risk and /api/forecast) contract and backward compatibility
"""

import unittest
from unittest.mock import patch, MagicMock
from app import app
from engine import (
    estimate_wbgt,
    _stull_wet_bulb,
    _globe_temperature,
    classify_wbgt_band,
    classify_thermal_stress,
    calculate_thermal_stress_index,
    classify_imd,
    fuse_risk,
    evaluate_operational_exposure,
    evaluate_acclimatization_schedule,
    evaluate_data_quality,
    generate_structured_explanation,
    generate_summary,
    assess_vulnerability,
    escalate_risk,
    get_work_rest,
    assess_full_risk_profile,
    FUSION_MATRIX,
    RISK_LEVELS,
    REGION_THRESHOLDS,
    METABOLIC_RATE_WATTS,
    CLOTHING_ADJUSTMENT_VALUES,
    ACCLIMATIZATION_STATUSES,
    ACCLIMATIZATION_SCHEDULES,
)


class TestWBGTPhysicsAndBoundaries(unittest.TestCase):
    """Verify Liljegren WBGT estimation, Stull wet bulb, and black globe energy balance."""

    def test_zero_solar_radiation_night_condition(self):
        """At zero solar radiation, globe temperature Tg equals air temperature Ta."""
        temp = 32.0
        tg = _globe_temperature(temp_c=temp, wind_ms=2.0, solar_wm2=0.0)
        self.assertAlmostEqual(tg, temp, places=2)

    def test_negative_solar_radiation_clamped(self):
        """Negative solar values should be clamped to 0.0 W/m²."""
        temp = 28.0
        wbgt_neg = estimate_wbgt(temp_c=temp, rh_pct=50.0, wind_ms=1.5, solar_wm2=-100.0)
        wbgt_zero = estimate_wbgt(temp_c=temp, rh_pct=50.0, wind_ms=1.5, solar_wm2=0.0)
        self.assertEqual(wbgt_neg, wbgt_zero)

    def test_wind_convective_cooling_effect(self):
        """Higher wind speed increases convective cooling, lowering black globe and WBGT in solar radiation."""
        temp = 40.0
        solar = 800.0
        wbgt_low_wind = estimate_wbgt(temp_c=temp, rh_pct=40.0, wind_ms=0.2, solar_wm2=solar)
        wbgt_high_wind = estimate_wbgt(temp_c=temp, rh_pct=40.0, wind_ms=5.0, solar_wm2=solar)
        self.assertGreater(wbgt_low_wind, wbgt_high_wind)

    def test_stull_wet_bulb_bounds(self):
        """Verify Stull wet bulb regression stays within physical range (Tw <= Ta)."""
        for t in [15.0, 25.0, 35.0, 45.0]:
            for rh in [10.0, 50.0, 90.0]:
                tw = _stull_wet_bulb(t, rh)
                self.assertLessEqual(tw, t + 0.1)
                self.assertGreater(tw, -20.0)

    def test_extreme_cold_and_heat_stability(self):
        """Calculations do not raise exceptions or crash at extreme limits."""
        wbgt_cold = estimate_wbgt(temp_c=-10.0, rh_pct=30.0, wind_ms=1.0, solar_wm2=0.0)
        self.assertIsInstance(wbgt_cold, float)
        self.assertLess(wbgt_cold, 10.0)

        wbgt_heat = estimate_wbgt(temp_c=52.0, rh_pct=80.0, wind_ms=0.5, solar_wm2=1000.0)
        self.assertIsInstance(wbgt_heat, float)
        self.assertGreater(wbgt_heat, 35.0)


class TestClassificationBoundaries(unittest.TestCase):
    """Verify exact boundary classifications for WBGT bands and IMD regional departures."""

    def test_wbgt_band_thresholds(self):
        self.assertEqual(classify_wbgt_band(26.9), "Safe")
        self.assertEqual(classify_wbgt_band(27.0), "Caution")
        self.assertEqual(classify_wbgt_band(29.3), "Caution")
        self.assertEqual(classify_wbgt_band(29.4), "Extreme Caution")
        self.assertEqual(classify_wbgt_band(31.0), "Extreme Caution")
        self.assertEqual(classify_wbgt_band(31.1), "Danger")
        self.assertEqual(classify_wbgt_band(32.1), "Danger")
        self.assertEqual(classify_wbgt_band(32.2), "Extreme Danger")

    def test_imd_plains_thresholds(self):
        # Plains threshold: 40.0°C
        self.assertEqual(classify_imd(39.9, "plains"), "Normal")
        self.assertEqual(classify_imd(40.0, "plains"), "Heat Day")
        self.assertEqual(classify_imd(41.9, "plains"), "Heat Day")
        self.assertEqual(classify_imd(42.0, "plains"), "Heatwave")
        self.assertEqual(classify_imd(43.9, "plains"), "Heatwave")
        self.assertEqual(classify_imd(44.0, "plains"), "Severe Heatwave")

    def test_imd_coastal_thresholds(self):
        # Coastal threshold: 37.0°C
        self.assertEqual(classify_imd(36.9, "coastal"), "Normal")
        self.assertEqual(classify_imd(37.0, "coastal"), "Heat Day")
        self.assertEqual(classify_imd(38.9, "coastal"), "Heat Day")
        self.assertEqual(classify_imd(39.0, "coastal"), "Heatwave")
        self.assertEqual(classify_imd(40.9, "coastal"), "Heatwave")
        self.assertEqual(classify_imd(41.0, "coastal"), "Severe Heatwave")

    def test_imd_hilly_thresholds(self):
        # Hilly threshold: 30.0°C
        self.assertEqual(classify_imd(29.9, "hilly"), "Normal")
        self.assertEqual(classify_imd(30.0, "hilly"), "Heat Day")
        self.assertEqual(classify_imd(31.9, "hilly"), "Heat Day")
        self.assertEqual(classify_imd(32.0, "hilly"), "Heatwave")
        self.assertEqual(classify_imd(33.9, "hilly"), "Heatwave")
        self.assertEqual(classify_imd(34.0, "hilly"), "Severe Heatwave")

    def test_unknown_region_defaults_to_plains(self):
        self.assertEqual(classify_imd(40.5, "unknown_region"), "Heat Day")
        self.assertEqual(classify_imd(38.0, "unknown_region"), "Normal")


class TestCategoricalRiskFusion(unittest.TestCase):
    """Verify the 20-cell lookup matrix maps valid bands to valid risk categories without addition."""

    def test_all_matrix_cells_valid(self):
        self.assertEqual(len(FUSION_MATRIX), 20)
        for (band, status), risk in FUSION_MATRIX.items():
            self.assertIn(risk, RISK_LEVELS)
            res = fuse_risk(band, status)
            self.assertEqual(res, risk)

    def test_safe_normal_is_low(self):
        self.assertEqual(fuse_risk("Safe", "Normal"), "Low")

    def test_extreme_danger_severe_heatwave_is_extreme(self):
        self.assertEqual(fuse_risk("Extreme Danger", "Severe Heatwave"), "Extreme")


class TestOperationalExposureEngine(unittest.TestCase):
    """Verify metabolic workload, clothing adjustments, and operational exposure tiers."""

    def test_metabolic_workload_categories(self):
        res_light = evaluate_operational_exposure(workload="light")
        self.assertEqual(res_light["metabolic_rate_watts"], 150)
        self.assertEqual(res_light["workload_category"], "light")

        res_heavy = evaluate_operational_exposure(workload="heavy")
        self.assertEqual(res_heavy["metabolic_rate_watts"], 360)

        res_very_heavy = evaluate_operational_exposure(workload="very_heavy")
        self.assertEqual(res_very_heavy["metabolic_rate_watts"], 480)

    def test_clothing_adjustment_values(self):
        res_light = evaluate_operational_exposure(clothing="light")
        self.assertEqual(res_light["clothing_adjustment_c"], 0.0)

        res_heavy = evaluate_operational_exposure(clothing="heavy")
        self.assertEqual(res_heavy["clothing_adjustment_c"], 2.0)

        res_ppe = evaluate_operational_exposure(clothing="ppe")
        self.assertEqual(res_ppe["clothing_adjustment_c"], 6.0)

    def test_operational_exposure_tiers(self):
        # Light work + light clothing -> Low operational exposure
        res_low = evaluate_operational_exposure(workload="light", clothing="light", acclimatization="fully")
        self.assertEqual(res_low["operational_exposure_level"], "Low")

        # Moderate work + light clothing -> Moderate
        res_mod = evaluate_operational_exposure(workload="moderate", clothing="light", acclimatization="fully")
        self.assertEqual(res_mod["operational_exposure_level"], "Moderate")

        # Heavy work + heavy clothing -> High
        res_high = evaluate_operational_exposure(workload="heavy", clothing="heavy", acclimatization="fully")
        self.assertEqual(res_high["operational_exposure_level"], "High")

        # Heavy work + PPE -> Very High
        res_vh = evaluate_operational_exposure(workload="heavy", clothing="ppe", acclimatization="fully")
        self.assertEqual(res_vh["operational_exposure_level"], "Very High")

    def test_shaded_environment_adds_protective_factor(self):
        res = evaluate_operational_exposure(is_sun_exposed=False)
        self.assertTrue(any("Shaded" in p or "indoor" in p for p in res["protective_factors"]))


class TestAcclimatizationFirstClassFactor(unittest.TestCase):
    """Verify acclimatization status handling and NIOSH/OSHA progressive ramp-up schedules."""

    def test_status_canonical_mapping(self):
        acc = evaluate_acclimatization_schedule("fully")
        self.assertEqual(acc["status"], "acclimatized")

        partial = evaluate_acclimatization_schedule("somewhat_used")
        self.assertEqual(partial["status"], "partially_acclimatized")

        unacc = evaluate_acclimatization_schedule("not_used")
        self.assertEqual(unacc["status"], "unacclimatized")

    def test_new_worker_5_day_schedule(self):
        expected_pcts = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
        for day, expected_pct in expected_pcts.items():
            res = evaluate_acclimatization_schedule("unacclimatized", day_in_heat=day)
            self.assertEqual(res["max_exposure_pct"], expected_pct)
            self.assertIn(f"Day {day}", res["schedule_recommendation"])

    def test_returning_worker_4_day_schedule(self):
        expected_pcts = {1: 50, 2: 60, 3: 80, 4: 100}
        for day, expected_pct in expected_pcts.items():
            res = evaluate_acclimatization_schedule("partially_acclimatized", day_in_heat=day, is_returning_worker=True)
            self.assertEqual(res["max_exposure_pct"], expected_pct)
            self.assertIn(f"Day {day}", res["schedule_recommendation"])

    def test_decision_support_disclaimer_present(self):
        res = evaluate_acclimatization_schedule("unacclimatized")
        self.assertTrue(res["is_decision_support"])
        self.assertIn("Occupational decision support", res["disclaimer"])


class TestDataQualityAndConfidence(unittest.TestCase):
    """Verify data quality checks, completeness, and qualitative confidence rating."""

    def test_observed_complete_weather_high_confidence(self):
        data = {"temp_c": 35.0, "rh_pct": 50.0, "wind_ms": 2.5, "solar_wm2": 600.0}
        conf = evaluate_data_quality(data, is_forecast=False)
        self.assertEqual(conf["level"], "High")
        self.assertGreaterEqual(conf["score"], 0.9)
        self.assertEqual(conf["completeness_pct"], 100.0)
        self.assertEqual(conf["missing_variables"], [])

    def test_forecast_data_medium_confidence(self):
        data = {"temp_c": 38.0, "rh_pct": 45.0, "wind_ms": 2.0, "solar_wm2": 700.0}
        conf = evaluate_data_quality(data, is_forecast=True)
        self.assertEqual(conf["level"], "Medium")
        self.assertTrue(conf["is_forecast"])

    def test_missing_required_variables_limited_confidence(self):
        data = {"temp_c": 35.0}  # missing rh, wind, solar
        conf = evaluate_data_quality(data, is_forecast=False)
        self.assertEqual(conf["level"], "Limited")
        self.assertIn("rh_pct", conf["missing_variables"])
        self.assertLessEqual(conf["score"], 0.5)

    def test_out_of_bounds_flags(self):
        data = {"temp_c": 65.0, "rh_pct": 120.0, "wind_ms": -2.0, "solar_wm2": -50.0}
        conf = evaluate_data_quality(data)
        self.assertGreater(len(conf["flags"]), 0)


class TestExplainabilityAndDrivers(unittest.TestCase):
    """Verify structured explanation dynamically generates actual calculated factors without generic boilerplate."""

    def test_drivers_and_protective_factors_presence(self):
        exp = generate_structured_explanation(
            location="Nagpur",
            temp_c=43.5,
            wbgt_c=32.5,
            wbgt_band="Extreme Danger",
            imd_status="Heatwave",
            base_risk="Very High",
            final_risk="Extreme",
            vulnerability="High",
            age_group="60+",
            experience="not_used",
            workload="heavy",
            clothing="ppe"
        )
        self.assertIn("summary", exp)
        self.assertIsInstance(exp["environmental_drivers"], list)
        self.assertIsInstance(exp["operational_drivers"], list)
        self.assertIsInstance(exp["vulnerability_drivers"], list)
        self.assertIsInstance(exp["protective_factors"], list)
        self.assertIsInstance(exp["recommended_actions"], list)

        # Environmental driver includes IMD and WBGT
        env_text = " ".join(exp["environmental_drivers"])
        self.assertIn("IMD Heatwave", env_text)
        self.assertIn("Extreme Danger", env_text)

        # Operational driver includes PPE and heavy work
        op_text = " ".join(exp["operational_drivers"])
        self.assertIn("PPE", op_text)
        self.assertIn("Heavy", op_text)

        # Vulnerability driver includes age 60+ and unacclimatized
        vuln_text = " ".join(exp["vulnerability_drivers"])
        self.assertIn("60+", vuln_text)
        self.assertIn("unacclimatized", vuln_text)


class TestWorkRestAndNonPrescriptiveHydration(unittest.TestCase):
    """Verify work/rest decision support and non-prescriptive hydration guidance."""

    def test_work_rest_screening_contract(self):
        for level in RISK_LEVELS:
            sched = get_work_rest(level)
            self.assertIn("work_minutes", sched)
            self.assertIn("rest_minutes", sched)
            self.assertIn("instruction", sched)
            self.assertEqual(sched["work_minutes"] + sched["rest_minutes"], 60)
            self.assertTrue(sched["is_decision_support"])

    def test_hydration_guidance_non_prescriptive(self):
        sched = get_work_rest("High")
        hyd = sched["hydration_guidance"]
        self.assertFalse(hyd["is_individualized_medical_prescription"])
        # Does NOT prescribe universal hard-coded volume like "Drink exactly 2.5 litres/hour"
        self.assertNotIn("Drink exactly", hyd["drinking_pattern"])
        self.assertIn("continuous access to cool potable water", hyd["general_principle"].lower())
        self.assertIn("electrolyte", hyd["electrolytes"].lower())


class TestUnifiedAssessFullRiskProfile(unittest.TestCase):
    """Verify assess_full_risk_profile coordinates all pipeline steps and returns primary + legacy schema."""

    def test_full_pipeline_output_structure(self):
        weather = {"temp_c": 38.0, "rh_pct": 55.0, "wind_ms": 2.0, "solar_wm2": 700.0}
        personal = {
            "workload": "heavy",
            "clothing": "light",
            "experience": "very_used",
            "age_group": "18–40",
            "age": 28,
        }
        res = assess_full_risk_profile(
            weather_data=weather,
            personal_inputs=personal,
            region="plains",
            location="Delhi",
            is_forecast=False
        )

        # 1. Primary Redesigned Categorical & Explainable Schema (Phase 6)
        self.assertIn("environmental_level", res)
        self.assertIn("operational_exposure_level", res)
        self.assertIn("overall_action_level", res)
        self.assertIn("drivers", res)
        self.assertIn("protective_factors", res)
        self.assertIn("recommended_actions", res)
        self.assertIn("confidence", res)
        self.assertIn("acclimatization_context", res)
        self.assertIn("work_rest_guidance", res)
        self.assertIn("hydration_guidance", res)
        self.assertTrue(res["is_decision_support"])

        # 2. Legacy Backward-Compatible Schema
        self.assertIn("risk_category", res)
        self.assertIn("base_risk", res)
        self.assertIn("personal_risk", res)
        self.assertIn("wbgt_c", res)
        self.assertIn("wbgt_band", res)
        self.assertIn("thermal_stress", res)
        self.assertIn("thermal_stress_index", res)
        self.assertIn("thermal_stress_level", res)
        self.assertIn("work_minutes", res)
        self.assertIn("rest_minutes", res)
        self.assertIn("instruction", res)
        self.assertIn("summary", res)


class TestAPIRouteIntegration(unittest.TestCase):
    """Verify live API routes /api/check-risk and /api/forecast conform to new model with backward compatibility."""

    def setUp(self):
        self.client = app.test_client()

    def test_api_check_risk_full_schema(self):
        payload = {
            "location": "Hyderabad",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "age_group": "18–40",
            "experience": "somewhat_used",
            "work_intensity": "heavy",
            "clothing": "light"
        }
        res = self.client.post("/api/check-risk", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Check primary Phase 6 fields
        self.assertIn("environmental_level", data)
        self.assertIn("operational_exposure_level", data)
        self.assertIn("overall_action_level", data)
        self.assertIn("drivers", data)
        self.assertIn("confidence", data)
        self.assertIn("acclimatization_context", data)
        self.assertIn("hydration_guidance", data)

        # Check backward compatibility fields
        self.assertIn("risk_category", data)
        self.assertIn("wbgt_c", data)
        self.assertIn("work_minutes", data)
        self.assertIn("rest_minutes", data)
        self.assertIn("summary", data)

    def test_api_forecast_enriched_days(self):
        res = self.client.get("/api/forecast?location=Hyderabad")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 5)
        day1 = data[0]

        # Check that each forecast day has both primary structured fields and legacy fields
        self.assertIn("environmental_level", day1)
        self.assertIn("overall_action_level", day1)
        self.assertIn("confidence", day1)
        self.assertIn("risk_category", day1)
        self.assertIn("max_temp_c", day1)
        self.assertIn("peak_wbgt_c" if "peak_wbgt_c" in day1 else "wbgt_c", day1)


if __name__ == "__main__":
    unittest.main()
