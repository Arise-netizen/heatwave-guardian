"""
Unit tests for HeatWave Guardian Risk Engine
Verifies:
- WBGT calculation (Safe vs Danger cases)
- WBGT band classification
- IMD heatwave classification across regions
- Fusion matrix lookup
- Vulnerability assessment and escalation
- Work/rest schedule
"""

import unittest
from engine import (
    estimate_wbgt,
    classify_wbgt_band,
    classify_imd,
    fuse_risk,
    assess_vulnerability,
    escalate_risk,
    get_work_rest,
    generate_summary,
    FUSION_MATRIX,
    RISK_LEVELS,
)


class TestRiskEngine(unittest.TestCase):

    def test_wbgt_estimation_safe_case(self):
        """Clearly Safe case: cool day, low humidity, moderate wind, no sun."""
        wbgt = estimate_wbgt(temp_c=22.0, rh_pct=40.0, wind_ms=3.0, solar_wm2=0.0)
        self.assertLess(wbgt, 27.0)
        self.assertEqual(classify_wbgt_band(wbgt), "Safe")

    def test_wbgt_estimation_danger_case(self):
        """Clearly Danger/Extreme Danger case: very hot, humid, low wind, intense solar."""
        wbgt = estimate_wbgt(temp_c=44.0, rh_pct=65.0, wind_ms=0.5, solar_wm2=900.0)
        self.assertGreaterEqual(wbgt, 32.2)
        self.assertEqual(classify_wbgt_band(wbgt), "Extreme Danger")

    def test_wbgt_bands(self):
        self.assertEqual(classify_wbgt_band(26.9), "Safe")
        self.assertEqual(classify_wbgt_band(27.0), "Caution")
        self.assertEqual(classify_wbgt_band(29.3), "Caution")
        self.assertEqual(classify_wbgt_band(29.4), "Extreme Caution")
        self.assertEqual(classify_wbgt_band(31.0), "Extreme Caution")
        self.assertEqual(classify_wbgt_band(31.1), "Danger")
        self.assertEqual(classify_wbgt_band(32.1), "Danger")
        self.assertEqual(classify_wbgt_band(32.2), "Extreme Danger")
        self.assertEqual(classify_wbgt_band(38.0), "Extreme Danger")

    def test_imd_classification(self):
        # Plains: threshold 40.0°C
        self.assertEqual(classify_imd(38.0, "plains"), "Normal")
        self.assertEqual(classify_imd(40.0, "plains"), "Heat Day")
        self.assertEqual(classify_imd(41.9, "plains"), "Heat Day")
        self.assertEqual(classify_imd(42.0, "plains"), "Heatwave")
        self.assertEqual(classify_imd(43.9, "plains"), "Heatwave")
        self.assertEqual(classify_imd(44.0, "plains"), "Severe Heatwave")
        self.assertEqual(classify_imd(48.0, "plains"), "Severe Heatwave")

        # Coastal: threshold 37.0°C
        self.assertEqual(classify_imd(36.5, "coastal"), "Normal")
        self.assertEqual(classify_imd(37.5, "coastal"), "Heat Day")
        self.assertEqual(classify_imd(39.0, "coastal"), "Heatwave")
        self.assertEqual(classify_imd(41.5, "coastal"), "Severe Heatwave")

        # Hilly: threshold 30.0°C
        self.assertEqual(classify_imd(29.0, "hilly"), "Normal")
        self.assertEqual(classify_imd(31.0, "hilly"), "Heat Day")
        self.assertEqual(classify_imd(33.0, "hilly"), "Heatwave")
        self.assertEqual(classify_imd(35.0, "hilly"), "Severe Heatwave")

    def test_fusion_matrix_coverage(self):
        # Check key cells per spec
        self.assertEqual(fuse_risk("Safe", "Normal"), "Low")
        self.assertEqual(fuse_risk("Safe", "Severe Heatwave"), "Moderate")
        self.assertEqual(fuse_risk("Caution", "Heatwave"), "Moderate")
        self.assertEqual(fuse_risk("Caution", "Severe Heatwave"), "High")
        self.assertEqual(fuse_risk("Danger", "Severe Heatwave"), "Very High")
        self.assertEqual(fuse_risk("Extreme Danger", "Severe Heatwave"), "Extreme")

        # Verify all 20 cells exist and are valid risk levels
        self.assertEqual(len(FUSION_MATRIX), 20)
        for key, val in FUSION_MATRIX.items():
            self.assertIn(val, RISK_LEVELS)

    def test_vulnerability_assessment(self):
        # High: not acclimatized AND heavy work
        self.assertEqual(assess_vulnerability(age=30, acclimatization="not", workload="heavy"), "High")
        self.assertEqual(assess_vulnerability(age=30, acclimatization="not", workload="very_heavy"), "High")
        # High: vulnerable age (<15 or >=65) AND not fully acclimatized
        self.assertEqual(assess_vulnerability(age=12, acclimatization="partially", workload="light"), "High")
        self.assertEqual(assess_vulnerability(age=70, acclimatization="not", workload="light"), "High")

        # Moderate: vulnerable age only
        self.assertEqual(assess_vulnerability(age=70, acclimatization="fully", workload="light"), "Moderate")
        # Moderate: heavy work only
        self.assertEqual(assess_vulnerability(age=30, acclimatization="fully", workload="heavy"), "Moderate")
        # Moderate: not fully acclimatized only
        self.assertEqual(assess_vulnerability(age=30, acclimatization="partially", workload="light"), "Moderate")

        # Low: standard age (15-64), light/moderate work, fully acclimatized
        self.assertEqual(assess_vulnerability(age=25, acclimatization="fully", workload="moderate"), "Low")
        self.assertEqual(assess_vulnerability(age=50, acclimatization="fully", workload="light"), "Low")

    def test_escalation(self):
        # Low vulnerability -> +0
        self.assertEqual(escalate_risk("Low", "Low"), "Low")
        self.assertEqual(escalate_risk("High", "Low"), "High")

        # Moderate vulnerability -> +1
        self.assertEqual(escalate_risk("Low", "Moderate"), "Moderate")
        self.assertEqual(escalate_risk("High", "Moderate"), "Very High")
        self.assertEqual(escalate_risk("Extreme", "Moderate"), "Extreme")

        # High vulnerability -> +2
        self.assertEqual(escalate_risk("Low", "High"), "High")
        self.assertEqual(escalate_risk("High", "High"), "Extreme")
        self.assertEqual(escalate_risk("Very High", "High"), "Extreme")
        self.assertEqual(escalate_risk("Extreme", "High"), "Extreme")

    def test_work_rest_schedule(self):
        sched_low = get_work_rest("Low")
        self.assertEqual(sched_low["work_minutes"], 60)
        self.assertEqual(sched_low["rest_minutes"], 0)

        sched_mod = get_work_rest("Moderate")
        self.assertEqual(sched_mod["work_minutes"], 45)
        self.assertEqual(sched_mod["rest_minutes"], 15)

        sched_high = get_work_rest("High")
        self.assertEqual(sched_high["work_minutes"], 30)
        self.assertEqual(sched_high["rest_minutes"], 30)

        sched_vh = get_work_rest("Very High")
        self.assertEqual(sched_vh["work_minutes"], 15)
        self.assertEqual(sched_vh["rest_minutes"], 45)

        sched_ext = get_work_rest("Extreme")
        self.assertEqual(sched_ext["work_minutes"], 0)
        self.assertEqual(sched_ext["rest_minutes"], 60)
        self.assertIn("STOP", sched_ext["instruction"])


if __name__ == "__main__":
    unittest.main()
